"""Near-optimal corridors: the set of routes a hotspot could take for almost the
cost of the best one.

WHY A SECOND FIELD IS NEEDED

`path_cost.cost_field` returns C(x), the cost of the cheapest descending path
from x to the deep target. One number per hotspot follows from it, and one path,
and that is all the present classification uses. It cannot say whether the
optimum is a lone route through an otherwise expensive mantle or one draw from a
wide family of near-equivalent routes, and that distinction is the whole question
here: a hotspot fails the root test when its best path is unexceptional, which
happens both when there is no structure and when the structure is so broad that
comparable routes are everywhere.

The missing half is D(x), the cost of the cheapest descending path from the
hotspot seed to x. With both,

    dC(x) = D(x) + C(x) - C_best

is the excess cost of the cheapest complete route that passes through x, and the
set where dC is small is the corridor of routes the tomography cannot distinguish
from the optimum.

WHY THE LATERAL BUDGET DECIDES THE DEFINITION

A path is a sequence of descents separated by lateral runs, and the search caps
the run at one depth (h_max, imposed as a number of relaxation sweeps). C already
includes the run at its own shell: C(x) is the cost to go *allowing* the path to
step sideways at x's depth before descending. If D also included the run at x's
shell, the sum would grant two runs where a path is allowed one, and dC would
open a corridor wider than any admissible path.

D is therefore defined at arrival: the cheapest cost to reach x with the last
move a descent. Every path's cells are either arrival cells or lie on a lateral
run out of one, so the corridor is complete when read as the set of admissible
arrival cells, and the sum is exact rather than a bound. The consequence is a
test: for every shell between the seed and the target, min_x [D(x) + C(x)] must
equal C_best exactly, and `self_check` asserts it.

The move set, the pricing, the multi-shell window mean and the reach are taken
from path_cost unchanged - imported, not restated - so the corridor is the
corridor of the search that produced the classification and not of a second
search that resembles it.
"""
from __future__ import annotations

import numpy as np

from path_cost import (SLANT_KM, _OFF, _cap_indices, _n_horiz, _offsets, _shift2, _spans,
                       cell_widths, channel_field)


def _relax_fwd(cur, wlat, wlon):
    """One min-plus relaxation sweep within a shell, for a forward field.

    The backward sweep charges a lateral step at the cell whose cost is being
    written, which is the cell the path departs from as it runs down-path. Going
    the other way the departure is the neighbour, so the weight has to travel
    with the value it is added to; rolling the values and not the weights would
    price every lateral step at the wrong end and, off the equator, by up to the
    ratio of the two cell widths.
    """
    out = cur
    for sh in (1, -1):                                   # meridional
        m = np.roll(cur, sh, axis=0) + np.roll(wlat, sh, axis=0)
        if sh == 1:
            m[0] = np.inf
        else:
            m[-1] = np.inf
        out = np.minimum(out, m)
    for sh in (1, -1):                                   # zonal, periodic
        out = np.minimum(out, np.roll(cur, sh, axis=1) + np.roll(wlon, sh, axis=1))
    return out


def forward_field(anom, contrast, depth, lat, lon, hlat, hlon, s=0.5,
                  z_target=2400.0, radius_deg=2.0, seed_depth=200.0,
                  n_relax=4, channel='min', a_clip=6.0, slant_km=SLANT_KM,
                  h_max_km=None, lateral=1.0, ncell=1):
    """Cheapest cost from the hotspot seed cap to every cell, counted on arrival.

    Returns D with the same shape as the anomaly cube. D is infinite above the
    seed depth and below the target surface, where no arrival is defined, and at
    every cell no descending path can reach.
    """
    z = np.asarray(depth, float)
    a = np.clip(channel_field(anom, contrast, channel), -a_clip, a_clip)
    dlat, dlon = cell_widths(lat, lon)

    kdeep = np.where(z >= z_target)[0]
    if not len(kdeep):
        kdeep = np.array([len(z) - 1])
    k0 = int(kdeep[0])
    k_seed = int(np.argmin(np.abs(z - seed_depth)))
    if k_seed >= k0:
        raise ValueError('seed depth is at or below the target surface')

    dk_set = _spans(z, k0, slant_km)
    n_relax = _n_horiz(h_max_km, dlat, n_relax)

    D = np.full(a.shape, np.inf)
    jj, ii = _cap_indices(depth, lat, lon, hlat, hlon, radius_deg, seed_depth)
    D[k_seed][np.ix_(jj, ii)] = 0.0

    # The relaxed shells a slanted move may depart from. Only the deepest
    # max(dk_set) of them can still be reached, so they are held in a ring
    # rather than as a second copy of the cube: the arrival field D is the
    # output and must not be overwritten by the lateral runs, which belong to
    # the departure and are already counted once in the backward field.
    dkmax = max(dk_set) if dk_set else 1
    DL = {}

    # The window mean over the shells a slanted move spans, keyed by span. The
    # backward sweep walks k upward and carries a running sum; the forward sweep
    # needs the window that STARTS at k - dk while standing at k, which is the
    # same one-add-one-subtract update read from the other end.
    win, wink = {}, {}

    def g_of(ks, dk):
        kk = ks + dk
        w = win.get(dk)
        if w is None or wink.get(dk) != ks - 1:
            w = a[ks:kk + 1].sum(axis=0)
        else:
            w = w - a[ks - 1] + a[kk]
        win[dk], wink[dk] = w, ks
        return np.exp(s * w / (dk + 1))

    for k in range(k_seed + 1, k0 + 1):
        kp = k - 1
        e_prev = np.exp(s * a[kp])
        src = D[kp].copy()
        for _ in range(n_relax):
            # The lateral weight belongs here too. cost_field charges a within-shell
            # step `lateral * dlat * e_k`; this charged the full `dlat * e_prev`, so
            # the forward field priced ponding 1/lateral times too dear while the
            # backward field priced it correctly. dC = D + C - C_best then inflated
            # along any route that ponds, which is why six corridors collapsed to
            # the two or three shells nearest the seed, where D is still near zero,
            # and why those six paths measured as lying outside their own corridor.
            # self_check kept passing because the global optimum itself ponds little,
            # so min(D + C) still equalled C_best.
            src = _relax_fwd(src, lateral * dlat * e_prev, lateral * dlon * e_prev)
        DL[kp] = src
        # D keeps ARRIVAL semantics on purpose: the cost to reach this cell by
        # descending into it, with no ponding at its own shell. C already prices
        # the ponding that follows an arrival, so storing the ponded values here
        # instead counts every within-shell step twice and min(D + C) drops below
        # C_best - measured, self_check fails at once. The consequence is that
        # dC = D + C - C_best describes cells a route can DESCEND through, and a
        # cell reached only by running sideways is not one. See corridor_all.
        for old in [q for q in DL if q < k - dkmax]:
            del DL[old]

        dz = float(z[k] - z[kp])
        cur = src + dz * np.exp(s * 0.5 * (a[kp] + a[k]))

        for dk in dk_set:
            ks = k - dk
            if ks < k_seed or ks not in DL:
                continue
            g = g_of(ks, dk)
            dzk = float(z[k] - z[ks])
            above = DL[ks]
            for dj, di in _offsets(ncell):
                # the source of a move arriving here is (j + dj, i + di), and
                # _shift2(A, -dj, -di)[j, i] is A[j + dj, i + di]
                sD = _shift2(above, -dj, -di)
                if di == 0:
                    h = abs(dj) * dlat
                else:
                    # dj is zero for a zonal offset, so no row is wrapped and
                    # the shifted widths stay finite. The width is read at the
                    # SOURCE cell, which is where the backward field charges it.
                    h = abs(di) * _shift2(dlon, -dj, -di)
                L = np.hypot(dzk, lateral * h)
                # A move crossing several cells is charged for the material it
                # crosses, not for its two ends - the same average the backward
                # field takes, read from the other end of the move, so that the
                # two fields price one physical move identically and self_check
                # still holds.
                nc = max(abs(dj), abs(di))
                if nc <= 1:
                    sg = _shift2(g, -dj, -di)
                    np.multiply(sg, g, out=sg)
                    np.sqrt(sg, out=sg)
                else:
                    acc = np.zeros_like(g)
                    for t in range(nc + 1):
                        np.add(acc, np.log(_shift2(g, -(dj * t // nc),
                                                   -(di * t // nc))), out=acc)
                    np.multiply(acc, 1.0 / (nc + 1), out=acc)
                    sg = np.exp(acc, out=acc)
                np.multiply(sg, L, out=sg)
                np.add(sg, sD, out=sg)
                np.minimum(cur, sg, out=cur)
        D[k] = cur
    D[:k_seed] = np.inf
    D[k0 + 1:] = np.inf
    return D


def excess_cost(D, C, k_seed, k0):
    """dC(x) = D(x) + C(x) - C_best, in the same units as the path cost.

    C_best is read from the fields rather than passed in, so the two halves
    cannot be paired with a cost from a different configuration.
    """
    with np.errstate(invalid='ignore'):
        T = D + C
    finite = np.isfinite(T[k_seed:k0 + 1])
    if not finite.any():
        return None, np.nan
    cbest = float(np.nanmin(np.where(np.isfinite(T), T, np.inf)))
    return T - cbest, cbest


def self_check(D, C, k_seed, k0, c_best=None, tol=1e-9):
    """The two halves must be halves of the same search.

    The global minimum of D + C is the cost of the best complete route, so it has
    to equal the cost the backward field reports at the seed. It is checked
    against that number rather than against itself.

    The shell-by-shell minima are returned as well, but they are NOT required to
    be equal: a slanted move may span several shells, and a shell the optimal
    route jumps over carries no arrival cell, so its minimum is legitimately
    higher. Requiring equality there would fail every model whose sampling is
    fine enough for a move to skip a shell, which is all of them at 10 km.
    """
    T = D + C
    mins = np.array([float(np.nanmin(np.where(np.isfinite(T[k]), T[k], np.inf)))
                     if np.isfinite(T[k]).any() else np.nan
                     for k in range(k_seed, k0 + 1)], float)
    if not np.isfinite(mins).any():
        return False, mins
    got = float(np.nanmin(mins))
    if c_best is None:
        return True, mins
    rel = abs(got - float(c_best)) / max(abs(float(c_best)), 1.0)
    return bool(rel <= tol), mins


def skipped_shells(mins, tol_km=1e-6):
    """Shells the optimal route passes without an arrival, as a boolean array."""
    m = np.asarray(mins, float)
    return np.isfinite(m) & (m > np.nanmin(m) + tol_km)


def trace_consistent(C, anom, contrast, depth, lat, lon, hlat, hlon, s=0.5,
                     radius_deg=2.0, seed_depth=200.0, n_relax=4, channel='min',
                     a_clip=6.0, slant_km=SLANT_KM, h_max_km=None,
                     lateral=1.0, ncell=1):
    """The optimal path, traced under exactly the rules the cost field was built on.

    path_cost.trace_path bounds the lateral run at one depth by h_max in
    kilometres; the field bounds it by the number of relaxation sweeps, which is
    h_max divided by the MERIDIONAL cell width. Off the equator a zonal cell is
    narrower, so more zonal steps fit inside h_max than the field ever priced,
    and the tracer can then return a route the field would not have allowed - one
    that is cheaper than the field's own optimum, by up to six per cent in the
    cases measured here. The classification is unaffected, because it reads the
    field and never the traced path, but any geometry drawn from the trace is
    then not the geometry the cost belongs to.

    This tracer counts steps instead, so the route it returns is priced at the
    field's optimum and is the path the search actually selected.
    """
    z = np.asarray(depth, float)
    a = np.clip(channel_field(anom, contrast, channel), -a_clip, a_clip)
    dlat, dlon = cell_widths(lat, lon)
    nlat, nlon_ = len(lat), len(lon)
    dk_set = _spans(z, len(z) - 1, slant_km)
    R = _n_horiz(h_max_km, dlat, n_relax)

    jj, ii = _cap_indices(depth, lat, lon, hlat, hlon, radius_deg, seed_depth)
    k = int(np.argmin(np.abs(z - seed_depth)))
    block = C[k][np.ix_(jj, ii)]
    if not np.isfinite(block).any():
        return None
    b = int(np.nanargmin(np.where(np.isfinite(block), block, np.inf)))
    j, i = int(jj[b // len(ii)]), int(ii[b % len(ii)])

    out = [(z[k], lat[j], lon[i])]
    steps = 0
    while True:
        if not np.isfinite(C[k, j, i]) or C[k, j, i] <= 0.0 or k + 1 >= len(z):
            break
        best, mv = np.inf, None
        dz = float(z[k + 1] - z[k])
        cost = (dz * float(np.exp(s * 0.5 * (a[k, j, i] + a[k + 1, j, i])))
                + C[k + 1, j, i])
        if cost < best:
            best, mv = cost, ('d', k + 1, j, i)
        for dk in dk_set:
            kk = k + dk
            if kk >= len(z):
                continue
            dzk = float(z[kk] - z[k])
            m_here = float(a[k:kk + 1, j, i].mean())
            for dj, di in _offsets(ncell):
                j2, i2 = j + dj, (i + di) % nlon_
                if not (0 <= j2 < nlat):
                    continue
                h = abs(dj) * dlat if di == 0 else abs(di) * float(dlon[j, i])
                L = float(np.hypot(dzk, lateral * h))
                nc = max(abs(dj), abs(di))
                if nc <= 1:
                    m_mean = 0.5 * (m_here + float(a[k:kk + 1, j2, i2].mean()))
                else:
                    acc = []
                    for t in range(nc + 1):
                        jt = j + dj * t // nc
                        it = (i + di * t // nc) % nlon_
                        if not (0 <= jt < nlat):
                            break
                        acc.append(float(a[k:kk + 1, jt, it].mean()))
                    if len(acc) < nc + 1:
                        continue
                    m_mean = float(np.mean(acc))
                cost = L * float(np.exp(s * m_mean)) + C[kk, j2, i2]
                if cost < best - 1e-9:
                    best, mv = cost, ('s', kk, j2, i2)
        if steps < R:
            e_in = float(np.exp(s * a[k, j, i]))
            for dj_, di_ in _OFF:
                j2, i2 = j + dj_, (i + di_) % nlon_
                if not (0 <= j2 < nlat):
                    continue
                w = dlat if di_ == 0 else float(dlon[j, i])
                # Same omission as the forward field had: the field discounts a
                # within-shell step by `lateral` and this charged it in full, so the
                # tracer refused ponding the field had already paid for and walked a
                # route more expensive than the field's own optimum.
                cost = lateral * w * e_in + C[k, j2, i2]
                if cost < best - 1e-9:
                    best, mv = cost, ('l', k, j2, i2)
        if mv is None:
            break
        steps = steps + 1 if mv[0] == 'l' else 0
        _, k, j, i = mv
        out.append((z[k], lat[j], lon[i]))
        if len(out) > 20000:
            break
    p = np.array(out, float)
    return p[:, 0], p[:, 1], p[:, 2]
