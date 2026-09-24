"""Least-cost path search for hotspot roots, with a matched null.

WHY THE AXIS FAMILY HAD TO GO

The tilted-axis scan represents a conduit as a straight line through the hotspot
with a prescribed deflection profile. Cross-sections show that this is too rigid:
beneath Hawaii, Kerguelen and Reunion there are large deep low-velocity bodies
that are laterally offset from the surface expression by more than any straight
axis can reach, and simply widening the tilt cap does not help - at 24 degrees
the verdicts are unchanged and at 36 degrees they get worse, because the null
rises as fast as the signal. The geometry model, not the aperture, was wrong.

THE REPLACEMENT

Following the graph formulation of Bao et al. (2026), a conduit is represented as
a least-cost path through the velocity field. A step of physical length L between
cells carries the cost

    w = sqrt(dz^2 + (lam * h)^2) * exp(s * a)

with a the mean shear-velocity anomaly of the two cells in per cent, s a
trade-off parameter, dz the depth gained, h the horizontal distance travelled and
lam a weight on horizontal travel. Slow material is cheap to traverse, so the
cheapest path follows low velocities; the length term penalises wandering, so a
laterally offset root is reached only when the intervening material is slow
enough to pay for the detour.

WHY lam EXISTS

With lam = 1 the length is the true Euclidean step, and following a conduit
leaning at an angle t from vertical costs 1/cos(t) times a vertical descent
through the same material. The detour is paid for only if the conduit is slower
than its surroundings by more than ln(1/cos t)/s: at s = 0.4 a 45 degree lean
needs a contrast of 1.7 per cent and a 60 degree lean needs 3.5 per cent, which
few conduits have. The search therefore abandons a strongly tilted conduit at
the point where the lean steepens and descends through whatever moderately slow
material lies below, which is what it did beneath Afar and Marion. This is a
property of minimising path length, not a defect of the implementation, and no
choice of s repairs it: raising s to follow tilt also makes every weak ambient
anomaly attractive.

lam < 1 discounts horizontal travel so that a tilted conduit can be followed
without making horizontal wandering free. It is a free parameter and is
calibrated against conduits of known tilt by tilt_recovery.py, not chosen by
eye. lam = 1 reproduces the earlier behaviour exactly.
Paths must descend monotonically, which is what separates a conduit from a tour
of the low-velocity network, and are otherwise free to move laterally.

Lengths are physical throughout. A step down costs the shell separation; a step
sideways costs the true cell width, which is the meridional spacing for a
latitude step and the parallel spacing, narrowing as cos(latitude), for a
longitude step. An earlier version allowed the descent step to displace the path
by one cell for the price of the vertical separation alone, which at one degree
made 111 km of lateral travel cost 30 km; that made every root reachable and the
geometry meaningless.

WHY A MOVE MAY SPAN SEVERAL SHELLS

Moving down one shell or sideways one cell cannot represent an intermediate
angle. With 10 km shells and 55 km cells a single move is either vertical or
within ten degrees of horizontal, so a conduit leaning at forty-five degrees has
to be built as a staircase - and a staircase costs the sum of its risers and its
treads, root-two times the distance it actually spans, however each step is
priced. That surcharge is worth about half a per cent of shear velocity, which is
most of the range real conduits occupy, and it does not fall evenly: a lean high
in the mantle is still followed because the slow column beneath it repays the
surcharge, while the same lean low down is abandoned because there is no column
left to repay anything. Beneath Hawaii the offset below 2000 km was invisible for
this reason, and a synthetic conduit of -2 per cent leaning over 2000 to 2600 km
was recovered not at all, against complete recovery of the identical conduit
leaning over 200 to 1300 km.

A move may therefore descend `dk` shells while stepping one cell sideways, and is
charged the true length of that segment. The set of `dk` is geometric rather than
exhaustive, which covers the range of angles at a fraction of the cost of a full
stencil, and the search composes them for anything in between.

WHAT MAKES IT AFFORDABLE

A separate search from every hotspot and every null location would be far too
slow. Instead the search is run once per configuration in reverse: a multi-source
sweep from the deep target region upward, by min-plus dynamic programming over
depth shells with in-shell relaxation. The result is a field C(z, lat, lon)
giving, for every cell, the cost of the cheapest descending path from that cell
to the deep mantle. Every hotspot and every null location is then a lookup into
that field, so the null costs nothing extra and can be large.

Because C is the exact cost to go, the optimal path itself is recovered by
greedy descent on C from the seed, with no need to store predecessors.

Lower cost means better connected. A hotspot counts as rooted when its cost is
below the 5th percentile of the costs at random locations under the identical
configuration - the same null logic as the earlier method, with the inequality
reversed.
"""
from __future__ import annotations

import numpy as np

R_E = 6371.0
DEG = np.pi / 180.0


def cell_widths(lat, lon):
    """Physical width of one grid cell, in km, for a latitude and a longitude step."""
    dlat = abs(float(lat[1] - lat[0])) * DEG * R_E
    coslat = np.clip(np.cos(np.asarray(lat, float) * DEG), 0.02, None)
    dlon = abs(float(lon[1] - lon[0])) * DEG * R_E * coslat      # varies with latitude
    return dlat, dlon[:, None] * np.ones((1, len(lon)))


# Vertical spans a single slanted move may cover, in kilometres. They are given
# as distances, not as counts of shells, because a count means a different move
# in every model: at SPiRaL's 61 km sampling sixteen shells is 976 km, against
# 160 km in RevealLO, and the search was then a different search in each model.
SLANT_KM = (20.0, 60.0, 160.0)
# Lateral offsets a slanted move may take. Only the four axis directions: adding
# the four diagonals doubles the work and changes nothing, because a diagonal is
# reached by composing two axis moves or by the in-shell relaxation, and both are
# already priced. Measured on injected conduits - shallow, ponded and deep leans
# all recover identically with four offsets and with eight, at 1.9 times the speed.
_OFF = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _offsets(ncell):
    """Lateral offsets for a slanted move, out to `ncell` cells.

    One cell was the whole move set. With a minimum vertical span of one shell, that
    caps the steepest move the search can express at about 64 degrees in this model,
    and a conduit leaning more steeply than its steepest available move cannot be
    followed at any price. Reaching two or three cells per shell raises the cap to
    about 76 and 81 degrees. The move is still charged its true length, so this buys
    reach rather than cheapness: nothing here makes wandering less expensive.
    """
    out = []
    for c in range(1, max(1, int(ncell)) + 1):
        out += [(c, 0), (-c, 0), (0, c), (0, -c)]
    return tuple(out)


def _shift2(A, dj, di, out=None):
    """Shift by (dj, di) into `out`. Longitude wraps; latitude does not, so rows
    that would wrap around the pole are marked unreachable instead.

    Written as slice copies into a caller-supplied buffer rather than np.roll,
    which allocates a fresh array on every call. The search makes a few dozen of
    these per depth shell and the allocations, not the arithmetic, dominated.
    """
    if out is None:
        out = np.empty_like(A)
    src = A
    if di:
        n = A.shape[1]
        src = np.empty_like(A)
        if di > 0:                       # element j moves to j + di, wrapping
            src[:, di:] = A[:, :n - di]
            src[:, :di] = A[:, n - di:]
        else:
            src[:, :n + di] = A[:, -di:]
            src[:, n + di:] = A[:, :-di]
    if not dj:
        out[...] = src
        return out
    if dj > 0:
        out[:dj] = np.inf
        out[dj:] = src[:-dj]
    else:
        out[dj:] = np.inf
        out[:dj] = src[-dj:]
    return out


def _relax(cur, wlat, wlon):
    """One min-plus relaxation sweep within a shell.

    Latitude and longitude steps are priced separately because a longitude cell
    is narrower than a latitude cell everywhere off the equator. Longitude wraps;
    latitude does not.
    """
    out = cur
    for sh in (1, -1):                                   # meridional
        m = np.roll(cur, sh, axis=0) + wlat
        if sh == 1:
            m[0] = np.inf
        else:
            m[-1] = np.inf
        out = np.minimum(out, m)
    for sh in (1, -1):                                   # zonal, periodic
        out = np.minimum(out, np.roll(cur, sh, axis=1) + wlon)
    return out


def channel_field(anom, contrast, channel='min'):
    """The anomaly the cost is built on, with the two-channel rule carried over.

    'contrast' finds narrow conduits that a lateral mean cannot resolve;
    'anom' finds conduits embedded in a slow province, where the local contrast
    against their surroundings is small. 'min' takes whichever is more
    favourable at each cell, as in the earlier method.
    """
    if channel == 'anom':
        a = anom
    elif channel == 'contrast':
        a = contrast
    else:
        a = np.minimum(anom, contrast)
    return np.where(np.isfinite(a), a, 0.0).astype(np.float64)


def _spans(z, k0, slant_km):
    """Shell counts matching the requested vertical spans for this model."""
    dz = float(np.median(np.diff(z[:k0 + 1]))) if k0 > 0 else 1.0
    out = sorted({max(1, int(round(v / dz))) for v in slant_km})
    return tuple(d for d in out if d <= k0)


def _descent_shell(C, a, z, k, k0, dk_set, dlat, dlon, s, lateral, ncell,
                   g_of=None, b1=None, b2=None):
    """Cost to go from every cell of shell k WITHOUT ponding: descend now.

    This is the field's `cur` before the within-shell relaxation, and it is what a
    tracer needs to choose a lateral run. C[k] already contains up to the whole
    ponding budget, so a tracer that steps sideways and then reads C[k] at the new
    cell has granted itself the budget twice; that is the disagreement that put six
    traced paths outside their own corridors.

    One implementation, used by the field and by the tracer, so a descent cannot be
    priced two ways.
    """
    if b1 is None:
        b1 = np.empty_like(C[0])
    if b2 is None:
        b2 = np.empty_like(C[0])
    if g_of is None:
        def g_of(k_, dk_):
            return np.exp(s * a[k_:k_ + dk_ + 1].mean(axis=0))

    dz = float(z[k + 1] - z[k])
    e1 = np.exp(s * 0.5 * (a[k] + a[k + 1]))
    cur = C[k + 1] + dz * e1                                # straight down
    for dk in dk_set:                                       # slanted, true length
        kk = k + dk
        if kk > k0:
            continue
        g = g_of(k, dk)
        dzk = float(z[kk] - z[k])
        below = C[kk]
        for dj, di in _offsets(ncell):
            L = np.hypot(dzk, lateral * np.hypot(dj * dlat, di * dlon))
            nc = max(abs(dj), abs(di))
            if nc <= 1:
                _shift2(g, dj, di, out=b1)
                np.multiply(b1, g, out=b1)
                np.sqrt(b1, out=b1)
            else:
                b1[...] = 0.0
                for t in range(nc + 1):
                    _shift2(g, dj * t // nc, di * t // nc, out=b2)
                    np.add(b1, np.log(b2, out=b2), out=b1)
                np.multiply(b1, 1.0 / (nc + 1), out=b1)
                np.exp(b1, out=b1)
            np.multiply(b1, L, out=b1)
            _shift2(below, dj, di, out=b2)
            np.add(b1, b2, out=b1)
            np.minimum(cur, b1, out=cur)
    return cur


def ponding_stages(cur0, a, k, dlat, dlon, s, lateral, budget):
    """The field's value at shell k for each remaining ponding budget, 0 to budget.

    stages[b] is what a path standing in shell k with b within-shell steps still
    allowed may expect to pay. stages[0] is descend-now; stages[budget] is C[k].
    Built by the same running minimum the field uses, so the two agree by
    construction rather than by inspection.
    """
    e_k = np.exp(s * a[k])
    wlat, wlon = lateral * dlat * e_k, lateral * dlon * e_k
    stages = [cur0]
    for _ in range(int(budget)):
        stages.append(np.minimum(stages[-1], _relax(stages[-1], wlat, wlon)))
    return stages


def _n_horiz(h_max_km, dlat, n_relax):
    """Relaxation sweeps matching a horizontal reach given in kilometres.

    Expressed as a distance rather than a count of cells, because a count is a
    different reach in every model: six cells is 334 km at half a degree and
    667 km at one. The reach bounds how far a path may run sideways at a single
    depth, which is what separates a conduit that ponds from a tour of the
    low-velocity network - the failure that made percolation useless here.
    """
    if h_max_km is None:
        return n_relax
    return max(1, int(round(float(h_max_km) / float(dlat))))


def cost_field(anom, contrast, depth, lat, lon, s=0.5, z_target=2400.0,
               n_relax=4, channel='min', a_clip=6.0, slant_km=SLANT_KM,
               h_max_km=None, lateral=1.0, ncell=1, forbid=None):
    """Cheapest descending-path cost from every cell to the deep target region.

    `forbid` is an optional boolean array, the shape of the anomaly, marking cells
    no route may pass through. It exists to answer one question: what is the
    cheapest path whose WORST material is no worse than a given level? Forbidding
    everything above that level and solving the ordinary problem answers it
    exactly, which turns a two-term objective - cost plus a penalty on the worst
    point - into a small family of ordinary searches whose results can be combined
    afterwards. Nothing else in the workflow passes it, so the default behaviour is
    unchanged.

    n_relax bounds the lateral displacement admitted within a single shell, in
    cells. It is not a bound on the displacement of the path, which accumulates
    over shells; it only forbids a path from stepping sideways across half the
    globe between two adjacent depths.
    """
    z = np.asarray(depth, float)
    a = np.clip(channel_field(anom, contrast, channel), -a_clip, a_clip)

    dlat, dlon = cell_widths(lat, lon)

    BAN = None if forbid is None else np.asarray(forbid, bool)
    C = np.full(a.shape, np.inf)
    kdeep = np.where(z >= z_target)[0]
    if not len(kdeep):
        kdeep = np.array([len(z) - 1])
    k0 = int(kdeep[0])
    C[k0] = 0.0                                  # the target surface, cost zero
    if BAN is not None:
        C[k0][BAN[k0]] = np.inf

    dk_set = _spans(z, k0, slant_km)
    n_relax = _n_horiz(h_max_km, dlat, n_relax)

    # A move spanning several shells must be charged for the material it passes
    # through, not merely for its endpoints. Pricing on the two ends alone lets a
    # path jump a fast barrier for nothing: a 200 km lid of +4 per cent raised the
    # cost of crossing it by 32 per cent at 10 km sampling and by 5 per cent at
    # 30 km, which is not a property of the mantle but of the move set. The
    # anomaly of a segment is therefore the mean over the shells it spans, held as
    # a running window so it costs one add and one subtract per shell rather than
    # a fresh sum, and exponentiated once per span rather than once per move.
    win = {}                                    # dk -> running sum over [k, k+dk]
    b1, b2 = np.empty_like(C[0]), np.empty_like(C[0])   # scratch, reused

    def g_of(k, dk):
        kk = k + dk
        w = win.get(dk)
        if w is None:
            w = a[k:kk + 1].sum(axis=0)
        else:
            w = w + a[k] - a[kk + 1]
        win[dk] = w
        return np.exp(s * w / (dk + 1))

    def descent_shell(k, b1, b2):
        return _descent_shell(C, a, z, k, k0, dk_set, dlat, dlon, s, lateral,
                              ncell, g_of=g_of, b1=b1, b2=b2)

    def _unused_descent_shell(k, b1, b2):
        """Cost to go from every cell of shell k WITHOUT ponding: descend now.

        This is `cur` before the within-shell relaxation. The tracer needs it to
        choose a lateral run, because C[k] already contains up to the full budget
        of ponding and reading it after a lateral step grants the budget twice.
        It is defined here, beside the field, so the two cannot price a descent
        differently - which is how the tracer and the field came to disagree.
        """
        dz = float(z[k + 1] - z[k])
        e1 = np.exp(s * 0.5 * (a[k] + a[k + 1]))
        cur = C[k + 1] + dz * e1                            # straight down
        for dk in dk_set:                                   # slanted, true length
            kk = k + dk
            if kk > k0:
                continue
            g = g_of(k, dk)
            dzk = float(z[kk] - z[k])
            below = C[kk]
            for dj, di in _offsets(ncell):
                L = np.hypot(dzk, lateral * np.hypot(dj * dlat, di * dlon))
                # A move crossing several cells is charged for the material it crosses,
                # not for its two ends. Pricing a three-cell step on its endpoints alone
                # would let the path hop a fast barrier for nothing, which is the same
                # error the vertical spans were fixed for.
                nc = max(abs(dj), abs(di))
                if nc <= 1:
                    _shift2(g, dj, di, out=b1)
                    np.multiply(b1, g, out=b1)
                    np.sqrt(b1, out=b1)
                else:
                    b1[...] = 0.0
                    for t in range(nc + 1):
                        _shift2(g, dj * t // nc, di * t // nc, out=b2)
                        np.add(b1, np.log(b2, out=b2), out=b1)
                    np.multiply(b1, 1.0 / (nc + 1), out=b1)
                    np.exp(b1, out=b1)
                np.multiply(b1, L, out=b1)
                _shift2(below, dj, di, out=b2)
                np.add(b1, b2, out=b1)
                np.minimum(cur, b1, out=cur)
        return cur

    for k in range(k0 - 1, -1, -1):
        cur = descent_shell(k, b1, b2)
        if BAN is not None:
            cur[BAN[k]] = np.inf
        e_k = np.exp(s * a[k])                              # horizontal ponding
        for _ in range(n_relax):
            # ponding within a shell is the same physical move as the horizontal
            # part of a slanted step and is discounted identically; charging it in
            # full while discounting the slanted step would buy tilt by making a
            # staircase of vertical drops and full-price sidesteps cheaper instead
            cur = np.minimum(cur, _relax(cur, lateral * dlat * e_k,
                                         lateral * dlon * e_k))
            # re-applied every sweep: a relaxation writes into its neighbours, so
            # masking once would let a banned cell pick up a finite value and be
            # relayed through on the next sweep
            if BAN is not None:
                cur[BAN[k]] = np.inf
        C[k] = cur
    return C


def site_cost(C, depth, lat, lon, hlat, hlon, radius_deg=2.0, seed_depth=200.0):
    """Cheapest path cost from the neighbourhood of a location at seed depth.

    A cap rather than a single cell, because a hotspot's surface position and the
    top of its conduit need not coincide to within one grid cell.
    """
    j, i = _cap_indices(depth, lat, lon, hlat, hlon, radius_deg, seed_depth)
    k = int(np.argmin(np.abs(np.asarray(depth, float) - seed_depth)))
    v = C[k][np.ix_(j, i)]
    v = v[np.isfinite(v)]
    return float(v.min()) if v.size else np.nan


def _cap_indices(depth, lat, lon, hlat, hlon, radius_deg, seed_depth):
    dj = max(1, int(round(radius_deg / abs(float(lat[1] - lat[0])))))
    jlat = int(np.argmin(np.abs(np.asarray(lat, float) - hlat)))
    jj = np.arange(max(0, jlat - dj), min(len(lat), jlat + dj + 1))
    cl = max(np.cos(hlat * DEG), 0.05)
    di = max(1, int(round(radius_deg / abs(float(lon[1] - lon[0])) / cl)))
    hlon = ((hlon + 180.0) % 360.0) - 180.0
    ilon = int(np.argmin(np.abs(np.asarray(lon, float) - hlon)))
    ii = np.arange(ilon - di, ilon + di + 1) % len(lon)
    return jj, ii


def trace_path(C, anom, contrast, depth, lat, lon, hlat, hlon, s=0.5,
               radius_deg=2.0, seed_depth=200.0, n_relax=4, channel='min',
               a_clip=6.0, slant_km=SLANT_KM, h_max_km=None, lateral=1.0, ncell=1,
               cache=None):
    """Recover the optimal path by greedy descent on the cost-to-go field.

    Returns depth, latitude and longitude of the path, in km and degrees. Because
    C is the exact cost to go, the move that attains C at each cell is on an
    optimal path, so no predecessor array is needed.

    `cache` is a dict the caller may hand in and reuse across sites. The descent
    values for a shell depend only on C, so they are the same for every path traced
    through one model; recomputing them per site made a trace cost about as much as
    the cost field itself. Pass one dict per model and the whole set is built once.
    Its memory is one array the size of C.
    """
    if cache is None:
        cache = {}
    z = np.asarray(depth, float)
    a = np.clip(channel_field(anom, contrast, channel), -a_clip, a_clip)
    dlat, dlon = cell_widths(lat, lon)
    nlat, nlon_ = len(lat), len(lon)
    dk_set = _spans(z, len(z) - 1, slant_km)

    jj, ii = _cap_indices(depth, lat, lon, hlat, hlon, radius_deg, seed_depth)
    k = int(np.argmin(np.abs(z - seed_depth)))
    block = C[k][np.ix_(jj, ii)]
    if not np.isfinite(block).any():
        return None
    b = int(np.nanargmin(np.where(np.isfinite(block), block, np.inf)))
    j, i = int(jj[b // len(ii)]), int(ii[b % len(ii)])

    out = [(z[k], lat[j], lon[i])]
    # Ponding budget, in steps, exactly as the field counts it. Tracking it in
    # kilometres was not the rule the field applied, and reading C[k] after a
    # sideways step granted the budget a second time, because C[k] already holds
    # up to the whole of it. The tracer therefore walked routes the field had
    # never priced and, for six hotspots, finished outside its own corridor. It
    # now descends on the staged values: with b steps left, the value is
    # stages[b], and stages[R] is C[k] itself.
    R = _n_horiz(h_max_km, dlat, n_relax)
    budget = R
    stages, stages_k = None, None
    while True:
        if not np.isfinite(C[k, j, i]) or C[k, j, i] <= 0.0 or k + 1 >= len(z):
            break
        if stages_k != k:
            cur0 = cache.get(k)
            if cur0 is None:
                cur0 = _descent_shell(C, a, z, k, len(z) - 1, dk_set, dlat, dlon,
                                      s, lateral, ncell)
                cache[k] = cur0
            stages = ponding_stages(cur0, a, k, dlat, dlon, s, lateral, R)
            stages_k = k
        dz = float(z[k + 1] - z[k])
        e_in = float(np.exp(s * a[k, j, i]))
        best, mv = np.inf, None
        # descend
        cost = dz * float(np.exp(s * 0.5 * (a[k, j, i] + a[k + 1, j, i]))) + C[k + 1, j, i]
        if cost < best:
            best, mv = cost, ('d', k + 1, j, i)
        # slanted descent, over the same move set and pricing the field was built
        # from, including the mean anomaly across every shell the move spans
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
                L = float(np.hypot(dzk, lateral * np.hypot(
                    dj * dlat, di * float(dlon[j, i]))))
                # the same averaging along the move that the cost field uses, so the
                # tracer scores a move exactly as the field priced it
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
        # step within the shell, up to the horizontal reach
        if budget > 0:
            for dj_, di_, w in ((1, 0, dlat), (-1, 0, dlat), (0, 1, dlon[j, i]),
                                (0, -1, dlon[j, i])):
                j2, i2 = j + dj_, (i + di_) % nlon_
                if not (0 <= j2 < nlat):
                    continue
                # cost_field discounts a within-shell step by `lateral`, and the
                # value on the far side is the one with a step already spent.
                cost = lateral * float(w) * e_in + stages[budget - 1][j2, i2]
                if cost < best - 1e-9:
                    best, mv = cost, ('l', k, j2, i2, float(w))
        if mv is None:
            break
        if mv[0] == 'l':
            budget -= 1                      # still at this depth
            _, k, j, i, _ = mv
        else:
            budget = R                       # descended: the reach resets
            _, k, j, i = mv
        out.append((z[k], lat[j], lon[i]))
        if len(out) > 20000:
            break
    p = np.array(out, float)
    return p[:, 0], p[:, 1], p[:, 2]
