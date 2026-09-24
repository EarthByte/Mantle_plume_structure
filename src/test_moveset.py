#!/usr/bin/env python3
"""The move set is threaded consistently, and it changes nothing at the old setting.

    python3 test_moveset.py

Two things can go quietly wrong when a search is given a wider move set, and both
would be invisible in the output.

The corridor is built from two fields: C, the cost from every cell down to the
target, and D, the cost from the hotspot seed out to every cell. They are
separate pieces of code that must price one physical move identically, or
dC = D + C - C_best is not an excess cost and the corridor widths mean nothing.
Multi-cell moves are charged for the material they cross rather than for their
two ends, and the two fields walk that material from opposite ends, so this is
exactly where they could drift apart. self_check is the statement that they have
not: the minimum of D + C must equal the cost C reports at the seed.

The second is a regression: at lateral 1.0 with one cell the new code must
reproduce the old bit for bit, or the published run is no longer reproducible.

The classify smoke then runs the classification end to end on a toy field, with
the sweep shrunk to seconds, and checks that both flags reach every cost field -
including inside the forked injection workers, where a missing argument would
otherwise surface hours into a real run - and that they are recorded in the
output, so no classification file can be mistaken for one from another move set.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OLD_CORRIDOR = 'to_delete/corridor.py.before_move_set'


def toy_field(nz=40, nlat=41, nlon=61, seed=3):
    """A smooth field with one strongly tilted slow conduit through it."""
    rng = np.random.default_rng(seed)
    depth = np.linspace(0.0, 2800.0, nz)
    lat = np.linspace(-20.0, 20.0, nlat)
    lon = np.linspace(-30.0, 30.0, nlon)
    a = rng.normal(0.0, 0.4, (nz, nlat, nlon))
    for _ in range(3):
        a = 0.25 * (np.roll(a, 1, 1) + np.roll(a, -1, 1) +
                    np.roll(a, 1, 2) + np.roll(a, -1, 2))
    for k in range(nz):
        j = int(nlat // 2 + 8 * k / nz)          # leans north with depth
        i = int(nlon // 2 + 14 * k / nz)         # and east
        a[k, max(j - 1, 0):j + 2, max(i - 1, 0):i + 2] -= 2.0
    return depth, lat, lon, a


def test_corridor():
    from path_cost import cost_field, site_cost
    from contrast import contrast_field
    import corridor as new

    depth, lat, lon, a = toy_field()
    con = contrast_field(a, lat, lon, 6.0)
    kw = dict(s=0.4, z_target=2400.0, n_relax=4, channel='anom', h_max_km=900.0)
    hla, hlo = float(lat[len(lat) // 2]), float(lon[len(lon) // 2])

    if os.path.exists(OLD_CORRIDOR):
        from importlib.machinery import SourceFileLoader
        loader = SourceFileLoader('corridor_old', OLD_CORRIDOR)
        spec = importlib.util.spec_from_loader('corridor_old', loader)
        old = importlib.util.module_from_spec(spec)
        loader.exec_module(old)
        Dn = new.forward_field(a, con, depth, lat, lon, hla, hlo, radius_deg=3.0,
                               seed_depth=200.0, lateral=1.0, ncell=1, **kw)
        Do = old.forward_field(a, con, depth, lat, lon, hla, hlo, radius_deg=3.0,
                               seed_depth=200.0, **kw)
        same = np.array_equal(np.nan_to_num(Dn, nan=-1, posinf=-2),
                              np.nan_to_num(Do, nan=-1, posinf=-2))
        print(f'  published setting reproduced bit for bit: {same}')
        assert same, 'lateral 1.0 ncell 1 changed - a regression, not a fix'
    else:
        print(f'  {OLD_CORRIDOR} absent; skipping the bit-for-bit check')

    z = np.asarray(depth, float)
    k0 = int(np.where(z >= 2400.0)[0][0])
    k_seed = int(np.argmin(np.abs(z - 200.0)))
    for lat_w, nc in ((1.0, 1), (0.60, 2), (0.60, 3), (0.80, 3)):
        C = cost_field(a, con, depth, lat, lon, lateral=lat_w, ncell=nc, **kw)
        D = new.forward_field(a, con, depth, lat, lon, hla, hlo, radius_deg=3.0,
                              seed_depth=200.0, lateral=lat_w, ncell=nc, **kw)
        cost = float(site_cost(C, depth, lat, lon, hla, hlo, 3.0, 200.0))
        _, cbest = new.excess_cost(D, C, k_seed, k0)
        ok, _ = new.self_check(D, C, k_seed, k0, c_best=cost)
        print(f'  lateral {lat_w:.2f} ncell {nc}: self-check '
              f'{"PASS" if ok else "FAIL"}  C_best {cbest:.4f} vs seed {cost:.4f}')
        assert ok, f'forward and backward fields disagree at lateral {lat_w}, ncell {nc}'


def test_classify_smoke():
    import pandas as pd

    out_dir = os.path.join('out', '_moveset_smoke')
    os.makedirs(out_dir, exist_ok=True)
    argv = sys.argv
    sys.argv = ['classify.py', '--file', 'toy.nc', '--tag', 'TOY', '--var', 'vs',
                '--out', out_dir, '--jobs', '1',
                '--lateral', '0.60', '--ncell', '2', '--suffix', '_smoke']
    try:
        import classify
        import path_cost

        nz, nlat, nlon = 24, 37, 73
        depth = np.linspace(0.0, 2880.0, nz)
        lat = np.linspace(-90.0, 90.0, nlat)
        lon = np.linspace(-180.0, 177.5, nlon)
        arr = np.random.default_rng(11).normal(0.0, 0.5, (nz, nlat, nlon))
        classify.load_anomaly = lambda *a_, **k_: (depth, lat, lon, arr)
        classify.dedupe_lon = lambda lo, ar: (lo, ar)
        classify.check_shells = lambda *a_, **k_: None

        classify.S_VALUES = (0.4,)
        classify.Z_TARGET = (2700.0,)
        classify.H_MAX = (900.0,)
        classify.RADII = (3.0,)
        classify.AMPS = (-1.0, -1.5)
        classify.SITES = classify.SITES[:1]
        classify.N_NULL = 20

        seen, real = [], path_cost.cost_field

        def spy(*a_, **k_):
            seen.append((k_.get('lateral'), k_.get('ncell')))
            return real(*a_, **k_)

        classify.cost_field = spy
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            classify.main()
        classify.cost_field = real
    finally:
        sys.argv = argv

    print(f'  cost_field calls: {len(seen)}; move sets seen: {sorted(set(seen))}')
    assert seen, 'cost_field was never called'
    assert set(seen) == {(0.60, 2)}, f'a cost field used the wrong move set: {set(seen)}'
    d = pd.read_csv(os.path.join(out_dir, 'classification_TOY_smoke.csv'))
    assert (d.lateral == 0.60).all() and (d.ncell == 2).all(), \
        'the classification does not record the move set it was computed under'
    print('  the classification records the move set it was computed under')


def toy_ponding(nz=40, nlat=41, nlon=81, seed=5):
    """A model whose cheapest route has to run sideways within one shell.

    Two vertical slow columns, offset in longitude, joined by a slow layer two
    shells thick. Nothing slow connects them diagonally, and a slanted move spans
    only so much depth, so the least-cost route must pond along the layer. A toy
    without that property cannot see a mispriced within-shell step at all: the
    first version of this test passed with the bug deliberately reintroduced.
    """
    rng = np.random.default_rng(seed)
    depth = np.linspace(0.0, 2800.0, nz)
    lat = np.linspace(-20.0, 20.0, nlat)
    lon = np.linspace(-40.0, 40.0, nlon)
    a = rng.normal(0.0, 0.15, (nz, nlat, nlon))
    for _ in range(3):
        a = 0.25 * (np.roll(a, 1, 1) + np.roll(a, -1, 1) +
                    np.roll(a, 1, 2) + np.roll(a, -1, 2))
    j = nlat // 2
    i_up, i_dn = nlon // 2 - 18, nlon // 2 + 18
    k_join = nz // 2
    a[k_join + 2:, :, :] += 2.5                              # fast below, so that
    a[:k_join, j - 1:j + 2, i_up - 1:i_up + 2] -= 3.0        # upper column
    a[k_join + 2:, j - 1:j + 2, i_dn - 1:i_dn + 2] -= 5.5    # lower column
    a[k_join:k_join + 2, j - 1:j + 2, i_up:i_dn + 1] -= 3.0  # the layer between
    # descending straight from the upper column is dear and the only cheap way on
    # is along the layer, which no slanted move can follow: two shells thick and
    # 36 cells long, against at most `ncell` cells of lateral travel per shell
    # descended. The route has to pond.
    return depth, lat, lon, a


def test_path_in_its_own_corridor():
    """The traced path must be the route the field found, not a dearer one.

    A greedy descent on the cost-to-go field is optimal only if the tracer prices
    every move exactly as the field does. Three implementations priced a
    within-shell step at full width while cost_field discounted it by `lateral`, so
    the tracer refused ponding the field had already paid for and the forward field
    charged it twice over. Six of 49 paths came out dearer than the field's own
    optimum, and their corridors collapsed to the two or three shells nearest the
    seed. Nothing caught it: the self-check compares the two fields with each other
    and the global optimum ponds little, so min(D + C) still equalled C_best.

    The invariant is one line: the excess cost dC is zero at every node of the
    traced path. It ties the tracer, the forward field and the backward field
    together without a fourth copy of the pricing to drift from.
    """
    from path_cost import cost_field, site_cost, trace_path
    from contrast import contrast_field
    import corridor as new
    import numpy as np

    depth, lat, lon, a = toy_ponding()
    con = contrast_field(a, lat, lon, 800.0)
    LAT, NC, HM, S, ZT = 0.60, 2, 900.0, 0.4, 2400.0
    hla, hlo = 0.0, -18.0

    C = cost_field(a, con, depth, lat, lon, s=S, z_target=ZT, n_relax=6,
                   channel='anom', h_max_km=HM, lateral=LAT, ncell=NC)
    tr = trace_path(C, a, con, depth, lat, lon, hla, hlo, s=S, radius_deg=3.0,
                    n_relax=6, channel='anom', h_max_km=HM, lateral=LAT, ncell=NC)
    assert tr is not None, 'the toy model produced no path'
    zz, la, lo = tr

    z = np.asarray(depth, float)
    k_seed = int(np.argmin(np.abs(z - 200.0)))
    k0 = int(np.where(z >= ZT)[0][0])
    D = new.forward_field(a, con, depth, lat, lon, hla, hlo, s=S, z_target=ZT,
                          radius_deg=3.0, seed_depth=200.0, n_relax=6,
                          channel='anom', h_max_km=HM, lateral=LAT, ncell=NC)
    dC, cbest = new.excess_cost(D, C, k_seed, k0)
    ok, _ = new.self_check(D, C, k_seed, k0,
                           c_best=float(site_cost(C, depth, lat, lon, hla, hlo,
                                                  3.0, 200.0)))
    assert ok, 'the two fields are not halves of one search'

    # A toy whose optimum never ponds cannot see this bug: the first two versions
    # of this test passed with the fault deliberately put back. Fail loudly if the
    # model stops exercising the move the test exists for.
    ponds = int(np.sum(np.diff(np.asarray(zz, float)) == 0))
    assert ponds >= 3, (
        f'this toy no longer exercises within-shell movement ({ponds} such steps), '
        f'so it cannot test the pricing this test exists for')

    ki = np.searchsorted(z, zz).clip(0, len(z) - 1)
    ii = np.abs(np.asarray(lat)[None, :] - la[:, None]).argmin(axis=1)
    ji = np.abs(((np.asarray(lon)[None, :] - lo[:, None] + 180) % 360) - 180
                ).argmin(axis=1)
    # The invariant holds at ARRIVAL nodes, and only there. D prices the cost of
    # descending into a cell with no ponding at its own shell, C prices ponding and
    # then descending, so dC = D + C - C_best describes cells a route can descend
    # through. A cell in the middle of a sideways run is not one, and its dC is
    # legitimately positive: the corridor does not represent ponding at all. That
    # is a property of the measure, not an error in it - storing the ponded values
    # in D instead makes min(D + C) fall below C_best and self_check fail at once.
    zz_a = np.asarray(zz, float)
    arrival = np.r_[True, np.diff(zz_a) != 0.0]
    v = np.asarray([dC[k, i, j] for k, i, j in zip(ki, ii, ji)], float)
    fin = np.isfinite(v)
    va = v[arrival & fin]
    worst = float(va.max()) if len(va) else float('nan')
    tol = 1e-6 * max(cbest, 1.0)
    assert worst <= tol, (
        f'the path leaves its own corridor at a node it DESCENDS into: max dC '
        f'{worst:.6g} against a tolerance of {tol:.3g} over {len(va)} arrival '
        f'nodes. The tracer and the fields are not pricing the same move set.')
    n_pond = int((~arrival & fin).sum())
    print(f'  {len(va)} arrival nodes, worst excess cost {worst:.3g} against '
          f'C_best {cbest:.1f}; {n_pond} ponding nodes outside the corridor by '
          f'construction')


if __name__ == '__main__':
    print('corridor: forward and backward fields price a move identically')
    test_corridor()
    print('\nclassify: the move set reaches every cost field, workers included')
    test_classify_smoke()
    print('\npath: the traced route is the route the field found')
    test_path_in_its_own_corridor()
    print('\nall move-set checks passed')
