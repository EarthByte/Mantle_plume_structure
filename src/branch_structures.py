#!/usr/bin/env python3
"""Does an imaged structure divide as it rises, at what depth, and into how many?

This is built from the question rather than from what happened to be on disk. An
earlier version counted branches of the least-cost CORRIDOR - the set of cells
lying on some descending route within a tolerance of the cheapest one - which is
a route bundle and not an imaged object, priced 99.8 per cent on slowness. Its
bifurcations say that two nearly equally cheap slow paths exist. That is a fact
about the ambient velocity field and cannot answer a question about plumes.

THE OBJECT. A conduit, if it is anything, is a localised low-velocity feature that
is slow relative to its immediate surroundings and continuous in depth. That is a
connected component of the prominence field - the laterally smoothed anomaly minus
the anomaly - which scores zero for anything broader than the smoothing length and
so ignores the provinces. Components are found in three dimensions with longitude
joined, without reference to any hotspot.

BRANCHING, DEFINED ON THAT OBJECT. Within one component, the number of separate
pieces present in each shell. A component that is one piece at 2500 km and two at
1500 km has divided between those depths. Because the component is connected in
three dimensions, the two pieces are the same structure and not two structures,
which is exactly the distinction a branch requires.

THE FLOOR IS MEASURED FOR THIS COUNTER. A resolution floor belongs to the method
that produced it, so the corridor counter's 450 km cannot be carried over. Pairs
of conduits are injected at known separations and counted by the identical code
used on the real field. Prominence is linear in the anomaly - it is a smoothing
minus the field - so the prominence of an injected conduit can be computed once
and added to the background field exactly, which makes the calibration cheap
enough to run at every threshold the measurement uses.

The script refuses to report branch geometry unless the floor has been measured
in the same run or supplied explicitly, because a branch count without one is not
a measurement.
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morphology as M
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from contrast import contrast_field, _smooth_shell
from prominence_map import label_3d_periodic

R_E, DEG = 6371.0, np.pi / 180.0


def pieces_in_shell(mask2d, LA, LO, floor_km, min_cells=4):
    """Separate pieces of one structure in one shell, merged below the floor."""
    lab, n = M._label_periodic(mask2d)
    out = []
    for q in range(1, n + 1):
        m = lab == q
        if m.sum() < min_cells:
            continue
        w = m.astype(float)
        clat, clon, _ = M.weighted_centroid(w, LA, LO)
        out.append(dict(lat=clat, lon=clon, cells=int(m.sum())))
    if len(out) <= 1:
        return out
    merged = []
    for c in sorted(out, key=lambda c: -c['cells']):
        for g in merged:
            if M.gc_km(g['lat'], g['lon'], c['lat'], c['lon']) < floor_km:
                t = g['cells'] + c['cells']
                g['lat'] = (g['lat'] * g['cells'] + c['lat'] * c['cells']) / t
                g['lon'] = (g['lon'] * g['cells'] + c['lon'] * c['cells']) / t
                g['cells'] = t
                break
        else:
            merged.append(dict(c))
    return merged


def prominence_of(g, lat, lon, sigma):
    """Prominence contributed by one horizontal pattern, computed exactly.

    Prominence is a smoothing minus the field and is therefore linear, so the
    contribution of an injected conduit can be added to the background field
    instead of recomputing the whole thing. A vertical conduit has the same
    horizontal shape at every depth, so this is computed once and reused.
    """
    dlat = float(abs(lat[1] - lat[0]))
    dlon = float(abs(lon[1] - lon[0]))
    return _smooth_shell(g, lat, dlat, dlon, sigma) - g


def floor_mode(P0, zz, lat, lon, a):
    """A Y injected into the real field: is the split seen, and at what depth?

    The previous version injected two PARALLEL conduits and asked how far apart
    they had to be before two pieces were counted. That calibrates the wrong
    thing. Two parallel conduits are two structures that never join, and counting
    pieces within one connected component cannot find a split that does not
    exist - which is why the curve was not monotonic, rising to 50 per cent at
    450 km and falling to 32 at 900 as the pair stopped being one component.

    A branch is a single conduit at depth that divides above some level. That is
    what is injected here: one Gaussian below the split depth, two limbs
    diverging linearly above it to a chosen separation at the top of the window.
    The test then asks two things the measurement needs - at what limb separation
    the split is detected at all, and how well the split DEPTH is recovered,
    which is one of the quantities being claimed.
    """
    rng = np.random.default_rng(a.seed)
    sites = []
    while len(sites) < a.floor_sites:
        la_ = float(np.degrees(np.arcsin(2 * rng.random() - 1)))
        if abs(la_) < 55.0:
            sites.append((la_, float(360.0 * rng.random() - 180.0)))
    LO, LA = np.meshgrid(lon, lat)
    LO = ((LO + 180.0) % 360.0) - 180.0
    zsplit = a.split_depth
    rows = []
    for si, (clat, clon) in enumerate(sites):
        cw = max(np.cos(np.radians(clat)), 0.2)
        for sep in [-1.0] + list(a.separations):
            cache = {}

            def prom_for(s_km):
                key = int(round(s_km / 100.0))
                if key not in cache:
                    if key == 0:
                        g = np.exp(-(M.gc_km(clat, clon, LA, LO) ** 2)
                                   / (2.0 * (a.inject_radius / 2.0) ** 2))
                    else:
                        off = 0.5 * (key * 100.0) / 111.2
                        d1 = M.gc_km(clat, clon - off / cw, LA, LO)
                        d2 = M.gc_km(clat, clon + off / cw, LA, LO)
                        r2 = 2.0 * (a.inject_radius / 2.0) ** 2
                        g = np.exp(-(d1 ** 2) / r2) + np.exp(-(d2 ** 2) / r2)
                    cache[key] = prominence_of(-a.inject_amp * g, lat, lon,
                                               a.sigma)
                return cache[key]

            rad = (max(sep, 0.0) / 2 + 700)
            jw = np.where(np.abs(lat - clat) <= rad / 111.2)[0]
            dl = ((lon - clon + 180.0) % 360.0) - 180.0
            iw = np.where(np.abs(dl) <= rad / 111.2 / cw)[0]
            if len(jw) < 6 or len(iw) < 6:
                continue
            sub = np.ix_(jw, iw)
            LAw, LOw = LA[sub], LO[sub]
            vol = []
            for k in range(P0.shape[0]):
                if sep < 0:
                    Pg = 0.0
                else:
                    frac = np.clip((zsplit - zz[k]) / max(zsplit - zz.min(), 1),
                                   0.0, 1.0)
                    Pg = prom_for(sep * frac)
                vol.append(((P0[k] + Pg) if sep >= 0 else P0[k])[sub]
                           >= a.threshold)
            vol = np.stack(vol)
            l3, n3 = ndi.label(vol)
            jc = int(np.argmin(np.abs(lat[jw] - clat)))
            ic = int(np.argmin(np.abs(((lon[iw] - clon + 180) % 360) - 180)))
            deep = [int(l3[kk_, jc, ic]) for kk_ in range(vol.shape[0])
                    if zz[kk_] > zsplit]
            deep = [o for o in deep if o > 0]
            q_own = max(set(deep), key=deep.count) if deep else 0
            cnt = []
            for k in range(P0.shape[0]):
                m = (l3[k] == q_own) if q_own else np.zeros_like(vol[k])
                cnt.append(len(pieces_in_shell(m, LAw, LOw, a.merge_floor)))
            cnt = np.asarray(cnt)
            above = zz < zsplit
            below = zz >= zsplit
            # recovered split depth: deepest sustained rise to two, read upward
            rec = np.nan
            run = 0
            for i2 in range(len(zz) - 1, -1, -1):
                run = run + 1 if cnt[i2] >= 2 else 0
                if run >= a.run_shells:
                    rec = float(zz[i2 + run - 1])
                    break
            rows.append(dict(site=si, separation_km=sep,
                             frac_two_above=float(np.mean(cnt[above] >= 2))
                             if above.any() else np.nan,
                             frac_two_below=float(np.mean(cnt[below] >= 2))
                             if below.any() else np.nan,
                             recovered_split_km=rec))
        print(f'  floor site {si + 1}/{len(sites)}', flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description='branching of imaged structures, with its own floor')
    ap.add_argument('--file', required=True)
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
    ap.add_argument('--nulls', type=int, default=300)
    ap.add_argument('--null-seed', type=int, default=7, dest='null_seed')
    ap.add_argument('--zmin', type=float, default=800.0)
    ap.add_argument('--zmax', type=float, default=2700.0)
    ap.add_argument('--sigma', type=float, default=800.0)
    ap.add_argument('--threshold', type=float, default=0.65)
    ap.add_argument('--min-extent', type=float, default=1000.0, dest='min_extent')
    ap.add_argument('--merge-floor', type=float, default=0.0, dest='merge_floor',
                    help='pieces closer than this are one; set from the floor')
    ap.add_argument('--separations', nargs='+', type=float,
                    default=[300., 450., 600., 900., 1200., 1600.])
    ap.add_argument('--inject-radius', type=float, default=250.0,
                    dest='inject_radius')
    ap.add_argument('--inject-amp', type=float, default=1.2, dest='inject_amp')
    ap.add_argument('--floor-sites', type=int, default=8, dest='floor_sites')
    ap.add_argument('--split-depth', type=float, default=1800.0,
                    dest='split_depth',
                    help='depth of the injected split, so that its recovery can '
                         'be checked against a known value')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--catchment', type=float, default=500.0)
    ap.add_argument('--run-shells', type=int, default=5, dest='run_shells')
    ap.add_argument('--mode', choices=['floor', 'measure', 'both'], default='both')
    a = ap.parse_args()

    depth, lat, lon, arr = load_anomaly(a.file, ModelSpec(a.tag, a.var),
                                        depth_max=a.depth_max, every=a.every)
    lon, arr = dedupe_lon(lon, arr)
    z = np.asarray(depth, float)
    k = np.where((z >= a.zmin) & (z <= a.zmax))[0]
    k = k[np.argsort(z[k])]
    zz = z[k]
    dz = float(np.median(np.abs(np.diff(zz))))
    print(f'{a.tag}: {len(zz)} shells {zz.min():.0f}-{zz.max():.0f} km, '
          f'{len(lat)}x{len(lon)}', flush=True)
    P0 = -contrast_field(arr[k], lat, lon, a.sigma)
    print(f'prominence field ready; threshold {a.threshold:g} per cent',
          flush=True)

    if a.mode in ('floor', 'both'):
        fd = floor_mode(P0, zz, lat, lon, a)
        fo = os.path.join(a.dir, f'branch_struct_floor_{a.tag}{a.suffix}.csv')
        fd.to_csv(fo, index=False)
        t = fd.groupby('separation_km').agg(
            two_above_split=('frac_two_above', 'median'),
            two_below_split=('frac_two_below', 'median'),
            recovered_split=('recovered_split_km', 'median')).round(2)
        t.index = ['nothing injected' if x < 0 else f'limbs {x:.0f} km apart'
                   for x in t.index]
        print(f'\nFLOOR: a Y injected with its split at {a.split_depth:.0f} km\n')
        print(t.to_string())
        g = fd[fd.separation_km > 0].groupby(
            'separation_km').frac_two_above.median()
        ok = g.index[g >= 0.5]
        floor = float(ok.min()) if len(ok) else np.nan
        print(f'\ntwo pieces resolved in most shells from '
              + (f'{floor:.0f} km apart' if np.isfinite(floor)
                 else 'no separation tested'))
        print(f'{fo}')
        if a.merge_floor <= 0 and np.isfinite(floor):
            a.merge_floor = floor
            print(f'merge floor set to the measured value, {floor:.0f} km')
    if a.mode == 'floor':
        return
    if a.merge_floor <= 0:
        sys.exit('no floor measured or supplied; refusing to report branch '
                 'geometry without one')

    mask = np.isfinite(P0) & (P0 >= a.threshold)
    lab, n = label_3d_periodic(mask)
    kk = np.repeat(np.arange(len(zz))[:, None, None], P0.shape[1], 1)
    kk = np.repeat(kk, P0.shape[2], 2)
    idx = np.arange(1, n + 1)
    ext = ((np.asarray(ndi.maximum(kk, lab, index=idx))
            - np.asarray(ndi.minimum(kk, lab, index=idx))) + 1) * dz
    keep = np.where(ext >= a.min_extent)[0] + 1
    print(f'\n{n} structures, {len(keep)} spanning at least '
          f'{a.min_extent:g} km', flush=True)

    LO, LA = np.meshgrid(lon, lat)
    LO = ((LO + 180.0) % 360.0) - 180.0
    srows, prof = [], []
    for q in keep:
        m3 = lab == q
        counts, cent = [], []
        for ki in range(len(zz)):
            ps = pieces_in_shell(m3[ki], LA, LO, a.merge_floor)
            counts.append(len(ps))
            cent.append(ps)
        counts = np.asarray(counts)
        up, zs = counts[::-1], zz[::-1]
        nmax, levels, bdepth = 1, 0, np.nan
        for mth in range(2, int(counts.max()) + 1 if counts.size else 2):
            run, best = 0, False
            for i2, v in enumerate(up >= mth):
                run = run + 1 if v else 0
                if run >= a.run_shells:
                    best = True
                    if mth == 2 and not np.isfinite(bdepth):
                        bdepth = float(zs[i2 - run + 1])
                    break
            if best:
                nmax, levels = mth, levels + 1
        spread = [float(np.mean([M.gc_km(
            np.mean([p['lat'] for p in ps]), np.mean([p['lon'] for p in ps]),
            p['lat'], p['lon']) for p in ps])) if len(ps) >= 2 else np.nan
            for ps in cent]
        srows.append(dict(structure=int(q), extent_km=float(ext[q - 1]),
                          pieces=nmax, branch_depth_km=bdepth, levels=levels,
                          median_spread_km=float(np.nanmedian(spread))
                          if np.isfinite(spread).any() else np.nan))
        prof.append(pd.DataFrame(dict(structure=int(q), z=zz, pieces=counts,
                                      spread_km=spread)))
    S = pd.DataFrame(srows)
    S.to_csv(os.path.join(a.dir,
             f'branch_structures_{a.tag}{a.suffix}.csv'), index=False)
    pd.concat(prof, ignore_index=True).to_csv(os.path.join(
        a.dir, f'branch_struct_profile_{a.tag}{a.suffix}.csv'), index=False)
    print(f'{int((S.pieces >= 2).sum())} of {len(S)} structures branch; '
          f'depths {S.branch_depth_km.min():.0f}-{S.branch_depth_km.max():.0f} km')

    # associate: each site takes the nearest structure within the catchment
    foot = {int(q): (lab == q).any(axis=0) for q in keep}
    sites = []
    H = pd.read_csv(a.hotspots, comment='#')
    for r in H.itertuples():
        sites.append((str(r.hotspot), float(r.lat), float(r.lon_180), 'hotspot'))
    rng = np.random.default_rng(a.null_seed)
    for qn in range(a.nulls):
        sites.append((f'null{qn:03d}',
                      float(np.degrees(np.arcsin(2 * rng.random() - 1))),
                      float(360.0 * rng.random() - 180.0), 'null_site'))
    arows = []
    for nm, la_, lo_, kind in sites:
        best, bq = np.inf, None
        for q, f in foot.items():
            jj, ii = np.where(f)
            if not len(jj):
                continue
            d = float(M.gc_km(la_, lo_, lat[jj], ((lon[ii] + 180) % 360) - 180).min())
            if d < best:
                best, bq = d, q
        r = S[S.structure == bq]
        arows.append(dict(site=nm, kind=kind, distance_km=best,
                          structure=bq,
                          pieces=int(r.pieces.iloc[0]) if len(r) else np.nan,
                          branch_depth_km=float(r.branch_depth_km.iloc[0]) if len(r) else np.nan,
                          levels=int(r.levels.iloc[0]) if len(r) else np.nan,
                          spread_km=float(r.median_spread_km.iloc[0]) if len(r) else np.nan))
    A = pd.DataFrame(arows)
    ao = os.path.join(a.dir, f'branch_assoc_{a.tag}{a.suffix}.csv')
    A.to_csv(ao, index=False)
    near = A[A.distance_km <= a.catchment]
    h, nl = near[near.kind == 'hotspot'], near[near.kind == 'null_site']
    from root_bias import perm_median_diff, fisher_one_sided
    print(f'\nsites with a structure within {a.catchment:g} km: '
          f'{len(h)} hotspots on {h.structure.nunique()} structures, '
          f'{len(nl)} nulls on {nl.structure.nunique()}')
    print('site-level comparison follows, but note that both populations draw '
          'from the\nsame structures, so it has little power; the structure-'
          'level test below is\nthe one to read\n')
    print(f'{"":24s} {"hotspots":>10s} {"nulls":>10s} {"p":>9s}')
    for lab_, col in (('pieces', 'pieces'), ('branch depth km', 'branch_depth_km'),
                      ('levels', 'levels'), ('spread km', 'spread_km')):
        x, y = h[col].dropna(), nl[col].dropna()
        if len(x) < 3 or len(y) < 3:
            continue
        p, _ = perm_median_diff(x, y, 50000)
        print(f'{lab_:24s} {np.median(x):10.0f} {np.median(y):10.0f} {p:9.4f}')
    for mth in (2, 3):
        a11 = int((h.pieces >= mth).sum()); a21 = int((nl.pieces >= mth).sum())
        if len(h) and len(nl):
            p = fisher_one_sided(a11, len(h) - a11, a21, len(nl) - a21)
            print(f'structure with {mth}+ pieces: hotspots {a11}/{len(h)} '
                  f'{100 * a11 / max(len(h),1):.1f}%, nulls {a21}/{len(nl)} '
                  f'{100 * a21 / max(len(nl),1):.1f}%, p = {p:.4f}')
    # the comparison that has power: structures WITH a hotspot on them against
    # structures without. A site-level test resamples the same objects and can
    # only report that they equal themselves.
    hs = set(h.structure.dropna().astype(int))
    S['has_hotspot'] = S.structure.isin(hs)
    w, wo = S[S.has_hotspot], S[~S.has_hotspot]
    print(f'\nSTRUCTURE-LEVEL: {len(w)} structures carry a hotspot within '
          f'{a.catchment:g} km, {len(wo)} do not\n')
    print(f'{"":24s} {"with":>8s} {"without":>9s} {"p":>9s}')
    for lab_, col in (('pieces', 'pieces'), ('branch depth km', 'branch_depth_km'),
                      ('levels', 'levels'), ('spread km', 'median_spread_km'),
                      ('extent km', 'extent_km')):
        x, y = w[col].dropna(), wo[col].dropna()
        if len(x) < 3 or len(y) < 3:
            print(f'{lab_:24s} {"--":>8s} {"--":>9s}')
            continue
        p, _ = perm_median_diff(x, y, 50000)
        print(f'{lab_:24s} {np.median(x):8.0f} {np.median(y):9.0f} {p:9.4f}')
    print(f'\n{ao}')


if __name__ == '__main__':
    main()
