#!/usr/bin/env python3
"""Is the slow anomaly beneath a hotspot vertically continuous to the base?

WHY THIS EXISTS. Every shape measurement in this study was made on the prominence
field, which is the anomaly minus its own 800 km smoothed version. That length was
chosen to isolate conduits 300 to 500 km wide, and it subtracts anything broader -
including the body of a plume, which in tomography is the part you can actually see.
The components left over are the residual around the edges of broad structure, and
they have no conduit shape because they are not the conduit.

This measures the anomaly directly. At each depth it takes the lowest value within a
radius of the site and asks how much of the lower mantle lies below a fixed level,
how long the longest interruption is, and how far up from the core-mantle boundary
an unbroken slow column reaches. No contrast field, no thresholded components, no
least-cost path. A route can thread between disconnected patches; a column cannot.

THE CONFOUND IS BUILT IN AND IS THE POINT. Hotspots overlie the slow provinces, so a
column that reaches the base may say nothing except that the base is slow there. The
comparison that matters is therefore against null sites MATCHED on the anomaly at
the base, and the summary reports it that way. If continuity survives that matching
it is a statement about the column; if it does not, it is the province result again
under another name.

  python3 column_continuity.py --pilot 6
  python3 column_continuity.py --tag RevealLO
"""
from __future__ import annotations

import argparse, os, sys, time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon

R_E, DEG = 6371.0, np.pi / 180.0


def depth_bins(arr, depth, zmin, zmax, dz):
    z = np.asarray(depth, float)
    edges = np.arange(zmin, zmax + dz, dz)
    out, zc = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = np.where((z >= lo) & (z < hi))[0]
        if len(k):
            out.append(np.nanmin(arr[k], axis=0))
            zc.append(0.5 * (lo + hi))
    return np.asarray(out), np.asarray(zc, float)


def column(vol, lat, lon, clat, clon, radius_km):
    """The lowest anomaly within `radius_km` of the site, at every depth."""
    dlat = (lat - clat) * DEG * R_E
    dl = ((lon - clon + 180.0) % 360.0) - 180.0
    dlon = dl * DEG * R_E * np.cos(np.radians(clat))
    j = np.where(np.abs(dlat) <= radius_km)[0]
    i = np.where(np.abs(dlon) <= radius_km)[0]
    if not len(j) or not len(i):
        return None
    D2 = dlon[i][None, :] ** 2 + dlat[j][:, None] ** 2
    m = D2 <= radius_km ** 2
    sub = vol[np.ix_(np.arange(vol.shape[0]), j, i)]
    out = np.where(m[None, :, :], sub, np.nan)
    return np.nanmin(out, axis=(1, 2))


def metrics(p, z, thr, gap_tol_km=0.0):
    """Continuity of a column profile at one threshold.

    reach is the shallowest depth an unbroken slow column reaches, walking up from
    the deepest bin and forgiving interruptions no longer than gap_tol_km.
    """
    dz = float(abs(z[1] - z[0]))
    below = p < thr
    frac = float(np.mean(below))

    runs, run = [], 0
    for b in below:
        run = 0 if b else run + 1
        runs.append(run)
    longest_gap = float(max(runs) * dz) if len(runs) else np.nan

    order = np.argsort(-z)                      # deepest first
    b = below[order]
    zz = z[order]
    if not b[0]:
        reach = np.nan
    else:
        k, miss, last = 0, 0, 0
        while k + 1 < len(b):
            if b[k + 1]:
                miss = 0
                last = k + 1
            else:
                miss += 1
                if miss * dz > gap_tol_km:
                    break
            k += 1
        reach = float(zz[last])          # always a bin that is itself below
    return frac, longest_gap, reach


def province_lookup(path, min_votes=3):
    if not os.path.exists(path):
        return None
    v = np.load(path)
    vote, plat, plon = v['vote'], v['lat'], v['lon']

    def inside(la, lo):
        j = int(np.abs(plat - la).argmin())
        i = int(np.abs(((plon - lo + 180.0) % 360.0) - 180.0).argmin())
        return bool(vote[j, i] >= min_votes)
    return inside


def find_model(tag):
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [os.environ.get('TOMO', ''), here, os.path.join(here, '..'),
             os.path.join(here, '..', '..'),
             os.path.join(here, '..', '..', 'REVEAL_mantle_tomography'),
             os.path.join(here, '..', '..', '..', 'REVEAL_mantle_tomography')]
    for r in roots:
        if r:
            c = os.path.abspath(os.path.join(r, f'{tag}.nc'))
            if os.path.exists(c):
                return c
    for r in roots:
        if r and os.path.isdir(r):
            for dp, _, names in os.walk(os.path.abspath(r)):
                if f'{tag}.nc' in names:
                    return os.path.join(dp, f'{tag}.nc')
    sys.exit(f'could not find {tag}.nc; pass it with --file')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default=None)
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--dir', default='out')
    ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
    ap.add_argument('--province', default='out/province_vote.npz')
    ap.add_argument('--zmin', type=float, default=800.0)
    ap.add_argument('--zmax', type=float, default=2850.0)
    ap.add_argument('--dz', type=float, default=50.0)
    ap.add_argument('--radii', nargs='+', type=float, default=[300.0, 400.0, 600.0])
    ap.add_argument('--thresholds', nargs='+', type=float,
                    default=[-0.4, -0.6, -0.8, -1.2])
    ap.add_argument('--gap-tol', type=float, default=100.0, dest='gap_tol',
                    help='an interruption this long or shorter does not break a column')
    ap.add_argument('--nulls', type=int, default=300)
    ap.add_argument('--null-seed', type=int, default=7, dest='null_seed')
    ap.add_argument('--base-depth', type=float, default=2700.0, dest='base_depth',
                    help='depth at which the basal anomaly is read, for matching')
    ap.add_argument('--pilot', type=int, default=0)
    a = ap.parse_args()

    if a.file is None:
        a.file = find_model(a.tag)
    print(f'loading {a.file}', flush=True)
    depth, lat, lon, arr = load_anomaly(a.file, ModelSpec(a.tag, a.var),
                                        depth_max=a.zmax + a.dz, every=a.every)
    lon, arr = dedupe_lon(lon, arr)
    vol, z = depth_bins(arr, depth, a.zmin, a.zmax, a.dz)
    del arr
    print(f'{len(z)} bins {z[0]:.0f}-{z[-1]:.0f} km, grid '
          f'{vol.shape[1]} x {vol.shape[2]}', flush=True)

    inside = province_lookup(a.province)
    sites = []
    H = pd.read_csv(a.hotspots, comment='#')
    if a.pilot:
        H = H.head(a.pilot)
    for r in H.itertuples():
        sites.append((str(r.hotspot), float(r.lat), float(r.lon_180), 'hotspot'))
    rng = np.random.default_rng(a.null_seed)
    for q in range(a.nulls if not a.pilot else min(a.nulls, 30)):
        sites.append((f'null{q:03d}',
                      float(np.degrees(np.arcsin(2 * rng.random() - 1))),
                      float(360.0 * rng.random() - 180.0), 'null_site'))

    kb = int(np.abs(z - a.base_depth).argmin())
    rows, t0 = [], time.time()
    for n, (name, la, lo, kind) in enumerate(sites, 1):
        for rad in a.radii:
            p = column(vol, lat, lon, la, lo, rad)
            if p is None:
                continue
            base = float(p[kb])
            for thr in a.thresholds:
                frac, gap, reach = metrics(p, z, thr, a.gap_tol)
                rows.append(dict(site=name, kind=kind, lat=la, lon=lo,
                                 radius_km=rad, threshold=thr,
                                 frac_below=frac, longest_gap_km=gap,
                                 reach_km=reach, base_anom=base,
                                 in_province=(None if inside is None
                                              else inside(la, lo))))
        if n % 25 == 0 or n == len(sites):
            print(f'  {n}/{len(sites)} sites  [{(time.time() - t0) / 60:.1f} min]',
                  flush=True)

    d = pd.DataFrame(rows)
    o = os.path.join(a.dir, f'column_continuity_{a.tag}.csv')
    d.to_csv(o, index=False)
    print(f'\n{o}\n')
    summarise(d, a)


def matched_nulls(d, hot, rad, thr, tol=0.15):
    """Null sites whose basal anomaly matches the hotspot distribution."""
    nul = d[(d.kind == 'null_site') & (d.radius_km == rad) & (d.threshold == thr)]
    lo, hi = np.nanpercentile(hot.base_anom, [10, 90])
    return nul[(nul.base_anom >= lo - tol) & (nul.base_anom <= hi + tol)]


def summarise(d, a):
    rng = np.random.default_rng(5)
    print(f'{"radius":>7s} {"thr":>6s} {"hotspots":>22s} {"all nulls":>20s} '
          f'{"matched nulls":>24s}')
    for rad in a.radii:
        for thr in a.thresholds:
            hot = d[(d.kind == 'hotspot') & (d.radius_km == rad)
                    & (d.threshold == thr)]
            nul = d[(d.kind == 'null_site') & (d.radius_km == rad)
                    & (d.threshold == thr)]
            mat = matched_nulls(d, hot, rad, thr)
            if not len(hot) or not len(nul):
                continue

            def p_of(g):
                x = hot.frac_below.to_numpy()
                y = g.frac_below.to_numpy()
                obs = np.median(x) - np.median(y)
                pool = np.r_[x, y]
                c = 0
                for _ in range(4000):
                    q = rng.permutation(len(pool))
                    if np.median(pool[q[:len(x)]]) - np.median(pool[q[len(x):]]) >= obs:
                        c += 1
                return (c + 1) / 4001.0

            print(f'{rad:7.0f} {thr:6.1f} '
                  f'{np.median(hot.frac_below):7.2f} n={len(hot):3d}      '
                  f'{np.median(nul.frac_below):6.2f} p={p_of(nul):.4f}   '
                  f'{np.median(mat.frac_below):6.2f} n={len(mat):3d} '
                  f'p={p_of(mat):.4f}')
    print('\nif the matched-null column loses the difference, continuity is the')
    print('province result restated and not a fact about the column')


if __name__ == '__main__':
    main()
