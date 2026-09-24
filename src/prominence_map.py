#!/usr/bin/env python3
"""Find the tube-like structures first, then ask where the hotspots are.

Every version of this analysis so far has anchored a column at a hotspot's
surface position and followed it downward. That assumes the answer. A conduit
leans, sometimes by more than a thousand kilometres, and a tracker that takes the
best local minimum one interval at a time will leave it the moment something
nearby is locally slower and will never come back. Worse, a path is
one-dimensional and a branching structure is not, so no amount of care in
following a column can represent a plume that divides. Comparing the column under
a hotspot with the column under a random point compares two arbitrary columns.

This inverts the order. The prominence field is computed everywhere, the
structures in it are found without reference to any hotspot, and only then is it
asked whether hotspots sit near them more often than random points do. Nothing
about where a conduit ought to be enters the detection.

PROMINENCE AS A FIELD. At every cell, the laterally smoothed anomaly minus the
anomaly itself. A feature much broader than the smoothing length has a value
equal to its own surroundings and scores zero, so a large low-velocity province
contributes nothing; a feature narrower than it stands out by however much it is
slower than its neighbourhood. This is the same quantity the contrast channel
already used, which is why that channel was the province-independent one, and it
is computed here with the same tested code rather than a second implementation.

STRUCTURES, NOT PATHS. The field is thresholded and its connected components are
found in three dimensions, with longitude joined so a structure crossing the date
line is one structure. A component may lean, widen, narrow, or branch: it is
whatever shape it is, and its depth extent is measured from what it occupies
rather than from a route chosen through it.

ASSOCIATION, NOT TRACKING. For each site the horizontal distance to the nearest
qualifying structure is measured over all depths. A hotspot whose plume leans a
thousand kilometres is still near its own structure; a random point is near one
only if the deep mantle is full of them. That comparison is the test, and it makes
no assumption about the geometry of what it is looking for.
"""
from __future__ import annotations

import argparse, os, sys
from math import comb

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morphology as M
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from contrast import contrast_field

R_E, DEG = 6371.0, np.pi / 180.0


def label_3d_periodic(mask):
    """Connected components in depth-latitude-longitude, longitude joined."""
    lab, n = ndi.label(mask)
    if n == 0:
        return lab, 0
    # merge labels that touch across the longitude seam
    a, b = lab[:, :, 0], lab[:, :, -1]
    pairs = set()
    m = (a > 0) & (b > 0)
    for x, y in zip(a[m].ravel(), b[m].ravel()):
        if x != y:
            pairs.add((min(int(x), int(y)), max(int(x), int(y))))
    if pairs:
        parent = list(range(n + 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for x, y in pairs:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)
        remap = np.arange(n + 1)
        for q in range(1, n + 1):
            remap[q] = find(q)
        lab = remap[lab]
    return lab, int(lab.max())


def fisher_one_sided(a11, a12, a21, a22):
    n, r1, c1 = a11 + a12 + a21 + a22, a11 + a12, a11 + a21
    hi = min(r1, c1)
    return float(sum(comb(r1, x) * comb(n - r1, c1 - x)
                     for x in range(a11, hi + 1)) / comb(n, c1))


def lab_inputs(zz, dz):
    return dict(n_shell=len(zz), dz=dz)


def structures(P, thr, info, min_extent):
    """Connected components above a threshold that span enough depth."""
    mask = np.isfinite(P) & (P >= thr)
    lab, n = label_3d_periodic(mask)
    if n == 0:
        return None, 0, 0.0
    kk = np.repeat(np.arange(info['n_shell'])[:, None, None], P.shape[1], 1)
    kk = np.repeat(kk, P.shape[2], 2)
    idx = np.arange(1, n + 1)
    ext = ((np.asarray(ndi.maximum(kk, lab, index=idx))
            - np.asarray(ndi.minimum(kk, lab, index=idx))) + 1) * info['dz']
    keep = np.where(ext >= min_extent)[0] + 1
    if not len(keep):
        return None, 0, 0.0
    foot = np.isin(lab, keep).any(axis=0)
    return foot, len(keep), float(foot.mean())


def scan(P, info, lat, lon, sites, a):
    """The association across thresholds, with the usable range marked.

    A threshold below percolation makes one structure of the whole field and
    every site sits on it; a threshold so high that only a handful of structures
    survive leaves nothing to be near. Both failures are visible in the footprint
    and the count, without reference to the association, so the range in which
    the construction means anything is decided before the answer is read.
    """
    from root_bias import perm_median_diff
    print(f'\n{"thr":>5s} {"structures":>10s} {"footprint":>10s} '
          f'{"hot med":>8s} {"null med":>9s} {"within 500 km":>21s} {"p":>8s}  note')
    rows = []
    for thr in a.scan:
        foot, nkeep, ff = structures(P, thr, info, a.min_extent)
        if foot is None:
            print(f'{thr:5.2f} {0:10d}   no structure spans the required depth')
            continue
        jj, ii = np.where(foot)
        fl = lat[jj]
        fo = ((lon[ii] + 180.0) % 360.0) - 180.0
        rec = []
        for nm, la_, lo_, kind in sites:
            rec.append((kind, float(M.gc_km(la_, lo_, fl, fo).min())))
        dd = pd.DataFrame(rec, columns=['kind', 'dist'])
        h = dd[dd.kind == 'hotspot'].dist
        nu = dd[dd.kind == 'null_site'].dist
        a11 = int((h <= 500).sum()); a21 = int((nu <= 500).sum())
        pf = fisher_one_sided(a11, len(h) - a11, a21, len(nu) - a21)
        note = ('percolates' if ff > 0.30
                else ('too few' if nkeep < 12 else ''))
        print(f'{thr:5.2f} {nkeep:10d} {100 * ff:9.1f}% {h.median():8.0f} '
              f'{nu.median():9.0f}   {a11:3d}/{len(h)} {100 * a11 / len(h):5.1f}% '
              f'{a21:4d}/{len(nu)} {100 * a21 / len(nu):5.1f}% {pf:8.4f}  {note}')
        rows.append(dict(threshold=thr, structures=nkeep, footprint=ff,
                         hot_median_km=float(h.median()),
                         null_median_km=float(nu.median()),
                         hot_within500=a11, null_within500=a21, p=pf,
                         usable=(note == '')))
    d = pd.DataFrame(rows)
    o = os.path.join(a.dir, f'prominence_scan_{a.tag}{a.suffix}.csv')
    d.to_csv(o, index=False)
    u = d[d.usable]
    if len(u):
        print(f'\nusable range {u.threshold.min():.2f} to {u.threshold.max():.2f}'
              f' per cent: {int((u.p < 0.05).sum())} of {len(u)} thresholds '
              f'significant at 0.05')
    else:
        print('\nno threshold leaves a usable field for this model')
    print(f'{o}')


def main():
    ap = argparse.ArgumentParser(
        description='find prominent structures, then locate hotspots on them')
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
    ap.add_argument('--sigma', type=float, default=800.0,
                    help='lateral smoothing that defines "the surroundings", km')
    ap.add_argument('--threshold', type=float, default=0.30,
                    help='prominence a cell needs to join a structure, per cent')
    ap.add_argument('--min-extent', type=float, default=1000.0,
                    dest='min_extent',
                    help='depth a structure must span to be counted, km')
    ap.add_argument('--catchment', nargs='+', type=float,
                    default=[500.0, 1000.0, 1500.0],
                    help='distances at which association is reported, km')
    ap.add_argument('--save-footprint', action='store_true',
                    dest='save_footprint',
                    help='write the structure footprint and the prominence '
                         'field summary for the figures')
    ap.add_argument('--scan', nargs='+', type=float, default=None,
                    help='scan these prominence thresholds instead of using one. '
                         'The amplitude of a prominence field depends on the '
                         'model, so a value calibrated on one model is not the '
                         'same object in another and each must be scanned on its '
                         'own terms.')
    a = ap.parse_args()

    depth, lat, lon, arr = load_anomaly(a.file, ModelSpec(a.tag, a.var),
                                        depth_max=a.depth_max, every=a.every)
    lon, arr = dedupe_lon(lon, arr)
    z = np.asarray(depth, float)
    k = np.where((z >= a.zmin) & (z <= a.zmax))[0]
    k = k[np.argsort(z[k])]
    zz = z[k]
    dz = float(np.median(np.abs(np.diff(zz))))
    print(f'{a.tag}: {len(zz)} shells from {zz.min():.0f} to {zz.max():.0f} km, '
          f'{len(lat)}x{len(lon)}, spacing {dz:.0f} km', flush=True)

    # prominence: the smoothed field minus the field, so a feature broader than
    # the smoothing length scores zero and a province contributes nothing
    P = -contrast_field(arr[k], lat, lon, a.sigma)
    print(f'prominence field: median {np.nanmedian(P):+.3f}, '
          f'90th percentile {np.nanquantile(P, 0.9):+.3f} per cent', flush=True)

    sites = []
    H = pd.read_csv(a.hotspots, comment='#')
    for r in H.itertuples():
        sites.append((str(r.hotspot), float(r.lat), float(r.lon_180), 'hotspot'))
    rng = np.random.default_rng(a.null_seed)
    for q in range(a.nulls):
        sites.append((f'null{q:03d}',
                      float(np.degrees(np.arcsin(2 * rng.random() - 1))),
                      float(360.0 * rng.random() - 180.0), 'null_site'))

    if a.scan:
        scan(P, lab_inputs(zz, dz), lat, lon, sites, a)
        return

    mask = np.isfinite(P) & (P >= a.threshold)
    lab, n = label_3d_periodic(mask)
    print(f'{n} connected structures above {a.threshold:g} per cent', flush=True)
    if n == 0:
        sys.exit('no structures at this threshold')

    # depth extent of each, and drop the ones too short to be a conduit
    kk = np.repeat(np.arange(len(zz))[:, None, None], P.shape[1], 1)
    kk = np.repeat(kk, P.shape[2], 2)
    kmin = ndi.minimum(kk, lab, index=np.arange(1, n + 1))
    kmax = ndi.maximum(kk, lab, index=np.arange(1, n + 1))
    size = ndi.sum(np.ones_like(lab, dtype=np.int32), lab,
                   index=np.arange(1, n + 1))
    ext = (np.asarray(kmax) - np.asarray(kmin) + 1) * dz
    keep = np.where(ext >= a.min_extent)[0] + 1
    print(f'{len(keep)} of them span at least {a.min_extent:g} km; '
          f'largest extent {ext.max():.0f} km, '
          f'median size {np.median(np.asarray(size)[keep - 1]):.0f} cells',
          flush=True)
    if not len(keep):
        sys.exit('no structure spans the required depth')

    kept = np.isin(lab, keep)
    # the horizontal footprint: a cell of any kept structure at any depth
    foot = kept.any(axis=0)
    if a.save_footprint:
        # the depth extent of the deepest-spanning structure over each cell, so
        # a map can show how far down the structure beneath a point reaches
        emap = np.zeros(foot.shape, np.float32)
        for q in keep:
            m = (lab == q).any(axis=0)
            emap = np.maximum(emap, m * ext[q - 1])
        fo = os.path.join(a.dir, f'prominence_footprint_{a.tag}{a.suffix}.npz')
        np.savez_compressed(fo, footprint=foot, extent_km=emap, lat=lat, lon=lon,
                            threshold=np.array([a.threshold]),
                            min_extent=np.array([a.min_extent]))
        print(f'wrote {fo}', flush=True)
    jj, ii = np.where(foot)
    flat_lat = lat[jj]
    flat_lon = ((lon[ii] + 180.0) % 360.0) - 180.0
    print(f'footprint covers {100.0 * foot.mean():.1f} per cent of the grid '
          f'by cell count', flush=True)

    sites = []
    H = pd.read_csv(a.hotspots, comment='#')
    for r in H.itertuples():
        sites.append((str(r.hotspot), float(r.lat), float(r.lon_180), 'hotspot'))
    rng = np.random.default_rng(a.null_seed)
    for q in range(a.nulls):
        sites.append((f'null{q:03d}',
                      float(np.degrees(np.arcsin(2 * rng.random() - 1))),
                      float(360.0 * rng.random() - 180.0), 'null_site'))

    rows = []
    for name, la_, lo_, kind in sites:
        d = M.gc_km(la_, lo_, flat_lat, flat_lon)
        q = int(np.argmin(d))
        j0, i0 = jj[q], ii[q]
        col = lab[:, j0, i0]
        lb = col[col > 0]
        e = float(ext[int(np.bincount(lb).argmax()) - 1]) if len(lb) else np.nan
        rows.append(dict(site=name, kind=kind, lat=la_, lon=lo_,
                         distance_km=float(d[q]), nearest_extent_km=e))
    d = pd.DataFrame(rows)
    out = os.path.join(a.dir, f'prominence_map_{a.tag}{a.suffix}.csv')
    d.to_csv(out, index=False)

    h, nl = d[d.kind == 'hotspot'], d[d.kind == 'null_site']
    print(f'\ndistance to the nearest structure spanning {a.min_extent:g} km')
    print(f'  hotspots median {h.distance_km.median():.0f} km, '
          f'nulls {nl.distance_km.median():.0f} km')
    from root_bias import perm_median_diff
    p, npm = perm_median_diff(h.distance_km, nl.distance_km, 100000)
    print(f'  p = {p:.4f} over {npm} resamples\n')
    print(f'{"within":>8s} {"hotspots":>18s} {"nulls":>18s} {"p":>9s}')
    for c in a.catchment:
        a11 = int((h.distance_km <= c).sum()); a12 = len(h) - a11
        a21 = int((nl.distance_km <= c).sum()); a22 = len(nl) - a21
        pf = fisher_one_sided(a11, a12, a21, a22)
        print(f'{c:7.0f}k {a11:6d}/{len(h)} {100.0 * a11 / len(h):7.1f}% '
              f'{a21:6d}/{len(nl)} {100.0 * a21 / len(nl):7.1f}% {pf:9.4f}')
    print(f'\n{out}')


if __name__ == '__main__':
    main()
