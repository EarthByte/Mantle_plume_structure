#!/usr/bin/env python3
"""Cross-model reproducibility of the corridors, at a scale the models share.

RevealLO samples the mantle at ten kilometres and the others at thirty to sixty,
so comparing corridors cell by cell would report disagreement that is a
difference of sampling. Both volumes are therefore block-averaged onto one coarse
grid before anything is compared, and the comparison is of where each model puts
the corridor rather than of which cells it fills.

Two overlaps are reported. The Jaccard index of the coarse corridors is the
strict one and falls quickly with any displacement. The centroid separation with
depth is the tolerant one and is the quantity a figure should carry, because two
models that agree on a structure displaced to the southwest have said the same
thing even when their corridors barely intersect.

REVEAL and RevealLO share data and methodology, so agreement between them
measures resolution rather than reproducibility, and the pair is reported
separately from the independent families for that reason.
"""
from __future__ import annotations
import argparse, itertools, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morphology as M

LINEAGE = {frozenset(('REVEAL', 'RevealLO')),
           frozenset(('RevealLO', 'RevealLO_30km')),
           frozenset(('REVEAL', 'RevealLO_30km'))}


def coarse(W, depth, lat, lon, zmax=2900.0, dz=100.0, dl=2.0,
           reduce='mean'):
    """Block-average a volume onto a FIXED global coarse grid.

    The grid is defined by absolute coordinate, not by the extent of the box the
    volume was stored in. Deriving the edges from each model's own box gives two
    models two different arrays - 111 longitude bins against 113 for the same
    hotspot, because their boxes were cut on different native samplings - and the
    comparison then either fails outright or, worse, succeeds against a
    misaligned neighbour. Every model lands on the same array here, and cells no
    model reaches are simply empty.

    `reduce` is 'mean' for a weight field and 'any' for a mask. The distinction
    decides whether the comparison means anything. A corridor occupies a small
    part of each coarse block, so the block MEAN of a thresholded corridor is far
    below the threshold almost everywhere it passes, and a Jaccard index built
    from mean-then-threshold is zero even where two models agree exactly - it was,
    at every site, including one carrying the same injected conduit in both. A
    coarse cell contains corridor if ANY fine cell in it does, which is what 'any'
    computes.

    Returns the coarse volume, the three edge vectors, and the depth range the
    volume actually covers, so the caller can restrict a comparison to the shells
    both models carry.
    """
    z = np.asarray(depth, float)
    lat = np.asarray(lat, float)
    lon = ((np.asarray(lon, float) + 180.0) % 360.0) - 180.0
    zb = np.arange(0.0, zmax + dz, dz)
    jb = np.arange(-90.0, 90.0 + dl, dl)
    ib = np.arange(-180.0, 180.0, dl)
    nk, nj, ni = len(zb), len(jb), len(ib)

    ki = np.clip((z / dz).astype(int), 0, nk - 1)
    ji = np.clip(((lat + 90.0) / dl).astype(int), 0, nj - 1)
    ii = ((lon + 180.0) / dl).astype(int) % ni

    flat = (ki[:, None, None] * (nj * ni) + ji[None, :, None] * ni
            + ii[None, None, :]).ravel()
    tot = np.bincount(flat, weights=np.asarray(W, float).ravel(),
                      minlength=nk * nj * ni)
    cnt = np.bincount(flat, minlength=nk * nj * ni)
    if reduce == 'any':
        out = (tot > 0).astype(float)
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            out = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    return (out.reshape(nk, nj, ni), zb, jb, ib,
            (float(z.min()), float(z.max())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--dir', default='out')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--field', default='cor0.02')
    ap.add_argument('--floor', type=float, default=0.5,
                    help='fraction of configurations a coarse cell must command')
    ap.add_argument('--zmax', type=float, default=2900.0,
                    help='base of the fixed comparison grid, km')
    ap.add_argument('--dz', type=float, default=100.0)
    ap.add_argument('--dl', type=float, default=2.0)
    a = ap.parse_args()

    store, sites = {}, None
    for t in a.tags:
        f = os.path.join(a.dir, f'morph_volumes_{t}{a.suffix}.npz')
        if not os.path.exists(f):
            print(f'  {t}: no volumes, skipped')
            continue
        store[t] = np.load(f)
        k = {q.split('|', 1)[1] for q in store[t].files if q.startswith('occ|')}
        sites = k if sites is None else (sites & k)
    if len(store) < 2:
        raise SystemExit('need at least two models with volumes')
    rows = []
    for s in sorted(x for x in (sites or ()) if not x.startswith('null')):
        C = {}
        for t, d in store.items():
            key = f'{a.field}|{s}'
            if key not in d.files:
                continue
            jb, ib = d[f'jb|{s}'], d[f'ib|{s}']
            W, zb, jj, ii, zr = coarse(d[key], d['depth'], d['lat'][jb],
                                       d['lon'][ib], a.zmax, a.dz, a.dl)
            Mk, _, _, _, _ = coarse((d[key] >= a.floor).astype(float), d['depth'],
                                    d['lat'][jb], d['lon'][ib], a.zmax, a.dz,
                                    a.dl, reduce='any')
            C[t] = (W, zb, jj, ii, zr, Mk)
        for t1, t2 in itertools.combinations(sorted(C), 2):
            W1, zb, jj, ii, zr1, M1 = C[t1]
            W2, _, _, _, zr2, M2 = C[t2]
            # only the shells both models carry: SEMUCB-WM1 reaches 2891 km and
            # RevealLO is cut at 2880, and counting the difference as
            # disagreement would penalise the deeper model for having data
            lo = max(zr1[0], zr2[0])
            hi = min(zr1[1], zr2[1])
            kk = np.where((zb >= lo - a.dz) & (zb <= hi))[0]
            if not len(kk):
                continue
            m1, m2 = M1[kk] > 0, M2[kk] > 0
            u = np.logical_or(m1, m2).sum()
            jac = float(np.logical_and(m1, m2).sum() / u) if u else np.nan
            # A strict overlap is zero whenever two corridors sit in adjacent
            # coarse cells, which happens to any narrow structure near a bin
            # edge whatever the models agree about - an injected vertical
            # conduit present in both scored zero for that reason alone. The
            # tolerant figure allows one coarse cell laterally and is what a
            # statement about agreement should rest on.
            from scipy.ndimage import maximum_filter
            d1 = maximum_filter(m1, size=(1, 3, 3), mode='nearest')
            d2 = maximum_filter(m2, size=(1, 3, 3), mode='nearest')
            n1, n2 = m1.sum(), m2.sum()
            jac1 = (float((m1 & d2).sum() / n1) if n1 else np.nan)
            jac2 = (float((m2 & d1).sum() / n2) if n2 else np.nan)
            jac_tol = float(np.nanmean([jac1, jac2]))
            LO, LA = np.meshgrid(ii, jj)
            sep = []
            for k in range(len(kk)):
                if not (m1[k].any() and m2[k].any()):
                    continue
                a1 = M.weighted_centroid(W1[kk[k]] * m1[k], LA, LO)
                a2 = M.weighted_centroid(W2[kk[k]] * m2[k], LA, LO)
                if np.isfinite(a1[0]) and np.isfinite(a2[0]):
                    sep.append(float(M.gc_km(a1[0], a1[1], a2[0], a2[1])))
            rows.append(dict(site=s, model_a=t1, model_b=t2, jaccard=jac,
                             overlap_within_one_cell=jac_tol,
                             centroid_sep_km=float(np.median(sep)) if sep else np.nan,
                             shells_compared=len(sep),
                             depth_overlap_km=f'{lo:.0f}-{hi:.0f}',
                             shared_lineage=frozenset((t1, t2)) in LINEAGE))
    df = pd.DataFrame(rows)
    if not len(df):
        raise SystemExit('no site was present in two models; nothing to compare')
    out = os.path.join(a.dir, f'morph_crossmodel{a.suffix}.csv')
    df.to_csv(out, index=False)
    for flag, lab in ((False, 'independent inversion families'),
                      (True, 'shared lineage (a resolution test, not a replication)')):
        g = df[df.shared_lineage == flag]
        if not len(g):
            continue
        print(f'\n{lab}:')
        piv = g.pivot_table(index='site', columns=['model_a', 'model_b'],
                            values='centroid_sep_km')
        piv.columns = [f'{x}/{y}'.replace('SEMUCB-WM1', 'SEMUCB')
                       .replace('RevealLO', 'RLO').replace('GLADM35', 'GLAD')
                       for x, y in piv.columns]
        print(piv.to_string(float_format=lambda x: f'{x:.0f}'))
        print(f'  median strict Jaccard {g.jaccard.median():.3f}, median '
              f'overlap within one coarse cell '
              f'{g.overlap_within_one_cell.median():.3f}, median centroid '
              f'separation {g.centroid_sep_km.median():.0f} km')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
