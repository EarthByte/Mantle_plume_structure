#!/usr/bin/env python3
"""How much do the provinces themselves differ between tomographic models?

A margin distance of -114 km is a measurement only if the margin is known to
better than 114 km. Agreement on the RANKING of a dozen hotspot endpoints does
not establish that: any province definition places Louisville outside and Samoa
deep inside, and those two alone will carry a rank correlation while the boundary
wanders by half a province. This measures the boundary directly.

  AREA OVERLAP. The Jaccard index of the two masks, weighted by cell area, and
  each model's province area as a fraction of the sphere. Two provinces that
  agree in position but differ in size score poorly here and should.

  BOUNDARY DISPLACEMENT. For every boundary cell of one model, the great-circle
  distance to the nearest boundary cell of the other. The median of that is the
  number a margin distance has to be compared against. It is computed exactly
  rather than through a Euclidean distance transform, which on a latitude and
  longitude grid is wrong by the cosine of the latitude.

  THE VOTE MAP. The count of models in which each cell is province. Individual
  models disagree about province boundaries - this is why cluster analysis across
  models exists - so a province defined by agreement among several is the
  defensible object, and the vote map is written out for use as one. REVEAL and
  RevealLO share data and methodology, so a vote taken over both double-counts
  one inversion; the pair is reported and can be collapsed with --lineage.

Every model is resampled onto a common grid first, so a comparison between a
model sampled at half a degree and one at two degrees is a comparison of
provinces and not of samplings.
"""
from __future__ import annotations

import argparse, itertools, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morphology as M
from tomo_io import ModelSpec, load_anomaly, dedupe_lon

R_E, DEG = 6371.0, np.pi / 180.0


def window_field(path, tag, var, z0, z1, glat, glon, every=2):
    """A model's mean anomaly over the depth window, on the common grid."""
    from scipy.interpolate import RegularGridInterpolator
    depth, lat, lon, arr = load_anomaly(path, ModelSpec(tag, var),
                                        depth_max=z1 + 40, every=every)
    lon, arr = dedupe_lon(lon, arr)
    z = np.asarray(depth, float)
    k = np.where((z >= z0) & (z <= z1))[0]
    if not len(k):
        raise SystemExit(f'{tag}: no shells between {z0:.0f} and {z1:.0f} km')
    f = np.nanmean(arr[k], axis=0)
    # Sorted AND deduplicated. dedupe_lon only removes a wrap meridian when the
    # span is exactly 360 degrees; after load_anomaly rolls a 0-360 file into
    # -180 to 180 a repeated value can survive, and the interpolator refuses a
    # non-strictly-ascending axis rather than silently averaging the pair.
    lo = np.asarray(lon, float)
    lo_u, keep_i = np.unique(lo, return_index=True)
    la = np.asarray(lat, float)
    if la[0] > la[-1]:
        la, f = la[::-1], f[::-1]
    F = RegularGridInterpolator((la, lo_u), f[:, keep_i], bounds_error=False,
                                fill_value=None)
    GLA, GLO = np.meshgrid(glat, glon, indexing='ij')
    return F(np.column_stack([GLA.ravel(), GLO.ravel()])).reshape(GLA.shape)


def mask_of(field, pct, min_cells, area):
    thr = float(np.nanpercentile(field, pct))
    lab, n = M._label_periodic(field <= thr)
    keep = [q for q in range(1, n + 1) if (lab == q).sum() >= min_cells]
    m = np.isin(lab, keep)
    return m, thr


def boundary(m):
    n = np.roll(m, 1, axis=1) & np.roll(m, -1, axis=1)
    n &= np.pad(m[:-1], ((1, 0), (0, 0)), constant_values=False)
    n &= np.pad(m[1:], ((0, 1), (0, 0)), constant_values=False)
    return m & ~n


def bdist(bA, bB, GLA, GLO, chunk=400):
    """Great-circle distance from each boundary cell of A to the nearest of B."""
    ja, ia = np.where(bA)
    jb, ib = np.where(bB)
    if not len(ja) or not len(jb):
        return np.array([])
    la_a, lo_a = GLA[ja, ia] * DEG, GLO[ja, ia] * DEG
    la_b, lo_b = GLA[jb, ib] * DEG, GLO[jb, ib] * DEG
    out = np.empty(len(ja))
    sb, cb = np.sin(la_b), np.cos(la_b)
    for s in range(0, len(ja), chunk):
        e = min(s + chunk, len(ja))
        c = (np.sin(la_a[s:e])[:, None] * sb[None, :]
             + np.cos(la_a[s:e])[:, None] * cb[None, :]
             * np.cos(lo_b[None, :] - lo_a[s:e][:, None]))
        out[s:e] = R_E * np.arccos(np.clip(c, -1, 1)).min(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True,
                    help='tag=path[=var] for each model')
    ap.add_argument('--z0', type=float, default=2600.0)
    ap.add_argument('--z1', type=float, default=2880.0)
    ap.add_argument('--pct', type=float, default=20.0)
    ap.add_argument('--grid', type=float, default=1.0,
                    help='common grid spacing in degrees')
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--min-cells', type=int, default=60, dest='min_cells')
    ap.add_argument('--lineage', nargs='+', default=['REVEAL', 'RevealLO',
                                                     'RevealLO_30km'],
                    help='tags sharing an inversion lineage; the vote counts '
                         'them once between them')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    glat = np.arange(-89.5, 90.0, a.grid)
    glon = np.arange(-180.0, 180.0, a.grid)
    GLA, GLO = np.meshgrid(glat, glon, indexing='ij')
    area = np.cos(GLA * DEG)

    masks, thrs = {}, {}
    for spec in a.models:
        parts = spec.split('=')
        tag, path = parts[0], parts[1]
        var = parts[2] if len(parts) > 2 else 'voigt'
        print(f'{tag}: reading', flush=True)
        f = window_field(path, tag, var, a.z0, a.z1, glat, glon, a.every)
        masks[tag], thrs[tag] = mask_of(f, a.pct, a.min_cells, area)
        print(f'  threshold {thrs[tag]:+.3f}, province area '
              f'{100 * float((masks[tag] * area).sum() / area.sum()):.1f} per cent '
              f'of the sphere', flush=True)

    tags = list(masks)
    rows = []
    print('\npairwise agreement between province definitions\n')
    print(f'{"pair":28s} {"area Jaccard":>13s} {"median boundary":>16s} '
          f'{"90th pct":>9s}')
    for t1, t2 in itertools.combinations(tags, 2):
        m1, m2 = masks[t1], masks[t2]
        inter = float(((m1 & m2) * area).sum())
        union = float(((m1 | m2) * area).sum())
        jac = inter / union if union else np.nan
        d = np.concatenate([bdist(boundary(m1), boundary(m2), GLA, GLO),
                            bdist(boundary(m2), boundary(m1), GLA, GLO)])
        med = float(np.median(d)) if d.size else np.nan
        p90 = float(np.percentile(d, 90)) if d.size else np.nan
        shared = (t1 in a.lineage and t2 in a.lineage)
        rows.append(dict(model_a=t1, model_b=t2, jaccard=jac,
                         boundary_median_km=med, boundary_p90_km=p90,
                         shared_lineage=shared))
        print(f'{t1 + " / " + t2:28s} {jac:13.3f} {med:13.0f} km {p90:8.0f} km'
              + ('   (shared lineage)' if shared else ''))

    # the vote map, counting a shared lineage once
    groups, seen = [], set()
    lin = [t for t in tags if t in a.lineage]
    if lin:
        groups.append(lin)
        seen.update(lin)
    groups += [[t] for t in tags if t not in seen]
    vote = np.zeros(GLA.shape, int)
    for g in groups:
        vote += np.any([masks[t] for t in g], axis=0).astype(int)
    n = len(groups)
    print(f'\nvote map over {n} independent families '
          f'({", ".join("+".join(g) for g in groups)})')
    for k in range(1, n + 1):
        print(f'  province in at least {k} of {n}: '
              f'{100 * float(((vote >= k) * area).sum() / area.sum()):5.2f} per '
              f'cent of the sphere')
    out = a.out or os.path.join(a.dir, 'province_vote.npz')
    np.savez_compressed(out, vote=vote.astype(np.int8), lat=glat, lon=glon,
                        n_families=np.array([n]),
                        thresholds=np.array([thrs[t] for t in tags]),
                        tags=np.array(tags))
    pd.DataFrame(rows).to_csv(os.path.join(a.dir, 'province_agreement.csv'),
                              index=False)
    ind = [r for r in rows if not r['shared_lineage']]
    if ind:
        print(f'\nacross independent families the boundaries differ by a median '
              f'of {np.median([r["boundary_median_km"] for r in ind]):.0f} km '
              f'(Jaccard {np.median([r["jaccard"] for r in ind]):.2f}).')
        print('A margin distance smaller than that is not resolved by these')
        print('models, whatever the rank correlation between them says.')
    print(f'\nwrote {os.path.basename(out)} and province_agreement.csv')


if __name__ == '__main__':
    main()
