#!/usr/bin/env python3
"""Turn the occupancy and corridor volumes into the morphology table.

Two tables come out. The profile carries one row per site, field and depth, and
is what the depth panels of the figures are drawn from. The summary reduces each
site and field to the handful of scalars the hypotheses are separated on, and
carries, for every one of them, the percentile of the value within the matched
null locations measured in the same run - so that a corridor 700 km wide is
reported as wide or narrow against corridors elsewhere in the same mantle rather
than against an intuition.

The nulls are matched on lowermost-mantle setting. A hotspot above a large low
velocity province is compared with random locations above material at least as
slow, because a corridor is wide wherever the deep mantle is slow and a global
null would attribute that to the hotspot.

CROSS-CONFIGURATION STABILITY WITHOUT KEEPING EVERY MASK

Storing one corridor per configuration for every site would be tens of gigabytes.
It is not necessary: if c(x) configurations out of n contain cell x, the sum of
the pairwise intersections over all pairs is the sum over cells of c(c-1)/2, and
the sum of the pairwise unions is the sum over cells of [n(n-1) - (n-c)(n-c-1)]/2.
Their ratio is the pooled Jaccard overlap, which is exact and needs only the
count field. It is a pooled figure and is named as one; it is not the mean of the
pairwise Jaccard values, which is not recoverable from counts.
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morphology as M

BANDS = (('upper', 200.0, 660.0), ('mid', 660.0, 1700.0),
         ('lower', 1700.0, 2700.0))


def pooled_jaccard(counts, n):
    """Pooled pairwise overlap of the per-configuration sets, from the counts."""
    c = np.asarray(counts, float).ravel()
    c = c[c > 0]
    if not c.size or n < 2:
        return np.nan
    inter = float(np.sum(c * (c - 1.0)))
    union = float(np.sum(n * (n - 1.0) - (n - c) * (n - c - 1.0)))
    return inter / union if union > 0 else np.nan


def fields_in(store, key, taus):
    out = {'occupancy': store[f'occ|{key}']}
    for t in taus:
        k = f'cor{t:g}|{key}'
        if k in store:
            out[f'corridor{t:g}'] = store[k]
    return out


def summarise(rows, W, depth, n_cfg, paths, hlat, hlon):
    s = {}
    for nm, z0, z1 in BANDS:
        sel = [r for r in rows if z0 <= r['depth_km'] <= z1 and r['weight'] > 0]
        for q in ('r50_km', 'r90_km', 'anisotropy', 'offset_km'):
            s[f'{q}_{nm}'] = (float(np.nanmedian([r[q] for r in sel]))
                              if sel else np.nan)
        s[f'ncomp_max_{nm}'] = (int(np.nanmax([r['n_components'] for r in sel]))
                                if sel else 0)
        s[f'main_share_{nm}'] = (float(np.nanmedian([r['main_share'] for r in sel]))
                                 if sel else np.nan)
    s['turning_total_deg'], s['turning_net_deg'] = M.azimuth_turning(rows)
    s['gap_fraction'] = M.gap_fraction(rows)
    s['pooled_jaccard'] = pooled_jaccard(np.round(W * n_cfg), n_cfg)
    s['volume_cells'] = int(np.sum(W > 0))
    s['weighted_cells'] = float(np.sum(W))
    if paths:
        s.update(M.path_metrics(paths, hlat, hlon))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--dir', default='out')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--comp-floor', type=float, default=0.10, dest='comp_floor',
                    help='agreement a cell must command to belong to a '
                         'component, as a fraction of the configurations')
    ap.add_argument('--match-k', type=int, default=15, dest='match_k',
                    help='nulls matched to each hotspot, taken as the k whose '
                         'lowermost-mantle anomaly is closest to it. A fixed '
                         'window was tried first and failed: random locations '
                         'almost never land inside a large low-velocity '
                         'province, so a hotspot at -2 per cent found fewer '
                         'than eight matches within half a per cent and fell '
                         'back to the global set, silently, for five of twelve '
                         'targets. Nearest-k always returns a set and reports '
                         'how badly matched it is. 0 disables matching')
    a = ap.parse_args()

    vol = os.path.join(a.dir, f'morph_volumes_{a.tag}{a.suffix}.npz')
    store = np.load(vol, allow_pickle=False)
    depth, lat, lon = store['depth'], store['lat'], store['lon']
    taus = store['taus']
    n_cfg = int(store['n_config'][0])
    keys = sorted({k.split('|', 1)[1] for k in store.files if k.startswith('occ|')})
    P = np.load(os.path.join(a.dir, f'morph_paths_{a.tag}{a.suffix}.npz'))

    prof_rows, sum_rows = [], []
    for key in keys:
        jb, ib = store[f'jb|{key}'], store[f'ib|{key}']
        hlat, hlon = store[f'site|{key}']
        deep = float(store[f'deep|{key}'][0]) if f'deep|{key}' in store else np.nan
        sub_lat, sub_lon = lat[jb], lon[ib]
        paths = [tuple(P[k]) for k in P.files if k.split('|')[0].replace(
            '/', '_').replace(' ', '_').replace('(', '').replace(')', '') == key]
        for fname, W in fields_in(store, key, taus).items():
            rows = M.profile(W, depth, sub_lat, sub_lon, hlat, hlon,
                             comp_floor=a.comp_floor)
            for r in rows:
                prof_rows.append(dict(site=key, field=fname, **r))
            s = summarise(rows, W, depth, n_cfg, paths, hlat, hlon)
            sum_rows.append(dict(site=key, field=fname, lat=hlat, lon=hlon,
                                 deep_dvs=deep, n_config=n_cfg, **s))

    prof = pd.DataFrame(prof_rows)
    summ = pd.DataFrame(sum_rows)

    # ---- matched-null percentiles
    isnull = summ.site.str.startswith('null')
    metrics = [c for c in summ.columns if c not in
               ('site', 'field', 'lat', 'lon', 'deep_dvs', 'n_config',
                'endpoint_lat', 'endpoint_lon', 'n_paths')
               and pd.api.types.is_numeric_dtype(summ[c])]
    pct = {}
    for _, r in summ[~isnull].iterrows():
        nul = summ[isnull & (summ.field == r.field)]
        mis = np.nan
        if a.match_k > 0 and np.isfinite(r.deep_dvs) and len(nul):
            dd = (nul.deep_dvs - r.deep_dvs).abs()
            m = nul.loc[dd.nsmallest(min(a.match_k, len(nul))).index]
            matched = len(m)
            mis = float((m.deep_dvs - r.deep_dvs).abs().max())
        else:
            m, matched = nul, len(nul)
        for c in metrics:
            v, ref = r[c], m[c].dropna().values
            pct[(r.site, r.field, c)] = (
                float(100.0 * np.mean(ref <= v)) if len(ref) and np.isfinite(v)
                else np.nan)
        pct[(r.site, r.field, '_n_matched')] = matched
        pct[(r.site, r.field, '_mismatch')] = mis
    if pct:
        add = []
        for _, r in summ.iterrows():
            row = {f'pct_{c}': pct.get((r.site, r.field, c), np.nan)
                   for c in metrics}
            row['n_matched_nulls'] = pct.get((r.site, r.field, '_n_matched'), np.nan)
            row['deep_mismatch_pct'] = pct.get((r.site, r.field, '_mismatch'), np.nan)
            add.append(row)
        summ = pd.concat([summ.reset_index(drop=True),
                          pd.DataFrame(add)], axis=1)

    prof.to_csv(os.path.join(a.dir, f'morph_profile_{a.tag}{a.suffix}.csv'),
                index=False)
    summ.to_csv(os.path.join(a.dir, f'morph_summary_{a.tag}{a.suffix}.csv'),
                index=False)
    named = summ[~summ.site.str.startswith('null')]
    show = ['site', 'field', 'r50_km_lower', 'r90_km_lower', 'anisotropy_lower',
            'ncomp_max_lower', 'turning_total_deg', 'gap_fraction',
            'pooled_jaccard']
    show = [c for c in show if c in named.columns]
    print(named[show].to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    print(f'\nwrote morph_profile_{a.tag}{a.suffix}.csv and '
          f'morph_summary_{a.tag}{a.suffix}.csv '
          f'({len(summ)} site-field rows, {int(isnull.sum())} of them nulls)')


if __name__ == '__main__':
    main()
