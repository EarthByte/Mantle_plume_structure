#!/usr/bin/env python3
"""Does the province result survive a MATCHED null, as the directional result did not?

Section 3.3.1 compares the fraction of hotspot paths ending inside a low-shear-velocity
province, 71.4 per cent, with the fraction for paths traced from 300 uniformly
distributed starting locations, 41.0 per cent. The null is well powered - 300 paths, not
30 - but uniform over the sphere, and uniform is not matched. Hotspots are overwhelmingly
oceanic, mostly at low to middle latitudes, and further from trenches than a random
point; a uniform null puts a large share of its starting points on cratons, under
continents and at high latitudes, where the descent begins somewhere no hotspot begins.

The directional result showed what that can cost. An ambient null of 30 gave no
preferred orientation and the paper concluded the lean was not an artefact of the search;
at 300 the ambient paths lean away from subduction as well, and matching them to
hotspots on distance to trenches raised their concentration further rather than lowering
it. The lesson is that a null has to resemble the thing it is a null for.

This tests the province contrast three ways, on the same paths:

  UNMATCHED   every null path, as the manuscript reports it.
  OCEANIC     null paths whose start is in oceanic lithosphere, since every hotspot
              in this sample that the comparison rests on is.
  MATCHED     for each hotspot, the nearest null starts in a space of absolute
              latitude and weighted distance to trenches, standardised, without
              replacement, so the null has the hotspots' geography and not the globe's.

If the contrast holds in all three the result is robust. If it collapses under matching,
it was geography.

    python3 province_null_test.py --tag RevealLO

Needs out/root_null_<tag>.csv traced at the CURRENT move set: the file shipped with the
paper predates the retrace, so the comparison is otherwise between a hotspot path from
one search and a null path from another. Re-trace it with

    python3 root_null.py --file <model>.nc --tag RevealLO --lateral 0.60 --ncell 2
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd
import provenance
import freshness
from scipy.stats import fisher_exact, mannwhitneyu

R_E, DEG, CUTOFF = 6371.0, math.pi / 180.0, 6000.0


def gc_km(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    return R_E * np.arccos(np.clip(np.sin(a) * np.sin(c)
                                   + np.cos(a) * np.cos(c) * np.cos(d - b), -1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--hinge', default='out/hinge_migration.csv')
    ap.add_argument('--age-grid', default=None, dest='age_grid',
                    help='seafloor age grid; a start with an age is oceanic. Without '
                         'one the oceanic test is skipped and says so')
    a = ap.parse_args()

    rt = pd.read_csv(os.path.join(a.dir, f'plume_roots_{a.tag}.csv'))
    # The null must come from the same search as the roots. root_null.py is in no
    # runner and its own defaults are the superseded move set, so its output stood five
    # days behind the roots it was compared against: every null percentage matched the
    # published value exactly while the hotspot percentage moved, and the odds ratio
    # collapsing from 2.4 to 1.25 said nothing about the mantle.
    freshness.require_newer(
        os.path.join(a.dir, f'root_null_{a.tag}.csv'),
        os.path.join(a.dir, f'conduit_paths_all_{a.tag}.json'),
        'the null root set',
        'Rerun root_null.py at the production move set: --lateral 0.60 --ncell 2.')
    nl = pd.read_csv(os.path.join(a.dir, f'root_null_{a.tag}.csv'))
    print(f'{len(rt)} hotspot paths, {len(nl)} null paths')
    if 'lateral' in nl.columns:
        print(f'  null traced at lateral {nl.lateral.iloc[0]}, ncell {int(nl.ncell.iloc[0])}')
    else:
        print('  !! the null file records no move set, so it predates the retrace.')
        print('     Re-trace it before believing anything below; see the docstring.')

    t = pd.read_csv(a.hinge)
    tlat, tlon = t.lat.to_numpy(float), t.lon.to_numpy(float)
    al = t.arc_km.to_numpy(float) if 'arc_km' in t.columns else np.ones(len(t))

    def tdist(lat, lon):
        dd = gc_km(lat, lon, tlat, tlon)
        sel = dd <= CUTOFF
        if sel.sum() < 20:
            return np.nan
        w = al[sel] / np.maximum(dd[sel], 200.0) ** 2
        return float((dd[sel] * w).sum() / w.sum())

    # hotspot starts are the surface positions; null starts are recorded
    hs = pd.read_csv('hotspots_courtillot2003.csv').dropna(
        subset=['lat', 'lon_180']).set_index('hotspot')
    rt = rt[rt.site.isin(hs.index)].copy()
    rt['slat'] = [float(hs.at[s, 'lat']) for s in rt.site]
    rt['slon'] = [float(hs.at[s, 'lon_180']) for s in rt.site]
    rt['td'] = [tdist(x, y) for x, y in zip(rt.slat, rt.slon)]
    nl = nl.copy()
    nl['td'] = [tdist(x, y) for x, y in zip(nl.start_lat, nl.start_lon)]

    ROWS = []

    def contrast(nsub, label):
        h_in = int(rt.root_in.astype(bool).sum()); h_n = len(rt)
        n_in = int(nsub.root_in.astype(bool).sum()); n_n = len(nsub)
        if n_n < 20:
            print(f'  {label:11s} only {n_n} null paths; not tested')
            return
        odds, P = fisher_exact([[h_in, h_n - h_in], [n_in, n_n - n_in]],
                               alternative='greater')
        mw = float(mannwhitneyu(rt.root_margin.dropna(),
                                nsub.root_margin.dropna(), alternative='less').pvalue)
        print(f'  {label:11s} hotspots {100*h_in/h_n:5.1f}%   null {100*n_in/n_n:5.1f}%'
              f'  (n={n_n:3d})   Fisher P = {P:.5f}   odds {odds:.2f}'
              f'   margin P = {mw:.4f}')
        # The signed margin mixes two things: whether a path ends inside at all, and
        # how far inside it goes. Reporting the medians beside the test says which,
        # and restricting to the paths that are inside removes the first so the
        # second can be tested on its own. Without this the margin result reads as
        # evidence about rooting depth when it is driven by membership.
        hm = float(rt.root_margin.median()); nm = float(nsub.root_margin.median())
        _row = dict(null=label, n_null=n_n, n_hot=h_n,
                    hot_pct=100 * h_in / h_n, null_pct=100 * n_in / n_n,
                    fisher_p=P, odds=float(odds), margin_p=mw,
                    margin_med_hot=hm, margin_med_null=nm)
        a_ = (-rt.root_margin[rt.root_in.astype(bool)].dropna()).to_numpy(float)
        b_ = (-nsub.root_margin[nsub.root_in.astype(bool)].dropna()).to_numpy(float)
        if len(a_) >= 10 and len(b_) >= 10:
            rg = np.random.default_rng(3)
            obs = float(np.median(a_) - np.median(b_))
            pool = np.concatenate([a_, b_]); kk = len(a_); cnt = 0
            for _ in range(20000):
                q = rg.permutation(pool)
                if abs(np.median(q[:kk]) - np.median(q[kk:])) >= abs(obs) - 1e-12:
                    cnt += 1
            pp = (cnt + 1) / 20001
            _row.update(depth_in_hot=float(np.median(a_)),
                        depth_in_null=float(np.median(b_)),
                        depth_diff=obs, depth_p=pp,
                        n_in_hot=len(a_), n_in_null=len(b_))
            print(f'  {"":11s}   median signed margin {hm:+.0f} vs {nm:+.0f} km;'
                  f'  inside only, depth {np.median(a_):.0f} vs {np.median(b_):.0f} km,'
                  f' difference {obs:+.0f} km, P = {pp:.3f}'
                  f'  (n {len(a_)} vs {len(b_)})')
        else:
            print(f'  {"":11s}   median signed margin {hm:+.0f} vs {nm:+.0f} km;'
                  f'  too few inside to test depth')
        ROWS.append(_row)

    print('\nfraction of paths ending inside a low-velocity province')
    contrast(nl, 'unmatched')

    if a.age_grid and os.path.exists(a.age_grid):
        import xarray as xr
        g = xr.open_dataset(a.age_grid)
        v = [c for c in g.data_vars][0]
        age = [float(g[v].interp(lon=x, lat=y).values) for x, y in
               zip(nl.start_lon, nl.start_lat)]
        contrast(nl[np.isfinite(age)], 'oceanic')
    else:
        print('  oceanic    skipped: no --age-grid given')

    # matched without replacement on |latitude| and weighted trench distance
    m = nl.dropna(subset=['td']).reset_index(drop=True)
    h = rt.dropna(subset=['td'])
    X = np.column_stack([np.abs(m.start_lat.to_numpy(float)), m.td.to_numpy(float)])
    Y = np.column_stack([np.abs(h.slat.to_numpy(float)), h.td.to_numpy(float)])
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-9)
    Xs, Ys = (X - mu) / sd, (Y - mu) / sd
    # One null path per hotspot matches well but leaves only 49 of them, and a
    # Fisher test on 49 against 49 has little power: an odds ratio of 2 can sit at
    # P = 0.07 purely for want of numbers. Matching k nearest keeps the geography and
    # buys back the power, so both are reported and the difference between them says
    # whether a marginal result is a weak effect or a small sample.
    for k in (1, 3, 5):
        taken, idx = set(), []
        for y in Ys:
            d = np.linalg.norm(Xs - y, axis=1)
            got = 0
            for j in np.argsort(d):
                if j not in taken:
                    taken.add(int(j)); idx.append(int(j)); got += 1
                    if got >= k:
                        break
        contrast(m.iloc[idx], f'matched x{k}')
    taken, idx = set(), []
    for y in Ys:
        d = np.linalg.norm(Xs - y, axis=1)
        for j in np.argsort(d):
            if j not in taken:
                taken.add(int(j)); idx.append(int(j)); break

    print('\ngeography of the two sets')
    print(f'  |latitude|  hotspots {np.median(np.abs(h.slat)):5.1f}   '
          f'null {np.median(np.abs(m.start_lat)):5.1f}   '
          f'matched {np.median(np.abs(m.iloc[idx].start_lat)):5.1f}')
    print(f'  trench km   hotspots {np.median(h.td):5.0f}   '
          f'null {np.median(m.td):5.0f}   matched {np.median(m.iloc[idx].td):5.0f}')

    # The manuscript quotes the matched x3 row. Writing the table means the audit
    # reads the same numbers rather than reimplementing the matching, which is how
    # two versions of one quantity got into the paper in the first place.
    out = os.path.join(a.dir, f'province_null_{a.tag}.csv')
    pd.DataFrame(ROWS).to_csv(out, index=False)
    provenance.stamp(out, n_rows=len(ROWS),
                     inputs=[os.path.join(a.dir, f'conduit_paths_all_{a.tag}.json')])
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
