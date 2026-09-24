"""Does ocean island geochemistry relate to conduit GEOMETRY?

A method that assigns each hotspot a deep source displaces it laterally by a few
hundred kilometres against hotspot separations of thousands, so any map of sources
reproduces the map of hotspots almost exactly. Correlations between such a map and a
spatially autocorrelated surface property therefore follow from the projection rather
than from plume structure, and that bound rules out relating geochemistry to conduit
POSITION - which is what an earlier test here did, finding nothing.

Geometry is not subject to that bound. Corridor width, tilt, lateral offset and
deflection depth are not reproductions of the hotspot map, so a relation between them
and lava composition would be a statement about plumes. This tests them, with the
false discovery rate controlled across the whole family, because the number of
isotope-by-geometry pairs is large enough that uncorrected tails are meaningless.
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd
import provenance
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument('--geochem', default='out/geochem_link_RevealLO.csv')
ap.add_argument('--params', default='out/plume_parameters_RevealLO.csv')
ap.add_argument('--groups', default='out/plume_groups_RevealLO.csv')
ap.add_argument('--min-n', type=int, default=12, dest='min_n')
ap.add_argument('--dir', default='out')
A = ap.parse_args()

g = pd.read_csv(A.geochem).set_index('hotspot')
p = pd.read_csv(A.params, index_col=0)
if os.path.exists(A.groups):
    p = p.join(pd.read_csv(A.groups, index_col=0)[['group']])
d = p.join(g, how='inner')
print(f'{len(d)} hotspots with both geometry and geochemistry')

geom = [c for c in ('w_shallow', 'w_mid', 'w_deep', 'offset', 'tilt') if c in d]
iso = [c for c in d.columns if c.startswith('med_') and d[c].notna().sum() >= A.min_n]
print(f'{len(geom)} geometric parameters against {len(iso)} isotope ratios '
      f'with at least {A.min_n} hotspots\n')

rows = []
for gc in geom:
    for ic in iso:
        m = d[[gc, ic]].dropna()
        if len(m) < A.min_n:
            continue
        r, pv = spearmanr(m[gc], m[ic])
        rows.append(dict(geometry=gc, isotope=ic, n=len(m), rho=r, p=pv))
r = pd.DataFrame(rows).sort_values('p').reset_index(drop=True)
r['q'] = r.p * len(r) / (r.index + 1)
r['q'] = r.q[::-1].cummin()[::-1].clip(upper=1.0)
_gg = os.path.join(A.dir, 'geochem_geometry_RevealLO.csv')
r.to_csv(_gg, index=False)
provenance.stamp(_gg, n_rows=len(r),
                 inputs=[os.path.join(A.dir, 'conduit_paths_all_RevealLO.json'),
                         os.path.join(A.dir, 'plume_roots_RevealLO.csv')])

print(f'{len(r)} pairs tested, Benjamini-Hochberg across all of them')
print(f'{"geometry":<12s}{"isotope":<24s}{"n":>4s}{"rho":>8s}{"p":>9s}{"q":>9s}')
for _, x in r.head(8).iterrows():
    print(f'{x.geometry:<12s}{x.isotope[:23]:<24s}{int(x.n):4d}{x.rho:+8.3f}'
          f'{x.p:9.4f}{x.q:9.3f}')
sig = r[r.q < 0.1]
print(f'\n{len(sig)} pairs survive a false discovery rate of 0.1')
if not len(sig):
    print('No relation between conduit geometry and lava composition survives')
    print('correction. Reported as a negative, since the projection bound already')
    print('rules out the position-based version and this rules out the geometric one.')

if 'group' in d and d.group.notna().sum() > 10:
    from scipy.stats import mannwhitneyu
    print('\nthe two conduit populations, tested on the same isotopes:')
    out = []
    for ic in iso:
        a = d[d.group == 1][ic].dropna(); b = d[d.group == 0][ic].dropna()
        if len(a) >= 4 and len(b) >= 4:
            u, pv = mannwhitneyu(a, b)
            out.append((ic, len(a), len(b), float(a.median()), float(b.median()), pv))
    out.sort(key=lambda t: t[-1])
    for ic, na, nb, ma, mb, pv in out[:5]:
        print(f'  {ic[:28]:<30s} group1 {ma:.4g} (n={na})  group0 {mb:.4g} (n={nb})  P = {pv:.3f}')
    if out and min(o[-1] for o in out) * len(out) >= 0.05:
        print('  nothing survives correction for the number of isotopes tested')
