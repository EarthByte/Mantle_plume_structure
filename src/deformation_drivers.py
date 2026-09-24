#!/usr/bin/env python3
"""Is how much a conduit has been deformed predictable from anything at the surface?

Conduit lean has a direction that published geodynamic modelling predicts. Its
magnitude is a separate question: offsets accumulated over a traced conduit range from
under 100 km to more than 1300 km, and if that spread has a cause visible at the
surface, the candidates are the speed of the overlying plate, distance to a ridge, the
ocean basin, and whether the conduit roots in a province.

This tests each, and also establishes how many independent dimensions the geometry
actually has. Lateral offset and tilt are not two parameters: tilt is the arctangent of
offset over the traced span, so they carry one quantity. Corridor width in three depth
bands is one quantity measured three times. A clustering fed all five will split on
deformation because deformation was entered twice, which is worth knowing before the
two groups it returns are read as populations.

Rank correlation throughout: it assumes no functional form and is not carried by one
extreme hotspot.

  python3 deformation_drivers.py
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--dir', default='out')
ap.add_argument('--apm-window', type=float, default=50.0, dest='apm_window')
A = ap.parse_args()

g = pd.read_csv(os.path.join(HERE, A.dir, f'plume_groups_{A.tag}.csv'), index_col=0)
sl = pd.read_csv(os.path.join(HERE, A.dir, f'plume_slant_{A.tag}.csv')).set_index('site')
apm = pd.read_csv(os.path.join(HERE, A.dir, 'plume_slant_apm.csv'))
w = A.apm_window if (apm.window == A.apm_window).any() else apm.window.min()
g = g.join(apm[apm.window == w].set_index('site')[['apm_mm_yr']])

rows = []
# tilt and offset are the same quantity
t = np.degrees(np.arctan2(sl.offset, sl.span))
rows.append(dict(what='tilt reconstructed from offset, max abs error (deg)',
                 value=float(np.abs(t - sl.tilt).max())))
r, p = spearmanr(g.offset, g.tilt)
rows.append(dict(what='rho offset vs tilt', value=float(r)))
# corridor width is one quantity measured three times
for a, b in (('w_shallow', 'w_mid'), ('w_mid', 'w_deep'), ('w_shallow', 'w_deep')):
    r, _ = spearmanr(g[a], g[b])
    rows.append(dict(what=f'rho {a} vs {b}', value=float(r)))
r, _ = spearmanr(g.w_deep, g.offset)
rows.append(dict(what='rho deep width vs offset', value=float(r)))

# does deformation track anything at the surface?
for col, lab in (('apm_mm_yr', 'plate speed'), ('d_any_ridge', 'spreading centre of any age'),
                 ('root_margin', 'depth into province')):
    if col not in g.columns:
        continue
    m = g[col].notna() & g.offset.notna()
    r, p = spearmanr(g.offset[m], g[col][m])
    rows.append(dict(what=f'rho offset vs {lab}', value=float(r)))
    rows.append(dict(what=f'P offset vs {lab}', value=float(p)))

# and do the two clusters differ on anything other than what defined them?
for col, lab in (('apm_mm_yr', 'plate speed'), ('d_any_ridge', 'spreading centre of any age'),
                 ('root_margin', 'depth into province')):
    if col not in g.columns:
        continue
    a = g.loc[g.group == 1, col].dropna(); b = g.loc[g.group == 0, col].dropna()
    if len(a) >= 3 and len(b) >= 3:
        rows.append(dict(what=f'P group split on {lab}',
                         value=float(mannwhitneyu(a, b).pvalue)))
for col in ('root_in', 'linked'):
    tb = pd.crosstab(g.group, g[col])
    if tb.shape == (2, 2):
        rows.append(dict(what=f'P group split on {col}',
                         value=float(fisher_exact(tb.values)[1])))
for col in ('offset', 'w_deep'):
    for grp in (1, 0):
        rows.append(dict(what=f'median {col} group {grp}',
                         value=float(g.loc[g.group == grp, col].median())))
rows.append(dict(what='n group 1', value=float((g.group == 1).sum())))

r = pd.DataFrame(rows)
r.to_csv(os.path.join(HERE, A.dir, f'deformation_drivers_{A.tag}.csv'), index=False)
print(r.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print('\ngroup 1:', ', '.join(sorted(g.index[g.group == 1])))
