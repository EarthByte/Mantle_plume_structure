"""Figure: deep endpoints against low-velocity provinces (Section 3.3).

Section 3.2 has Figure 4 as its visual summary; Section 3.3 had none, which is what
prompted this (Fichtner, review of G3_MS_v13, comment 44). Three panels, all in
RevealLO, all read from files already written by the pipeline so the figure cannot
disagree with the text: (a) the share of endpoints inside a large low-shear-velocity
province, hotspots against the uniform null and the three matched nulls
(out/province_null_RevealLO.csv); (b) the median signed distance from the nearest
province boundary for the same populations, negative denoting the interior
(out/province_null_RevealLO.csv); (c) the separation between the two endpoints of
hotspots whose two best-fitting descents lie within the 3% cost margin
(out/competing_routes_RevealLO.csv). No title; the argument is in the caption.

    python3 fig_province.py --tag RevealLO
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as FS

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--margin-pct', type=float, default=3.0, dest='margin_pct')
ap.add_argument('--placed-cm', type=float, default=16.01, dest='placed_cm')
ap.add_argument('--out', default='../figures/fig10_province')
A = ap.parse_args()
FS.apply(8.5)

pn = pd.read_csv(os.path.join(A.dir, f'province_null_{A.tag}.csv')).set_index('null')
rt = pd.read_csv(os.path.join(A.dir, f'plume_roots_{A.tag}.csv'))
cr = pd.read_csv(os.path.join(A.dir, f'competing_routes_{A.tag}.csv'))

NULLS = ['unmatched', 'matched x1', 'matched x3', 'matched x5']
NLABEL = {'unmatched': 'uniform', 'matched x1': '×1', 'matched x3': '×3', 'matched x5': '×5'}

fig, (a, b, c) = plt.subplots(1, 3, figsize=(A.placed_cm * FS.CM, 7.6 * FS.CM))

# (a) share of endpoints inside a province: hotspots (one bar, repeated for reference)
# against each null population
x = np.arange(len(NULLS))
hot_pct = float(pn.loc['unmatched', 'hot_pct'])
null_pct = [float(pn.loc[n, 'null_pct']) for n in NULLS]
a.axhline(hot_pct, color=FS.INK, lw=1.5, zorder=2)
a.bar(x, null_pct, 0.6, color='white', edgecolor=FS.BLU, lw=1.2, zorder=3)
a.set_ylim(0, 75)
a.set_xticks(x)
a.set_xticklabels([NLABEL[n] for n in NULLS], fontsize=8.5)
a.set_ylabel('endpoints in a\nprovince (%)')
a.text(0.02, hot_pct / 75 + 0.05, 'hotspots (61.2%, n=49)', transform=a.transAxes,
       fontsize=8.0, color=FS.INK, va='bottom', ha='left')

# (b) median signed margin: hotspots against the same four null populations
hot_marg = float(pn.loc['unmatched', 'margin_med_hot'])
null_marg = [float(pn.loc[n, 'margin_med_null']) for n in NULLS]
b.axhline(0, color=FS.GRY, lw=0.8, zorder=1)
b.axhline(hot_marg, color=FS.INK, lw=1.5, zorder=2)
b.bar(x, null_marg, 0.6, color='white', edgecolor=FS.BLU, lw=1.2, zorder=3)
b.set_ylim(min(hot_marg, 0) - 100, max(null_marg) * 1.35)
b.set_xticks(x)
b.set_xticklabels([NLABEL[n] for n in NULLS], fontsize=8.5)
b.set_ylabel('signed distance from province\nboundary, median (km, − = interior)')
b.text(0.02, 0.95, 'hotspots (−283 km)', transform=b.transAxes, fontsize=8.5,
       color=FS.INK, va='top', ha='left')

# (c) separation between the two competing descents, among hotspots within the margin
sub = cr[cr.margin_pct <= A.margin_pct]
sep = sub.separation_km.to_numpy(float)
bins = np.arange(0, 3300, 250)
c.hist(sep, bins=bins, color='white', edgecolor=FS.BLU, lw=1.2, zorder=3)
med = float(np.median(sep))
c.axvline(med, color=FS.INK, lw=1.5, zorder=4)
c.annotate(f'median {med:.0f} km', xy=(med + 60, c.get_ylim()[1] * 0.92), fontsize=8.5,
           color=FS.INK, ha='left', va='top')
c.set_xlabel('competing-descent\nseparation (km)')
c.set_ylabel(f'hotspots (of {len(sub)}\nwithin {A.margin_pct:.0f}% margin)')

a.set_xlabel('ambient null population')
b.set_xlabel('ambient null population')

for ax, letter in zip((a, b, c), 'abc'):
    ax.text(-0.20, 1.04, letter, transform=ax.transAxes, fontsize=13, fontweight='bold',
            ha='left', va='bottom')
fig.tight_layout(w_pad=1.4)
FS.check(fig, placed_cm=A.placed_cm)
for ext in ('pdf', 'png'):
    fig.savefig(f'{A.out}.{ext}', bbox_inches='tight')
print(f'wrote {A.out}.pdf and .png; text checks passed')
