"""Supporting figure: root fraction against longevity of the surface record."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longevity import CATALOGUE, DOCUMENTED

CM = 1 / 2.54
import figstyle as F
F.apply(9.0)
SMALL = 8.5                     # the secondary size used on this figure
INK, ACC, BLU, GRY = F.INK, F.ACC, F.BLU, F.GRY
OUT = 'figures'

LAB = {'age_cat': 'catalogue hotspot age (Ma)',
       'age_doc': 'oldest dated volcanism (Ma)'}

# Only the points that carry the argument are labelled - the best-connected
# hotspots, and the long-lived ones with no root - and each is placed by hand.
# A rule-based offset cannot separate Tahiti from Pitcairn, which are close in
# both age and root fraction, nor Hawaii from Louisville on the zero row.
# (dx as a fraction of the x span, dy in root fraction, horizontal alignment)
PLACE = {
    'age_cat': {
        'Tahiti/Society':           (-0.015, -0.075, 'right'),
        'Pitcairn':                 (+0.015, +0.055, 'left'),
        'Macdonald (Cook-Austral)': (0.0, -0.075, 'center'),
        'Jan Mayen':                (-0.01, +0.075, 'right'),
        'Marion':                   (-0.01, +0.075, 'right'),
        'Hawaii':                   (0.0, +0.055, 'center'),
        'Louisville':               (0.0, +0.130, 'center'),
    },
    'age_doc': {
        'Tahiti/Society':           (-0.015, -0.075, 'right'),
        'Pitcairn':                 (+0.015, +0.055, 'left'),
        'Macdonald (Cook-Austral)': (0.0, -0.075, 'center'),
        'Louisville':               (-0.012, +0.055, 'right'),
        'Hawaii':                   (+0.012, +0.130, 'left'),
        'Kerguelen(Heard)':         (0.0, +0.055, 'center'),
        'Great Meteor/New England': (+0.02, +0.205, 'left'),
        'Tristan':                  (-0.005, +0.285, 'right'),
    },
}


def halo(t, lw=2.0):
    t.set_path_effects([pe.withStroke(linewidth=lw, foreground='white')])
    return t


d = pd.read_csv('out/comparison_all_models.csv')
d['age_cat'] = d.hotspot.map(CATALOGUE)
d['age_doc'] = d.hotspot.map(lambda h: DOCUMENTED.get(h, (np.nan,))[0])

fig, ax = plt.subplots(1, 2, figsize=(14.82 * CM, 6.8 * CM),
                       constrained_layout=True)
for k, col in enumerate(('age_cat', 'age_doc')):
    a = ax[k]
    m = d.dropna(subset=[col, 'f_mean'])
    span = m[col].max() - m[col].min()
    a.axhspan(0.8, 1.02, color='#f3e2de', zorder=0)
    a.axhspan(-0.02, 0.2, color='#e5eaf0', zorder=0)
    sc = a.scatter(m[col], m.f_mean, s=26, c=m.n_deep, cmap='YlOrRd',
                   vmin=0, vmax=5, edgecolor=INK, linewidth=0.6, zorder=3)
    for _, r in m[m.hotspot.isin(PLACE[col])].iterrows():
        dx, dy, ha = PLACE[col][r.hotspot]
        halo(a.text(r[col] + dx * span, r.f_mean + dy,
                    str(r.hotspot).split('(')[0].split('/')[0].strip(),
                    fontsize=SMALL, ha=ha, va='center', color=INK, zorder=4))
    rho, p = spearmanr(m[col], m.f_mean)
    # The statistic sits above the frame; inside it, it collided with the run of
    # labels along the zero row. Provenance of the ages is in the caption.
    a.set_title(f'rho = {rho:+.2f},   p = {p:.2f},   n = {len(m)}', loc='left',
                fontsize=SMALL, color=INK, pad=5)
    a.set_xlabel(LAB[col])
    a.set_xlim(-0.06 * m[col].max(), 1.12 * m[col].max())
    a.set_ylim(-0.03, 1.06)
    if k == 0:
        # shorter than the axis is tall: at reading size the old wording
        # overran the frame at both ends and sat under the panel letter
        a.set_ylabel('mean root fraction')
    halo(a.text(-0.14, 1.06, 'ab'[k], transform=a.transAxes, fontsize=13,
                fontweight='bold', va='bottom', ha='right'))
cb = fig.colorbar(sc, ax=ax, shrink=0.82, ticks=[0, 1, 2, 3, 4, 5], pad=0.02)
cb.set_label('Independent models finding a root', fontsize=SMALL)
cb.ax.tick_params(labelsize=SMALL)
F.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'fig_longevity.{ext}'), bbox_inches='tight',
                facecolor='white')
print('wrote fig_longevity')
