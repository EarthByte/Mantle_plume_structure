#!/usr/bin/env python3
"""Figure 2: what the method does, in one picture.

Five steps, each a short noun phrase and a small drawing of the thing itself. A
workflow diagram earns its place only if a reader who never opens the methods can
still follow the results, so nothing here is a parameter, a file name or a threshold:
those belong in the text. The drawings are schematic and carry no data.

  python3 fig_workflow.py --out ../figures/fig2_workflow
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as FS

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='../figures/fig2_workflow')
A = ap.parse_args()

FS.apply(9.5)
fig, axes = plt.subplots(1, 5, figsize=(FS.PLACED_CM * FS.CM, 4.8 * FS.CM))
fig.subplots_adjust(left=0.055, right=0.988, top=0.92, bottom=0.20, wspace=0.38)

# One-word labels. A workflow panel that needs a phrase to be understood has not been
# drawn clearly enough, and a phrase long enough to wrap drops that panel's label below
# its neighbours.
STEPS = ('hotspot', 'descending path', 'corridor', 'injection test', 'orientation')

zz = np.linspace(0, 1, 60)
axis_x = 0.52 + 0.20 * (1 - zz) ** 2 - 0.20


def frame(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(FS.GRY); sp.set_linewidth(0.6)


for k, ax in enumerate(axes):
    frame(ax)
    if k < 4:
        ax.fill_betweenx(zz, axis_x - 0.15, axis_x + 0.15, color=FS.BLU,
                         alpha=0.18, linewidth=0)
    if k == 2:
        ax.fill_betweenx(zz, axis_x - 0.07, axis_x + 0.07, color=FS.ACC,
                         alpha=0.28, linewidth=0)
    if 1 <= k <= 3:
        ax.plot(axis_x, zz, color=FS.ACC, lw=1.8)
    if k < 4:
        ax.plot([axis_x[-1]], [1.0], marker='v', ms=6.5, color=FS.INK,
                clip_on=False, zorder=5)
    if k == 3:
        # a synthetic of known shape put into the same field, and what comes back:
        # drawn as one object recovered, not as a second plume beside the first
        ax.plot(axis_x, zz, color=FS.INK, lw=2.6, ls=(0, (1.2, 1.6)), zorder=6)
    if k == 4:
        for ang, ln in ((104, 0.30), (126, 0.24), (86, 0.17), (148, 0.13)):
            t = np.radians(ang)
            ax.plot([0.5, 0.5 + ln * np.cos(t)], [0.5 + 0.0, 0.55 + ln * np.sin(t)],
                    color=FS.BLU, lw=3.0, solid_capstyle='round', zorder=3)
        t = np.radians(112)
        ax.annotate('', xy=(0.5 + 0.40 * np.cos(t), 0.55 + 0.40 * np.sin(t)),
                    xytext=(0.5, 0.55), zorder=4,
                    arrowprops=dict(arrowstyle='-|>', color=FS.INK, lw=1.3))
    ax.set_xlabel(STEPS[k], fontsize=9.5, labelpad=6)

for k in range(4):
    fig.patches.append(FancyArrowPatch(
        (axes[k].get_position().x1 + 0.006, 0.56),
        (axes[k + 1].get_position().x0 - 0.006, 0.56),
        transform=fig.transFigure, arrowstyle='-|>', mutation_scale=11,
        color=FS.GRY, lw=1.0, shrinkA=0, shrinkB=0))

# depth sense, once, outside every panel
p0 = axes[0].get_position()
fig.patches.append(FancyArrowPatch((0.030, p0.y1), (0.030, p0.y0),
                                   transform=fig.transFigure, arrowstyle='-|>',
                                   mutation_scale=10, color=FS.GRY, lw=0.9))
fig.text(0.021, 0.5 * (p0.y0 + p0.y1), 'depth', fontsize=8.5, color=FS.GRY,
         rotation=90, va='center', ha='right')

FS.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(f'{A.out}.{ext}', bbox_inches='tight')
print(f'wrote {A.out}.pdf and .png; text checks passed')
