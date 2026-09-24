#!/usr/bin/env python3
"""Figure: which conduit shapes the root test sees, and which it does not.

Each row is an injected morphology and each column a question asked of it: does
the least-cost search root the site, does the traced path follow the injected
axis, and does the near-optimal corridor describe the shape correctly. Reading
across a row says what would happen to a plume of that shape; reading down the
first column says which shapes the published classification is able to detect at
all.

The distinction the caption must carry: this measures the analysis applied to a
model, not the ability of seismic data to place such a structure in a model.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle
from figstyle import CM, INK, GRY
import synth_shapes as S

PRETTY = {'vertical': 'vertical tube', 'uniform': 'uniform lean',
          'transition': 'transition-zone step', 's_shaped': 'S-shaped',
          'reversing': 'reversing azimuth', 'bifurcating': 'bifurcating',
          'sheet': 'sheet', 'tapered': 'amplitude tapered',
          'gapped': 'gap at 1400-1800 km', 'narrow_in_broad': 'narrow in a province',
          'twin_branch': 'twin branches'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--out', default='figures/fig_synth_morph.pdf')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    figstyle.apply(10.0)

    df = pd.read_csv(os.path.join(a.dir, f'synth_morph_{a.tag}{a.suffix}.csv'))
    cal = df[df.amp <= -1.0]
    shapes = [s for s in S.NAMES if s in set(cal['shape'])]
    g = cal.groupby('shape')
    det = g.detected.mean().reindex(shapes)
    rec = (g.recovered.mean().reindex(shapes) if 'recovered' in cal
           else pd.Series(np.nan, index=shapes))
    cols = [('detected by\nthe root test', det),
            ('path follows\nthe injected axis', rec)]
    extra = []
    for key, lab, fmt in (('corr_r50_km', 'corridor r50 (km)', '{:.0f}'),
                          ('corr_anisotropy', 'anisotropy', '{:.2f}'),
                          ('corr_ncomp_max', 'components', '{:.0f}')):
        if key in df.columns and df[key].notna().any():
            extra.append((lab, df[df[key].notna()].groupby('shape')[key]
                          .median().reindex(shapes), fmt))

    ncol = len(cols) + len(extra)
    fig, axes = plt.subplots(1, ncol, sharey=True,
                             figsize=(17.5 * CM, (0.66 * len(shapes) + 4.2) * CM))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.235, right=0.985, bottom=0.22, top=0.775,
                        wspace=0.24)
    y = np.arange(len(shapes))
    for ax, (lab, v) in zip(axes, cols):
        ax.barh(y, v.values, height=0.62, color=[INK if q >= 0.75 else
                                                 ('#B4442E' if q < 0.5 else GRY)
                                                 for q in v.values])
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 1.0])
        ax.set_title(lab, fontsize=9.5, pad=6)
        for q, val in zip(y, v.values):
            if np.isfinite(val):
                ax.text(min(val + 0.04, 0.99), q, f'{val:.2f}', va='center',
                        ha='left' if val < 0.7 else 'right', fontsize=8.5,
                        color='white' if val >= 0.7 else INK)
    for ax, (lab, v, fmt) in zip(axes[len(cols):], extra):
        m = float(np.nanmax(v.values)) if np.isfinite(v.values).any() else 1.0
        ax.barh(y, v.values, height=0.62, color=GRY)
        ax.set_xlim(0, m * 1.28)
        ax.set_title(lab, fontsize=9.5, pad=6)
        for q, val in zip(y, v.values):
            if np.isfinite(val):
                ax.text(val + m * 0.04, q, fmt.format(val), va='center',
                        ha='left', fontsize=8.5, color=INK)
        ax.set_xticks([0, float(f'{m:.2g}')])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([PRETTY.get(s, s) for s in shapes], fontsize=9.5)
    axes[0].set_ylim(len(shapes) - 0.5, -0.5)
    for ax in axes:
        ax.grid(axis='x', color=figstyle.GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.text(0.235, 0.045, 'Injected at the calibration amplitude. These rates '
             'describe this analysis applied to a model,\nnot the ability of '
             'seismic data to place such a structure in one.', fontsize=9,
             color=INK, linespacing=1.35)
    figstyle.check(fig, placed_cm=17.5)
    fig.savefig(a.out, bbox_inches='tight')
    fig.savefig(a.out.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
