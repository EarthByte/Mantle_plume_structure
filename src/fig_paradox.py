#!/usr/bin/env python3
"""Figure: geological persistence and tomographic distinctness are not the same
observable.

Panel (a) puts the dated duration of each hotspot's volcanic record against the
fraction of retained configurations under which the least-cost search roots it,
and panel (b) puts the root fraction of each track class beside the others in
every model. Nothing on either panel is derived from the morphology analysis, and
the track classes on both are the ones fixed before any corridor was computed.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle
from figstyle import CM, INK, BLU, ACC, GRY

COL = {'P1': ACC, 'P2': BLU, 'P3': GRY}
LBL = {'P1': 'P1  long and supported', 'P2': 'P2  gapped or disputed',
       'P3': 'P3  short or ambiguous'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='out')
    ap.add_argument('--models', nargs='+',
                    default=['RevealLO', 'REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1'])
    ap.add_argument('--out', default='figures/fig_paradox.pdf')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    figstyle.apply(10.5)

    cls = pd.read_csv(os.path.join(a.dir, 'track_classes.csv'))
    models = [m for m in a.models
              if os.path.exists(os.path.join(a.dir, f'classification_{m}.csv'))]
    if not models:
        raise SystemExit('no classification tables found')
    base = pd.read_csv(os.path.join(a.dir, f'classification_{models[0]}.csv'))
    m = base.merge(cls[['hotspot', 'track_class', 'duration_myr']], on='hotspot')

    fig = plt.figure(figsize=(17.5 * CM, 8.4 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.34,
                          left=0.105, right=0.975, bottom=0.26, top=0.905)

    ax = fig.add_subplot(gs[0, 0])
    d = m.dropna(subset=['duration_myr'])
    for cl in ('P3', 'P2', 'P1'):
        g = d[d.track_class == cl]
        ax.scatter(g.duration_myr, g.root_fraction, s=34 if cl != 'P3' else 20,
                   facecolor=COL[cl], edgecolor='white', linewidth=0.6,
                   zorder=3 if cl == 'P1' else 2, label=LBL[cl])
    # The four P1 hotspots the search does not root all sit at zero, so labelling
    # them in place piles four names on one point. They are fanned upward at
    # fixed heights with leaders instead, which keeps the reading unambiguous
    # without moving any datum.
    p1 = d[d.track_class == 'P1'].sort_values('duration_myr')
    heights = [0.30, 0.45, 0.60, 0.75, 0.90]
    for q, (_, r) in enumerate(p1.iterrows()):
        y = r.root_fraction if r.root_fraction > 0.6 else heights[q % len(heights)]
        xt = 168.0
        # The leader is drawn as a line and the name as plain text rather than as
        # an annotation with arrowprops: the bounding box of an annotation
        # includes its arrow, so figstyle.check would measure the leader as part
        # of the label and report a collision at every anchor point.
        ax.plot([r.duration_myr, xt * 0.985], [r.root_fraction, y], lw=0.6,
                color=GRY, zorder=1, solid_capstyle='butt')
        ax.text(xt, y, r.hotspot.split('(')[0].strip(), fontsize=9, color=INK,
                va='center', ha='left')
    ax.set_xscale('log')
    ax.set_xlabel('dated duration of the volcanic record (Myr)')
    ax.set_ylabel('fraction of configurations\nrooting the hotspot')
    ax.set_ylim(-0.08, 1.16)
    ax.set_xlim(3, 460)
    ax.set_xticks([5, 10, 20, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(loc='center left', frameon=False, handletextpad=0.4,
              borderpad=0.1, labelspacing=0.3, fontsize=9,
              bbox_to_anchor=(-0.015, 0.62))
    fig.text(0.012, 0.955, '(a)', fontsize=11, fontweight='bold', color=INK)

    ax2 = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(4)
    for xi, tag in enumerate(models):
        c = pd.read_csv(os.path.join(a.dir, f'classification_{tag}.csv'))
        mm = c.merge(cls[['hotspot', 'track_class']], on='hotspot')
        for oi, cl in enumerate(('P1', 'P2', 'P3')):
            v = mm[mm.track_class == cl].root_fraction.values
            if not len(v):
                continue
            x = xi + (oi - 1) * 0.26
            ax2.scatter(x + 0.05 * rng.standard_normal(len(v)), v, s=13,
                        facecolor=COL[cl], edgecolor='none', alpha=0.75)
            ax2.plot([x - 0.11, x + 0.11], [v.mean()] * 2, color=INK, lw=1.6,
                     solid_capstyle='butt', zorder=4)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([t.replace('SEMUCB-WM1', 'SEMUCB')
                         .replace('GLADM35', 'GLAD-M35') for t in models],
                        rotation=32, ha='right')
    ax2.set_ylabel('root fraction')
    ax2.set_ylim(-0.08, 1.16)
    ax2.set_xlim(-0.55, len(models) - 0.45)
    fig.text(0.582, 0.955, '(b)', fontsize=11, fontweight='bold', color=INK)
    for oi, cl in enumerate(('P1', 'P2', 'P3')):
        ax2.text(0.02 + 0.10 * oi, 0.955, cl, transform=ax2.transAxes,
                 color=COL[cl], fontsize=10, fontweight='bold')

    figstyle.check(fig, placed_cm=17.5)
    fig.savefig(a.out, bbox_inches='tight')
    fig.savefig(a.out.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
