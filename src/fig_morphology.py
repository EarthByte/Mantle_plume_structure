#!/usr/bin/env python3
"""Figure: the near-optimal corridor beneath each target, hotspot by hotspot.

One small panel per hotspot rather than twelve curves on one axis. Twelve series
cannot be separated by colour - no seven-hue categorical scale survives a check
for colour-vision deficiency, let alone twelve - and the comparison the reader
has to make is between the shapes of the profiles, which small multiples put side
by side without asking anyone to hold a colour key in their head.

Each panel carries the corridor width against depth as a filled envelope, the
displacement of its centroid from the hotspot as a line, and behind both the
interquartile range of the same two quantities at the matched null locations. A
hotspot whose corridor is unremarkable has its curves inside the grey; one whose
corridor is genuinely broad, or genuinely displaced, leaves it.

Panels are red where the least-cost search does not root the hotspot and blue
where it does, so the reader can see at a glance whether the two groups differ in
shape - which is the question the reframed paper turns on.
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle
from figstyle import CM, INK, BLU, ACC, GRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--field', default='corridor0.02')
    ap.add_argument('--out', default='figures/fig_morphology.pdf')
    ap.add_argument('--zmax', type=float, default=2800.0)
    ap.add_argument('--xmax', type=float, default=None,
                    help='common horizontal limit in km; default is the 98th '
                         'percentile over every panel, so one broad corridor '
                         'cannot flatten the rest')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    figstyle.apply(10.0)

    prof = pd.read_csv(os.path.join(a.dir, f'morph_profile_{a.tag}{a.suffix}.csv'))
    prof = prof[prof.field == a.field]
    cls = pd.read_csv(os.path.join(a.dir, 'track_classes.csv'))
    cl = pd.read_csv(os.path.join(a.dir, f'classification_{a.tag}.csv'))
    key = {h.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', ''): h
           for h in cls.hotspot}
    tcls = dict(zip(cls.hotspot, cls.track_class))
    rootf = dict(zip(cl.hotspot, cl.root_fraction))
    named = [s for s in prof.site.unique() if not s.startswith('null')]
    named.sort(key=lambda s: ({'P1': 0, 'P2': 1}.get(tcls.get(key.get(s, s)), 2),
                              -rootf.get(key.get(s, s), 0.0)))
    nul = prof[prof.site.str.startswith('null') & (prof.weight > 0)]

    n = len(named)
    ncol = 4
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True,
                             figsize=(17.5 * CM, (2.9 * nrow + 4.6) * CM))
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(left=0.095, right=0.985,
                        bottom=0.030 + 0.62 / nrow, top=0.945,
                        wspace=0.18, hspace=0.34)

    if a.xmax is None:
        v = prof[prof.weight > 0][['r90_km', 'offset_km']].values
        v = v[np.isfinite(v)]
        a.xmax = float(np.nanpercentile(v, 98)) if v.size else 1000.0
    band = None
    if len(nul):
        g = nul.groupby('depth_km')
        band = (g.r90_km.quantile(0.25), g.r90_km.quantile(0.75),
                g.offset_km.quantile(0.25), g.offset_km.quantile(0.75))

    for q, s in enumerate(named):
        ax = axes[q // ncol, q % ncol]
        name = key.get(s, s)
        rf = rootf.get(name, np.nan)
        col = BLU if (np.isfinite(rf) and rf >= 0.8) else ACC
        if band is not None:
            ax.fill_betweenx(band[0].index, band[0].values, band[1].values,
                             color=GRY, alpha=0.30, lw=0, zorder=1)
            ax.plot(band[3].values, band[3].index, color=GRY, lw=0.8, ls=':',
                    zorder=2)
        d = prof[(prof.site == s) & (prof.weight > 0)].sort_values('depth_km')
        ax.fill_betweenx(d.depth_km.values, 0.0, d.r90_km.values, color=col,
                         alpha=0.30, lw=0, zorder=3)
        ax.plot(d.r90_km.values, d.depth_km.values, color=col, lw=1.2, zorder=4)
        ax.plot(d.offset_km.values, d.depth_km.values, color=INK, lw=1.0,
                ls='--', zorder=5)
        ax.set_ylim(a.zmax, 100)
        ax.set_xlim(0, a.xmax)
        ax.grid(axis='y', color=figstyle.GRID, lw=0.6)
        short = name.split('(')[0].strip().replace('Great Meteor/New England',
                                                   'Great Meteor')
        ax.set_title(f'{short}   {tcls.get(name, "")}   '
                     f'root {rf:.2f}'.rstrip(), fontsize=9.5, color=INK, pad=3)
    for q in range(n, nrow * ncol):
        axes[q // ncol, q % ncol].set_visible(False)
    for c in range(ncol):
        axes[-1, c].set_xlabel('kilometres', fontsize=9.5)
    for r in range(nrow):
        axes[r, 0].set_ylabel('depth (km)', fontsize=9.5)

    handles = [plt.Line2D([], [], color=ACC, lw=2.4,
                          label='corridor width (r90), not rooted'),
               plt.Line2D([], [], color=BLU, lw=2.4,
                          label='corridor width (r90), rooted'),
               plt.Line2D([], [], color=INK, lw=1.2, ls='--',
                          label='centroid offset from the hotspot'),
               plt.Line2D([], [], color=GRY, lw=6, alpha=0.5,
                          label='matched nulls, interquartile width'),
               plt.Line2D([], [], color=GRY, lw=1.0, ls=':',
                          label='matched nulls, upper quartile offset')]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.006), handlelength=2.0,
               columnspacing=1.6, labelspacing=0.4)
    figstyle.check(fig, placed_cm=17.5)
    fig.savefig(a.out, bbox_inches='tight')
    fig.savefig(a.out.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
