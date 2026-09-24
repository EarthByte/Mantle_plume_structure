#!/usr/bin/env python3
"""Figure: where the corridor sits, seen from above, in each tomographic model.

Each panel is one hotspot, seen in plan view, with the hotspot at the origin.
The line traces the displacement of the corridor centroid from the surface down
to the base of the mantle, once for every model, and its colour carries depth.
Four independent inversion families are drawn solid; REVEAL is drawn faint
because it shares data and methodology with RevealLO and agreement between them
is a resolution test rather than a replication.

This is the quantity the cross-model test found to be reproducible. The width,
anisotropy and dominance of the same corridors are not: between independent
families their per-hotspot rankings correlate at about zero, against 0.6 to 0.9
between REVEAL and RevealLO. Nothing about the shape of a corridor is drawn here
for that reason, and the panels carry only where the structure is, not what it
looks like.
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle
from figstyle import CM, INK, GRY

DEG = np.pi / 180.0
FAINT = ('REVEAL', 'RevealLO_30km')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='out')
    ap.add_argument('--out', default='figures/fig_displacement.pdf')
    ap.add_argument('--zmax', type=float, default=2800.0)
    ap.add_argument('--lim', type=float, default=None,
                    help='half-width of each panel in km; default is the 95th '
                         'percentile of the displacements drawn')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    figstyle.apply(10.0)

    per = pd.read_csv(os.path.join(a.dir, 'morph_displacement_by_model.csv'))
    agr = pd.read_csv(os.path.join(a.dir, 'morph_displacement.csv'))
    per = per[per.depth_km <= a.zmax]
    per['east'] = per.offset_km * np.sin(per.azimuth_deg * DEG)
    per['north'] = per.offset_km * np.cos(per.azimuth_deg * DEG)
    sites = sorted(per.site.unique())
    models = sorted(per.model.unique())
    if a.lim is None:
        a.lim = float(np.nanpercentile(per.offset_km, 95)) * 1.15

    agree = {s: int(g.agrees.sum()) for s, g in agr.groupby('site')}
    total = {s: int(len(g)) for s, g in agr.groupby('site')}

    ncol = 4
    nrow = math.ceil(len(sites) / ncol)
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True,
                             figsize=(17.5 * CM, (4.0 * nrow + 4.4) * CM))
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.028 + 0.60 / nrow,
                        top=0.955, wspace=0.17, hspace=0.44)
    cmap = plt.get_cmap('viridis')
    norm = matplotlib.colors.Normalize(200.0, a.zmax)

    for q, s in enumerate(sites):
        ax = axes[q // ncol, q % ncol]
        ax.axhline(0, color=figstyle.GRID, lw=0.8, zorder=1)
        ax.axvline(0, color=figstyle.GRID, lw=0.8, zorder=1)
        for r in (500.0, 1000.0, 1500.0):
            if r < a.lim:
                ax.add_patch(plt.Circle((0, 0), r, fill=False, lw=0.6,
                                        edgecolor=figstyle.GRID, zorder=1))
        for m in models:
            g = per[(per.site == s) & (per.model == m)].sort_values('depth_km')
            if len(g) < 2:
                continue
            pts = np.column_stack([g.east.values, g.north.values])
            seg = np.stack([pts[:-1], pts[1:]], axis=1)
            faint = m in FAINT
            lc = LineCollection(seg, cmap=cmap, norm=norm,
                                linewidths=1.0 if faint else 1.9,
                                alpha=0.35 if faint else 1.0, zorder=2 if faint else 3)
            lc.set_array(g.depth_km.values[:-1])
            ax.add_collection(lc)
            ax.plot(pts[-1, 0], pts[-1, 1], 'o', ms=3.5 if not faint else 2.2,
                    color=cmap(norm(g.depth_km.values[-1])),
                    markeredgecolor='white', markeredgewidth=0.5,
                    zorder=4 if not faint else 2)
        ax.plot(0, 0, marker='*', ms=8, color=INK, zorder=5)
        ax.set_xlim(-a.lim, a.lim)
        ax.set_ylim(-a.lim, a.lim)
        ax.set_aspect('equal')
        name = (s.replace('_', ' ').replace('Cook-Austral', '')
                .replace('Great Meteor New England', 'Great Meteor')
                .replace('KerguelenHeard', 'Kerguelen')
                .replace('Tahiti Society', 'Tahiti').strip())
        ax.set_title(f'{name}\n{agree.get(s, 0)} of {total.get(s, 0)} depth bins '
                     f'agree', fontsize=9.5, color=INK, pad=3, linespacing=1.25)
    for q in range(len(sites), nrow * ncol):
        axes[q // ncol, q % ncol].set_visible(False)
    for c in range(ncol):
        axes[-1, c].set_xlabel('east of the hotspot (km)', fontsize=9)
    for r in range(nrow):
        axes[r, 0].set_ylabel('north (km)', fontsize=9)

    cax = fig.add_axes([0.085, 0.030 + 0.30 / nrow, 0.34, 0.010])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, orientation='horizontal')
    cb.set_label('depth (km)', fontsize=9)
    cb.ax.tick_params(labelsize=8.5)
    fig.text(0.50, 0.028 + 0.30 / nrow,
             'star: hotspot.  rings at 500, 1000 and 1500 km.\n'
             'faint line: REVEAL, which shares a lineage with RevealLO.',
             fontsize=9, color=GRY, linespacing=1.35)
    figstyle.check(fig, placed_cm=17.5)
    fig.savefig(a.out, bbox_inches='tight')
    fig.savefig(a.out.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
