"""Figure S2: corridor width against depth for every hotspot, one panel each.

The main text gives the median across all 49 with its interquartile range, which
hides how differently individual conduits behave. This is the per-hotspot form:
each panel carries that hotspot's profile against the global median, so a reader
can see which conduits are better constrained than typical and at what depths, and
can check any claim made about a named hotspot. Ordered alphabetically. No titles.
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figstyle as FS

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--ncols', type=int, default=7)
ap.add_argument('--out', default='../figures/figS2_corridor_profiles')
A = ap.parse_args()
FS.apply(8.5)

pr = pd.read_csv(os.path.join(A.dir, f'corridor_profiles_{A.tag}.csv'))
sm = pd.read_csv(os.path.join(A.dir, f'corridor_summary_{A.tag}.csv'))
sites = sorted(sm[sm.ok == True].site)
med = pr.groupby('depth').width_km.median()
zg, mg = med.index.to_numpy(float), med.to_numpy(float)
xmax = float(np.ceil(pr.width_km.quantile(0.999) / 100) * 100)

nc = A.ncols
nr = int(np.ceil(len(sites) / nc))
fig, axes = plt.subplots(nr, nc, figsize=(FS.PLACED_CM * FS.CM,
                                          FS.PLACED_CM * FS.CM * nr / nc * 0.92),
                         sharex=True, sharey=True)
axes = np.atleast_2d(axes)
for i, nm in enumerate(sites):
    ax = axes[i // nc, i % nc]
    g = pr[pr.site == nm]
    ax.plot(mg, zg, color=FS.GRY, lw=1.0, zorder=1)
    ax.plot(g.width_km.to_numpy(), g.depth.to_numpy(), color=FS.ACC, lw=1.3,
            zorder=2)
    # A panel that is simply blank reads as a plotting failure. Where the route
    # ponds there is no corridor to measure below the shells it descends into,
    # and the panel should say so rather than leave the reader guessing.
    _fin = int(np.isfinite(pd.to_numeric(g.width_km, errors='coerce')).sum())
    if _fin < 0.5 * max(len(g), 1):
        ax.text(0.5, 0.5, 'no corridor\n(route ponds)', transform=ax.transAxes,
                ha='center', va='center', fontsize=7.5, color=FS.GRY)
    for zz in (660, 1500):
        ax.axhline(zz, color=FS.INK, lw=0.5, ls=(0, (2, 2)), zorder=0)
    # Panels are small; the full names are in Table S1. A name that overflows its
    # panel is worse than an abbreviated one.
    short = nm.split('/')[0].split('(')[0].strip()
    if len(short) > 13:
        short = short[:12] + '.'
    ax.text(0.05, 0.03, short, transform=ax.transAxes, fontsize=FS.FLOOR_PT + 0.5,
            va='bottom',
            ha='left', color=FS.INK, bbox=FS.mask())
    ax.set_xlim(0, xmax)
    ax.set_ylim(2750, 150)
for j in range(len(sites), nr * nc):
    axes[j // nc, j % nc].axis('off')
for r in range(nr):
    axes[r, 0].set_ylabel('depth (km)')
for c_ in range(nc):
    axes[-1, c_].set_xlabel('width (km)')
fig.tight_layout(w_pad=0.7, h_pad=0.7)
FS.check(fig, floor=7.0)
for ext in ('pdf', 'png'):
    fig.savefig(f'{A.out}.{ext}', bbox_inches='tight')
print(f'wrote {A.out}.pdf and .png: {len(sites)} hotspots, {nr} by {nc}')
