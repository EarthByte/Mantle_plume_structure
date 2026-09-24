"""Figure: what a corridor is, and why its width is an error bar rather than a size.

The left panel shows corridor width against depth: the median across all 49 conduits
with its interquartile range, and three named hotspots that behave differently -
the narrowest at every depth, one that reverses between shallow and deep, and one of
the broadest. The right panel shows the same widths measured at three tolerances, so
that a reader can see the number is a property of the threshold as much as of the
mantle, which is the reason no width in this paper is presented as a conduit
diameter. No title.
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figstyle as FS

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--named', default='Marion,Eifel,Australia E')
ap.add_argument('--out', default='../figures/fig_corridor')
A = ap.parse_args()
FS.apply(10.5)

pr = pd.read_csv(os.path.join(A.dir, f'corridor_profiles_{A.tag}.csv'))
sm = pd.read_csv(os.path.join(A.dir, f'corridor_summary_{A.tag}.csv'))
sm = sm[sm.ok == True]

fig, (a, b) = plt.subplots(1, 2, figsize=(FS.PLACED_CM * FS.CM, 8.0 * FS.CM),
                           gridspec_kw={'width_ratios': [1.35, 1.0]})

g = pr.groupby('depth').width_km
med, q1, q3 = g.median(), g.quantile(.25), g.quantile(.75)
z = med.index.to_numpy(float)
a.fill_betweenx(z, q1.to_numpy(), q3.to_numpy(), color=FS.GRY, alpha=0.25,
                linewidth=0, zorder=1)
a.plot(med.to_numpy(), z, color=FS.INK, lw=2.2, zorder=4)
# Labels are anchored at well separated depths rather than all at the deepest
# point, where three of them would land on top of each other.
LABEL_Z = (2650.0, 2050.0, 1450.0)
for nm, col, lz in zip([x.strip() for x in A.named.split(',')],
                       (FS.BLU, FS.ACC, '#4a7c1f'), LABEL_Z):
    s = pr[pr.site == nm]
    if not len(s):
        print(f'  {nm}: not in the profile file; not drawn')
        continue
    # A path whose route ponds has no corridor below the shells it descends into,
    # so its profile is NaN almost everywhere. Anchoring a label at a NaN width
    # puts it nowhere and takes the collision check with it. Anchor on what is
    # actually drawn, and say plainly when there is nothing to draw.
    fin = s[np.isfinite(pd.to_numeric(s.width_km, errors='coerce'))]
    if not len(fin):
        print(f'  {nm}: no corridor at any depth, so no profile to draw. '
              f'Choose another site for this panel with --named.')
        continue
    if len(fin) < len(s):
        print(f'  {nm}: corridor defined at {len(fin)} of {len(s)} shells')
    a.plot(fin.width_km.to_numpy(), fin.depth.to_numpy(), color=col, lw=1.5,
           alpha=0.95, zorder=3)
    k = (fin.depth - lz).abs().idxmin()
    a.annotate(nm, xy=(float(fin.loc[k, 'width_km']), float(fin.loc[k, 'depth'])),
               xytext=(6, 0), textcoords='offset points', fontsize=9.0,
               color=col, ha='left', va='center', bbox=FS.mask())
# Drawn against the right edge: at the left they sit at the same heights as the
# depth ticks, so the reader sees 1000 twice on the same line.
for zz in (410, 660, 1000):
    a.axhline(zz, color=FS.INK, lw=0.7, ls=(0, (3, 3)), zorder=0)
    a.annotate(f'{zz}', xy=(1.0, zz), xycoords=a.get_yaxis_transform(),
               xytext=(-3, 3), textcoords='offset points', fontsize=8.5,
               color=FS.INK, ha='right')
a.set_ylim(2750, 150)
a.set_xlabel('corridor width (km)')
a.set_ylabel('depth (km)')

taus = ['0.01', '0.02', '0.05']
vals = [sm[f'width_med_{t}'].dropna().to_numpy(float) for t in taus]
bp = b.boxplot(vals, vert=True, widths=0.55, patch_artist=True,
               medianprops=dict(color=FS.INK, lw=1.8),
               flierprops=dict(marker='o', ms=3.0, mfc='none', mec=FS.GRY))
for patch, col in zip(bp['boxes'], (FS.BLU, FS.ACC, FS.GRY)):
    patch.set_facecolor(col); patch.set_alpha(0.35); patch.set_edgecolor(col)
b.set_xticklabels([f'{t}' for t in taus])
b.set_xlabel('tolerance on excess cost')
b.set_ylabel('corridor width (km)')
for i, v in enumerate(vals):
    b.annotate(f'{np.median(v):.0f}', xy=(i + 1, np.median(v)), xytext=(12, -4),
               textcoords='offset points', fontsize=9.0, color=FS.INK)

for ax, ltr in zip((a, b), 'ab'):
    ax.text(-0.17, 1.03, ltr, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='bottom', bbox=FS.mask())
fig.tight_layout(w_pad=2.2)
FS.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(f'{A.out}.{ext}', bbox_inches='tight')
print(f'wrote {A.out}.pdf and .png; text checks passed')
print(f'  median width by tolerance: ' +
      ', '.join(f'{t} -> {np.median(v):.0f} km' for t, v in zip(taus, vals)))
