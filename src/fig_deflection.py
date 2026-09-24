"""Figure: where traced conduits deflect, and how broadly.

The distribution is rebuilt from the stored per-path deflections exactly as
bao_pdf.py forms it - each path normalised to unit weight before summing, so that a
path with many deflections does not dominate one with few. The argument the figure
has to carry is about the width of the distribution rather than its peak, and the
spread does not behave like an interface. The band fractions are therefore drawn on
the figure and not left to the caption. No title.
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
ap.add_argument('--bin-km', type=float, default=50.0, dest='bin_km')
ap.add_argument('--z-top', type=float, default=260.0, dest='z_top')
ap.add_argument('--placed-cm', type=float, default=9.5, dest='placed_cm')
ap.add_argument('--out', default='../figures/fig_deflection')
A = ap.parse_args()
FS.apply(10.0)

d = pd.read_csv(os.path.join(A.dir, f'bao_pdf_{A.tag}.csv'))
edges = np.arange(A.z_top, 2900.0 + A.bin_km, A.bin_km)
centres = 0.5 * (edges[:-1] + edges[1:])
pdf = np.zeros(len(centres))
for _, g in d.groupby('site'):
    h = np.histogram(g.depth.to_numpy(float), bins=edges)[0].astype(float)
    if h.sum() > 0:
        pdf += h / h.sum()
pdf /= max(pdf.sum(), 1e-9)
# Band fractions are taken from the deflection depths themselves with exact band
# edges, each path at unit weight, exactly as mid_mantle_census.py forms them, so the
# figure's labels are the numbers the text quotes. Summing the 50 km bins put 1000 km
# inside a bin and made the figure disagree with the census by a few tenths of a
# per cent.
_share = {}
for _, g in d.groupby('site'):
    z_ = g.depth.to_numpy(float)
    for lo, hi in ((660, 1000), (1000, 1500), (1500, 2200)):
        _share[(lo, hi)] = _share.get((lo, hi), 0.0) + ((z_ >= lo) & (z_ < hi)).mean()
_n = d.site.nunique()
_share = {k: v / _n for k, v in _share.items()}

BANDS = [(660, 1000, FS.BLU), (1000, 1500, FS.ACC), (1500, 2200, FS.GRY)]
fig, ax = plt.subplots(figsize=(A.placed_cm * FS.CM, 8.0 * FS.CM))
# Band fractions go on the left, against the axis, where the distribution is thin.
# Placed on the right they crowd the peak marker and its label, which the overlap
# check tolerates at its 1.5 pt threshold but which reads as collided.
for lo, hi, col in BANDS:
    m = (centres >= lo) & (centres < hi)
    ax.axhspan(lo, hi, color=col, alpha=0.10, linewidth=0, zorder=0)
    ax.text(0.035, 0.5 * (lo + hi), f'{100 * _share[(lo, hi)]:.1f}%',
            transform=ax.get_yaxis_transform(), ha='left', va='center',
            fontsize=9.5, color=col, zorder=5, bbox=FS.mask())
ax.plot(pdf, centres, color=FS.INK, lw=1.6, zorder=3)
ax.fill_betweenx(centres, 0, pdf, color=FS.INK, alpha=0.12, zorder=2)
pk = float(centres[int(np.argmax(pdf))])
ax.plot([pdf.max()], [pk], 'o', ms=6, color=FS.ACC, zorder=4)
# The label goes to the RIGHT of the peak marker, into space made for it, rather than
# up and to the left where it crossed the curve: the peak is the one place on this
# figure where masking the data would hide the thing being pointed at.
ax.annotate(f'peak {pk:.0f} km', xy=(pdf.max(), pk), xytext=(8, 0),
            textcoords='offset points', ha='left', va='center',
            fontsize=9.5, color=FS.ACC)
ax.axhline(800, color=FS.INK, lw=1.1, ls=(0, (4, 3)), zorder=1)
ax.annotate('published 800 km', xy=(0.30, 800), xycoords=ax.get_yaxis_transform(),
            xytext=(0, 7), textcoords='offset points', ha='center',
            fontsize=9.0, color=FS.INK, bbox=FS.mask())
# a little below the deepest deflection, so the curve does not run into the
# corner where the first x tick label sits
ax.set_ylim(2460, A.z_top)
ax.set_xlim(0, pdf.max() * 1.50)   # room to the right of the peak for its label
ax.set_xlabel('fraction of deflections')
ax.set_ylabel('depth (km)')

fig.tight_layout()
FS.check(fig, placed_cm=A.placed_cm)
for ext in ('pdf', 'png'):
    fig.savefig(f'{A.out}.{ext}', bbox_inches='tight')
print(f'wrote {A.out}.pdf and .png; peak {pk:.0f} km; text checks passed')
for lo, hi, _ in BANDS:
    m = (centres >= lo) & (centres < hi)
    print(f'  {lo}-{hi} km carries {100 * _share[(lo, hi)]:.1f}%')
