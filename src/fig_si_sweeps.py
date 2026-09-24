"""Figure S1: the injection calibration in full, every site rather than the medians.

The main text figure shows medians, which is what the argument rests on but which
hides how much the individual sites scatter. This shows every injected corridor as a
point, so a reader can judge whether a median rests on a tight group or a broad one,
and can see that the continuity curves in four models are separated by less than the
scatter within any one of them. Four panels, one per sweep. No titles.
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figstyle as FS
import model_names

R_E, DEG = 6371.0, np.pi / 180.0
ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--out', default='../figures/figS1_sweeps')
A = ap.parse_args()
FS.apply(10.0)

MODELS = [('RevealLO', FS.ACC), ('RevealLO_30km', FS.GRY),
          ('GLADM35', FS.BLU), ('SEMUCB-WM1', '#4a7c1f')]


def sweep(tag, kind):
    f = [x for x in sorted(glob.glob(os.path.join(
        A.dir, f'corridor_width_calibration_{tag}_{kind}sweep*.csv')))
        if 'coarse' not in x]
    if not f:
        return None
    c = pd.read_csv(f[-1])
    return (c[c.injected_radius > 0].dropna(subset=['width']),
            c[c.injected_radius == 0].dropna(subset=['width']))


def scatter_median(ax, x, y, colour, jitter=0.0, marker='o'):
    rng = np.random.default_rng(0)
    xs = np.asarray(x, float)
    if jitter:
        xs = xs + rng.normal(0, jitter, size=len(xs))
    ax.plot(xs, y, marker, ms=3.0, mfc='none', mec=colour, mew=0.8, alpha=0.75,
            zorder=2, ls='none')
    d = pd.DataFrame({'x': np.asarray(x, float), 'y': np.asarray(y, float)})
    g = d.groupby('x').y.median()
    ax.plot(g.index, g.values, '-', color=colour, lw=2.0, zorder=3)
    return g


fig, axes = plt.subplots(2, 2, figsize=(FS.PLACED_CM * FS.CM, 12.6 * FS.CM))
(a, b), (c, d) = axes

cal = pd.read_csv(os.path.join(A.dir, 'corridor_width_calibration_RevealLO.csv'))
ci = cal[cal.injected_radius > 0].dropna(subset=['width'])
scatter_median(a, ci.injected_fwhm, ci.width, FS.ACC, jitter=6.0)
a.set_xlabel('injected width (km FWHM)')
a.set_ylabel('corridor width (km)')

am = sweep('RevealLO', 'amp')
if am:
    scatter_median(b, np.abs(am[0].amp.to_numpy(float)), am[0].width, FS.BLU,
                   jitter=0.012)
b.set_xlabel('conduit amplitude (per cent slow)')
b.set_ylabel('corridor width (km)')

tl = sweep('RevealLO', 'tilt')
if tl:
    scatter_median(c, tl[0].tilt.to_numpy(float) * R_E * DEG, tl[0].width, FS.ACC,
                   jitter=12.0, marker='s')
c.set_xlabel('lateral offset over the column (km)')
c.set_ylabel('corridor width (km)')

_key = {}
for tag, col in MODELS:
    s = sweep(tag, 'duty')
    if not s:
        continue
    scatter_median(d, 100 * s[0].duty.to_numpy(float), s[0].width, col, jitter=1.2)
    _key[tag], = d.plot([], [], '-o', color=col, ms=4, label=model_names.short(tag))
d.set_xlabel('column occupied (per cent)')
d.set_ylabel('corridor width (km)')

for ax, ltr in zip((a, b, c, d), 'abcd'):
    ax.text(-0.02, 1.05, ltr, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='bottom', ha='right')
fig.tight_layout(w_pad=2.4, h_pad=2.2)
# After tight_layout, never before: the layout moves the axes, and a legend
# measured against the data in the old positions is measured against nothing.
# Outside the axes, in the gap above the panel. Inside, the only clear position
# needed 42 per cent more y range than the data use, which spends a third of the
# panel on empty space to house four words.
# Two columns fill downwards, so the entries are given column by column: RevealLO and
# GLAD-M35 on the left, the long 'RevealLO, 30 km' and SEMUCB-WM1 on the right. Read
# across, the top row pairs the two RevealLO samplings. Ordered as MODELS, the two
# longest labels shared a column pair and the key came out wider than the panel,
# reaching left across the panel letter.
_order = [t for t in ('RevealLO', 'GLADM35', 'RevealLO_30km', 'SEMUCB-WM1') if t in _key]
d.legend([_key[t] for t in _order], [model_names.short(t) for t in _order],
         loc='lower right', bbox_to_anchor=(1.0, 1.005), frameon=False,
         fontsize=8.5, handlelength=1.4, ncol=2, columnspacing=1.1,
         handletextpad=0.5, borderaxespad=0.0)
FS.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(f'{A.out}.{ext}', bbox_inches='tight')
print(f'wrote {A.out}.pdf and .png')
print(f'  radius sweep: {len(ci)} injected corridors shown individually')
