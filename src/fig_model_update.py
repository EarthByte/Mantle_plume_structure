#!/usr/bin/env python3
"""Figure: where RevealLO differs from REVEAL, by depth.

Reads out/model_update_RevealLO_REVEAL.csv (model_update.py), so the figure cannot
disagree with the numbers quoted from it. Four panels sharing a depth axis, from 80 km
to 2880 km (the crust and Moho depths above are left out: their anomalies are several
times larger and say nothing about the mantle update):
  (a) RMS shear-velocity anomaly of each model;
  (b) the RMS of RevealLO - REVEAL as a share of REVEAL's RMS - how much of the
      anomaly field the update changed;
  (c) the correlation of the two anomaly fields;
  (d) the RMS difference over the Southern and Northern hemispheres, the test of the
      report's claim that the added data mainly improve the south.
Dashed lines mark 410, 660 and 1000 km as in Figure 3; the dotted line marks 2300 km,
where the report's taper returns the update to REVEAL. No title; the argument is in
the caption.

    python3 fig_model_update.py --out ../figures/fig_model_update
"""
from __future__ import annotations
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figstyle as FS

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--new', default='RevealLO')
ap.add_argument('--old', default='REVEAL')
ap.add_argument('--zmin', type=float, default=80.0)
ap.add_argument('--placed-cm', type=float, default=16.01, dest='placed_cm')
ap.add_argument('--out', default='../figures/fig_model_update')
A = ap.parse_args()
FS.apply(8.5)

d = pd.read_csv(os.path.join(A.dir, f'model_update_{A.new}_{A.old}.csv'))
d = d[(d.depth_km >= A.zmin) & d.rms_diff.notna()]
z = d.depth_km.to_numpy()

fig, ax = plt.subplots(1, 4, figsize=(A.placed_cm * FS.CM, 9.0 * FS.CM), sharey=True)
a, b, c, e = ax

a.plot(d.rms_new, z, color=FS.BLU, lw=1.5, label=A.new)
a.plot(d.rms_old, z, color=FS.ACC, lw=1.5, ls=(0, (4, 2)), label=A.old)
a.set_xlim(0, 4.0)
a.set_xlabel('RMS anomaly (%)')
a.set_ylabel('depth (km)')
FS.legend(a, loc='center right', handlelength=1.4, handletextpad=0.5, borderpad=0.3)

b.plot(d.ratio, z, color=FS.INK, lw=1.5)
b.set_xlim(0, 1.1)
b.set_xticks([0, 0.5, 1.0])
b.set_xlabel(f'RMS difference /\nRMS {A.old}')

c.plot(d['corr'], z, color=FS.INK, lw=1.5)
c.set_xlim(0.7, 1.0)
c.set_xlabel('correlation')

e.plot(d.rms_diff_south, z, color=FS.INK, lw=1.5, label='south')
e.plot(d.rms_diff_north, z, color=FS.GRY, lw=1.5, ls=(0, (4, 2)), label='north')
e.set_xlim(0, 1.7)
e.set_xlabel('RMS difference (%)')
FS.legend(e, loc='center right', handlelength=1.4, handletextpad=0.5, borderpad=0.3)

for k, x in enumerate(ax):
    for zz in (410, 660, 1000):
        x.axhline(zz, color=FS.GRY, lw=0.7, ls=(0, (3, 2)), zorder=0)
    x.axhline(2300, color=FS.GRY, lw=0.9, ls=(0, (1, 1.5)), zorder=0)
    x.text(-0.12 if k == 0 else -0.05, 1.03, 'abcd'[k], transform=x.transAxes,
           fontsize=13, fontweight='bold', ha='left', va='bottom')
b.text(0.97, 2270, 'taper begins', ha='right', va='bottom', fontsize=8.5,
       color=FS.INK, bbox=FS.mask())
a.set_ylim(2891, 0)
a.set_yticks([0, 500, 1000, 1500, 2000, 2500])

fig.tight_layout(w_pad=0.8)
FS.check(fig, placed_cm=A.placed_cm)
for ext in ('pdf', 'png'):
    fig.savefig(f'{A.out}.{ext}', bbox_inches='tight')
print(f'wrote {A.out}.pdf and .png; text checks passed')
