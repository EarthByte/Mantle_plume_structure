"""Figure: where, in depth, hotspot conduits differ from ambient slow structure.

Everything is read from out/mid_mantle_summary.csv, written by mid_mantle_census.py,
so the figure cannot disagree with the tables. Four panels: (a) the share of sites
beneath which the connected slow column reaches 2600 km, hotspots against random
sites, by model; (b) prominence of the local velocity minimum by depth band, hotspots
against null sites, by model; (c) apparent tilt of the traced paths by depth band,
hotspots against ambient paths in the paper model, with the other models' hotspot
paths for comparison; (d) the path-weighted share of deflections by depth band. The
depth range of the mid-mantle viscosity increase (Rudolph et al., 2015) is shaded in
(b)-(d). No title; the argument is in the caption.
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as FS
import model_names

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--placed-cm', type=float, default=16.01, dest='placed_cm')
ap.add_argument('--out', default='../figures/fig6_bands')
A = ap.parse_args()
FS.apply(10.0)

S = pd.read_csv(os.path.join(A.dir, 'mid_mantle_summary.csv'))
ORDER = ['RevealLO', 'REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1']
LABEL = model_names.DISPLAY
BANDS = ['400-660', '660-1000', '1000-1500', '1500-2200', '2200-2700']
MID = np.array([530, 830, 1250, 1850, 2450], float)
COL = {'RevealLO': FS.INK, 'REVEAL': FS.BLU, 'GLADM35': FS.ACC,
       'SPiRaL': '#5B8C5A', 'SEMUCB-WM1': '#8E6BB0', 'RevealLO_30km': FS.GRY}
# Tilt and deflection depend on the node spacing of the trace, so (c) and (d) compare
# the paper model with its own 30 km resampling only; the other models' values are in
# Table S5 with that caveat.
PATH_TAGS = ['RevealLO', 'RevealLO_30km']
JUMP = (800, 1200)


def series(tag, q, col):
    d = S[(S.tag == tag) & (S.quantity == q)].set_index('band')
    return np.array([d[col].get(b, np.nan) for b in BANDS], float)


fig, axs = plt.subplots(2, 2, figsize=(A.placed_cm * FS.CM, 16.0 * FS.CM))
(a, b), (c, d) = axs

# (a) connected columns reaching 2600 km, by model
tags = [t for t in ORDER if ((S.tag == t) & (S.quantity == 'root_ge_2600_share')).any()]
x = np.arange(len(tags))
h = [100 * float(S[(S.tag == t) & (S.quantity == 'root_ge_2600_share')].hotspot.iloc[0]) for t in tags]
n = [100 * float(S[(S.tag == t) & (S.quantity == 'root_ge_2600_share')].ambient.iloc[0]) for t in tags]
a.barh(x + 0.19, h, 0.36, color=[COL[t] for t in tags], zorder=3)
a.barh(x - 0.19, n, 0.36, color='white', edgecolor=[COL[t] for t in tags], lw=1.2, zorder=3)
a.set_yticks(x)
a.set_yticklabels([LABEL[t] for t in tags])
a.invert_yaxis()
a.set_xlabel('connected slow column to 2600 km (% of sites)')
a.set_xlim(0, 60)
FS.legend(a, loc='lower right', handles=[plt.Rectangle((0, 0), 1, 1, color=FS.INK),
                  plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=FS.INK, lw=1.2)],
         labels=['hotspots (49)', 'random sites (300)'], frameon=False)

# (b) prominence by band
for t in ORDER:
    if not ((S.tag == t) & (S.quantity == 'prominence')).any():
        continue
    hb = series(t, 'prominence', 'hotspot'); nb = series(t, 'prominence', 'ambient')
    m = np.isfinite(hb)
    b.plot(hb[m], MID[m], '-o', color=COL[t], ms=5, lw=1.5, zorder=3)
    b.plot(nb[m], MID[m], '--o', color=COL[t], ms=5, lw=1.2, mfc='white', zorder=3)
b.set_xlabel('prominence of the minimum (%)')
b.set_xlim(left=0)

# (c) apparent tilt by band
for t in PATH_TAGS:
    if not ((S.tag == t) & (S.quantity == 'tilt_deg')).any():
        continue
    hb = series(t, 'tilt_deg', 'hotspot'); m = np.isfinite(hb) & (hb > 0)
    c.plot(hb[m], MID[m], '-o', color=COL[t], ms=5, lw=1.5 if t == A.tag else 1.0,
           alpha=1.0 if t == A.tag else 0.7, zorder=3)
    if t == A.tag:
        ab = series(t, 'tilt_deg', 'ambient'); m2 = np.isfinite(ab)
        c.plot(ab[m2], MID[m2], '--o', color=COL[t], ms=5, lw=1.2, mfc='white', zorder=3)
c.set_xlabel('apparent tilt of the path (°)')
c.set_xlim(0, 32)

# (d) deflection share by band
for t in PATH_TAGS:
    if not ((S.tag == t) & (S.quantity == 'deflection_share')).any():
        continue
    hb = 100 * series(t, 'deflection_share', 'hotspot'); m = np.isfinite(hb)
    d.plot(hb[m], MID[m], '-o', color=COL[t], ms=5, lw=1.5 if t == A.tag else 1.0,
           alpha=1.0 if t == A.tag else 0.7, zorder=3)
    if t == A.tag:
        ab = 100 * series(t, 'deflection_share', 'ambient'); m2 = np.isfinite(ab)
        d.plot(ab[m2], MID[m2], '--o', color=COL[t], ms=5, lw=1.2, mfc='white', zorder=3)
d.set_xlabel('share of deflections (%)')
d.set_xlim(0, 65)

for ax in (b, c, d):
    ax.axhspan(*JUMP, color=FS.GRY, alpha=0.18, lw=0, zorder=0)
    ax.set_ylim(2700, 400)
    ax.set_yticks([660, 1000, 1500, 2200, 2700])
for ax in (b, c):
    ax.set_ylabel('depth (km)')
d.set_ylabel('depth (km)')
b.annotate('viscosity increase', xy=(0.98, 1000), xycoords=b.get_yaxis_transform(),
           ha='right', va='center', fontsize=9.0, color=FS.INK, bbox=FS.mask())

handles = [Line2D([], [], color=COL[t], lw=1.5, marker='o', ms=5, label=LABEL[t]) for t in ORDER]
handles += [Line2D([], [], color=FS.GRY, lw=1.0, marker='o', ms=5, label='RevealLO, 30 km (c, d)')]
handles += [Line2D([], [], color=FS.INK, lw=1.5, marker='o', ms=5, label='hotspots'),
            Line2D([], [], color=FS.INK, lw=1.2, ls='--', marker='o', ms=5, mfc='white',
                   label='random sites or ambient paths')]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.0))
for ax, letter in zip((a, b, c, d), 'abcd'):
    ax.text(-0.16, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight='bold',
            ha='left', va='bottom')
fig.tight_layout(rect=(0, 0.11, 1, 1))
FS.check(fig, placed_cm=A.placed_cm)
for ext in ('pdf', 'png'):
    fig.savefig(f'{A.out}.{ext}', bbox_inches='tight')
print(f'wrote {A.out}.pdf and .png; text checks passed')
