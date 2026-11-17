#!/usr/bin/env python3
"""The slab geometry of Mather et al. (2026) laid over the tomography.

  maps     RevealLO and REVEAL at 500, 660 and 1100 km, with the slab-free window and
           the 48-25 Ma ribbon of the rebuilt slab footprint, and the present-day
           primary conduit Mather et al. propose
  profile  the anomaly within 250 km of that conduit, by depth, in five models
  ribbon   ribbon minus window, by depth, in five models, at the latitudes of the
           hotspots (45-28 S)

Reads out/tasman_slab_footprint.npz (tasman_slab_footprint.py),
out/tasman_slab_compare.csv and out/tasman_plume_site_profile.csv
(tasman_slab_compare.py), and the regional volumes of tasman_split.py.

  python3 fig_tasman_slab.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as FS
import model_names

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--out', default='../figures/figS4_tasman_slab',
                help='path stem; .pdf and .png are written')
A = ap.parse_args()
os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
FS.apply(10.0)
try:
    from cmcrameri import cm as _cm
    CMAP = _cm.roma
except ImportError:
    CMAP = plt.get_cmap('RdYlBu')
LIM = 1.8
ORDER = ['RevealLO', 'REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1']
COL = {'RevealLO': FS.INK, 'REVEAL': FS.BLU, 'GLADM35': FS.ACC,
       'SPiRaL': '#5B8C5A', 'SEMUCB-WM1': '#8E6BB0'}
LBL = model_names.DISPLAY
PLUME = (-36.7, 161.1)
TASMANTID = [(-22.3, 155.3), (-23.2, 155.5), (-24.3, 155.3), (-24.9, 155.1),
             (-25.6, 155.3), (-26.4, 155.0), (-27.6, 155.2), (-28.2, 155.5),
             (-29.0, 155.3), (-30.6, 156.1), (-33.1, 156.1), (-36.6, 156.2)]
LORDHOWE = [(-17.6, 158.4), (-18.6, 158.4), (-22.4, 159.2), (-23.4, 159.3),
            (-25.2, 159.3), (-26.6, 159.4), (-29.5, 159.1), (-31.5, 159.1)]
HOT = [(-33.0, 159.0), (-39.0, 156.0)]
EXT = (140.0, 180.0, -50.0, -12.0)
PC = ccrs.PlateCarree()

K = np.load(os.path.join(A.dir, 'tasman_slab_classes.npz'))
flon, flat = K['lon'], K['lat']
window = K['window_wide'].astype(float)
ribbon = K['ribbon'].astype(float)

fig = plt.figure(figsize=(18.0 * FS.CM, 13.0 * FS.CM))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.06], width_ratios=[1, 1, 1, 1.15],
                      hspace=0.42, wspace=0.10, left=0.07, right=0.90, top=0.97,
                      bottom=0.17)
letters = iter('abcdefgh')
for r, tag in enumerate(['RevealLO', 'REVEAL']):
    V = np.load(os.path.join(A.dir, f'tasman_split_volume_{tag}.npz'))
    z, la, lo, a = V['depth'], V['lat'], V['lon'], V['anom']
    for c, zq in enumerate((500, 660, 1100)):
        ax = fig.add_subplot(gs[r, c], projection=PC)
        k = int(np.argmin(np.abs(z - zq)))
        pc = ax.pcolormesh(lo, la, a[k], cmap=CMAP, vmin=-LIM, vmax=LIM,
                           shading='nearest', transform=PC, rasterized=True)
        ax.contour(flon, flat, window, levels=[0.5], colors='white', linewidths=1.6,
                   transform=PC, zorder=4)
        ax.contour(flon, flat, window, levels=[0.5], colors=FS.INK, linewidths=0.7,
                   linestyles='--', transform=PC, zorder=5)
        ax.contourf(flon, flat, ribbon, levels=[0.5, 1.5], colors='none',
                    hatches=['////'], transform=PC, zorder=4)
        ax.contour(flon, flat, ribbon, levels=[0.5], colors='white', linewidths=1.6,
                   transform=PC, zorder=4)
        ax.contour(flon, flat, ribbon, levels=[0.5], colors=FS.INK, linewidths=0.8,
                   transform=PC, zorder=5)
        ax.set_extent(EXT, crs=PC)
        ax.coastlines('50m', linewidth=0.5, color=FS.INK)
        for pts in (TASMANTID, LORDHOWE):
            ax.scatter([p[1] for p in pts], [p[0] for p in pts], s=8, marker='.',
                       color=FS.INK, transform=PC, zorder=6)
        ax.scatter([p[1] for p in HOT], [p[0] for p in HOT], marker='*', s=50,
                   color=FS.INK, edgecolor='white', lw=0.5, transform=PC, zorder=7)
        ax.scatter([PLUME[1]], [PLUME[0]], s=70, facecolor='none', edgecolor='white',
                   lw=2.2, transform=PC, zorder=7)
        ax.scatter([PLUME[1]], [PLUME[0]], s=70, facecolor='none', edgecolor=FS.ACC,
                   lw=1.2, transform=PC, zorder=8)
        ax.text(0.03, 0.04, f'{float(z[k]):.0f} km', transform=ax.transAxes, ha='left',
                va='bottom', bbox=FS.mask(0.85), fontsize=10)
        ax.text(0.03, 0.96, next(letters), transform=ax.transAxes, ha='left', va='top',
                fontweight='bold', fontsize=11, bbox=FS.mask(0.85))
        if c == 0:
            ax.text(-0.06, 0.5, LBL[tag], transform=ax.transAxes, rotation=90,
                    ha='right', va='center', fontsize=11)
cax = fig.add_subplot(gs[2, 0:3])
cb = fig.colorbar(pc, cax=cax, orientation='horizontal', extend='both')
cb.set_label('Shear-velocity anomaly from the shell mean (%)')

P = pd.read_csv(os.path.join(A.dir, 'tasman_plume_site_profile.csv'))
C = pd.read_csv(os.path.join(A.dir, 'tasman_slab_compare.csv'))
for r, (L, xl) in enumerate((('g', 'Anomaly at the proposed conduit (%)'),
                             ('h', 'Ribbon minus window (%)'))):
    ax = fig.add_subplot(gs[r, 3])
    for tag in ORDER:
        if r == 0:
            g = P[P.model == tag]
            ax.plot(g.cap, g.depth, color=COL[tag], lw=1.4, marker='o', ms=3,
                    label=LBL[tag])
        else:
            g = C[C.model == tag]
            ax.plot(g.ribbon_minus_window, g.depth, color=COL[tag], lw=1.2)
    ax.axvline(0, color=FS.GRY, lw=0.8)
    ax.set_ylim(2800, 300)
    ax.set_xlim(-2.4, 1.2)
    ax.set_xlabel(xl)
    ax.set_ylabel('Depth (km)')
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.text(0.04, 0.97, L, transform=ax.transAxes, ha='left', va='top',
            fontweight='bold', fontsize=11, bbox=FS.mask(0.85))
    if r == 0:
        h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, frameon=False, handlelength=1.4,
           columnspacing=1.0, bbox_to_anchor=(0.5, 0.0))
FS.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(f'{A.out}.{ext}', dpi=300)
print(f'wrote {A.out}')
