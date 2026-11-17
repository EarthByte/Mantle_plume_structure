#!/usr/bin/env python3
"""The rebuilt slab footprint beneath the SW Pacific, for comparison with Mather et al.
(2026, Fig. 4): (a) time of the most recent subduction at each cell, (b) cumulative
slab area density, with the window, the 48-25 Ma ribbon, the two chains, the two
hotspots and the proposed conduit.

Reads out/tasman_slab_footprint.npz and out/tasman_slab_classes.npz.

  python3 fig_tasman_footprint.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import cartopy.crs as ccrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as FS

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--out', default='../figures/tasman_split')
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
FS.apply(10.0)

PLUME = (-36.7, 161.1)
TASMANTID = [(-22.3, 155.3), (-23.2, 155.5), (-24.3, 155.3), (-24.9, 155.1),
             (-25.6, 155.3), (-26.4, 155.0), (-27.6, 155.2), (-28.2, 155.5),
             (-29.0, 155.3), (-30.6, 156.1), (-33.1, 156.1), (-36.6, 156.2)]
LORDHOWE = [(-17.6, 158.4), (-18.6, 158.4), (-22.4, 159.2), (-23.4, 159.3),
            (-25.2, 159.3), (-26.6, 159.4), (-29.5, 159.1), (-31.5, 159.1)]
HOT = [(-33.0, 159.0), (-39.0, 156.0)]
PC = ccrs.PlateCarree()
PR = ccrs.PlateCarree(central_longitude=180.0)
EXT = (140.0, 185.0, -52.0, -12.0)

F = np.load(os.path.join(A.dir, 'tasman_slab_footprint.npz'))
K = np.load(os.path.join(A.dir, 'tasman_slab_classes.npz'))
lon, lat = F['lon'], F['lat']

fig = plt.figure(figsize=(18.0 * FS.CM, 9.6 * FS.CM))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.05], hspace=0.30, wspace=0.20,
                      left=0.075, right=0.925, top=0.97, bottom=0.15)
panels = [(F['last_time'], mc.Normalize(0, 120), plt.get_cmap('YlGnBu'),
           'Most recent subduction (Ma)', 'a'),
          (np.where(F['area_density'] > 0, F['area_density'], np.nan),
           mc.LogNorm(1e6, 1e10), plt.get_cmap('PuBuGn'), 'Slab area density (kg/m²)', 'b')]
for c, (f, norm, cmap, lab, L) in enumerate(panels):
    ax = fig.add_subplot(gs[0, c], projection=PR)
    pm = ax.pcolormesh(lon, lat, f, norm=norm, cmap=cmap, shading='nearest',
                       transform=PC, rasterized=True)
    ax.contour(K['lon'], K['lat'], K['window_wide'].astype(float), levels=[0.5],
               colors=FS.INK, linewidths=0.9, linestyles='--', transform=PC, zorder=4)
    ax.contourf(K['lon'], K['lat'], K['ribbon'].astype(float), levels=[0.5, 1.5],
                colors='none', hatches=['////'], transform=PC, zorder=4)
    ax.contour(K['lon'], K['lat'], K['ribbon'].astype(float), levels=[0.5],
               colors=FS.INK, linewidths=0.9, transform=PC, zorder=5)
    ax.coastlines('50m', linewidth=0.5, color=FS.INK)
    for pts, mk in ((TASMANTID, 'D'), (LORDHOWE, 's')):
        ax.scatter([p[1] for p in pts], [p[0] for p in pts], s=10, marker=mk,
                   facecolor='white', edgecolor=FS.INK, lw=0.6, transform=PC, zorder=6)
    ax.scatter([p[1] for p in HOT], [p[0] for p in HOT], marker='*', s=70,
               color=FS.INK, edgecolor='white', lw=0.5, transform=PC, zorder=7)
    ax.scatter([PLUME[1]], [PLUME[0]], s=90, facecolor='none', edgecolor='white',
               lw=2.4, transform=PC, zorder=7)
    ax.scatter([PLUME[1]], [PLUME[0]], s=90, facecolor='none', edgecolor=FS.ACC,
               lw=1.3, transform=PC, zorder=8)
    ax.set_extent(EXT, crs=PC)
    gl = ax.gridlines(crs=PC, draw_labels=True, lw=0.3, color=FS.GRY,
                      xlocs=range(140, 190, 10), ylocs=range(-50, -9, 10))
    gl.top_labels = False
    gl.right_labels = c == 1
    gl.left_labels = c == 0
    gl.xlabel_style = gl.ylabel_style = {'size': 9.5}
    ax.text(0.03, 0.97, L, transform=ax.transAxes, ha='left', va='top',
            fontweight='bold', fontsize=11, bbox=FS.mask(0.85))
    cax = fig.add_subplot(gs[1, c])
    cb = fig.colorbar(pm, cax=cax, orientation='horizontal',
                      extend='max' if c == 1 else 'neither')
    cb.set_label(lab)
FS.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(A.out, f'fig_tasman_footprint.{ext}'), dpi=300)
print('wrote fig_tasman_footprint')
