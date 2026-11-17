#!/usr/bin/env python3
"""Figures for the Tasmantid-Lord Howe test (tasman_split.py).

  maps     depth slices through RevealLO and REVEAL beneath the Tasman Sea, with the
           two chains, the two hotspots and the two traced descents
  section  a north-south section along 160 E through both models
  pair     separation of the two descents with depth in five models against start
           pairs in the same region, and the share of their near-optimal corridors
           against every other pair of hotspots within 1500 km

Reads out/tasman_split_<tag>.json and out/tasman_split_volume_<tag>.npz.

  python3 fig_tasman_split.py
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as FS
import model_names

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--out', default='../figures')
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
FS.apply(10.0)

try:                                    # the house sections use GMT's roma
    from cmcrameri import cm as _cm
    CMAP = _cm.roma
except ImportError:
    CMAP = plt.get_cmap('RdYlBu')
LIM = 1.8                               # per cent, as in the section atlas
ORDER = ['RevealLO', 'REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1']
COL = {'RevealLO': FS.INK, 'REVEAL': FS.BLU, 'GLADM35': FS.ACC,
       'SPiRaL': '#5B8C5A', 'SEMUCB-WM1': '#8E6BB0'}
LBL = model_names.DISPLAY

# Dated edifices of the two chains, positions read from Seton et al. (2019, Geological
# Magazine 156, Fig. 2), which compiles the Tasmantid ages of Crossingham et al. (2017)
# and the Lord Howe ages; read from the figure, good to about 0.3 degrees.
TASMANTID = [(-22.3, 155.3), (-23.2, 155.5), (-24.3, 155.3), (-24.9, 155.1),
             (-25.6, 155.3), (-26.4, 155.0), (-27.6, 155.2), (-28.2, 155.5),
             (-29.0, 155.3), (-30.6, 156.1), (-33.1, 156.1), (-36.6, 156.2)]
LORDHOWE = [(-17.6, 158.4), (-18.6, 158.4), (-22.4, 159.2), (-23.4, 159.3),
            (-25.2, 159.3), (-26.6, 159.4), (-29.5, 159.1), (-31.5, 159.1)]
HS = pd.read_csv('hotspots_courtillot2003.csv').set_index('hotspot')
HOT = {'Lord Howe': 'Lord Howe (Tasman East)', 'Tasmantid': 'Tasmanid (Tasman central)'}
MK = {'Lord Howe': dict(marker='o', s=34), 'Tasmantid': dict(marker='s', s=30)}
EXT = (140.0, 180.0, -50.0, -12.0)


def load(tag):
    J = json.load(open(os.path.join(A.dir, f'tasman_split_{tag}.json')))
    V = np.load(os.path.join(A.dir, f'tasman_split_volume_{tag}.npz'))
    return J, V


def at_depth(p, zq):
    z = np.asarray(p['depth'], float)
    k = int(np.clip(np.searchsorted(z, zq, side='right') - 1, 0, len(z) - 1))
    return float(p['lat'][k]), float(p['lon'][k])


def slice_map(ax, V, zq):
    z, la, lo, a = V['depth'], V['lat'], V['lon'], V['anom']
    k = int(np.argmin(np.abs(z - zq)))
    pc = ax.pcolormesh(lo, la, a[k], cmap=CMAP, vmin=-LIM, vmax=LIM, shading='nearest',
                       transform=ccrs.PlateCarree(), rasterized=True)
    return pc, float(z[k])


def furniture(ax):
    ax.set_extent(EXT, crs=ccrs.PlateCarree())
    ax.coastlines('50m', linewidth=0.5, color=FS.INK)
    for pts, m in ((TASMANTID, 'D'), (LORDHOWE, '^')):
        ax.scatter([p[1] for p in pts], [p[0] for p in pts], s=9, marker='.',
                   color=FS.INK, transform=ccrs.PlateCarree(), zorder=4)


# --------------------------------------------------------------------------- maps
DEPTHS = [400, 660, 1100, 2000, 2700]
fig = plt.figure(figsize=(18.0 * FS.CM, 9.4 * FS.CM))
gs = fig.add_gridspec(3, len(DEPTHS), height_ratios=[1, 1, 0.07], hspace=0.10,
                      wspace=0.06, left=0.07, right=0.99, top=0.98, bottom=0.18)
letters = iter('abcdefghij')
for r, tag in enumerate(['RevealLO', 'REVEAL']):
    J, V = load(tag)
    for c, zq in enumerate(DEPTHS):
        ax = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
        pc, zr = slice_map(ax, V, zq)
        furniture(ax)
        for h, p in J['paths'].items():
            ax.plot(p['lon'], p['lat'], color=FS.INK, lw=0.8,
                    ls='-' if h == 'Lord Howe' else '--', transform=ccrs.PlateCarree(),
                    zorder=5)
            y, x = at_depth(p, zr)
            ax.scatter([x], [y], facecolor='white', edgecolor=FS.INK, lw=1.0,
                       transform=ccrs.PlateCarree(), zorder=6, **MK[h])
        for h, n in HOT.items():
            ax.scatter([HS.loc[n].lon_180], [HS.loc[n].lat], marker='*', s=55,
                       color=FS.INK, edgecolor='white', lw=0.5,
                       transform=ccrs.PlateCarree(), zorder=7)
        if c == 0:
            ax.plot([178.0, 150.0], [-14.0, -46.0], color=FS.INK, lw=0.7, ls=(0, (4, 2)),
                    transform=ccrs.Geodetic(), zorder=4)
        ax.text(0.03, 0.04, f'{zr:.0f} km', transform=ax.transAxes, ha='left',
                va='bottom', bbox=FS.mask(0.85), fontsize=10)
        ax.text(0.03, 0.96, next(letters), transform=ax.transAxes, ha='left', va='top',
                fontweight='bold', fontsize=11, bbox=FS.mask(0.85))
        if c == 0:
            ax.text(-0.06, 0.5, LBL[tag], transform=ax.transAxes, rotation=90,
                    ha='right', va='center', fontsize=11)
cax = fig.add_subplot(gs[2, 1:4])
cb = fig.colorbar(pc, cax=cax, orientation='horizontal', extend='both')
cb.set_label('Shear-velocity anomaly from the shell mean (%)')
FS.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(A.out, f'fig_tasman_split_maps.{ext}'), dpi=300)
plt.close(fig)

# ------------------------------------------------------------------------ section
# A great circle cut north-east to south-west: from the slow lower mantle beneath the
# Norfolk Basin, across the fast body at about 1100 km (the lower-mantle slab of
# Schellart et al., 2009), to the two hotspots and the Tasman Sea.
SEC_A, SEC_B = (-14.0, 178.0), (-46.0, 150.0)
R_E = 6371.0


def _xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def great_circle(a, b, n):
    p, q = _xyz(*a), _xyz(*b)
    w = np.arccos(np.clip(p @ q, -1, 1))
    f = np.linspace(0, 1, n)[:, None]
    v = (np.sin((1 - f) * w) * p + np.sin(f * w) * q) / np.sin(w)
    return (np.degrees(np.arcsin(v[:, 2])), np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360,
            f[:, 0] * w * R_E)


def project(la, lo):
    """Distance along the section and distance off it, km."""
    p, q = _xyz(*SEC_A), _xyz(*SEC_B)
    nrm = np.cross(p, q); nrm /= np.linalg.norm(nrm)
    v = _xyz(np.asarray(la, float), np.asarray(lo, float))
    off = np.degrees(np.arcsin(np.clip(v @ nrm, -1, 1))) * np.pi / 180 * R_E
    vp = v - np.outer(v @ nrm, nrm); vp /= np.linalg.norm(vp, axis=-1, keepdims=True)
    along = np.arctan2(np.cross(p, vp) @ nrm, vp @ p) * R_E
    return along, off


gla, glo, gx = great_circle(SEC_A, SEC_B, 240)
fig, axs = plt.subplots(1, 2, figsize=(18.0 * FS.CM, 8.8 * FS.CM), sharey=True,
                        gridspec_kw=dict(wspace=0.06, left=0.11, right=0.97,
                                         top=0.97, bottom=0.33))
for ax, tag, L in zip(axs, ['RevealLO', 'REVEAL'], 'ab'):
    J, V = load(tag)
    z, la, lo, a = V['depth'], V['lat'], V['lon'], V['anom']
    jj = np.array([int(np.argmin(np.abs(la - x))) for x in gla])
    ii = np.array([int(np.argmin(np.abs(((lo - x + 180) % 360) - 180))) for x in glo])
    sec = np.where((z >= 80.0)[:, None], a[:, jj, ii], np.nan)
    pc = ax.pcolormesh(gx, z, sec, cmap=CMAP, vmin=-LIM, vmax=LIM,
                       shading='nearest', rasterized=True)
    for zz in (410, 660, 1000):
        ax.axhline(zz, color=FS.INK, lw=0.5, ls=':')
    for h, p in J['paths'].items():
        al, _ = project(p['lat'], p['lon'])
        ax.plot(al, p['depth'], color=FS.INK, lw=1.2,
                ls='-' if h == 'Lord Howe' else '--')
    for h, n in HOT.items():
        al, _ = project([HS.loc[n].lat], [HS.loc[n].lon_180])
        ax.scatter(al, [0], marker='v', s=42, color=FS.INK, clip_on=False, zorder=6)
    ax.set_xlim(0, gx[-1])
    ax.set_ylim(2890, 0)
    ax.set_xlabel('Distance from 14° S, 178° E (km)')
    ax.text(0.02, 0.97, L, transform=ax.transAxes, ha='left', va='top',
            fontweight='bold', fontsize=11, bbox=FS.mask(0.85))
    ax.text(0.98, 0.03, LBL[tag], transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, bbox=FS.mask(0.85))
axs[0].set_ylabel('Depth (km)')
cax = fig.add_axes([0.30, 0.13, 0.45, 0.035])
cb = fig.colorbar(pc, cax=cax, orientation='horizontal', extend='both')
cb.set_label('Shear-velocity anomaly from the shell mean (%)')
FS.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(A.out, f'fig_tasman_split_section.{ext}'), dpi=300)
plt.close(fig)

# --------------------------------------------------------------------------- pair
fig = plt.figure(figsize=(18.0 * FS.CM, 9.6 * FS.CM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.45, left=0.11,
                      right=0.99, top=0.93, bottom=0.24)
ax = fig.add_subplot(gs[0])
for tag in ORDER:
    J, _ = load(tag)
    zg = np.asarray(J['null_sep_depth'])
    ax.plot(np.asarray(J['pair_sep']), zg, color=COL[tag], lw=1.6, label=LBL[tag])
    ax.plot(np.asarray(J['null_sep_pct']['50']), zg, color=COL[tag], lw=0.8, ls=':')
ax.axvline(J['merge_km'], color=FS.GRY, lw=0.8)
ax.set_ylim(2700, 200)
ax.set_xlim(-60, 2300)
ax.set_xlabel('Separation of the two descents (km)')
ax.set_ylabel('Depth (km)')
fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=5,
           frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=1.6)
ax.text(-0.02, 1.03, 'a', transform=ax.transAxes, ha='right', va='bottom',
        fontweight='bold', fontsize=11)
for k, (tag, L) in enumerate((('RevealLO', 'b'), ('REVEAL', 'c'))):
    J, _ = load(tag)
    ax = fig.add_subplot(gs[k + 1])
    R = pd.DataFrame(J['corridor']['reference'])
    ax.scatter(R.upper, R.lower, s=16, color=FS.GRY, edgecolor='white', lw=0.4,
               zorder=3)
    ax.scatter([J['corridor']['pair_upper']], [J['corridor']['pair_lower']], s=60,
               marker='*', color=COL[tag], zorder=4)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_aspect('equal')
    ax.set_xlabel('Shared, 200–1000 km')
    if k == 0:
        ax.set_ylabel('Shared, 1000–2700 km')
    ax.text(-0.02, 1.03, L, transform=ax.transAxes, ha='right', va='bottom',
            fontweight='bold', fontsize=11)
    ax.text(0.97, 0.03, LBL[tag], transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10)
FS.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(A.out, f'fig_tasman_split_pair.{ext}'), dpi=300)
plt.close(fig)
print('wrote fig_tasman_split_maps, fig_tasman_split_section, fig_tasman_split_pair')
