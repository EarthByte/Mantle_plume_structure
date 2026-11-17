#!/usr/bin/env python3
"""At what depth, if any, does the tomography see the slab ribbon of Mather et al. (2026)?

Reads the rebuilt slab footprint (tasman_slab_footprint.py) and the regional volumes
written by tasman_split.py, and compares the shear-velocity anomaly, shell by shell,
inside three map-view classes in the Tasman region (50-10 S, 145-180 E):

  ribbon   cells that received slab subducted 48-25 Ma (the d'Entrecasteaux ribbon),
           above a floor of 1e7 kg/m2 so that a single grazing trench step does not count
  window   cells that received no slab at any time since 200 Ma
  other    every other cell that received slab

If the ribbon still sits where it was deposited, its cells should be faster than the
window cells at the depth it has reached. Mather et al. place it near the transition
zone. The profile of ribbon minus window answers where it is fastest.

Writes out/tasman_slab_compare.csv (model, depth, ribbon, window, other, n cells).

  python3 tasman_slab_compare.py
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--footprint', default='out/tasman_slab_footprint.npz')
ap.add_argument('--floor', type=float, default=1.0e7, help='kg/m2 for a ribbon cell')
ap.add_argument('--box', default='-45,-28,145,180',
                help='lat0,lat1,lon0,lon1 of the comparison. The default is the latitude band '
                     'of the present hotspots and of the ribbon beside them, so that ribbon and '
                     'window are compared at the same latitudes: north of about 28 S the '
                     'transition zone is fast everywhere, which would favour any class that '
                     'reaches further north.')
A = ap.parse_args()

F = np.load(A.footprint)
flon, flat = F['lon'], F['lat']
R_E = 6371.0e3
cell = (R_E ** 2) * np.radians(flon[1] - flon[0]) ** 2 * np.cos(np.radians(flat))[:, None]
rib_d = F['mass_25_48'] / cell
tot_d = F['area_density']
BOX = tuple(float(x) for x in A.box.split(','))
inbox = ((flat >= BOX[0]) & (flat <= BOX[1]))[:, None] & \
        ((flon >= BOX[2]) & (flon <= BOX[3]))[None, :]
# The footprint is deposited at trench points, 1 Myr apart, on a 0.5 degree grid, so
# slab-covered ground carries isolated empty cells that are artefacts of the sampling,
# not windows. A morphological closing (one pass of a 3 x 3 element) removes them;
# what stays empty is a window. Within the latitudes of the present hotspots
# (45-31 S) the 48-25 Ma material is a thin line along the north-east edge of the
# window, the trace of the d'Entrecasteaux / New Caledonia trench; it is widened by one
# cell (about 50 km) so that the comparison does not hinge on rasterisation. North of
# 31 S the same subduction interval covers a broad belt beneath New Caledonia and the
# Norfolk Ridge, which is kept as its own class.
from scipy import ndimage
_st = np.ones((3, 3), bool)
slab = ndimage.binary_closing(tot_d > 0, structure=_st, iterations=1)
band = ((flat >= BOX[0]) & (flat <= BOX[1]))[:, None]
# The ribbon of Mather et al. is the young slab beside the window (their Fig. 4b,e):
# 48-25 Ma material within three cells (about 150 km) of the window's edge. The rest of
# the 48-25 Ma material in the band is the broad belt to the east.
# The Tasman window is the empty region containing 40 S, 158 E; small empty patches
# elsewhere are other gaps in the deposit and are not part of this comparison.
_lab, _n = ndimage.label(~slab)
_j0, _i0 = int(np.argmin(np.abs(flat + 40.0))), int(np.argmin(np.abs(flon - 158.0)))
win = _lab == _lab[_j0, _i0]
near = ndimage.binary_dilation(win, structure=_st, iterations=3) & ~win
dep = (rib_d >= A.floor) & inbox
line = ndimage.binary_dilation(dep & near, structure=_st) & inbox & ~win
CLS = {'ribbon': line,
       'belt': dep & ~line,
       'window': inbox & win,
       'other': inbox & slab & ~line & ~dep}
for k, m in CLS.items():
    print(f'  {k:7s} {int(m.sum()):5d} cells')
# The window over the whole region, for the figure (the classes above are confined to
# the hotspot latitudes).
_wide = (((flat >= -52) & (flat <= -10))[:, None] & ((flon >= 138) & (flon <= 182))[None, :])
_keep = _wide & win
np.savez_compressed(os.path.join(A.dir, 'tasman_slab_classes.npz'), lon=flon, lat=flat,
                    window_wide=_keep, **{k: v for k, v in CLS.items()})

rows = []
for tag in ['RevealLO', 'REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1']:
    V = np.load(os.path.join(A.dir, f'tasman_split_volume_{tag}.npz'))
    z, la, lo, a = V['depth'], V['lat'], V['lon'], V['anom']
    jj = np.array([int(np.argmin(np.abs(la - y))) for y in flat])
    ii = np.array([int(np.argmin(np.abs(lo - x))) for x in flon])
    for k in range(len(z)):
        if z[k] < 100 or z[k] > 2800:
            continue
        s = a[k][np.ix_(jj, ii)]
        r = dict(model=tag, depth=float(z[k]))
        for c, m in CLS.items():
            r[c] = float(np.nanmean(s[m]))
            r[f'n_{c}'] = int(m.sum())
        rows.append(r)
d = pd.DataFrame(rows)
d['ribbon_minus_window'] = d.ribbon - d.window
d['belt_minus_window'] = d.belt - d.window
d.to_csv(os.path.join(A.dir, 'tasman_slab_compare.csv'), index=False)
d['other_minus_window'] = d.other - d.window
d['ribbon_minus_other'] = d.ribbon - d.other
for tag, g in d.groupby('model', sort=False):
    g = g.set_index('depth')
    for col in ('ribbon_minus_window', 'ribbon_minus_other', 'belt_minus_window'):
        b = g[col].idxmax()
        print(f'{tag:11s} {col:20s} max at {b:5.0f} km ({g[col].max():+.2f}%): '
              + ' '.join(f'{int(q)}:{g[col].iloc[np.argmin(np.abs(g.index - q))]:+.2f}'
                         for q in (300, 500, 660, 800, 1000, 1200, 1500, 2000, 2500)))

# The present-day primary conduit Mather et al. propose (their Fig. 4c,f, red circle),
# located by a similarity transform from the two youngest dated edifices in that panel
# (Gascoyne, 36.6 S 156.2 E; Lord Howe Island, 31.6 S 159.1 E); good to about a degree.
PLUME = (-36.7, 161.1)
R_KM = 6371.0
prow = []
for tag in ['RevealLO', 'REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1']:
    V = np.load(os.path.join(A.dir, f'tasman_split_volume_{tag}.npz'))
    z, la, lo, a = V['depth'], V['lat'], V['lon'], V['anom']
    LA, LO = np.meshgrid(la, lo, indexing='ij')
    p0, p1 = np.radians(PLUME[0]), np.radians(LA)
    dist = R_KM * np.arccos(np.clip(np.sin(p0) * np.sin(p1) + np.cos(p0) * np.cos(p1)
                                    * np.cos(np.radians(LO - PLUME[1])), -1, 1))
    cap = dist <= 250.0
    for q in (300, 400, 500, 660, 800, 1000, 1200, 1500, 2000, 2500, 2800):
        k = int(np.argmin(np.abs(z - q)))
        v = float(np.nanmean(a[k][cap]))
        sh = a[k][np.isfinite(a[k])]
        prow.append(dict(model=tag, depth=float(q), shell=float(z[k]), cap=v, pct=float((sh < v).mean())))
P = pd.DataFrame(prow)
P.to_csv(os.path.join(A.dir, 'tasman_plume_site_profile.csv'), index=False)
print(f'\nmean anomaly within 250 km of the proposed conduit {PLUME} (regional shell percentile)')
print(P.assign(s=P.cap.map('{:+.2f}'.format) + ' (' + (100 * P.pct).round().astype(int).astype(str) + ')')
       .pivot(index='depth', columns='model', values='s').reindex(columns=['RevealLO', 'REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1']).to_string())
