"""Anomaly beneath a hotspot as a function of depth, and the model around it.

The classification asks whether a path is cheaper than the null. This asks the
simpler questions behind it: how slow is the slowest material available beneath a
hotspot at each depth within a cap of the given radius, what the cap averages,
and what the same field looks like once the long-wavelength background is removed
by the local contrast. It is what the discussion quotes to show where a path
fails, and it is reported from the model at its native sampling rather than from
a decimated copy.

Averaged over a depth range, the cap minimum and the cap mean are very different
quantities: the cap is mostly background, so beneath Hawaii between 1000 and
2000 km the slowest material available averages -0.88 per cent while the cap
averages -0.06. A path takes the slowest route it can find, so the first is the
one that bears on whether a connection exists, and it is what the discussion
quotes.

The root-mean-square anomaly of each depth shell is carried in the same file,
under the hotspot name "(shell)", because the supplement compares the models on
that quantity and it costs nothing once the cube is loaded.
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contrast import contrast_field
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
warnings.filterwarnings('ignore')

R_E, DEG = 6371.0, np.pi / 180.0

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--radius', type=float, default=500.0, help='cap radius in km')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--sites', default='Hawaii,Kerguelen,Reunion,Tahiti,Macdonald,Iceland',
                help='comma-separated, matched on the start of the name')
ap.add_argument('--tag', default=None,
                help='model tag; names the output file when --out is not given')
ap.add_argument('--sigma-km', type=float, default=800.0, dest='sigma_km',
                help='smoothing length of the local-contrast background; the '
                     'default matches SIGMA in classify.py')
ap.add_argument('--out', default=None)
A = ap.parse_args()
if A.out is None:
    A.out = f'out/site_profiles_{A.tag}.csv' if A.tag else 'out/site_profiles.csv'

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec('m', A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
print(f'{arr.shape[0]} shells {depth.min():.0f}-{depth.max():.0f} km, '
      f'{arr.shape[1]}x{arr.shape[2]}', flush=True)

cf = contrast_field(arr, lat, lon, A.sigma_km)

hs = pd.read_csv(A.hotspots)
LO, LA = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
LO = ((LO + 180) % 360) - 180

rows = []
for want in [s.strip() for s in A.sites.split(',')]:
    m = hs[hs.hotspot.astype(str).str.lower().str.startswith(want.lower()[:6])]
    if not len(m):
        print(f'  {want}: not in the hotspot table'); continue
    r = m.iloc[0]
    d = R_E * np.arccos(np.clip(
        np.sin(float(r.lat) * DEG) * np.sin(LA * DEG) +
        np.cos(float(r.lat) * DEG) * np.cos(LA * DEG) *
        np.cos((LO - float(r.lon_180)) * DEG), -1, 1))
    cap = d <= A.radius
    prof = np.array([np.nanmin(np.where(cap, s, np.nan)) for s in arr])
    mean = np.array([np.nanmean(np.where(cap, s, np.nan)) for s in arr])
    cmin = np.array([np.nanmin(np.where(cap, s, np.nan)) for s in cf])
    for z, a, m, c in zip(depth, prof, mean, cmin):
        rows.append(dict(hotspot=str(r.hotspot), depth_km=float(z),
                         min_dvs=float(a), mean_dvs=float(m),
                         min_contrast=float(c)))
    show = [100, 200, 300, 410, 500, 660, 800, 1000, 1500, 2000, 2400, 2800]
    print(f'  {str(r.hotspot)[:24]:26s} '
          + '  '.join(f'{z}:{prof[int(np.argmin(abs(depth - z)))]:+.2f}' for z in show),
          flush=True)
    for z0, z1 in ((1000, 2000), (2000, 2880)):
        w = (depth >= z0) & (depth <= z1)
        # The discussion quotes the first of these. They are far apart - the cap
        # is mostly background, so its mean is much weaker than the slowest
        # material in it, and a path takes the slowest route available.
        print(f'{"":28s} {z0}-{z1} km: slowest available '
              f'{np.nanmean(prof[w]):+.2f} %,  cap mean '
              f'{np.nanmean(mean[w]):+.2f} %', flush=True)

# Whole-shell root-mean-square, for the comparison of models in the supplement.
rms = np.sqrt(np.nanmean(np.square(arr.reshape(arr.shape[0], -1)), axis=1))
for z, v in zip(depth, rms):
    rows.append(dict(hotspot='(shell)', depth_km=float(z), min_dvs=np.nan,
                     mean_dvs=np.nan, min_contrast=np.nan, rms_dvs=float(v)))
print('  root-mean-square anomaly by depth: '
      + '  '.join(f'{z}:{rms[int(np.argmin(abs(depth - z)))]:.2f}'
                  for z in (1000, 1500, 2000, 2400, 2800)), flush=True)

os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
pd.DataFrame(rows).to_csv(A.out, index=False)
print('wrote', A.out)
