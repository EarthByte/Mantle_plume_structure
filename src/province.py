"""How far a hotspot sits inside a large low-velocity province.

Hotspot geochemistry has long been associated with the two great slow regions of
the lowermost mantle, and any measure of depth of connection that is built by
following an assumed conduit through slow material will inherit that association
whether or not the conduit is real. Separating the two requires the province
membership to be measured on its own, which is what this does: the mean shear
velocity anomaly of the lowermost mantle beneath each hotspot, in a cap of the
given radius, over a depth window that stops above the core-mantle boundary.

The value is also expressed as a percentile against the same random locations the
classification uses, so that it reads on the same scale as everything else here:
a low percentile means few random places sit over material this slow.
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
warnings.filterwarnings('ignore')

R_E, DEG = 6371.0, np.pi / 180.0

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--z0', type=float, default=2600.0,
                help='top of the window averaged over, km')
ap.add_argument('--z1', type=float, default=2880.0,
                help='base of the window, km; keep above the core-mantle boundary')
ap.add_argument('--radius', type=float, default=800.0, help='cap radius, km')
ap.add_argument('--n-null', type=int, default=300, dest='n_null')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--dir', default='out')
A = ap.parse_args()

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
k = np.where((depth >= A.z0) & (depth <= A.z1))[0]
if not len(k):
    raise SystemExit(f'no shells between {A.z0:.0f} and {A.z1:.0f} km')
deep = np.nanmean(arr[k], axis=0)
print(f'{A.tag}: averaging {len(k)} shells, {depth[k].min():.0f}-{depth[k].max():.0f} km',
      flush=True)

LO, LA = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
LO = ((LO + 180) % 360) - 180


def cap_mean(hlat, hlon):
    d = R_E * np.arccos(np.clip(
        np.sin(hlat * DEG) * np.sin(LA * DEG) +
        np.cos(hlat * DEG) * np.cos(LA * DEG) * np.cos((LO - hlon) * DEG), -1, 1))
    m = d <= A.radius
    return float(np.nanmean(deep[m])) if m.any() else np.nan


rng = np.random.default_rng(51)                 # the null the rest of the study uses
nlo = 360 * rng.random(A.n_null) - 180
nla = np.degrees(np.arcsin(2 * rng.random(A.n_null) - 1))
nul = np.array([cap_mean(a_, b_) for a_, b_ in zip(nla, nlo)], float)
nul = nul[np.isfinite(nul)]

hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180'])
rows = []
for _, r in hs.iterrows():
    v = cap_mean(float(r.lat), float(r.lon_180))
    rows.append(dict(hotspot=str(r.hotspot), dVs_deep=v,
                     pct_vs_null=float(100.0 * np.mean(nul < v)) if np.isfinite(v) else np.nan))
out = pd.DataFrame(rows).sort_values('dVs_deep')
os.makedirs(A.dir, exist_ok=True)
out.to_csv(os.path.join(A.dir, f'province_{A.tag}.csv'), index=False)

print(f'null of {len(nul)} random caps: median {np.median(nul):+.3f} per cent\n')
print('lowermost-mantle anomaly beneath each hotspot, slowest first')
print('(a low percentile means few random places sit over material this slow)\n')
print(f"  {'hotspot':28s} {'dVs':>8s} {'pct':>6s}")
for _, r in out.head(15).iterrows():
    print(f'  {str(r.hotspot)[:26]:28s} {r.dVs_deep:+7.3f}% {r.pct_vs_null:5.1f}')
print('  ...')
for _, r in out.tail(5).iterrows():
    print(f'  {str(r.hotspot)[:26]:28s} {r.dVs_deep:+7.3f}% {r.pct_vs_null:5.1f}')
print(f'\nwrote province_{A.tag}.csv')
