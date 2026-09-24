"""The tracer's false-positive branch spectrum, measured away from hotspots.

Injection calibration showed the ambient field produces a branch at roughly a
fifth of sites, and that those cluster near 1000 km - which is where Rudolph et
al. place their viscosity increase and therefore where a real signal is expected.
Any claim that plume branches pile up at 1000 km has to be made against this
background, so the background is measured rather than assumed.

This is not a null test of whether hotspots differ from random mantle. It is the
depth spectrum of the instrument's own false positives, needed to correct a
histogram that will be built from real plumes.
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon, check_shells
from plume_core import Volume, Tracer, per_depth_threshold, R_E, DEG
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=2)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--pct', type=float, default=15.0)
ap.add_argument('--min-area', type=float, default=2.0e5, dest='min_area')
ap.add_argument('--box', type=float, default=28.0)
ap.add_argument('--sites', type=int, default=200)
ap.add_argument('--away-km', type=float, default=1500.0, dest='away',
                help='keep sites at least this far from any hotspot')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--dir', default='out')
A = ap.parse_args()

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
check_shells(depth, arr, A.tag)
lat = np.asarray(lat, float)
lon = ((np.asarray(lon, float) + 180) % 360) - 180
o = np.argsort(lon); lon, arr = lon[o], arr[:, :, o]
THR = per_depth_threshold(arr, lat, A.pct)

hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180'])
hla = hs.lat.to_numpy(float); hlo = hs.lon_180.to_numpy(float)


def far(la, lo):
    d = R_E * np.arccos(np.clip(
        np.sin(la * DEG) * np.sin(hla * DEG) +
        np.cos(la * DEG) * np.cos(hla * DEG) * np.cos((hlo - lo) * DEG), -1, 1))
    return d.min() >= A.away


rng = np.random.default_rng(97)
sites, tries = [], 0
while len(sites) < A.sites and tries < A.sites * 60:
    tries += 1
    la = float(np.degrees(np.arcsin(2 * rng.random() - 1)))
    lo = float(360 * rng.random() - 180)
    if abs(la) > 80 or not far(la, lo):
        continue
    sites.append((la, lo))
print(f'{len(sites)} sites at least {A.away:.0f} km from any hotspot', flush=True)

rows = []
for k, (hlat, hlon) in enumerate(sites):
    jl = np.where(np.abs(lat - hlat) <= A.box)[0]
    dl = np.abs(((lon - hlon + 180) % 360) - 180)
    il = np.where(dl <= A.box / max(np.cos(hlat * DEG), 0.2))[0]
    if len(jl) < 20 or len(il) < 20:
        continue
    v = Volume(depth, lat[jl], lon[il],
               np.asarray(arr[:, jl][:, :, il], float),
               min_area_km2=A.min_area, thresholds=THR)
    t = Tracer(v).trace(hlat, hlon)
    if not t or not t.get('found'):
        rows.append(dict(lat=hlat, lon=hlon, found=False, n_branch=0))
        continue
    for d_ in (t['branch_depths'] or [np.nan]):
        rows.append(dict(lat=hlat, lon=hlon, found=True,
                         n_branch=t['n_branch'], branch_depth=d_,
                         root_km=t['root_km'], max_strands=t['max_strands']))
    if (k + 1) % 40 == 0:
        print(f'  {k + 1}/{len(sites)}', flush=True)

d = pd.DataFrame(rows)
p = os.path.join(A.dir, f'plume_ambient_{A.tag}.csv')
d.to_csv(p, index=False)
n_site = d.groupby(['lat', 'lon']).ngroups
withb = d[np.isfinite(d.get('branch_depth', pd.Series(dtype=float)))]
print(f'\n{n_site} sites traced; a branch appeared at '
      f'{withb.groupby(["lat", "lon"]).ngroups} of them '
      f'({100 * withb.groupby(["lat", "lon"]).ngroups / max(n_site, 1):.0f}%)')
if len(withb):
    print('\nfalse-positive branch depths:')
    for lo_, hi_ in [(300, 660), (660, 900), (900, 1100), (1100, 1500),
                     (1500, 2000), (2000, 2900)]:
        m = (withb.branch_depth >= lo_) & (withb.branch_depth < hi_)
        print(f'  {lo_:5d}-{hi_:5d} km  {int(m.sum()):4d}  '
              f'({100 * m.mean():4.0f}% of all false positives)')
    print(f'\nmedian {withb.branch_depth.median():.0f} km, '
          f'quartiles {withb.branch_depth.quantile(.25):.0f}-'
          f'{withb.branch_depth.quantile(.75):.0f} km')
print(f'\nwrote {p}', flush=True)
