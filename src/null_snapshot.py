"""Save the null distribution of path costs for one calibrated configuration.

The classification compares each hotspot with this distribution, so it is worth
showing. Recomputed rather than stored during classification because it is one
cost field, and the median calibrated configuration is not known until
calibration has finished.
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from contrast import contrast_field
from path_cost import cost_field, site_cost
import path_config
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--dir', default='out')
ap.add_argument('--ncell', type=int, default=1,
                help='lateral cells a single slanted move may cross; '
                     'slant_calibrate.py chose 2')
ap.add_argument('--lateral', type=float, default=1.0,
                help='weight on the lateral component of a move length; '
                     'slant_calibrate.py chose 0.60')
ap.add_argument('--n', type=int, default=2000)
A = ap.parse_args()
f, var, every = A.file, A.var, A.every

# The configuration is READ from out/path_config_<tag>.json, never re-derived.
# Re-deriving it as the median of the retained set makes it depend on which
# configurations happened to clear a 75 per cent floor on eighteen injections,
# and after the move-set retrace that median moved from s=0.4 z=2700 r=3.0 to
# s=0.3 z=2800 r=2.0. A null traced under one configuration and hotspots under
# another are two different searches and must not be compared.
c = path_config.load(A.tag, A.dir)
print(f'{A.tag}: s={c.s} z_target={c.z_target:.0f} {c.channel} radius={c.radius}')

depth, lat, lon, arr = load_anomaly(f, ModelSpec(A.tag, var),
                                    depth_max=A.depth_max, every=every)
lon, arr = dedupe_lon(lon, arr)
con = contrast_field(arr, lat, lon, 800.0)
C = cost_field(arr, con, depth, lat, lon, s=float(c.s), z_target=float(c.z_target),
               n_relax=6, channel=str(c.channel),
               lateral=float(A.lateral), ncell=int(A.ncell))

rng = np.random.default_rng(47)
nlo = 360 * rng.random(A.n) - 180
nla = np.degrees(np.arcsin(2 * rng.random(A.n) - 1))
nul = np.array([site_cost(C, depth, lat, lon, la, lo, float(c.radius), 200.0)
                for la, lo in zip(nla, nlo)], float)

hs = pd.read_csv('hotspots_courtillot2003.csv')
hs['cost'] = [site_cost(C, depth, lat, lon, float(r.lat), float(r.lon_180),
                        float(c.radius), 200.0) for _, r in hs.iterrows()]
np.savez(os.path.join(A.dir, f'null_{A.tag}.npz'), null=nul,
         cost=hs.cost.values, hotspot=hs.hotspot.values,
         p5=np.nanpercentile(nul, 5), s=c.s, z_target=c.z_target,
         channel=str(c.channel), radius=c.radius)
print(f'null p5 = {np.nanpercentile(nul, 5):.0f}, median = {np.nanmedian(nul):.0f}; '
      f'{int((hs.cost < np.nanpercentile(nul, 5)).sum())} hotspots below it')
