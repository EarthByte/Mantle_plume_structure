#!/usr/bin/env python3
"""How much worse is the least-cost path's worst point than it needed to be?

Two numbers per hotspot, in per cent anomaly, lower being slower and better:

  achievable   the best worst-point over all descending routes from the seed cap,
               from the bottleneck field
  accepted     the worst point the current least-cost path actually passes through

If the two agree the least-cost path is already the most continuous route and the
criterion question is moot for that site. If accepted is much worse than
achievable, the path is crossing material it did not have to, which is the
complaint the sections raise by eye, measured.

  python3 bottleneck_report.py --file <model.nc> --tag RevealLO
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from contrast import contrast_field
from path_cost import channel_field
from path_bottleneck import bottleneck_field, site_bottleneck
import path_config

SIGMA = 800.0

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--ncell', type=int, default=2)
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--sites', default='', help='default: every hotspot')
ap.add_argument('--dir', default='out')
ap.add_argument('--accepted-only', action='store_true', dest='accepted_only',
                help='skip the bottleneck field and report only what the traced path '
                     'accepts. The field is a full dynamic program and costs what a '
                     'cost field costs; the accepted side is a lookup. Every model can '
                     'afford the second every run, and the paper model then gets the '
                     'first in the continuity step, overwriting this file with the '
                     'achievable column filled in.')
A = ap.parse_args()

c = path_config.load(A.tag, A.dir)
hm = getattr(c, 'h_max', None)
hm = None if hm is None or (isinstance(hm, float) and hm != hm) else float(hm)

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
con = contrast_field(arr, lat, lon, SIGMA)
if A.accepted_only:
    B = None
    print('accepted side only; the bottleneck field is not built', flush=True)
else:
    print('building the bottleneck field', flush=True)
    B = bottleneck_field(arr, con, depth, lat, lon, z_target=float(c.z_target),
                         n_relax=6, channel=str(c.channel), h_max_km=hm,
                         ncell=A.ncell)
ach = channel_field(arr, con, str(c.channel))

# A cell is called a gap when its channel value is above this. It is not a tuning
# parameter: zero is "no slower than the local background", and a small negative
# number is the closest thing to a neutral cell the field contains.
GAP = -0.10

z = np.asarray(depth, float)
lat = np.asarray(lat, float); lon = np.asarray(lon, float)
hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180']).set_index('hotspot')
paths = json.load(open(os.path.join(A.dir, f'conduit_paths_all_{A.tag}.json')))
want = [s.strip() for s in A.sites.split(',') if s.strip()] or list(paths)

rows = []
for s in want:
    if s not in paths or s not in hs.index:
        continue
    hla, hlo = float(hs.at[s, 'lat']), float(hs.at[s, 'lon_180'])
    best = (np.nan if B is None else
            site_bottleneck(B, depth, lat, lon, hla, hlo, float(c.radius), 200.0))
    q = paths[s]
    zz = np.asarray(q['depth'], float)
    la = np.asarray(q['lat'], float); lo = np.asarray(q['lon'], float)
    ki = np.searchsorted(z, zz).clip(0, len(z) - 1)
    ii = np.abs(lat[None, :] - la[:, None]).argmin(axis=1)
    ji = np.abs(((lon[None, :] - lo[:, None] + 180) % 360) - 180).argmin(axis=1)
    v = np.asarray([ach[k, i, j] for k, i, j in zip(ki, ii, ji)], float)
    v = v[np.isfinite(v)]
    acc = float(v.max()) if len(v) else np.nan
    worst_z = float(zz[int(np.argmax(v))]) if len(v) else np.nan
    # What the route ends on. A path can be continuous all the way to somewhere
    # neutral, and that is a different failure from crossing a gap, so the two are
    # reported side by side rather than one standing in for the other.
    kd = np.where(z >= float(c.z_target))[0]
    dj = np.abs(lat - la[-1])
    di = np.abs(((lon - lo[-1] + 180.0) % 360.0) - 180.0) * np.cos(np.radians(la[-1]))
    m = np.hypot(dj[:, None], di[None, :]) <= 3.0
    cap = float(np.nanmean(arr[kd][:, m])) if len(kd) and m.any() else np.nan
    rows.append(dict(site=s, achievable=best, accepted=acc, penalty=acc - best,
                     worst_at_km=worst_z, mean=float(v.mean()) if len(v) else np.nan,
                     n=len(v), n_gap=int((v > GAP).sum()),
                     frac_gap=float((v > GAP).mean()) if len(v) else np.nan,
                     cap_anom=cap))

d = pd.DataFrame(rows).sort_values(
    'penalty' if not A.accepted_only else 'accepted', ascending=False)
out = os.path.join(A.dir, f'bottleneck_report_{A.tag}.csv')
d.to_csv(out, index=False)
print(f'\n{"site":26s} {"achievable":>11s} {"accepted":>9s} {"penalty":>8s} '
      f'{"worst at":>9s} {"gaps":>8s} {"cap":>7s}')
for _, r in d.head(14).iterrows():
    print(f'{r.site:26s} {r.achievable:10.3f}% {r.accepted:8.3f}% '
          f'{r.penalty:7.3f}% {r.worst_at_km:8.0f}k {r.n_gap:4.0f}/{r.n:<3.0f} '
          f'{r.cap_anom:+6.2f}%')
print(f'\n{int((d.accepted > GAP).sum())} of {len(d)} routes cross a cell that is not '
      f'slow at all; {int((d.worst_at_km > 400).sum())} of those worst cells are below '
      f'400 km, so they are not the lid at the seed')
print(f'{int((d.cap_anom > -1.0).sum())} of {len(d)} end on deep material weaker than '
      f'-1 per cent, {int((d.cap_anom > 0).sum())} of them on material that is fast')
if not A.accepted_only:
    print(f'median penalty {d.penalty.median():.3f}%, '
          f'{int((d.penalty > 0.5).sum())} sites above 0.5%')
print(f'wrote {out}')
