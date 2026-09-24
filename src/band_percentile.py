#!/usr/bin/env python3
"""How slow, relative to its own depth shell, is the material a path runs through,
interval by interval?

track_support.py measures this in three bands (200-660, 660-1500, 1500-2700 km) for
the paper model against its ambient paths. This is the same measurement in the five
bands mid_mantle_census.py uses around the mid-mantle viscosity increase, for any
model, with the ambient paths included where they exist. The anomaly at each node is
expressed as a percentile of its depth shell, so a value of 0.10 means the slowest
tenth of that shell, and the path's value in a band is the median over its nodes
there.

    python3 band_percentile.py --file <model.nc> --tag RevealLO --var voigt --every 1

Writes out/band_percentile_<tag>.csv (site, population, one column per band);
mid_mantle_census.py reads it into out/mid_mantle_summary.csv.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd
import provenance
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from mid_mantle_census import BANDS

R_E, DEG = 6371.0, np.pi / 180.0


def measure(paths, z, lat, lon, order, anom):
    out = {}
    for name, q in paths.items():
        pz = np.asarray(q['depth'], float)
        pla = np.asarray(q['lat'], float)
        plo = np.asarray(q['lon'], float)
        if len(pz) < 3:
            continue
        o = np.argsort(pz); pz, pla, plo = pz[o], pla[o], plo[o]
        pct = np.empty(len(pz))
        for n, (zz, la, lo_) in enumerate(zip(pz, pla, plo)):
            k = int(np.argmin(np.abs(z - zz)))
            j = int(np.argmin(np.abs(lat - la)))
            i = int(np.argmin(np.abs(lon - lo_)))
            pct[n] = np.searchsorted(order[k], float(anom[k, j, i])) / order.shape[1]
        rec = {}
        for nm, a, b in BANDS:
            m = (pz >= a) & (pz < b)
            rec[nm] = float(np.median(pct[m])) if m.any() else np.nan
        out[name] = rec
    return pd.DataFrame(out).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
    ap.add_argument('--dir', default='out')
    a = ap.parse_args()

    pth = os.path.join(a.dir, f'conduit_paths_all_{a.tag}.json')
    amb = os.path.join(a.dir, f'ambient_paths_{a.tag}.json')
    depth, lat, lon, arr = load_anomaly(a.file, ModelSpec(a.tag, a.var),
                                        depth_max=a.depth_max, every=a.every)
    lon, arr = dedupe_lon(lon, arr)
    z, lat, lon = (np.asarray(v, float) for v in (depth, lat, lon))
    order = np.sort(arr.reshape(arr.shape[0], -1), axis=1)
    print(f'{a.tag}: {arr.shape[0]} shells', flush=True)

    H = measure(json.load(open(pth)), z, lat, lon, order, arr)
    H.insert(0, 'population', 'hotspot')
    inputs = [pth]
    if os.path.exists(amb):
        A_ = measure(json.load(open(amb)), z, lat, lon, order, arr)
        A_.insert(0, 'population', 'ambient')
        H = pd.concat([H, A_]); inputs.append(amb)
    H.index.name = 'site'
    out = os.path.join(a.dir, f'band_percentile_{a.tag}.csv')
    H.to_csv(out)
    provenance.stamp(out, inputs=inputs)
    print(H.groupby('population')[[nm for nm, _, _ in BANDS]].median().round(3).to_string())
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
