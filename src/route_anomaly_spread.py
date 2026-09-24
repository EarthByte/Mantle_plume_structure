#!/usr/bin/env python3
"""How well is the anomaly averaged along a traced route known?

Two descents are treated as indistinguishable when they differ in the anomaly the cost
integrates by less than a stated amount. That amount was fixed as a judgement, with the
note that the better number is the spread of the same quantity across the models. This
measures it: every hotspot's traced RevealLO route is held fixed, and the channel the
cost integrates - the local contrast, clipped as the cost clips it - is averaged along
it, weighted by the length the objective charges, in each model in turn.

The spread across models is the uncertainty a reader should attach to any statement
about which of two routes is slower. It is compared with two other scales: the numerical
reproducibility of the same quantity under resampling of one model, which is far
smaller, and the difference between the first and second descents at each hotspot.

One model per invocation, because this machine holds one model at a time; the results
accumulate in out/route_anomaly_spread.csv and --report summarises them.

  python3 route_anomaly_spread.py --model GLADM35
  python3 route_anomaly_spread.py --report
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from contrast import contrast_field
from path_cost import channel_field

R_E, DEG = 6371.0, np.pi / 180.0
ROOT = os.path.expanduser('~/mnt/Muller_mantle_tomography_subduction')
if not os.path.isdir(ROOT):
    ROOT = os.path.expanduser('~/Documents/Papers/in_prep/Muller_mantle_tomography_subduction')
TOMO = os.path.join(ROOT, 'REVEAL_mantle_tomography')
MODELS = {
    'RevealLO':   (os.path.join(TOMO, 'RevealLO.nc'), 'voigt', 1, 2880.0),
    'REVEAL':     (os.path.join(ROOT, 'REVEAL_vs_full.nc'), 'voigt', 1, 2880.0),
    'GLADM35':    (os.path.join(TOMO, 'GLADM35.nc'), 'voigt', 2, 2890.0),
    'SPiRaL':     (os.path.join(TOMO, 'SPiRaL.nc'), 'voigt', 1, 2891.0),
    'SEMUCB-WM1': (os.path.join(TOMO, 'SEMUCB-WM1.nc'), 'vs', 1, 2891.0),
}
LATERAL = 0.60


def gc(a1, o1, a2, o2):
    p1, p2 = a1 * DEG, a2 * DEG
    dl = (o2 - o1) * DEG
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_E * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--dir', default='out')
    A = ap.parse_args()
    OUT = os.path.join(A.dir, 'route_anomaly_spread.csv')

    if A.model:
        f, var, every, dmax = MODELS[A.model]
        paths = json.load(open(os.path.join(A.dir, 'conduit_paths_all_RevealLO.json')))
        depth, lat, lon, arr = load_anomaly(f, ModelSpec(A.model, var),
                                            depth_max=dmax, every=every)
        lon, arr = dedupe_lon(lon, arr)
        con = contrast_field(arr, lat, lon, 800.0)
        ach = np.clip(channel_field(arr, con, 'contrast'), -6.0, 6.0)
        del arr, con
        rows = []
        zmax = float(depth.max())
        for site, p in paths.items():
            z = np.asarray(p['depth'], float); la = np.asarray(p['lat'], float)
            lo = np.asarray(p['lon'], float)
            o = np.argsort(z); z, la, lo = z[o], la[o], lo[o]
            k = z <= zmax                      # GLAD-M35 stops shallower
            z, la, lo = z[k], la[k], lo[k]
            ki = np.abs(depth[None, :] - z[:, None]).argmin(axis=1)
            ji = np.abs(lat[None, :] - la[:, None]).argmin(axis=1)
            ii = np.abs(((lon[None, :] - lo[:, None] + 180) % 360) - 180).argmin(axis=1)
            v = ach[ki, ji, ii]
            L = np.hypot(np.diff(z), LATERAL * gc(la[:-1], lo[:-1], la[1:], lo[1:]))
            am = 0.5 * (v[:-1] + v[1:])
            rows.append(dict(model=A.model, site=site,
                             mean_a=float(np.average(am, weights=L))))
        new = pd.DataFrame(rows)
        if os.path.exists(OUT):
            old = pd.read_csv(OUT)
            new = pd.concat([old[old.model != A.model], new], ignore_index=True)
        new.to_csv(OUT + '.part', index=False); os.replace(OUT + '.part', OUT)
        print(f'{A.model}: {len(rows)} routes averaged, wrote {OUT}')

    if A.report:
        d = pd.read_csv(OUT)
        w = d.pivot(index='site', columns='model', values='mean_a')
        print(f'models present: {list(w.columns)}  sites {len(w)}')
        sd = w.std(axis=1, ddof=1)
        rg = w.max(axis=1) - w.min(axis=1)
        print(f'  across models, per hotspot: median s.d. {sd.median():.3f} %, '
              f'median range {rg.median():.3f} %')
        c = pd.read_csv(os.path.join(A.dir, 'competing_routes_RevealLO.csv'))
        c = c[c.alt_reached == True]
        gap = (c.mean_a - c.alt_mean_a).abs()
        print(f'  first vs second descent, RevealLO: median |gap| {gap.median():.3f} %')
        print(f'  gap below the cross-model s.d. of the same site: '
              f'{int((gap.values < sd.reindex(c.site).values).sum())} of {len(c)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
