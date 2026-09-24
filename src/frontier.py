#!/usr/bin/env python3
"""The trade-off between cheapness and continuity, measured once per hotspot.

THE OBJECTIVE. Two earlier attempts failed for the same reason: continuity on its
own gives every hotspot the same answer, because a route that pays nothing for
distance can always reach good material somewhere. The length term was doing
necessary work. So the object is a two-term objective - the ordinary path cost plus
a penalty on the WORST material the route accepts:

    total = sum over moves of L exp(s a)  +  lambda * max over the route of a

WHY A FRONTIER AND NOT A LAMBDA. The maximum is not decomposable, so this cannot be
solved directly by the shell sweep. It does not need to be. For a fixed level, the
cheapest route whose worst point is no worse than that level is an ORDINARY search
with everything above the level forbidden, which cost_field now does exactly. Solve
that for a ladder of levels and each hotspot gets a curve of (worst point, cost).
The optimum for any lambda is a point on that curve, so lambda becomes a choice made
AFTER the compute rather than a parameter baked into it - and calibrating it over a
range costs nothing more.

The ends of the ladder are the two criteria that failed on their own: the widest
level is today's least-cost path, and the tightest level at which a route still
reaches the target is the pure bottleneck.

On a machine with little memory to spare, run one level per invocation: successive
solves in one process can leave freed cost fields unreturned, and the 3 GB Cowork
sandbox failed on its fourth. A desktop can take the whole ladder in one run. Either
way the file is written after each level and resumed on the next run:

  for L in 100 35 20 12 8 5; do
    python3 frontier.py --file <model.nc> --tag RevealLO --levels $L
  done
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from contrast import contrast_field
from path_cost import cost_field, site_cost, trace_path, channel_field
import path_config

R_E, DEG, SIGMA = 6371.0, np.pi / 180.0, 800.0

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--lateral', type=float, default=0.60)
ap.add_argument('--ncell', type=int, default=2)
ap.add_argument('--levels', default='1,2,3,5,8,12,20,35,100',
                help='per-shell percentiles of the anomaly. 100 forbids nothing and '
                     'reproduces the present path exactly, which is the control.')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--sites', default='')
ap.add_argument('--dir', default='out')
ap.add_argument('--no-resume', dest='resume', action='store_false',
                help='recompute every level instead of keeping the ones the output '
                     'file already holds')
A = ap.parse_args()

c = path_config.load(A.tag, A.dir)
hm = getattr(c, 'h_max', None)
hm = None if hm is None or (isinstance(hm, float) and hm != hm) else float(hm)

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
con = contrast_field(arr, lat, lon, SIGMA)
ach = channel_field(arr, con, str(c.channel))
z = np.asarray(depth, float)
lat = np.asarray(lat, float); lon = np.asarray(lon, float)
hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180']).set_index('hotspot')
want = [s.strip() for s in A.sites.split(',') if s.strip()] or list(hs.index)


def gc_km(la1, lo1, la2, lo2):
    p1, p2 = la1 * DEG, la2 * DEG
    dl = (lo2 - lo1) * DEG
    h = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2)
    return 2 * R_E * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


# Each level is written as it completes. A ladder is hours of compute and a run
# that has to start again from the beginning is a run nobody finishes; more to the
# point, a level that has been solved is a result, and holding it in memory until
# the last one finishes is the only way to lose it. Resuming skips levels the file
# already holds, so an interrupted ladder is continued rather than repeated, and
# the file is replaced through a temporary so an interrupted write cannot destroy
# the levels already in it.
OUT = os.path.join(A.dir, f'frontier_{A.tag}.csv')

# The configuration every row was computed under, written into the file. Resume
# without this is worse than no resume: the file that was here on 20 September held
# nine levels computed on the anom channel five days and a whole retrace earlier, and
# a resumed ladder would have skipped four of five requested levels and produced a
# file four parts one cost field and one part another, with nothing saying so.
CFG = dict(cfg_s=float(c.s), cfg_z_target=float(c.z_target),
           cfg_channel=str(c.channel), cfg_h_max=(hm if hm is not None else float('nan')),
           cfg_radius=float(c.radius), cfg_lateral=float(A.lateral),
           cfg_ncell=int(A.ncell))


def _same_config(d_):
    for k, v in CFG.items():
        if k not in d_.columns:
            return f'{k} is absent, so the file predates this check'
        u = d_[k].dropna().unique()
        if isinstance(v, str):
            if len(u) and str(u[0]) != v:
                return f'{k} is {u[0]!r} in the file and {v!r} now'
        elif v != v:                       # NaN: the file must be NaN too
            if len(u):
                return f'{k} is {u[0]} in the file and unset now'
        elif len(u) and abs(float(u[0]) - float(v)) > 1e-9:
            return f'{k} is {u[0]} in the file and {v} now'
    return None


done = pd.read_csv(OUT) if (A.resume and os.path.exists(OUT)) else None
if done is not None:
    why = _same_config(done)
    if why:
        raise SystemExit(
            f'{OUT} was computed under a different configuration: {why}.\n'
            f'Resuming would mix two cost fields in one file. Move it aside and '
            f'run again:\n  mkdir -p to_delete && mv {OUT} to_delete/')
have = set(done.level.round(6)) if done is not None else set()
rows = [] if done is None else done.to_dict('records')


def _flush():
    d_ = pd.DataFrame(rows)
    tmp = OUT + '.part'
    d_.to_csv(tmp, index=False)
    os.replace(tmp, OUT)


for lev in [float(x) for x in A.levels.split(',')]:
    if round(lev, 6) in have:
        print(f'level p{lev:g}: already in {OUT}, skipping', flush=True)
        continue
    if lev >= 100:
        ban = None
    else:
        thr = np.nanpercentile(ach.reshape(ach.shape[0], -1), lev, axis=1)
        ban = ach > thr[:, None, None]
    print(f'level p{lev:g}: '
          f'{"nothing forbidden" if ban is None else f"{100*ban.mean():.1f}% of cells forbidden"}',
          flush=True)
    C = cost_field(arr, con, depth, lat, lon, s=float(c.s), z_target=float(c.z_target),
                   n_relax=6, channel=str(c.channel), h_max_km=hm,
                   lateral=A.lateral, ncell=A.ncell, forbid=ban)
    cache = {}
    for name in want:
        hla, hlo = float(hs.at[name, 'lat']), float(hs.at[name, 'lon_180'])
        sc = float(site_cost(C, depth, lat, lon, hla, hlo, float(c.radius), 200.0))
        if not np.isfinite(sc):
            rows.append(dict(site=name, level=lev, reached=False, **CFG))
            continue
        tr = trace_path(C, arr, con, depth, lat, lon, hla, hlo, s=float(c.s),
                        radius_deg=float(c.radius), n_relax=6,
                        channel=str(c.channel), h_max_km=hm, lateral=A.lateral,
                        ncell=A.ncell, cache=cache)
        if tr is None:
            rows.append(dict(site=name, level=lev, reached=False, **CFG))
            continue
        zz, la, lo = (np.asarray(v, float) for v in tr)
        o = np.argsort(zz); zz, la, lo = zz[o], la[o], lo[o]
        ki = np.searchsorted(z, zz).clip(0, len(z) - 1)
        ii = np.abs(lat[None, :] - la[:, None]).argmin(axis=1)
        ji = np.abs(((lon[None, :] - lo[:, None] + 180) % 360) - 180).argmin(axis=1)
        v = np.asarray([ach[k, i, j] for k, i, j in zip(ki, ii, ji)], float)
        v = v[np.isfinite(v)]
        d = gc_km(hla, hlo, la, lo)
        # where the route ends and how slow it is on average. Without these a
        # reader can see that a tighter level costs more but not what it bought,
        # which is the only reason to look at the frontier at all.
        # what the route ends on. A tighter level is worth its extra cost only if
        # the route it selects finishes in deep slow material; a route that is
        # continuous all the way to somewhere neutral has bought nothing.
        cj = np.abs(lat - la[-1])
        ci = np.abs(((lon - lo[-1] + 180.) % 360.) - 180.) * np.cos(np.radians(la[-1]))
        cm = np.hypot(cj[:, None], ci[None, :]) <= 3.0
        kdp = np.where(z >= float(c.z_target))[0]
        cap = (float(np.nanmean(arr[kdp][:, cm]))
               if len(kdp) and cm.any() else float('nan'))
        rows.append(dict(**CFG, site=name, level=lev, reached=True, cost=sc,
                         worst=float(v.max()) if len(v) else np.nan,
                         mean=float(v.mean()) if len(v) else np.nan,
                         end_lat=float(la[-1]), end_lon=float(lo[-1]), cap_anom=cap,
                         offset_km=float(d[-1]), d660_km=float(np.interp(660., zz, d)),
                         d1000_km=float(np.interp(1000., zz, d))))
    del C
    _flush()
    print(f'level p{lev:g}: written', flush=True)

d = pd.DataFrame(rows)
out = OUT
ok = d[d.reached == True]
print(f'\n{len(ok)} of {len(d)} site-level combinations produced a path')
print('\ncoverage: sites with any route to the target, by level')
for l_, r_ in d.groupby('level').reached.agg(['sum', 'count']).iterrows():
    print(f'  p{l_:<5g} {int(r_["sum"]):3d} of {int(r_["count"])}')
print(f'\n{"site":24s} ' + ' '.join(f'{l:>7g}' for l in sorted(ok.level.unique())))
print('  cost of the cheapest route whose worst point is under each level')
for s_ in sorted(ok.site.unique())[:12]:
    g = ok[ok.site == s_].set_index('level')
    print(f'{s_:24s} ' + ' '.join(
        f'{g.cost.get(l, float("nan")):7.0f}' for l in sorted(ok.level.unique())))
print(f'\nwrote {out}')
print('Pick lambda afterwards: the chosen route at weight lambda is the level\n'
      'minimising cost + lambda * worst. Calibration sweeps lambda over this file.')
