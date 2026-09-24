#!/usr/bin/env python3
"""Trace every hotspot at a stated move set and channel, and report how far the
path strays from its own volcano.

WHY THIS EXISTS. The move set was calibrated by tilt_recovery.py, which asks only
whether a conduit of known tilt is followed. Nothing in that criterion measures the
opposite error - a path leaving a conduit that is there. The zero-tilt rows of the
calibration measure exactly that, and they disagree sharply between ncell 1 and
ncell 2, so the move set in production was chosen on evidence that never tested it.

This supplies the missing side of the criterion at the level the reader sees:
distance from the hotspot against depth, for all sites at once, so that a change
which repairs the failing sites can be checked against the sites already right.

It writes only its own report and reads the frozen configuration without altering
it; --channel overrides the channel for this run alone and is stamped into the
output. Compare against out/production_offsets_RevealLO.csv, the same measurement
taken on the traced paths behind the current figures.

    python3 moveset_check.py --file <RevealLO.nc> --lateral 0.6 --ncell 1
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
from path_cost import cost_field, trace_path
import path_config
import provenance

R_E, DEG, SIGMA = 6371.0, np.pi / 180.0, 800.0
DEPTHS = (300., 400., 500., 660., 800., 1000., 1500., 2000., 2500.)

ap = argparse.ArgumentParser()
ap.add_argument('--file', default=None)
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--lateral', type=float, default=None)
ap.add_argument('--ncell', type=int, default=None)
ap.add_argument('--measure', default=None,
                help='measure an existing traced path file written by paths.py --all '
                     'instead of tracing anything, and write the result as '
                     'production_offsets_<tag>.csv. This is how the comparison baseline '
                     'is regenerated after a retrace, so it is reproducible rather than '
                     'whatever was measured by hand at the time.')
ap.add_argument('--channel', default=None,
                help='override the frozen channel for this run only')
ap.add_argument('--s', default=None,
                help='comma-separated s values to scan, overriding the frozen s for '
                     'this run only. The model is loaded once and a cost field is '
                     'built and released per value.')
ap.add_argument('--sites', default=None,
                help='comma-separated subset; default is every hotspot in the table')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--dir', default='out')
ap.add_argument('--out', default=None)
A = ap.parse_args()

R_E_ = R_E


def _gc(la1, lo1, la2, lo2):
    p1, p2 = la1 * DEG, la2 * DEG
    dl = (lo2 - lo1) * DEG
    h = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2)
    return 2 * R_E_ * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


if A.measure:
    import json as _json
    _hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180']).set_index('hotspot')
    _P = _json.load(open(A.measure))
    _rows = []
    for _n, _rec in _P.items():
        if _n not in _hs.index:
            continue
        _z = np.asarray(_rec['depth'], float)
        _la = np.asarray(_rec['lat'], float)
        _lo = np.asarray(_rec['lon'], float)
        _o = np.argsort(_z); _z, _la, _lo = _z[_o], _la[_o], _lo[_o]
        _d = _gc(float(_hs.at[_n, 'lat']), float(_hs.at[_n, 'lon_180']), _la, _lo)
        _r = dict(site=_n, max_km=float(_d.max()), max_deg=float(_d.max() / 111.19),
                  end_km=float(_d[-1]), zmax=float(_z.max()),
                  um_max=float(_d[_z <= 660].max()) if (_z <= 660).any() else np.nan)
        for _t in DEPTHS:
            _r[f'd{int(_t)}'] = float(np.interp(_t, _z, _d))
        _rows.append(_r)
    _D = pd.DataFrame(_rows).sort_values('max_deg', ascending=False)
    _out = A.out or os.path.join(A.dir, f'production_offsets_{A.tag}.csv')
    _D.to_csv(_out, index=False)
    provenance.stamp(_out, measured_from=A.measure, n_paths=len(_D),
                     inputs=[A.measure, A.hotspots])
    print(f'measured {len(_D)} paths from {A.measure}\n')
    _hdr = ' '.join(f'{int(_t):>5d}' for _t in DEPTHS)
    print(f'{"site":26s} {_hdr}  um_max   max  max_deg')
    for _, _r in _D.iterrows():
        _cells = ' '.join(f'{_r[f"d{int(_t)}"]:5.0f}' for _t in DEPTHS)
        print(f'{_r.site:26s} {_cells} {_r.um_max:6.0f} {_r.max_km:6.0f} {_r.max_deg:7.1f}')
    print(f'\nmedian max offset {_D.max_deg.median():.1f} deg, '
          f'{int((_D.max_deg > 15).sum())} sites beyond 15 deg')
    print(f'wrote {_out}')
    raise SystemExit(0)

if A.file is None or A.lateral is None or A.ncell is None:
    raise SystemExit('--file, --lateral and --ncell are required unless --measure is given')

c = path_config.load(A.tag, A.dir)
hm = getattr(c, 'h_max', None)
hm = None if hm is None or (isinstance(hm, float) and hm != hm) else float(hm)
ch = A.channel or str(c.channel)
S_LIST = ([float(x) for x in A.s.split(',') if x.strip()] if A.s else [float(c.s)])
print(f'tag {A.tag}  z_target {c.z_target}  radius {c.radius}  h_max {hm}')
print(f'move set lateral {A.lateral} ncell {A.ncell}   channel {ch}'
      f'{"  (frozen is " + str(c.channel) + ")" if A.channel else ""}')
print(f's {", ".join(f"{v:g}" for v in S_LIST)}'
      f'{"  (frozen is " + f"{float(c.s):g}" + ")" if A.s else ""}\n', flush=True)

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
con = contrast_field(arr, lat, lon, SIGMA)
hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180']).set_index('hotspot')
want = ([s.strip() for s in A.sites.split(',') if s.strip()] if A.sites
        else list(hs.index))


def gc_km(la1, lo1, la2, lo2):
    p1, p2 = la1 * DEG, la2 * DEG
    dl = (lo2 - lo1) * DEG
    h = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2)
    return 2 * R_E * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


rows = []
for sv in S_LIST:
    print(f'building the cost field at s={sv:g}', flush=True)
    C = cost_field(arr, con, depth, lat, lon, s=sv, z_target=float(c.z_target),
                   n_relax=6, channel=ch, h_max_km=hm,
                   lateral=A.lateral, ncell=A.ncell)
    cache = {}
    for n, name in enumerate(want, 1):
        if name not in hs.index:
            print(f'  {name}: not in the hotspot table')
            continue
        hla, hlo = float(hs.at[name, 'lat']), float(hs.at[name, 'lon_180'])
        tr = trace_path(C, arr, con, depth, lat, lon, hla, hlo, s=sv,
                        radius_deg=float(c.radius), n_relax=6, channel=ch,
                        h_max_km=hm, lateral=A.lateral, ncell=A.ncell, cache=cache)
        base = dict(site=name, channel=ch, lateral=A.lateral, ncell=A.ncell, s=sv)
        if tr is None:
            rows.append(base)
            continue
        z, la, lo = (np.asarray(v, float) for v in tr)
        o = np.argsort(z); z, la, lo = z[o], la[o], lo[o]
        d = gc_km(hla, hlo, la, lo)
        r = dict(base, max_km=float(d.max()), max_deg=float(d.max() / 111.19),
                 end_km=float(d[-1]), zmax=float(z.max()),
                 um_max=float(d[z <= 660].max()) if (z <= 660).any() else np.nan)
        for t_ in DEPTHS:
            r[f'd{int(t_)}'] = float(np.interp(t_, z, d))
        rows.append(r)
        if n % 10 == 0:
            print(f'  {n}/{len(want)}', flush=True)
    del C, cache

D = pd.DataFrame(rows)
stag = ('s' + '-'.join(f'{v:g}' for v in S_LIST)) if A.s else f's{S_LIST[0]:g}'
out = A.out or os.path.join(
    A.dir, f'moveset_{A.tag}_lat{A.lateral:g}_nc{A.ncell}_{ch}_{stag}.csv')
D.to_csv(out, index=False)
provenance.stamp(out, config=c, lateral=A.lateral, ncell=A.ncell,
                 channel_override=A.channel, s_scan=A.s,
                 inputs=[A.file, A.hotspots])

ok = D.dropna(subset=['max_deg'])
ref = os.path.join(A.dir, f'production_offsets_{A.tag}.csv')
PR = (pd.read_csv(ref)[['site', 'max_deg']].rename(columns={'max_deg': 'prod_deg'})
      if os.path.exists(ref) else None)

for sv in S_LIST:
    g = ok[ok.s == sv].sort_values('max_deg', ascending=False)
    print(f'\n=========== s = {sv:g}   (lateral {A.lateral}, ncell {A.ncell}, '
          f'channel {ch}) ===========')
    print(f'{len(g)} traced, {int((D.s == sv).sum()) - len(g)} untraced; '
          f'median max offset {g.max_deg.median():.1f} deg, '
          f'{int((g.max_deg > 15).sum())} sites beyond 15 deg')
    if PR is not None:
        m = g.merge(PR, on='site', how='inner')
        m['change'] = m.max_deg - m.prod_deg
        worse = m[m.change > 0.5].sort_values('change', ascending=False)
        print(f'  improved {int((m.change < -0.5).sum())}, '
              f'unchanged {int((m.change.abs() <= 0.5).sum())}, '
              f'degraded {int((m.change > 0.5).sum())} of {len(m)}')
        best = m.sort_values('change').head(10)
        print('  most improved:')
        for _, r in best.iterrows():
            print(f'    {r.site:26s} {r.prod_deg:6.1f} -> {r.max_deg:6.1f}   {r.change:+6.1f}')
        if len(worse):
            print('  degraded:')
            for _, r in worse.iterrows():
                print(f'    {r.site:26s} {r.prod_deg:6.1f} -> {r.max_deg:6.1f}   {r.change:+6.1f}')

print('\n--- the sites under discussion, max offset in degrees by s ---\n')
WATCH = ['Iceland', 'Darfur', 'Crozet/Pr. Edward', 'Baja/Guadalupe', 'Afar',
         'Marquesas', 'Discovery', 'Pitcairn', 'Hawaii', 'Louisville',
         'Kerguelen(Heard)', 'Reunion', 'Galapagos', 'Tristan']
print(f'{"site":26s} ' + ' '.join(f'{v:>7g}' for v in S_LIST) + '   production')
for name in WATCH:
    g = ok[ok.site == name]
    if not len(g):
        continue
    cells = []
    for sv in S_LIST:
        h = g[g.s == sv]
        cells.append(f'{h.max_deg.iloc[0]:7.1f}' if len(h) else '      -')
    pd_ = (PR[PR.site == name].prod_deg.iloc[0]
           if PR is not None and (PR.site == name).any() else float('nan'))
    print(f'{name:26s} ' + ' '.join(cells) + f'   {pd_:9.1f}')
print(f'\nwrote {out}')
