#!/usr/bin/env python3
"""Choose the move set and lateral weight, on hotspots against a matched ambient set.

The search could not express a tilt steeper than about 64 degrees, because a slanted
move descended at least one shell and crossed at most one cell. `ncell` widens that;
`lateral` discounts horizontal travel. Both let the path reach further, and both let it
wander further, so neither can be chosen on how far the path travels.

THE CRITERION, AND WHY IT IS THIS ONE

Lateral offset carries no information: hotspot offsets are indistinguishable from the
offsets of paths traced at ambient sites where there is no conduit at all (496 against
480 km, P = 0.87, ambient_null.py). A setting cannot therefore be chosen by the offsets
it produces. What a real conduit offers that ambient mantle does not is slow material,
so the measure is how much slower the material along a hotspot path is than along an
ambient path traced at the same setting.

PRE-REGISTERED, FIXED BEFORE THE FIRST RUN

  Measure   gain = median mean anomaly along hotspot paths
                 - median mean anomaly along ambient paths, same setting.
            More negative is better. Ambient sites are at least 1500 km from any
            hotspot and are the same set at every setting.

  Admit     a setting only if hotspot paths are slower than ambient at P < 0.05,
            one-sided Mann-Whitney. A gain with no support behind it is noise.

  Choose    among admitted settings, the largest gain in magnitude. Where two are
            within 10 per cent of the best, take the more conservative: larger
            lateral first, then smaller ncell. A setting that reaches further should
            have to earn it.

  Report    the offset distributions too, hotspot and ambient, so that a setting which
            buys its gain by wandering is visible rather than hidden behind one number.

  If no setting is admitted, the finding is that the method cannot follow steep
  conduits on this data and that is what the paper says.

  python3 slant_calibrate.py --file <model.nc> --pilot
  python3 slant_calibrate.py --file <model.nc>          # resumes; run until complete
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from inject import gc_km
from path_cost import cost_field, trace_path
from contrast import contrast_field

R_E, DEG, SEED = 6371.0, math.pi / 180.0, 200.0
HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--var', default='voigt')
ap.add_argument('--settings', default='1.0:1,1.0:3,0.8:3,0.6:3,0.6:2',
                help='lateral:ncell pairs')
ap.add_argument('--ambient', type=int, default=30)
ap.add_argument('--seed', type=int, default=51)
ap.add_argument('--crop-deg', type=float, default=32.0, dest='crop_deg')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--pilot', action='store_true', help='8 hotspots and 8 ambient sites')
ap.add_argument('--no-resume', dest='resume', action='store_false')
ap.set_defaults(resume=True)
ap.add_argument('--report-only', action='store_true', dest='report_only')
ap.add_argument('--dir', default='out')
A = ap.parse_args()

SET = [(float(x.split(':')[0]), int(x.split(':')[1]))
       for x in A.settings.split(',') if x.strip()]
OUT = os.path.join(HERE, A.dir, f'slant_calibrate_{A.tag}.csv')
COLS = ['lateral', 'ncell', 'kind', 'site', 'lat', 'lon', 'offset_km',
        'mean_anom', 'edge_deg']

det = pd.read_csv(os.path.join(HERE, A.dir, f'detection_{A.tag}.csv'))
keep = det[(det.n_cal >= 4) & (det.detect_cal >= 0.75)].sort_values(
    ['s', 'z_target', 'channel', 'radius']).reset_index(drop=True)
c = keep.iloc[len(keep) // 2]
hm = getattr(c, 'h_max', None)
hm = None if hm is None or (isinstance(hm, float) and hm != hm) else float(hm)

hs = pd.read_csv(os.path.join(HERE, A.hotspots)).dropna(
    subset=['lat', 'lon_180']).set_index('hotspot')
rng = np.random.default_rng(A.seed)
amb = []
while len(amb) < A.ambient:
    la, lo = float(rng.uniform(-55, 55)), float(rng.uniform(-180, 180))
    if np.min(gc_km(la, lo, hs.lat.to_numpy(float), hs.lon_180.to_numpy(float))) > 1500:
        amb.append((f'amb{len(amb):02d}', la, lo))
JOBS = ([('hotspot', s, float(hs.at[s, 'lat']), float(hs.at[s, 'lon_180']))
         for s in hs.index] + [('ambient', n, la, lo) for n, la, lo in amb])
if A.pilot:
    JOBS = [j for j in JOBS if j[0] == 'hotspot'][:8] + \
           [j for j in JOBS if j[0] == 'ambient'][:8]

done, rows = set(), []
if A.resume and os.path.exists(OUT):
    _d = pd.read_csv(OUT)
    rows = _d.to_dict('records')
    done = {(round(float(r['lateral']), 3), int(r['ncell']), str(r['site']))
            for r in rows}
    print(f'resuming: {len(done)} traces already done', flush=True)

if not A.report_only:
    depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var), depth_max=2880.0)
    lon, arr = dedupe_lon(lon, arr)
    arr = np.asarray(arr, float)
    z = np.asarray(depth, float)
    LATV, LONV = np.asarray(lat, float), np.asarray(lon, float)
    ztar = float(c.z_target)
    print(f'model loaded; {len(JOBS)} sites x {len(SET)} settings', flush=True)

    for lam, ncell in SET:
        for kind, name, la, lo in JOBS:
            if (round(lam, 3), ncell, name) in done:
                continue
            jj = np.where(np.abs(LATV - la) <= A.crop_deg)[0]
            dl = ((LONV - lo + 180.0) % 360.0) - 180.0
            ii = np.where(np.abs(dl) <= A.crop_deg / max(math.cos(la * DEG), 0.25))[0]
            sub = np.ascontiguousarray(arr[:, jj[:, None], ii[None, :]])
            sla, slo = LATV[jj], LONV[ii]
            # The contrast is computed from the INJECTED sub-volume, not the original,
            # or the injected conduit is invisible to a channel defined as departure from
            # a local background - which is the channel every model is now frozen on.
            # classify.py has always done this; these two injection scripts passed None,
            # harmless while the frozen channel was the raw anomaly and fatal once it was
            # not. The sigma is the 800 km used everywhere else in the workflow.
            c2 = (contrast_field(sub, sla, slo, 800.0)
                  if str(c.channel) != 'anom' else None)
            C = cost_field(sub, c2, z, sla, slo, s=float(c.s), z_target=ztar,
                           channel=str(c.channel), h_max_km=hm, lateral=lam,
                           ncell=ncell)
            # No descent cache here: every site is traced through its own cropped
            # cost field, so there is nothing to share between sites. The trace now
            # costs about as much as the cropped field it walks on.
            pz, pla, plo = trace_path(C, sub, c2, z, sla, slo, la, lo,
                                      s=float(c.s), radius_deg=float(c.radius),
                                      seed_depth=SEED, channel=str(c.channel),
                                      h_max_km=hm, lateral=lam, ncell=ncell)
            pz, pla, plo = (np.asarray(pz, float), np.asarray(pla, float),
                            np.asarray(plo, float))
            o = np.argsort(pz); pz, pla, plo = pz[o], pla[o], plo[o]
            m = (pz >= SEED) & (pz <= ztar)
            off = float(gc_km(pla[m][0], plo[m][0], pla[m][-1], plo[m][-1]))
            vals = [float(sub[int(np.argmin(np.abs(z - q))),
                              int(np.argmin(np.abs(sla - u))),
                              int(np.argmin(np.abs(slo - v)))])
                    for q, u, v in zip(pz, pla, plo)]
            rows.append(dict(lateral=lam, ncell=ncell, kind=kind, site=name,
                             lat=la, lon=lo, offset_km=off,
                             mean_anom=float(np.mean(vals)),
                             edge_deg=min(float(np.min(np.abs(pla - sla[0]))),
                                          float(np.min(np.abs(pla - sla[-1]))))))
            pd.DataFrame(rows)[COLS].to_csv(OUT, index=False)

d = pd.DataFrame(rows)[COLS] if rows else pd.DataFrame(columns=COLS)
want = len(JOBS) * len(SET)
print(f'\n{len(d)} of {want} traces')
if len(d) < want:
    print('run again to continue\n')

print(f'\n{"lateral":>7s} {"ncell":>5s} {"hotspot":>9s} {"ambient":>9s} {"gain":>7s} '
      f'{"P":>7s} {"n h/a":>7s}   {"offset hot":>10s} {"offset amb":>10s}')
tab = []
_dropped = []
for (lam, nc), g in d.groupby(['lateral', 'ncell']):
    h = g[g.kind == 'hotspot']; a = g[g.kind == 'ambient']
    # A trace that returned no usable anomaly is a MISSING measurement, not a
    # value. The median already skipped it while the Mann-Whitney did not, so one
    # failed ambient trace turned P into nan and the setting was dropped from the
    # admitted list without a word. At pilot size that silently disqualified
    # lateral 1.00, which on the remaining seven ambient traces sits at P = 0.02:
    # a failed trace was deciding the move set. Drop them from both, and say so.
    n_h0, n_a0 = len(h), len(a)
    h = h[np.isfinite(h.mean_anom)]
    a = a[np.isfinite(a.mean_anom)]
    if n_h0 - len(h) or n_a0 - len(a):
        _dropped.append((lam, nc, n_h0 - len(h), n_a0 - len(a)))
    if len(h) < 5 or len(a) < 5:
        print(f'{lam:7.2f} {nc:5d}   too few usable traces '
              f'({len(h)} hotspot, {len(a)} ambient); not tested')
        continue
    gain = float(h.mean_anom.median() - a.mean_anom.median())
    P = float(mannwhitneyu(h.mean_anom, a.mean_anom, alternative='less').pvalue)
    tab.append((lam, nc, gain, P))
    print(f'{lam:7.2f} {nc:5d} {h.mean_anom.median():8.3f}% {a.mean_anom.median():8.3f}% '
          f'{gain:6.3f}% {P:7.4f} {len(h):3d}/{len(a):<3d}   '
          f'{h.offset_km.median():7.0f} km {a.offset_km.median():7.0f} km')
if _dropped:
    print('\ntraces with no usable anomaly, excluded from both the median and the test:')
    for lam, nc, dh, da in _dropped:
        print(f'  lateral {lam:.2f} ncell {nc}: {dh} hotspot, {da} ambient')
    print('  A setting is compared on fewer traces than another only where this '
          'says so.')
adm = [t for t in tab if t[3] < 0.05]
if not adm:
    print('\nNo setting admitted: hotspot paths are not slower than ambient at P < 0.05.')
elif len(d) == want:
    best = min(adm, key=lambda t: t[2])
    tol = best[2] * 0.90
    ok = [t for t in adm if t[2] <= tol]
    pick = sorted(ok, key=lambda t: (-t[0], t[1]))[0]
    print(f'\nbest gain {best[2]:.3f}% at lateral {best[0]:.2f} ncell {best[1]}; '
          f'within 10 per cent the most conservative is '
          f'lateral {pick[0]:.2f} ncell {pick[1]}')
    print(f'CHOSEN lateral = {pick[0]:.2f}, ncell = {pick[1]}')
else:
    print('\nincomplete; no choice made')
