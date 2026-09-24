#!/usr/bin/env python3
"""Does a traced path reproduce the tilt of a conduit whose tilt is known?

The injection calibration measured whether corridor WIDTH tracks injected width.
It never measured whether the recovered PATH tracks injected tilt, and tilt is what
the paper's central result is about. Cross-sections beneath Afar and Marion show the
gap: the conduit leans strongly, the traced path abandons it near the transition zone
and descends through unrelated moderately slow material instead.

That behaviour follows from the cost. A step costs its length times exp(s*a), so
following a conduit leaning at t from vertical costs 1/cos(t) times a vertical descent
through the same material, and the detour is paid for only if the conduit is slower
than its surroundings by more than ln(1/cos t)/s. At the calibrated s = 0.4 that is
1.7 per cent at 45 degrees and 3.5 per cent at 60 degrees. Few conduits are that
strong, so the search lets go of them. Raising s does not fix it, because a value high
enough to buy the detour also makes every weak ambient anomaly attractive.

path_cost.py now carries a weight `lateral` on horizontal travel, with 1.0 reproducing
the old behaviour exactly. This script calibrates it.

PRE-REGISTERED, FIXED BEFORE THE FIRST RUN.

  Injected geometry: conduits of the sweep radius and amplitude used by the width
  calibration, at lateral offsets of 0, 5, 10, 15 and 20 degrees of arc measured at the
  base, profile 'uniform' so the lean is constant with depth and the injected offset is
  unambiguous.

  Measure: the horizontal distance in km between the path at the seed depth and the
  path at the target depth, against the injected offset over the same interval.

  Selection rule: among the lateral weights swept, choose the largest weight whose
  median absolute error in recovered offset, taken over all injected tilts above zero,
  is within 10 per cent of the best weight's. Largest, not best, because a smaller
  weight buys tilt recovery by making horizontal travel cheap, and the tie-break should
  favour the most conservative value that is not measurably worse.

  Veto, applied before the rule: a weight is disqualified if the median recovered
  offset at zero injected tilt exceeds 200 km. A search that wanders when nothing was
  injected is not measuring tilt, and no improvement at high tilt can compensate.

  If no weight satisfies the veto, the finding is that the method cannot be repaired by
  this parameter and the tilt results must be withdrawn rather than rescaled.

  python3 tilt_recovery.py --file <model.nc> --pilot
  python3 tilt_recovery.py --file <model.nc>
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np, pandas as pd
import provenance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from inject import inject_tilted, gc_km
from path_cost import cost_field, trace_path, channel_field
from contrast import contrast_field
import path_config

R_E, DEG, SEED_DEPTH = 6371.0, math.pi / 180.0, 200.0

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--sites', type=int, default=6)
ap.add_argument('--seed', type=int, default=51)
ap.add_argument('--tilts', default='0,5,10,15,20')
ap.add_argument('--lateral', default='1.0,0.8,0.6,0.5,0.4,0.3')
ap.add_argument('--radius', type=float, default=400.0)
ap.add_argument('--amp', type=float, default=-1.5)
ap.add_argument('--azimuth', type=float, default=45.0)
ap.add_argument('--ncell', type=int, default=1,
                help='lateral cells a single slanted move may span')
ap.add_argument('--crop-deg', type=float, default=28.0, dest='crop_deg',
                help='half-width of the local box the cost field is built on. The '
                     'injection is local and a descending path cannot leave the box '
                     'within the depth range, so building the field globally repeats '
                     'the same work for every combination and made the sweep a day '
                     'long. Paths are checked against the crop edge and reported if '
                     'they approach it.')
ap.add_argument('--no-resume', dest='resume', action='store_false',
                help='start over rather than continuing an interrupted sweep')
ap.set_defaults(resume=True)
ap.add_argument('--pilot', action='store_true',
                help='2 sites and 3 tilts, to check the machinery before the full run')
ap.add_argument('--dir', default='out')
ap.add_argument('--levels', default='100',
                help='continuity levels to sweep, as per-shell percentiles of the '
                     'cost channel. 100 forbids nothing and is the present search '
                     'exactly, so the control sits inside the ladder. The rule that '
                     'selects among them is fixed in CONTINUITY_PREREGISTRATION.md.')
ap.add_argument('--save-paths', default=None, dest='save_paths',
                help='also write the full traced geometry to this file inside --dir, '
                     'in the same shape as conduit_paths_all_<tag>.json. The ambient '
                     'run is the null for track_support.py, and a null made of summary '
                     'numbers cannot be measured the same way the hotspots are')
ap.add_argument('--s', type=float, default=None,
                help='run under this s instead of the frozen one, for this run '
                     'only. The frozen file is never written. Use it to ask '
                     'whether a candidate s still recovers a conduit of known '
                     'tilt, which is the cost of lowering it.')
ap.add_argument('--out', default=None,
                help='output file name inside --dir. The ambient control is read by '
                     'ambient_null.py as ambient_null_<tag>.csv, so an ambient run '
                     'writes there rather than to the sweep table')
A = ap.parse_args()

TILTS = [float(x) for x in A.tilts.split(',')]
LATERAL = [float(x) for x in A.lateral.split(',')]
if A.pilot:
    TILTS = TILTS[:3]
    A.sites = 2

# The configuration is READ from out/path_config_<tag>.json, never re-derived.
# Re-deriving it as the median of the retained set makes it depend on which
# configurations happened to clear a 75 per cent floor on eighteen injections,
# and after the move-set retrace that median moved from s=0.4 z=2700 r=3.0 to
# s=0.3 z=2800 r=2.0. A null traced under one configuration and hotspots under
# another are two different searches and must not be compared.
c = path_config.load(A.tag, A.dir)
if A.s is not None:
    c = c.copy()
    c['s'] = float(A.s)
hm = getattr(c, 'h_max', None)
hm = None if hm is None or (isinstance(hm, float) and hm != hm) else float(hm)
print(f'calibrated configuration: s={c.s} z_target={c.z_target:.0f} '
      f'channel={c.channel} radius={c.radius} h_max={hm}'
      + ('   [s OVERRIDDEN for this run; the frozen file is unchanged]'
         if A.s is not None else ''), flush=True)

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
arr = np.asarray(arr, dtype=np.float64)
z = np.asarray(depth, float)
LATV, LONV = np.asarray(lat, float), np.asarray(lon, float)
print(f'model loaded: {arr.shape}', flush=True)

hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180'])
rng = np.random.default_rng(A.seed)
sites = []
while len(sites) < A.sites:
    la = float(rng.uniform(-55, 55))
    lo = float(rng.uniform(-180, 180))
    d = gc_km(la, lo, hs.lat.to_numpy(float), hs.lon_180.to_numpy(float))
    if np.min(d) > 1500.0:            # clear of any real hotspot conduit
        sites.append((la, lo))
print(f'{len(sites)} ambient injection sites, all beyond 1500 km of a hotspot',
      flush=True)


def offset_km(zz, la, lo, z0, z1):
    """Horizontal distance between the path at z0 and at z1."""
    o = np.argsort(zz); zz, la, lo = zz[o], la[o], lo[o]
    m = (zz >= z0) & (zz <= z1)
    if m.sum() < 2:
        return float('nan')
    return float(gc_km(la[m][0], lo[m][0], la[m][-1], lo[m][-1]))


def crop(la, lo):
    """Indices of a local box around a site."""
    jm = np.abs(LATV - la) <= A.crop_deg
    dl = ((LONV - lo + 180.0) % 360.0) - 180.0
    im = np.abs(dl) <= A.crop_deg / max(math.cos(la * DEG), 0.25)
    return np.where(jm)[0], np.where(im)[0]


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), A.dir,
                   A.out or f'tilt_recovery_{A.tag}.csv')
# ncell belongs here beside lateral: the two together are the move set, and a file
# recording only one of them cannot say which search produced it.
COLS = ['level', 'lateral', 'ncell', 'tilt_deg', 'lat', 'lon', 'injected_km', 'recovered_km',
        'error_km', 'edge_deg', 'lean_deep', 'offset_660_1500', 'mean_anom']

# Rows are appended as they are measured and a combination already on disk is skipped.
# The sweep is long enough that something will interrupt it, and starting again from
# the beginning each time is how a calibration never gets finished.
done = set()
rows = []
if A.resume and os.path.exists(OUT):
    _d = pd.read_csv(OUT)
    for _, r in _d.iterrows():
        done.add((round(float(r.get('level', 100.0)), 3),
                  round(float(r.lateral), 3), round(float(r.tilt_deg), 3),
                  round(float(r.lat), 4)))
        _r = {k: r[k] for k in COLS if k in _d.columns}
        _r.setdefault('ncell', 1)               # files written before ncell existed
        _r.setdefault('level', 100.0)           # and before the continuity axis existed
        _r.setdefault('offset_660_1500', float('nan'))
        rows.append(_r)
    print(f'resuming: {len(done)} combinations already measured', flush=True)


def flush(rs):
    _f = pd.DataFrame(rs)
    for _c in COLS:
        if _c not in _f.columns:
            _f[_c] = float('nan')
    _f[COLS].to_csv(OUT, index=False)
    provenance.stamp(OUT, config=c, lateral=A.lateral, ncell=A.ncell,
                     amp=A.amp, tilts=A.tilts, sites=A.sites, seed=A.seed,
                     radius_km=A.radius, s_override=A.s,
                     injects_a_conduit=bool(float(A.amp) != 0.0),
                     inputs=[A.file, A.hotspots])


_paths = {}
ztar = float(c.z_target)
LEVELS = [float(x) for x in A.levels.split(',') if x.strip()]
for lev in LEVELS:
  for lam in LATERAL:
    for tilt in TILTS:
        for (la, lo) in sites:
            if (round(lev, 3), round(lam, 3), round(tilt, 3), round(la, 4)) in done:
                continue
            jj, ii = crop(la, lo)
            sub = np.ascontiguousarray(arr[:, jj[:, None], ii[None, :]])
            sla, slo = LATV[jj], LONV[ii]
            a2 = inject_tilted(sub, z, sla, slo, la, lo, A.radius, A.amp, tilt,
                               azimuth=A.azimuth, profile='uniform')
            # The contrast is computed from the INJECTED sub-volume, not the original,
            # or the injected conduit is invisible to a channel defined as departure from
            # a local background - which is the channel every model is now frozen on.
            # classify.py has always done this; these two injection scripts passed None,
            # harmless while the frozen channel was the raw anomaly and fatal once it was
            # not. The sigma is the 800 km used everywhere else in the workflow.
            c2 = (contrast_field(a2, sla, slo, 800.0)
                  if str(c.channel) != 'anom' else None)
            # The continuity constraint, if any, is built from the INJECTED
            # sub-volume for the same reason the contrast is: a level defined on
            # the original field would forbid the conduit that was just put there.
            if lev >= 100.0:
                ban = None
            else:
                ach = channel_field(a2, c2, str(c.channel))
                thr = np.nanpercentile(ach.reshape(ach.shape[0], -1), lev, axis=1)
                ban = ach > thr[:, None, None]
            C = cost_field(a2, c2, z, sla, slo, s=float(c.s), z_target=ztar,
                           channel=str(c.channel), h_max_km=hm, lateral=lam,
                           ncell=A.ncell, forbid=ban)
            _tr = trace_path(C, a2, c2, z, sla, slo, la, lo,
                             s=float(c.s), radius_deg=float(c.radius),
                             seed_depth=SEED_DEPTH, channel=str(c.channel),
                             h_max_km=hm, lateral=lam, ncell=A.ncell)
            if _tr is None:
                # No admissible route from this site at this continuity level. That
                # is a measurement - the level has stranded an injected conduit -
                # and it belongs in the file as one, not as a crash. It carries no
                # offset, so it is excluded from the error statistic and counted by
                # the coverage veto instead.
                print(f'    no route at level p{lev:g}, tilt {tilt:.0f}', flush=True)
                rows.append(dict(level=float(lev), lateral=lam,
                                 ncell=int(A.ncell), tilt_deg=tilt, lat=la, lon=lo,
                                 injected_km=float('nan'), recovered_km=float('nan'),
                                 error_km=float('nan'), edge_deg=float('nan'),
                                 lean_deep=float('nan'),
                                 offset_660_1500=float('nan'),
                                 mean_anom=float('nan')))
                flush(rows)
                continue
            pz, pla, plo = _tr
            _pl = np.asarray(pla, float)
            edge = min(float(np.min(np.abs(_pl - sla[0]))),
                       float(np.min(np.abs(_pl - sla[-1]))))
            rec = offset_km(np.asarray(pz, float), np.asarray(pla, float),
                            np.asarray(plo, float), SEED_DEPTH, ztar)
            # injected offset over the same depth interval, profile 'uniform'
            f0 = (SEED_DEPTH - 100.0) / (2800.0 - 100.0)
            f1 = (ztar - 100.0) / (2800.0 - 100.0)
            inj = float(gc_km(la + tilt * f0 * math.cos(A.azimuth * DEG),
                              lo + tilt * f0 * math.sin(A.azimuth * DEG)
                              / max(math.cos(la * DEG), 0.2),
                              la + tilt * f1 * math.cos(A.azimuth * DEG),
                              lo + tilt * f1 * math.sin(A.azimuth * DEG)
                              / max(math.cos(la * DEG), 0.2)))
            _vals = [float(a2[int(np.argmin(np.abs(z - _q))),
                              int(np.argmin(np.abs(sla - _u))),
                              int(np.argmin(np.abs(slo - _v)))])
                     for _q, _u, _v in zip(np.asarray(pz, float),
                                           np.asarray(pla, float),
                                           np.asarray(plo, float))]
            _anom = float(np.mean(_vals)) if _vals else float('nan')
            _z = np.asarray(pz, float); _o = np.argsort(_z)
            _z, _a2, _b2 = _z[_o], np.asarray(pla, float)[_o], np.asarray(plo, float)[_o]
            _m = (_z >= 660.0) & (_z <= 1500.0)
            if _m.sum() >= 4:
                _e = R_E * DEG * (((_b2[_m] - _b2[_m][0] + 180) % 360) - 180) \
                    * math.cos(_a2[_m][0] * DEG)
                _n = R_E * DEG * (_a2[_m] - _a2[_m][0])
                _lean = (math.degrees(math.atan2(_e[-1], _n[-1])) % 360.0
                         if math.hypot(_e[-1], _n[-1]) >= 20 else float('nan'))
                # The offset accumulated over the SAME window the lean is measured
                # in. Section 3.1 calls Louisville's 93 km over 660-1500 km small,
                # and small is only meaningful against paths with no conduit in
                # them, which is what these are. Without this the file records the
                # offset over 200-2700 km only and that comparison cannot be made.
                _off_w = float(gc_km(_a2[_m][0], _b2[_m][0], _a2[_m][-1], _b2[_m][-1]))
            else:
                _lean = float('nan')
                _off_w = float('nan')
            if A.save_paths:
                _k = f'{la:.4f},{lo:.4f}' if (tilt == 0 and A.amp == 0) else \
                     f'{la:.4f},{lo:.4f}|lat{lam:g}|t{tilt:g}'
                _paths[_k] = dict(lat=np.asarray(pla, float).tolist(),
                                  lon=np.asarray(plo, float).tolist(),
                                  depth=np.asarray(pz, float).tolist(),
                                  site_lat=float(la), site_lon=float(lo),
                                  lateral=float(lam), ncell=int(A.ncell),
                                  tilt_deg=float(tilt), amp=float(A.amp))
            rows.append(dict(level=float(lev), lateral=lam,
                             ncell=int(A.ncell), tilt_deg=tilt,
                             lat=la, lon=lo,
                             injected_km=inj, recovered_km=rec,
                             error_km=rec - inj, edge_deg=edge, lean_deep=_lean,
                             offset_660_1500=_off_w, mean_anom=_anom))
            if edge < 3.0:
                print(f'    path came within {edge:.1f} deg of the box edge; '
                      'widen --crop-deg before trusting this row', flush=True)
            flush(rows)
            print(f'  lateral {lam:.2f}  tilt {tilt:4.1f}  injected {inj:7.0f} km  '
                  f'recovered {rec:7.0f} km', flush=True)

d = pd.DataFrame(rows)
if not len(d) or not d.recovered_km.notna().any():
    raise SystemExit('no paths recovered; refusing to write an empty calibration')
flush(rows)
out = OUT
print(f'\nwrote {out}\n')
_want = len(LATERAL) * len(TILTS) * len(sites)
if len(d) < _want:
    print(f'{len(d)} of {_want} combinations measured; run again to continue\n')
if 'level' in d.columns and d.level.nunique() > 1:
    # The continuity axis, under the rule fixed in CONTINUITY_PREREGISTRATION.md:
    # lowest median absolute offset error among the levels passing the drift veto,
    # and the level changes from 100 only on a margin of at least 10 per cent.
    print(f'\n{"level":>7s} {"zero-tilt drift":>16s} {"median |error|":>15s} '
          f'{"verdict":>10s}')
    tab = []
    for lv, g in d.groupby('level'):
        dr = float(g[g.tilt_deg == 0].recovered_km.median())
        er = float(g[g.tilt_deg > 0].error_km.abs().median())
        nstr = int(g.recovered_km.isna().sum())
        tab.append((lv, dr, er, dr <= 200.0 and nstr == 0, nstr))
    for lv, dr, er, ok, nstr in sorted(tab, reverse=True):
        why = '' if ok else ('  stranded' if nstr else '  vetoed')
        print(f'{lv:7g} {dr:13.0f} km {er:12.0f} km {why:>10s}'
              + (f'   {nstr} of {len(d[d.level == lv])} injections had no route'
                 if nstr else ''))
    okt = [t for t in tab if t[3]]
    ctl = [t for t in tab if abs(t[0] - 100.0) < 1e-9]
    if not okt or not ctl:
        print('\nNo admissible level, or the control was not swept; by the rule '
              'fixed in advance the level does not change.')
    else:
        bst = min(okt, key=lambda t: t[2])
        e100 = ctl[0][2]
        if bst[2] <= 0.90 * e100 and abs(bst[0] - 100.0) > 1e-9:
            print(f'\nbest median error {bst[2]:.0f} km at level p{bst[0]:g} against '
                  f'{e100:.0f} km at p100, an improvement of '
                  f'{100 * (1 - bst[2] / e100):.0f} per cent')
            print(f'CHOSEN level = p{bst[0]:g}  (drift and detection vetoes still to '
                  'be applied from frontier.py and classify.py)')
        else:
            print(f'\nbest median error {bst[2]:.0f} km at level p{bst[0]:g} against '
                  f'{e100:.0f} km at p100: inside the 10 per cent margin')
            print('CHOSEN level = p100, no change')

print(f'{"lateral":>8s} {"zero-tilt drift":>16s} {"median |error|":>15s} {"verdict":>10s}')
best, table = None, []
for lam, g in d.groupby('lateral'):
    drift = float(g[g.tilt_deg == 0].recovered_km.median())
    err = float(g[g.tilt_deg > 0].error_km.abs().median())
    ok = drift <= 200.0
    table.append((lam, drift, err, ok))
    if ok and (best is None or err < best[2]):
        best = (lam, drift, err)
for lam, drift, err, ok in sorted(table, reverse=True):
    print(f'{lam:8.2f} {drift:13.0f} km {err:12.0f} km '
          f'{"" if ok else "  vetoed":>10s}')
if best is None:
    print('\nNo lateral weight passes the zero-tilt veto. By the rule fixed in advance '
          'the tilt results are not repairable by this parameter.')
else:
    tol = best[2] * 1.10
    chosen = max(lam for lam, drift, err, ok in table if ok and err <= tol)
    print(f'\nbest median error {best[2]:.0f} km at lateral {best[0]:.2f}; '
          f'within 10 per cent, the largest qualifying weight is {chosen:.2f}')
    print(f'CHOSEN lateral = {chosen:.2f}')

if A.save_paths:
    import json
    _pp = os.path.join(os.path.dirname(os.path.abspath(__file__)), A.dir, A.save_paths)
    json.dump(_paths, open(_pp, 'w'))
    print(f'wrote {A.save_paths} with {len(_paths)} traced paths')
