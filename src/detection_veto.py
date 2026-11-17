#!/usr/bin/env python3
"""The detection veto for the continuity level: does the constraint cost detection?

CONTINUITY_PREREGISTRATION.md fixes three vetoes on the continuity level. Drift and
coverage have been measured and p35 passes both. The third is detection - injection
recovery at the frozen configuration must not fall below the 0.75 floor that defines
the retained set - and classify.py, which measures it, has no level axis.

This is that measurement and nothing else. It reuses classify.py's primitives and
constants without modification - the same four injection sites, the same amplitudes
and deflection styles, the same 300 null sites drawn from the same seed, the same 5th
percentile threshold and seed depth - evaluated at the FROZEN configuration only,
with the continuity level as the one axis that varies. That is the single-axis screen
the pre-registration asks for, and it is how the channel was decided.

p100 IS THE CONTROL. At p100 nothing is forbidden, so this must reproduce classify's
own detect_cal for the frozen configuration exactly - 18 of 18 for RevealLO. If it does
not, the script is not measuring what classify measures and the p35 number means
nothing; that comparison is printed and a mismatch stops the run.

Only the calibration amplitudes are injected, because detect_cal is computed from
those alone; the weaker ones feed the sensitivity curves, which the veto does not use.

It writes out/detection_veto_<tag>.csv and nothing that any other script reads, so it
cannot disturb the published classification. It resumes: a finished injection is not
repeated.

  python3 detection_veto.py --file <model.nc> --tag RevealLO --levels 100,35 --jobs 4
"""
from __future__ import annotations
import argparse, itertools, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import path_config
import classify as K
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from contrast import contrast_field
from path_cost import cost_field, site_cost, channel_field
from inject import inject_tilted

_W = {}


def _ban(anom, con, channel, level):
    if level >= 100.0:
        return None
    a = channel_field(anom, con, channel)
    thr = np.nanpercentile(a.reshape(a.shape[0], -1), level, axis=1)
    return a > thr[:, None, None]


def _field(anom, con, level):
    c = _W['cfg']
    return cost_field(anom, con, _W['depth'], _W['lat'], _W['lon'], s=float(c.s),
                      z_target=float(c.z_target), n_relax=K.N_RELAX,
                      channel=str(c.channel), h_max_km=_W['hm'],
                      lateral=_W['lateral'], ncell=_W['ncell'],
                      forbid=_ban(anom, con, str(c.channel), level))


def _one(job):
    level, idx = job
    (nm, la, lo), amp, (tl, prof) = _W['plan'][idx]
    a2 = inject_tilted(_W['arr'], _W['depth'], _W['lat'], _W['lon'], la, lo,
                       K.SYN_RADIUS, amp, tl,
                       profile=('uniform' if prof == 'uniform' else 'transition'))
    c2 = contrast_field(a2, _W['lat'], _W['lon'], K.SIGMA)
    C = _field(a2, c2, level)
    c = float(site_cost(C, _W['depth'], _W['lat'], _W['lon'], la, lo,
                        float(_W['cfg'].radius), K.SEED_DEPTH))
    del C, a2, c2
    return level, idx, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
    ap.add_argument('--lateral', type=float, default=0.60)
    ap.add_argument('--ncell', type=int, default=2)
    ap.add_argument('--levels', default='100,35')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--max-jobs', type=int, default=0, dest='max_jobs',
                    help='stop after this many injections and exit cleanly, for '
                         'machines that cap a single process; it resumes')
    A = ap.parse_args()

    cfg = path_config.load(A.tag, A.dir)
    hm = getattr(cfg, 'h_max', None)
    hm = None if hm is None or (isinstance(hm, float) and hm != hm) else float(hm)
    depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                        depth_max=A.depth_max, every=A.every)
    lon, arr = dedupe_lon(lon, arr)
    con = contrast_field(arr, lat, lon, K.SIGMA)
    levels = [float(x) for x in A.levels.split(',') if x.strip()]
    _W.update(arr=arr, depth=depth, lat=lat, lon=lon, cfg=cfg, hm=hm,
              lateral=float(A.lateral), ncell=int(A.ncell))

    plan = [p for p in itertools.product(K.SITES, K.AMPS, K.SYN_GEOM)
            if p[1] <= K.CAL_AMP]
    _W['plan'] = plan

    # null thresholds and the pre-injection status of the four sites, per level,
    # with the same null draw classify.py makes
    rng = np.random.default_rng(42)
    nlon = 360.0 * rng.random(K.N_NULL) - 180.0
    nlat = np.degrees(np.arcsin(2 * rng.random(K.N_NULL) - 1))
    OUT = os.path.join(A.dir, f'detection_veto_{A.tag}.csv')
    prev = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    thr, pre = {}, {}
    for L in levels:
        rec = prev[(prev.get('kind', pd.Series(dtype=str)) == 'threshold')
                   & (prev.get('level', pd.Series(dtype=float)) == L)] \
            if len(prev) else pd.DataFrame()
        if len(rec):
            thr[L] = float(rec.value.iloc[0])
            pre[L] = [float(x) for x in str(rec.pre.iloc[0]).split('|')]
            print(f'  p{L:g}: threshold {thr[L]:.0f} (resumed)', flush=True)
            continue
        C = _field(arr, con, L)
        nul = np.array([site_cost(C, depth, lat, lon, a_, o_, float(cfg.radius),
                                  K.SEED_DEPTH) for a_, o_ in zip(nlat, nlon)], float)
        thr[L] = float(np.nanpercentile(nul, K.NULL_PCT))
        pre[L] = [float(site_cost(C, depth, lat, lon, a_, o_, float(cfg.radius),
                                  K.SEED_DEPTH)) for _, a_, o_ in K.SITES]
        del C
        prev = pd.concat([prev, pd.DataFrame([dict(
            kind='threshold', level=L, value=thr[L],
            pre='|'.join(f'{x:.6g}' for x in pre[L]))])], ignore_index=True)
        prev.to_csv(OUT + '.part', index=False); os.replace(OUT + '.part', OUT)
        print(f'  p{L:g}: null p{K.NULL_PCT:g} threshold {thr[L]:.0f}', flush=True)

    done = set()
    if len(prev) and 'idx' in prev:
        q = prev[prev.kind == 'injection']
        done = {(float(l), int(i)) for l, i in zip(q.level, q.idx)}
    jobs = [(L, i) for L in levels for i in range(len(plan)) if (L, i) not in done]
    if A.max_jobs:
        jobs = jobs[:A.max_jobs]
    print(f'{len(jobs)} injection(s) to run, {len(done)} already done', flush=True)

    def _record(level, idx, cst):
        nonlocal prev
        (nm, la, lo), amp, (tl, prof) = plan[idx]
        si = [s_[0] for s_ in K.SITES].index(nm)
        informative = not (pre[level][si] < thr[level])
        hit = (bool(np.isfinite(cst) and cst < thr[level]) if informative else None)
        prev = pd.concat([prev, pd.DataFrame([dict(
            kind='injection', level=level, idx=idx, site=nm, amp=amp,
            profile=prof, cost=cst, hit=hit)])], ignore_index=True)
        prev.to_csv(OUT + '.part', index=False); os.replace(OUT + '.part', OUT)
        print(f'  p{level:g} {nm:26s} amp={amp:+.2f} {prof:10s} '
              f'{"uninformative" if hit is None else ("HIT" if hit else "miss")}',
              flush=True)

    if A.jobs > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(A.jobs) as pool:
            for level, idx, cst in pool.imap_unordered(_one, jobs):
                _record(level, idx, cst)
    else:
        for j in jobs:
            _record(*_one(j))

    q = prev[prev.kind == 'injection']
    print(f'\n{"level":>6s} {"detect_cal":>11s} {"n_cal":>6s}  against the 0.75 floor')
    res = {}
    for L in levels:
        h = [x for x in q[q.level == L].hit if x is not None and x == x]
        n_all = int((q.level == L).sum())
        if n_all < len(plan):
            print(f'{L:6g}  incomplete: {n_all} of {len(plan)} injections')
            continue
        h = [bool(x) for x in h]
        res[L] = (float(np.mean(h)) if h else float('nan'), len(h))
        print(f'{L:6g} {res[L][0]:11.3f} {res[L][1]:6d}  '
              f'{"PASSES" if res[L][0] >= K.MIN_DETECT else "FAILS"}')

    # the control: p100 must reproduce classify's own number for the frozen config
    if 100.0 in res:
        det = pd.read_csv(os.path.join(A.dir, f'detection_{A.tag}.csv'))
        m = det[(det.s == float(cfg.s)) & (det.z_target == float(cfg.z_target))
                & (det.channel == str(cfg.channel)) & (det.radius == float(cfg.radius))]
        if hm is not None:
            m = m[m.h_max == hm]
        if len(m):
            pub = float(m.detect_cal.iloc[0]); npub = int(m.n_cal.iloc[0])
            ok = abs(pub - res[100.0][0]) < 1e-9 and npub == res[100.0][1]
            print(f'\ncontrol: p100 gives {res[100.0][0]:.3f} over {res[100.0][1]}; '
                  f'classify.py gives {pub:.3f} over {npub} - '
                  f'{"REPRODUCED" if ok else "MISMATCH"}')
            if not ok:
                raise SystemExit('the control does not reproduce classify.py, so the '
                                 'other levels cannot be read')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
