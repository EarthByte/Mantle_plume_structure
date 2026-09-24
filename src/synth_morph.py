#!/usr/bin/env python3
"""What the root test, the ensemble and the corridor do to structures that are
coherent but are not pipes.

The calibration behind the published classification injects a vertical conduit,
one deflected across the transition zone and one leaning at a constant angle. All
three are single continuous tubes, so the detection rate it establishes is the
detection rate for a tube. If a long-lived plume can be expressed as a curve, a
branch, a sheet or a segmented column, then the classification's failures may be
failures of the geometry model rather than absences of structure - and that can
be measured rather than argued, by injecting those shapes and asking which of
them the test sees.

The same three quantities are recorded for every injected shape:

  DETECTED   does the site fall below the fifth percentile of the null, under
             this configuration, as the classification requires
  RECOVERED  does the traced path follow the injected axis, measured as the
             median distance between them over the depth range of the conduit
  CORRIDOR   what the near-optimal corridor says the shape is - its width, its
             anisotropy and the number of components it separates - so that a
             sheet injected as a sheet can be checked to read as a sheet

The third is what makes the analysis falsifiable. A corridor description that
cannot tell an injected sheet from an injected tube is not a description of the
mantle.

A NECESSARY LIMITATION, STATED HERE AND CARRIED INTO THE PAPER

Injecting a structure into an already inverted model tests this analysis, not the
seismic inversion. These numbers say what would be recovered if a structure of
that amplitude and scale were present in the model. They say nothing about
whether the seismic data would have placed it there, and no statement about the
resolving power of the tomography may be built on them.
"""
from __future__ import annotations

import argparse, itertools, os, sys, warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon, check_shells
from contrast import contrast_field
from path_cost import cost_field, site_cost
from corridor import forward_field, excess_cost, trace_consistent
import morphology as M
import synth_shapes as S
from morph_run import retained, stratify, _hmax, box_indices

warnings.filterwarnings('ignore')

SIGMA, N_RELAX, N_NULL, NULL_PCT, SEED_DEPTH = 800.0, 6, 300, 5.0, 200.0
SITES = [('open ocean', 20.0, -156.0), ('province interior', -25.0, -165.0),
         ('province margin', -21.0, 56.0), ('control', 35.0, -40.0)]
AMPS = (-0.75, -1.0, -1.5)
CAL_AMP = -1.0
TAU = 0.02

_W = {}


def _case(q):
    arr, depth, lat, lon = _W['arr'], _W['depth'], _W['lat'], _W['lon']
    (nm, la, lo), amp, shape, radius = _W['plan'][q]
    a2 = S.inject(arr, depth, lat, lon, la, lo, shape, amp,
                  radius_km=radius, offset_deg=_W['offset'])
    c2 = contrast_field(a2, lat, lon, SIGMA)
    axis = S.limbs(shape, np.asarray(depth, float), la, lo,
                   offset_deg=_W['offset'], radius_km=radius)
    rows = []
    for ci, cfg in enumerate(_W['cfgs']):
        s, zt, ch, hm, rd = (cfg['s'], cfg['z_target'], cfg['channel'],
                             cfg['h_max'], cfg['radius'])
        k = (s, zt, ch, hm, rd)
        if _W['pre'][k][[x[0] for x in SITES].index(nm)] < _W['thresh'][k]:
            continue                       # already rooted here: uninformative
        C = cost_field(a2, c2, depth, lat, lon, s=s, z_target=zt,
                       n_relax=N_RELAX, channel=ch, h_max_km=hm)
        c = site_cost(C, depth, lat, lon, la, lo, rd, SEED_DEPTH)
        r = dict(site=nm, amp=amp, shape=shape, conduit_radius_km=radius,
                 config=ci, s=s, z_target=zt,
                 channel=ch, h_max=hm, radius=rd, cost=float(c),
                 threshold=float(_W['thresh'][k]),
                 detected=bool(np.isfinite(c) and c < _W['thresh'][k]))
        tr = trace_consistent(C, a2, c2, depth, lat, lon, la, lo, s=s,
                              radius_deg=rd, n_relax=N_RELAX, channel=ch,
                              h_max_km=hm)
        if tr is not None:
            pz, pla, plo = tr
            dev = []
            for zz, aa, oo in zip(pz, pla, plo):
                kk = int(np.argmin(np.abs(np.asarray(depth, float) - zz)))
                dev.append(min(float(M.gc_km(float(l[0][kk]), float(l[1][kk]),
                                             aa, oo)) for l in axis))
            r['axis_distance_km'] = float(np.median(dev))
            r['recovered'] = bool(r['axis_distance_km'] <= radius)
        if _W['corridor'] and abs(amp - CAL_AMP) < 1e-9:
            D = forward_field(a2, c2, depth, lat, lon, la, lo, s=s, z_target=zt,
                              radius_deg=rd, seed_depth=SEED_DEPTH,
                              n_relax=N_RELAX, channel=ch, h_max_km=hm)
            z = np.asarray(depth, float)
            k0 = int(np.where(z >= zt)[0][0]); ks = int(np.argmin(np.abs(z - SEED_DEPTH)))
            dC, cb = excess_cost(D, C, ks, k0)
            W = (np.isfinite(dC) & (dC <= TAU * cb)).astype(np.float32)
            jb, ib = box_indices(lat, lon, la, lo, 40.0)
            rows_p = M.profile(W[:, jb][:, :, ib], depth, lat[jb], lon[ib], la, lo,
                               comp_floor=0.5)
            sel = [x for x in rows_p if 1000.0 <= x['depth_km'] <= 2600.0
                   and x['weight'] > 0]
            if sel:
                r['corr_r50_km'] = float(np.nanmedian([x['r50_km'] for x in sel]))
                r['corr_anisotropy'] = float(np.nanmedian([x['anisotropy'] for x in sel]))
                r['corr_ncomp_max'] = int(np.nanmax([x['n_components'] for x in sel]))
            del D, dC, W
        del C
        rows.append(r)
    print(f'  {shape:16s} {nm:18s} amp={amp} radius={radius:.0f} km', flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--configs', type=int, default=8)
    ap.add_argument('--shapes', nargs='+', default=list(S.NAMES))
    ap.add_argument('--sites', type=int, default=len(SITES),
                    help='how many of the four injection sites to use')
    ap.add_argument('--amps', nargs='+', type=float, default=list(AMPS))
    ap.add_argument('--radii', nargs='+', type=float, default=[300.0],
                    help='conduit radii to inject, km. Sweeping this is how the '
                         'resolution floor of the analysis is measured in the '
                         'model actually used, which is what has to carry the '
                         'interpretation once agreement with a coarser model is '
                         'no longer the standard: a coarse model cannot '
                         'represent a 150 km conduit at all, so its silence '
                         'about one is not evidence')
    ap.add_argument('--offset', type=float, default=10.0,
                    help='total lateral displacement of the injected axis, degrees')
    ap.add_argument('--corridor', action='store_true')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--suffix', default='')
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)

    keep = stratify(retained(a.dir, a.tag), a.configs)
    cfgs = [dict(s=float(r.s), z_target=float(r.z_target), channel=str(r.channel),
                 h_max=_hmax(r), radius=float(r.radius)) for _, r in keep.iterrows()]
    depth, lat, lon, arr = load_anomaly(a.file, ModelSpec(a.tag, a.var),
                                        depth_max=a.depth_max, every=a.every)
    lon, arr = dedupe_lon(lon, arr)
    check_shells(depth, arr, a.tag)
    con = contrast_field(arr, lat, lon, SIGMA)

    rng = np.random.default_rng(42)                 # the classification's own draw
    nlon = 360.0 * rng.random(N_NULL) - 180.0
    nlat = np.degrees(np.arcsin(2 * rng.random(N_NULL) - 1))
    thresh, pre = {}, {}
    for cfg in cfgs:
        s, zt, ch, hm, rd = (cfg['s'], cfg['z_target'], cfg['channel'],
                             cfg['h_max'], cfg['radius'])
        C = cost_field(arr, con, depth, lat, lon, s=s, z_target=zt,
                       n_relax=N_RELAX, channel=ch, h_max_km=hm)
        nul = np.array([site_cost(C, depth, lat, lon, la, lo, rd, SEED_DEPTH)
                        for la, lo in zip(nlat, nlon)], float)
        k = (s, zt, ch, hm, rd)
        thresh[k] = float(np.nanpercentile(nul, NULL_PCT))
        pre[k] = np.array([site_cost(C, depth, lat, lon, la, lo, rd, SEED_DEPTH)
                           for _, la, lo in SITES], float)
        del C
        print(f'  observed s={s} zt={zt:.0f} {ch} h{hm if hm else 0:.0f} r{rd}: '
              f'null p5 = {thresh[k]:.0f}', flush=True)

    plan = list(itertools.product(SITES[:a.sites], a.amps, a.shapes, a.radii))
    print(f'{len(plan)} injections x {len(cfgs)} configurations', flush=True)
    _W.update(arr=arr, depth=depth, lat=lat, lon=lon, cfgs=cfgs, thresh=thresh,
              pre=pre, plan=plan, offset=a.offset, corridor=bool(a.corridor))
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.jobs) as pool:
            got = pool.map(_case, range(len(plan)), chunksize=1)
    else:
        got = [_case(q) for q in range(len(plan))]
    df = pd.DataFrame([r for rows in got for r in rows])
    out = os.path.join(a.dir, f'synth_morph_{a.tag}{a.suffix}.csv')
    df.to_csv(out, index=False)

    print('\ndetection and recovery at the calibration amplitude, by shape:')
    cal = df[df.amp <= CAL_AMP]
    g = cal.groupby('shape').agg(n=('detected', 'size'),
                                 detected=('detected', 'mean'),
                                 recovered=('recovered', 'mean'),
                                 axis_km=('axis_distance_km', 'median'))
    order = [s for s in S.NAMES if s in g.index]
    print(g.loc[order].to_string(float_format=lambda x: f'{x:.3g}'))
    if len(a.radii) > 1 or len(a.amps) > 1:
        print('\nthe floor: detection rate by conduit radius and amplitude')
        piv = df.pivot_table(index='conduit_radius_km', columns='amp',
                             values='detected', aggfunc='mean')
        print(piv.to_string(float_format=lambda x: f'{x:.2f}'))
        print('\n  and the same for whether the traced path follows the axis')
        piv2 = df.pivot_table(index='conduit_radius_km', columns='amp',
                              values='recovered', aggfunc='mean')
        print(piv2.to_string(float_format=lambda x: f'{x:.2f}'))
    if a.corridor and 'corr_r50_km' in df:
        c = df[df.corr_r50_km.notna()].groupby('shape').agg(
            r50=('corr_r50_km', 'median'), anis=('corr_anisotropy', 'median'),
            ncomp=('corr_ncomp_max', 'median'))
        print('\ncorridor description of each injected shape:')
        print(c.loc[[s for s in S.NAMES if s in c.index]].to_string(
            float_format=lambda x: f'{x:.3g}'))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
