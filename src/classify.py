"""Stage 42 - calibrate the least-cost path search on a model, then classify on it.

Replaces s23_classify.py, which scanned a family of straight tilted axes. The
axis family could not reach the deep low-velocity bodies that sit hundreds of
kilometres from the surface hotspot, and widening its aperture did not help. Here
the conduit geometry is not prescribed at all: it is the cheapest descending path
through the velocity field, and the lateral offset it adopts is an output.

  1. SENSITIVITY. Synthetic conduits of 300 km radius and four amplitudes are
     injected at four sites spanning the situations the search must handle, each
     straight and tilted. A case is informative only where the unperturbed field
     does not already root the site, so that detection measures the conduit and
     not its surroundings. Recovery is tabulated separately for each amplitude,
     because it turns out to be close to a step: the calibration amplitude
     CAL_AMP and stronger are recovered almost always, weaker conduits almost
     never. A configuration is kept if it recovers at least MIN_DETECT of the
     cases at CAL_AMP or stronger, and the weaker amplitudes measure where the
     search stops seeing, which is the sensitivity floor the classification
     inherits.

  2. SPECIFICITY. For each surviving configuration the identical search is
     evaluated at random locations, and a hotspot counts as rooted only if its
     path is cheaper than the 5th percentile of that configuration's own null.

  3. ENSEMBLE. The reported quantity is the fraction of surviving configurations
     under which the hotspot is rooted.

The configurations differ in what the cheapest path is asked to optimise. For
small s the cost integral weights the mean anomaly along the path; as s grows,
exp(s*a) is increasingly dominated by the fastest cell traversed, so the cheapest
path approaches the one whose slowest bottleneck is strongest, with length as the
tiebreak. The grid spans that range, and calibration decides how much of it is
usable: in practice the bottleneck end fails the detection floor, because a path
selected on its single worst cell is too easily matched by a path that threads
the low-velocity provinces, so the null rises as fast as the signal.
"""
from __future__ import annotations

import argparse, itertools, os, sys, warnings
import numpy as np
import provenance
import pandas as pd
import path_config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon, check_shells
from contrast import contrast_field
from path_cost import cost_field, site_cost
from inject import inject_tilted

warnings.filterwarnings('ignore')

SIGMA = 800.0
N_NULL = 300
N_RELAX = 6
NULL_PCT = 5.0
MIN_DETECT = 0.75
DECISIVE_HI, DECISIVE_LO = 0.80, 0.20

S_VALUES = (0.3, 0.4, 0.5, 0.65, 0.8, 1.2)
# The target surface. It reaches to the deepest shell the model carries, because
# the whole target is free: any path arriving anywhere on it has arrived, so the
# gain from stepping sideways at depth z is only what it saves on the column
# still below, and that goes to zero at the target. With the target at 2500 km a
# conduit leaning entirely below 1800 km was recovered at 445 km of 800; at
# 2800 km it is recovered at 667, and its lateral travel below 2000 km doubles.
Z_TARGET = (2300.0, 2500.0, 2700.0, 2800.0)

# How far a path may run sideways at a single depth, in kilometres. Swept rather
# than asserted: it is the parameter that separates a conduit which ponds from a
# tour of the low-velocity network, and calibration against the ponded synthetics
# decides it. Left unbounded, the route beneath Afar ran 1386 km at 200 km depth
# before descending at all.
H_MAX = (300.0, 900.0)
# 'min' takes whichever of the anomaly and the local contrast is more
# favourable at each cell; 'anom' is the shell-relative anomaly alone; 'contrast'
# is the anomaly minus its own 800 km background alone. The third was never swept
# because 'min' was assumed to cover it, and measuring what 'min' actually
# resolves to showed otherwise: along the recovered paths the anomaly is the
# cheaper branch in 94 per cent of cells, so the contrast contributes a three per
# cent discount and no geometry. A search on the anomaly answers "is this mantle
# slow"; a search on the contrast answers "does this stand out from its
# surroundings", and those are different questions about a hotspot sitting inside
# a large low-velocity province. --channels makes the third runnable.
# All three, always. The screen exists to choose among channels, so a channel missing
# from it cannot be chosen - and worse, a configuration frozen on a channel the screen
# omits looks like a configuration with no retained set at all. That happened: run_all.sh
# calls classify.py without --channels, so a screen over min and anom alone overwrote a
# merged table, and the ensemble - correctly restricted to the frozen channel - then found
# nothing and reported no classification for any model.
CHANNELS = ('min', 'anom', 'contrast')
SEED_DEPTH = 200.0
RADII = (2.0, 3.0)

SITES = [('open ocean', 20.0, -156.0), ('Pacific province interior', -25.0, -165.0),
         ('African province margin', -21.0, 56.0), ('control', 35.0, -40.0)]
AMPS = (-0.5, -0.75, -1.0, -1.5)
# Injected geometries: a vertical conduit, one whose offset is taken up across
# the transition zone, and one carrying the same offset as a steady lean. The
# third is the case a search that pays for lateral travel finds hardest, and
# until now it was absent - so the reported detection rate described only
# conduits whose segments happen to lie along the grid.
SYN_GEOM = ((0.0, 'vertical'), (10.0, 'transition'), (10.0, 'uniform'))
PROFILES = ('vertical', 'transition', 'uniform')
CAL_AMP = -1.0
SYN_RADIUS = 300.0


_W = {}                                     # read-only state shared with workers


def _inject_case(idx):
    """One injected conduit, all configurations. Returns (idx, {config: hit}).

    Runs in a forked worker, reading the model from `_W`. Returns only the
    verdicts, a few hundred bytes, rather than any field.
    """
    arr, depth = _W['arr'], _W['depth']
    lat, lon = _W['lat'], _W['lon']
    thresh, site_pre = _W['thresh'], _W['site_pre']
    (nm, la, lo), amp, (tl, prof) = _W['plan'][idx]
    si = [s_[0] for s_ in SITES].index(nm)
    a2 = inject_tilted(arr, depth, lat, lon, la, lo, SYN_RADIUS, amp, tl,
                       profile=('uniform' if prof == 'uniform' else 'transition'))
    c2 = contrast_field(a2, lat, lon, SIGMA)
    res = {}
    for s, zt, ch, hm in _W['fields']:
        C = cost_field(a2, c2, depth, lat, lon, s=s, z_target=zt,
                       n_relax=N_RELAX, channel=ch, h_max_km=hm,
                       lateral=_W['lateral'], ncell=_W['ncell'])
        for rd in RADII:
            k = (s, zt, ch, hm, rd)
            if site_pre[k][si] < thresh[k]:
                res[k] = None                   # site already rooted: uninformative
            else:
                c = site_cost(C, depth, lat, lon, la, lo, rd, SEED_DEPTH)
                res[k] = bool(np.isfinite(c) and c < thresh[k])
        del C
    del a2, c2
    print(f'  {nm} amp={amp} {prof} tilt={tl:.0f}', flush=True)
    return idx, res


def configs(depth, channels=CHANNELS):
    zmax = float(np.max(depth))
    out = []
    for s, zt, ch, hm, rd in itertools.product(S_VALUES, Z_TARGET, channels,
                                               H_MAX, RADII):
        if zt > zmax:                          # GLAD-M35 stops shallower
            continue
        out.append(dict(s=s, z_target=zt, channel=ch, h_max=hm, radius=rd))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
    ap.add_argument('--out', default='out')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--channels', nargs='+', default=list(CHANNELS),
                    choices=('min', 'anom', 'contrast'),
                    help='cost channels to sweep. The default reproduces the '
                         'published classification; --channels contrast runs the '
                         'control that separates a conduit standing out from its '
                         'surroundings from mantle that is merely slow')
    ap.add_argument('--suffix', default='',
                    help='appended to every output name, so a control run cannot '
                         'overwrite the classification it is a control for')
    ap.add_argument('--jobs', type=int, default=1,
                    help='injections to run at once. Each worker holds about '
                         'twice the model in memory, so 8 needs roughly 10 GB '
                         'for RevealLO at native sampling')
    ap.add_argument('--depth-max', type=float, default=2880.0,
                    dest='depth_max',
                    help='ignore shells deeper than this, in km. The default '
                         'sits just above the core-mantle boundary at 2891 km, '
                         'because a file that samples the whole Earth radius '
                         'carries shells in which the shear velocity collapses')
    ap.add_argument('--ncell', type=int, default=1,
                    help='lateral cells a single slanted move may cross. The '
                         'move set with one cell cannot express a tilt steeper '
                         'than about 64 degrees, so conduits that lean harder '
                         'than that are unreachable however cheap they are. '
                         'slant_calibrate.py chose 2')
    ap.add_argument('--lateral', type=float, default=1.0,
                    help='weight on the lateral component of a move length. '
                         'slant_calibrate.py chose 0.60. The default reproduces '
                         'the behaviour every result before that calibration '
                         'was computed under')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    depth, lat, lon, arr = load_anomaly(a.file, ModelSpec(a.tag, a.var),
                                        depth_max=a.depth_max, every=a.every)
    n0 = len(lon)
    lon, arr = dedupe_lon(lon, arr)
    if len(lon) != n0:
        print('  dropped repeated wrap meridian', flush=True)
    check_shells(depth, arr, a.tag)
    print(f'{a.tag}: {len(depth)} shells {depth.min():.0f}-{depth.max():.0f} km, '
          f'{len(lat)}x{len(lon)}', flush=True)

    c_obs = contrast_field(arr, lat, lon, SIGMA)
    print(f'channels swept: {", ".join(a.channels)}', flush=True)
    cfgs = configs(depth, tuple(a.channels))
    print(f'{len(cfgs)} candidate configurations', flush=True)

    rng = np.random.default_rng(42)
    nlon = 360.0 * rng.random(N_NULL) - 180.0
    nlat = np.degrees(np.arcsin(2 * rng.random(N_NULL) - 1))
    hs = pd.read_csv(a.hotspots).dropna(subset=['lat', 'lon_180']).reset_index(drop=True)

    # ---- observed field: null thresholds, hotspot costs, and the pre-injection
    #      status of the four injection sites, all in one pass per cost field
    fields = sorted({(c['s'], c['z_target'], c['channel'], c['h_max'])
                     for c in cfgs})
    thresh, hcost, site_pre = {}, {}, {}
    for s, zt, ch, hm in fields:
        C = cost_field(arr, c_obs, depth, lat, lon, s=s, z_target=zt,
                       n_relax=N_RELAX, channel=ch, h_max_km=hm,
                       lateral=float(a.lateral), ncell=int(a.ncell))
        for rd in RADII:
            nul = np.array([site_cost(C, depth, lat, lon, la, lo, rd, SEED_DEPTH)
                            for la, lo in zip(nlat, nlon)], float)
            q = float(np.nanpercentile(nul, NULL_PCT))
            thresh[(s, zt, ch, hm, rd)] = q
            hcost[(s, zt, ch, hm, rd)] = np.array(
                [site_cost(C, depth, lat, lon, float(r.lat), float(r.lon_180), rd,
                           SEED_DEPTH) for _, r in hs.iterrows()], float)
            site_pre[(s, zt, ch, hm, rd)] = np.array(
                [site_cost(C, depth, lat, lon, la, lo, rd, SEED_DEPTH)
                 for _, la, lo in SITES], float)
        del C
        print(f'  observed s={s} zt={zt:.0f} {ch} h{hm:.0f}: '
              f'null p{NULL_PCT:.0f} = '
              f'{thresh[(s, zt, ch, hm, RADII[0])]:.0f}', flush=True)

    # ---- synthetic conduits
    #
    # Every injection is independent of every other, and there are 48 of them
    # against 36 cost fields each, so this is the whole run. numpy's elementwise
    # arithmetic is single-threaded, which left fifteen of sixteen cores idle;
    # the injections are therefore handed to a process pool. Workers are forked
    # after the model is loaded, so the anomaly cube is shared copy-on-write
    # rather than pickled to each of them - the arrays are never written, only
    # read. Each worker builds its own injected copy and its own cost field, so
    # peak memory is about twice the model per worker.
    print(f'injecting synthetic conduits on {a.jobs} core(s)', flush=True)
    plan = list(itertools.product(SITES, AMPS, SYN_GEOM))
    cases = [dict(site=nm, amp=amp, tilt=tl, profile=prof)
             for (nm, _, _), amp, (tl, prof) in plan]
    _W.update(arr=arr, depth=depth, lat=lat, lon=lon, fields=fields,
              thresh=thresh, site_pre=site_pre, plan=plan,
              lateral=float(a.lateral), ncell=int(a.ncell))
    if a.jobs > 1:
        import multiprocessing as mp
        ctx = mp.get_context('fork')            # fork: inherit, do not pickle
        with ctx.Pool(a.jobs) as pool:
            done = pool.map(_inject_case, range(len(plan)), chunksize=1)
    else:
        done = [_inject_case(i) for i in range(len(plan))]
    det_hits = {k: [] for k in thresh}          # config -> list of True/False/None
    for idx, res in sorted(done):               # sorted: order must match `cases`
        for k in det_hits:
            det_hits[k].append(res[k])
    _W.clear()

    cs = pd.DataFrame(cases)
    rows, curves = [], []
    for c in cfgs:
        k = (c['s'], c['z_target'], c['channel'], c['h_max'], c['radius'])
        h = pd.Series(det_hits[k], dtype=object)
        r = dict(**c)
        for amp in AMPS:
            v = [x for x in h[cs.amp.values == amp] if x is not None]
            r[f'detect_{abs(amp):g}'] = float(np.mean(v)) if v else np.nan
            r[f'n_{abs(amp):g}'] = len(v)
            curves.append(dict(**c, amp=amp, n=len(v),
                               detect=float(np.mean(v)) if v else np.nan))
        v = [x for x, m in zip(h, cs.amp.values <= CAL_AMP) if m and x is not None]
        r['n_cal'], r['detect_cal'] = len(v), (float(np.mean(v)) if v else np.nan)
        # split the calibration rate by deflection style, so a configuration that
        # only works on grid-aligned conduits cannot hide inside the average
        for prof in PROFILES:
            m = (cs.profile.values == prof) & (cs.amp.values <= CAL_AMP)
            v = [x for x, mm in zip(h, m) if mm and x is not None]
            r[f'n_{prof}'] = len(v)
            r[f'detect_{prof}'] = float(np.mean(v)) if v else np.nan
        rows.append(r)
    det = pd.DataFrame(rows)
    _dt = os.path.join(a.out, f'detection_{a.tag}{a.suffix}.csv')
    det.to_csv(_dt, index=False)
    provenance.stamp(_dt, channels=list(a.channels), lateral=a.lateral,
                     ncell=a.ncell, suffix=a.suffix, inputs=[a.file, a.hotspots])
    pd.DataFrame(curves).to_csv(
        os.path.join(a.out, f'sensitivity_{a.tag}{a.suffix}.csv'), index=False)
    print('\nrecovery by amplitude, median over configurations:')
    for amp in AMPS:
        print(f'  {amp:+.2f} %: {det[f"detect_{abs(amp):g}"].median():.2f}')
    print('recovery by conduit geometry, at calibration amplitude:')
    for prof in PROFILES:
        print(f'  {prof:>10s}: {det[f"detect_{prof}"].median():.2f} '
              f'(n={int(det[f"n_{prof}"].max())})')
    # The ensemble is taken over the frozen channel alone; path_config.retained explains
    # why. Before anything is frozen there is no channel to restrict to and the whole
    # retained set is used, which is the right behaviour for the very first screen.
    _ens_ch = path_config.ensemble_channel(a.tag, a.out)
    keep = path_config.retained(det, _ens_ch)
    if _ens_ch:
        print(f'ensemble restricted to the frozen channel {_ens_ch!r}')
    print(f'\n{len(keep)} of {len(det)} configurations reach the '
          f'{MIN_DETECT:.0%} detection floor')
    if not len(keep):
        # This is an inconsistency between the screen and the frozen configuration, not a
        # result. It means the model is frozen on a channel the screen did not cover, or
        # on one where nothing clears the floor, and either way no classification can be
        # written. Returning quietly let the whole workflow carry on through aggregation
        # and figures with six models' classifications left at their previous values, and
        # the run reached the figures before anyone noticed.
        raise SystemExit(
            f'{a.tag}: no configuration reaches the {MIN_DETECT:.0%} floor'
            + (f' on the frozen channel {_ens_ch!r}. The screen covered '
               f'{sorted(det.channel.unique())}. Either screen that channel '
               f'(classify.py --channels ...) or re-freeze onto one that is screened.'
               if _ens_ch else '. The screen found nothing admissible at all.'))

    ex = np.zeros((len(hs), len(keep)), bool)
    cst = np.full((len(hs), len(keep)), np.nan)
    for j, (_, c) in enumerate(keep.iterrows()):
        k = (c.s, c.z_target, c.channel, c.h_max, c.radius)
        cst[:, j] = hcost[k]
        ex[:, j] = np.isfinite(hcost[k]) & (hcost[k] < thresh[k])

    hs['root_fraction'] = ex.mean(axis=1)
    hs['cost_median'] = np.nanmedian(cst, axis=1)
    hs['ensemble_class'] = hs.root_fraction.map(
        lambda f: 'deeper than null' if f >= DECISIVE_HI
        else ('not deeper than null' if f <= DECISIVE_LO else 'ambiguous'))
    cols = ['hotspot', 'lat', 'lon_180', 'root_fraction', 'cost_median',
            'ensemble_class'] + (['count'] if 'count' in hs.columns else [])
    out = hs[cols].sort_values('root_fraction', ascending=False)
    # Provenance. The move set changes which conduits the search can reach at
    # all, so a classification file is meaningless without it and two files from
    # different move sets must never be mistaken for one another.
    out['lateral'] = float(a.lateral)
    out['ncell'] = int(a.ncell)
    # Provenance the staleness guard can actually read. It used to record the move set
    # only, so a change to the channel or to the ensemble rule left it invisible and the
    # guard skipped the step.
    out['ensemble_channel'] = _ens_ch or 'all'
    out['n_ensemble'] = int(len(keep))
    _cl = os.path.join(a.out, f'classification_{a.tag}{a.suffix}.csv')
    out.to_csv(_cl, index=False)
    provenance.stamp(_cl, channels=list(a.channels), lateral=a.lateral,
                     ncell=a.ncell, suffix=a.suffix,
                     n_retained=int(len(keep)), inputs=[a.file, a.hotspots])

    print('\n' + '=' * 88)
    print(hs.ensemble_class.value_counts().to_string())
    print(out.head(22).to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    if 'count' in hs.columns:
        from scipy.stats import spearmanr
        m = hs.dropna(subset=['count', 'root_fraction'])
        rho, p = spearmanr(m['count'], m.root_fraction)
        print(f'\nCourtillot criteria count vs root fraction: rho = {rho:+.3f}, '
              f'p = {p:.4f}, n = {len(m)}')
    print(f'\nwrote classification_{a.tag}{a.suffix}.csv')


if __name__ == '__main__':
    main()
