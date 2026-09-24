"""Can axis coherence detect a conduit at the places the claim is about?

The root-depth walk was cleared at random sites and then applied to hotspots, and
that is why it failed: hotspots are selected for sitting over the largest slow
structures in the lower mantle, which is the property that breaks a connectivity
measure. So this calibration runs at hotspot locations by default and the random
sites are the option, not the other way round.

A tube of known axis, radius and amplitude is added to the real field. The
question is whether the scatter of the recovered track separates a site with a
conduit from the same site without one. If the two distributions overlap, the
statistic cannot detect a conduit in the presence of the real ambient field and
there is nothing further to discuss.
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon, check_shells
from axis_core import AxisTracker, coherence, disc_benchmark, R_E, DEG
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=2)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--z0', type=float, default=800.0)
ap.add_argument('--z1', type=float, default=2850.0)
ap.add_argument('--radius', default='300,400,600,800,1000,1200',
                help='search radii to compare, km. The range must run high enough for '
                     'the coarse models: a 300 km disc chosen on RevealLO half-degree '
                     'cells is one or two cells across on SEMUCB-WM1 at two degrees, '
                     'where almost every minimum is then rim-pinned and discarded')
ap.add_argument('--conduit-radius', default='200,400', dest='cr', help='injected tube sigma, km')
ap.add_argument('--amps', default='0.3,0.6,1.2', help='injected amplitudes, per cent')
ap.add_argument('--tilt', type=float, default=0.0, help='injected lean, km per 1000 km depth')
ap.add_argument('--axial-tols', default='150,300,450', dest='tols',
                help='distances from the fitted line counted as on-axis, km')
ap.add_argument('--freeze', action='store_true',
                help='write axis_config_frozen.json with the best configuration. The '
                     'measurement will not run without it, and will not run if it is '
                     'edited afterwards, because the file carries its own checksum.')
ap.add_argument('--min-cells', type=int, default=12, dest='min_cells',
                help='interior cells a configuration must have. Below this the disc '
                     'cannot tell interior from rim and the statistic silently floors '
                     'at zero, which is what produced the all-zero REVEAL result')
ap.add_argument('--max-ambient-zero', type=float, default=0.5, dest='max_zero',
                help='the largest share of ambient sites allowed to score exactly zero. '
                     'A high AUC against an ambient that is zero everywhere is trivially '
                     'easy and is NOT evidence the statistic can rank real sites: REVEAL '
                     'scored AUC 0.99 with 75 per cent of its ambient sites at zero, and '
                     'then separated hotspots from nulls not at all, because both were '
                     'zero. Dynamic range in the ambient is a separate requirement from '
                     'sensitivity to an injection, and both are needed')
ap.add_argument('--min-auc', type=float, default=0.75, dest='min_auc',
                help='a model whose best configuration cannot separate an injected '
                     'conduit this well is not measurable and is refused a frozen '
                     'configuration rather than being given a meaningless one')
ap.add_argument('--detect-amp', type=float, default=0.6, dest='detect_amp',
                help='the amplitude the configuration is chosen to detect; the weakest '
                     'conduit we require the statistic to see, not the easiest')
ap.add_argument('--at-random', action='store_true', dest='at_random')
ap.add_argument('--sites', type=int, default=49)
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--dir', default='out')
ap.add_argument('--suffix', default='')
A = ap.parse_args()

RAD = [float(x) for x in A.radius.split(',')]
TOLS = [float(x) for x in A.tols.split(',')]
CR = [float(x) for x in A.cr.split(',')]
AMPS = [float(x) for x in A.amps.split(',')]
os.makedirs(A.dir, exist_ok=True)

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
check_shells(depth, arr, A.tag)
lat = np.asarray(lat, float)
lon = ((np.asarray(lon, float) + 180) % 360) - 180
o = np.argsort(lon); lon, arr = lon[o], arr[:, :, o]
kz = np.where((depth >= A.z0) & (depth <= A.z1))[0]
print(f'{A.tag}: window {depth[kz].min():.0f}-{depth[kz].max():.0f} km, {len(kz)} shells', flush=True)

if A.at_random:
    rg = np.random.default_rng(202)
    SITES = [(f'site{i:03d}', float(np.degrees(np.arcsin(2 * rg.random() - 1))),
              float(360 * rg.random() - 180)) for i in range(A.sites)]
else:
    hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180'])
    SITES = [(str(r['hotspot']), float(r['lat']), float(r['lon_180'])) for _, r in hs.iterrows()]
print(f'{len(SITES)} sites, {"random" if A.at_random else "HOTSPOTS"}', flush=True)

rows = []
cells = {}
for rad in RAD:
    trk = AxisTracker(depth, lat, lon, arr, rad)
    _c = []
    for _n, _la, _lo in SITES[:12]:
        _b = trk._box(_la, _lo)
        if _b is not None:
            _c.append(int((_b[2] <= rad * 0.85).sum()))
    cells[rad] = int(np.median(_c)) if _c else 0
    if cells[rad] < A.min_cells:
        print(f'  radius {rad:.0f} km: only {cells[rad]} interior cells, below '
              f'--min-cells {A.min_cells}; skipped', flush=True)
        continue
    bench = disc_benchmark(rad, len(kz))
    for name, hlat, hlon in SITES:
        b = trk._box(hlat, hlon)
        if b is None:
            continue
        jl, il, d = b
        t_amb = trk.track(hlat, hlon, kz)
        for tol in TOLS:
            amb = coherence(t_amb, len(kz), axial_tol_km=tol)
            rows.append(dict(site=name, radius=rad, amp=0.0, cr=np.nan, tol=tol,
                             bench=bench, **{f'a_{k}': v for k, v in amb.items()}))
        sub0 = np.asarray(arr[:, jl][:, :, il], float)
        for cr in CR:
            for amp in AMPS:
                # A tube whose axis may lean, built in the same local frame the
                # tracker measures in, so an injected tilt and a recovered tilt
                # are the same quantity.
                dz = (depth - depth[kz].min())[:, None, None]
                shift = A.tilt * dz / 1000.0
                dlon = ((lon[il] - hlon + 180) % 360) - 180
                ex = (R_E * DEG * dlon * np.cos(hlat * DEG))[None, None, :]
                ny = (R_E * DEG * (lat[jl] - hlat))[None, :, None]
                rr2 = (ex - shift) ** 2 + ny ** 2
                fld = sub0 - amp * np.exp(-0.5 * rr2 / cr ** 2)
                t_inj = trk.track(hlat, hlon, kz, sub=fld)
                for tol in TOLS:
                    c = coherence(t_inj, len(kz), axial_tol_km=tol)
                    rows.append(dict(site=name, radius=rad, amp=amp, cr=cr, tol=tol,
                                     bench=bench, **{f'a_{k}': v for k, v in c.items()}))
    print(f'  radius {rad:.0f} km done, {cells[rad]} interior cells '
          f'(no-axis benchmark {bench:.0f} km)', flush=True)

df = pd.DataFrame(rows)
p = os.path.join(A.dir, f'axis_test_{A.tag}{A.suffix}.csv')
df.to_csv(p, index=False)


def auc(pos, neg):
    """P(a site with a conduit scores lower scatter than one without). 0.5 is useless."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    return float((np.sum(pos[:, None] < neg[None, :])
                  + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))


best = None
print(f'\nconfiguration search: separating an injected conduit from its absence,')
print(f'at the SAME hotspot sites, scored on the weakest conduit we require '
      f'({A.detect_amp:.1f}% amplitude)')
print(f'{"radius":>7s} {"tol":>5s} {"tube":>6s} {"amp":>5s} {"axial amb":>10s} '
      f'{"axial inj":>10s} {"AUC":>6s}')
for rad in [r for r in RAD if cells.get(r, 0) >= A.min_cells]:
    for tol in TOLS:
        a0 = df[(df.radius == rad) & (df.amp == 0) & (df.tol == tol)]
        for cr in CR:
            for amp in AMPS:
                s_ = df[(df.radius == rad) & (df.amp == amp)
                        & (df.cr == cr) & (df.tol == tol)]
                if not len(s_):
                    continue
                a = auc(-s_.a_axial_frac, -a0.a_axial_frac)
                print(f'{rad:7.0f} {tol:5.0f} {cr:6.0f} {amp:5.2f} '
                      f'{np.nanmedian(a0.a_axial_frac):10.2f} '
                      f'{np.nanmedian(s_.a_axial_frac):10.2f} {a:6.2f}')
                # The configuration is chosen on the WEAKEST conduit we require the
                # statistic to detect, not the strongest it happens to manage. A
                # config tuned on a 1.2 per cent conduit would be chosen for a case
                # every configuration handles and would say nothing about the hard one.
                if abs(amp - A.detect_amp) < 1e-9 and np.isfinite(a):
                    if best is None or a > best['auc']:
                        best = dict(radius=rad, axial_tol_km=tol, cr=cr, amp=amp,
                                    auc=float(a),
                                    ambient=float(np.nanmedian(a0.a_axial_frac)),
                                    injected=float(np.nanmedian(s_.a_axial_frac)))

print('\nAn AUC near 0.5 means the statistic cannot see the conduit at all.')
if best is not None:
    print(f'\nbest at {A.detect_amp:.1f}%: radius {best["radius"]:.0f} km, '
          f'axial tolerance {best["axial_tol_km"]:.0f} km, AUC {best["auc"]:.2f} '
          f'(ambient {best["ambient"]:.2f} vs injected {best["injected"]:.2f})')

if best is not None:
    _a0 = df[(df.radius == best['radius']) & (df.amp == 0) & (df.tol == best['axial_tol_km'])]
    _z = float(np.mean(_a0.a_axial_frac.to_numpy(float) == 0.0)) if len(_a0) else 1.0
    if _z > A.max_zero:
        print(f'\nREFUSING to freeze: {100 * _z:.0f}% of ambient sites score exactly '
              f'zero at the best configuration, above --max-ambient-zero '
              f'{100 * A.max_zero:.0f}%. The AUC of {best["auc"]:.2f} is against an '
              f'ambient with no dynamic range, so the statistic cannot rank real sites '
              f'against each other even though it can see a strong injected conduit. '
              f'This model cannot support the comparison.')
        best = None

if best is not None and best['auc'] < A.min_auc:
    print(f'\nREFUSING to freeze: best AUC {best["auc"]:.2f} at {A.detect_amp:.1f}% is '
          f'below --min-auc {A.min_auc:.2f}. This model cannot detect a conduit of the '
          f'weakest amplitude required, at any configuration tried, so it cannot be '
          f'asked whether hotspots have one. That is a statement about the model, and '
          f'it belongs in the paper rather than being papered over with a '
          f'configuration that returns numbers.')
    best = None

if A.freeze and best is not None:
    import hashlib, json, datetime
    cfg = dict(radius_km=best['radius'], axial_tol_km=best['axial_tol_km'],
               z0=A.z0, z1=A.z1, every=A.every, depth_max=A.depth_max,
               rim_frac=0.85, interior_cells=cells.get(best['radius'], 0),
               chosen_on=dict(model=A.tag, amp=best['amp'],
               conduit_radius_km=best['cr'], auc=best['auc'],
               ambient_axial=best['ambient'], injected_axial=best['injected'],
               sites='hotspots' if not A.at_random else 'random'),
               frozen_utc=datetime.datetime.utcnow().isoformat(timespec='seconds'))
    body = json.dumps(cfg, sort_keys=True)
    cfg['checksum'] = hashlib.sha256(body.encode()).hexdigest()[:16]
    fp = os.path.join(A.dir, f'axis_config_frozen_{A.tag}.json')
    if os.path.exists(fp):
        print(f'\nREFUSING to overwrite {fp}. A configuration frozen once and then '
              f'refrozen after seeing a result is not frozen. Move the existing file '
              f'aside deliberately if you really mean to redo this.')
    else:
        with open(fp, 'w') as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        print(f'\nfroze {fp}  checksum {cfg["checksum"]}')
        print('The measurement reads its parameters from this file and will refuse to '
              'run if it is edited.')
print(f'\nwrote {p}', flush=True)
