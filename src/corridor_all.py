"""Near-optimal corridors for every hotspot, at the calibrated configuration.

A corridor is the set of routes the tomography cannot distinguish from the best
one: dC(x) = D(x) + C(x) - C_best, small where a route through x costs almost
nothing extra. It does two jobs at once.

It is the ERROR BAR on the traced path. Two fields correlated at 0.94 gave conduit
leans 62 degrees apart at Hawaii, so a single least-cost path is not a conduit; the
corridor states directly how much of the mantle the data allow the path to occupy,
without inventing a perturbation scheme.

And it is the measurement of ISOLATION versus NETWORK. A narrow closed corridor is a
distinct conduit; a broad one, especially one sharing cells with a neighbour's, is
part of an interconnected structure. Bao et al. emphasise networks of interconnected
plume-like structures in East Africa and the Indian Ocean, where the African province
dominates; a global survey should recover the whole spectrum with their region at one
end of it.

morph_run.py already computes corridors but selects configurations by even spacing
through a table sorted differently from paths.py, so no --configs value reproduces the
configuration the paths were traced under without running 28 of them. This uses that
configuration directly, calling the same corridor.py functions, so corridors and paths
describe the same search.
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon, check_shells
from contrast import contrast_field
from path_cost import cost_field, site_cost
from corridor import forward_field, excess_cost, self_check
import path_config
import provenance
warnings.filterwarnings('ignore')

R_E, DEG = 6371.0, np.pi / 180.0
SEED_DEPTH, N_RELAX, SIGMA = 200.0, 6, 800.0

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--taus', default='0.01,0.02,0.05',
                help='corridor thresholds as a fraction of the best cost')
ap.add_argument('--mask-tau', type=float, default=0.02, dest='mask_tau',
                help='the threshold whose mask is stored for overlap analysis')
ap.add_argument('--coarsen', type=int, default=4,
                help='factor by which the stored mask is thinned in each dimension')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--ncell', type=int, default=1,
                help='lateral cells a single slanted move may span. 1 was the whole move set and caps the steepest expressible tilt near 64 degrees in this model; 2 raises it to 76. Chosen with --lateral by slant_calibrate.py on hotspot paths against a matched ambient set.')
ap.add_argument('--lateral', type=float, default=1.0,
                help='weight on horizontal travel in the path cost. 1.0 is the Euclidean step length used before the tilt calibration; 0.40 is the value chosen by tilt_recovery.py under its pre-registered rule. Left at 1.0 the search abandons strongly tilted conduits, recovering a median 72 per cent of an injected lateral offset against 93 per cent at 0.40.')
ap.add_argument('--dir', default='out')
ap.add_argument('--refreeze', action='store_true',
                help='re-derive the configuration and overwrite the frozen one; '
                     'never incidental to another change')
ap.add_argument('--min-coverage', type=float, default=0.5, dest='min_coverage',
                help='fraction of the depth column the corridor must reach for its '
                     'width to be reported. Below it the width is written as NaN and '
                     'width_defined is False, so a path whose route ponds - and whose '
                     'corridor is therefore absent rather than narrow - cannot enter '
                     'a width statistic unnoticed.')
ap.add_argument('--suffix', default='')
ap.add_argument('--diagnose', default='',
                help='comma-separated site names to report the excess-cost '
                     'distribution for. Six sites returned corridors of tens '
                     'of cells where a typical site returns tens of thousands, '
                     'and a width statistic cannot be read off a corridor that '
                     'small. This says whether the cause is a steep cost '
                     'surface or a sparse reachable set.')
ap.add_argument('--float32', action='store_true',
                help='halve the memory, from about 3 GB to 1.5. NOT the default: '
                     'corridor.self_check requires the minimum of D + C to equal the '
                     'best cost to 1e-9 relative, and float32 carries only about 1e-7, '
                     'so the check fails at 7.9e-08 on every site - measured, not '
                     'guessed. The corridors themselves are unaffected, but a corridor '
                     'that has not passed the check is not one, so this is opt-in and '
                     'the failures are reported rather than tolerated.')
A = ap.parse_args()
TAUS = [float(x) for x in A.taus.split(',')]
# Fraction of the depth column a corridor must reach before its width is a number
# at all. See the note where it is applied.
MIN_COVERAGE = float(A.min_coverage)

# Read from out/path_config_<tag>.json, the same file paths.py reads, so the two
# cannot drift apart and neither re-derives it. See path_config.py.
c = path_config.load(A.tag, A.dir, refreeze=A.refreeze)
hm = getattr(c, 'h_max', None)
hm = None if hm is None or (isinstance(hm, float) and hm != hm) else float(hm)

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
check_shells(depth, arr, A.tag)
# Five full-resolution arrays are alive at the peak - anomaly, contrast, cost,
# forward field and excess cost - which is 3 GB in float64 on a 289 x 361 x 721
# grid. Float32 halves that and costs nothing here: the cost field is a sum of
# positive increments and the corridor threshold is a per cent of it, so seven
# significant figures are ample.
DT = np.float32 if A.float32 else np.float64
if A.float32:
    print('float32 requested: the self-check will fail at about 8e-08 relative, '
          'which is precision and not an error in the corridors', flush=True)
arr = np.asarray(arr, dtype=DT)
con = np.asarray(contrast_field(arr, lat, lon, SIGMA), dtype=DT)
C = np.asarray(cost_field(arr, con, depth, lat, lon, s=float(c.s),
                          z_target=float(c.z_target), n_relax=N_RELAX,
                          channel=str(c.channel), h_max_km=hm, lateral=float(A.lateral), ncell=int(A.ncell)), dtype=DT)
z = np.asarray(depth, float)
k0 = int(np.where(z >= float(c.z_target))[0][0]) if (z >= float(c.z_target)).any() else len(z) - 1
k_seed = int(np.argmin(np.abs(z - SEED_DEPTH)))
print(f'cost field built; seed shell {z[k_seed]:.0f} km, target shell {z[k0]:.0f} km',
      flush=True)

dlat = abs(lat[1] - lat[0]) * DEG * R_E
dlon = abs(lon[1] - lon[0]) * DEG * R_E
cell_km2 = dlat * dlon * np.cos(np.asarray(lat, float) * DEG)
dz = float(np.median(np.diff(z)))

DIAGNOSE = {x.strip() for x in A.diagnose.split(',') if x.strip()}
hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180'])
if DIAGNOSE:
    _miss = DIAGNOSE - set(hs.hotspot.astype(str))
    if _miss:
        raise SystemExit(f'--diagnose names no such site: {sorted(_miss)}')
rows, prof_rows, masks = [], [], {}
for i, r in hs.reset_index(drop=True).iterrows():
    name, hla, hlo = str(r.hotspot), float(r.lat), float(r.lon_180)
    cost = float(site_cost(C, depth, lat, lon, hla, hlo, float(c.radius), SEED_DEPTH))
    if not np.isfinite(cost):
        rows.append(dict(site=name, ok=False)); continue
    D = forward_field(arr, con, depth, lat, lon, hla, hlo, s=float(c.s),
                      z_target=float(c.z_target), radius_deg=float(c.radius),
                      seed_depth=SEED_DEPTH, n_relax=N_RELAX,
                      channel=str(c.channel), h_max_km=hm,
                      lateral=float(A.lateral), ncell=int(A.ncell))
    dC, cbest = excess_cost(D, C, k_seed, k0)
    if dC is None:
        del D
        rows.append(dict(site=name, ok=False)); continue
    ok, _ = self_check(D, C, k_seed, k0, c_best=cost)
    if name in DIAGNOSE:
        _fD = np.isfinite(D[k_seed:k0 + 1]); _fC = np.isfinite(C[k_seed:k0 + 1])
        _both = _fD & _fC
        _n = _fD.size
        _d = np.asarray(dC, float)[k_seed:k0 + 1]
        _dv = _d[np.isfinite(_d)]
        print(f'\n  -- {name}: C_best {cbest:.1f}, self-check {"pass" if ok else "FAIL"}')
        print(f'     reachable cells   forward {100*_fD.sum()/_n:6.2f}%   '
              f'backward {100*_fC.sum()/_n:6.2f}%   both {100*_both.sum()/_n:6.2f}%'
              f'   of {_n} cells')
        if _dv.size:
            _q = np.percentile(_dv, [0, 1, 5, 25, 50])
            print(f'     excess cost dC    min {_q[0]:.3g}  p1 {_q[1]:.3g}  '
                  f'p5 {_q[2]:.3g}  p25 {_q[3]:.3g}  median {_q[4]:.3g}')
            for _t in TAUS:
                print(f'       tau {_t:<5g} tolerance {_t*cbest:8.2f}  '
                      f'cells {int((_dv <= _t*cbest).sum()):8d}  '
                      f'({100*(_dv <= _t*cbest).sum()/max(_both.sum(),1):6.3f}% of reachable)')
            # A steep surface and a sparse reachable set look identical in a cell
            # count. They separate here: if almost every reachable cell is already
            # inside the corridor, the corridor is small because little is
            # reachable, and the width is a property of the move set, not the model.
            _in = int((_dv <= TAUS[1] * cbest).sum()) if len(TAUS) > 1 else 0
            _frac = _in / max(int(_both.sum()), 1)
            _verdict = ('sparse reachable set - the width is a move-set artefact'
                        if _both.sum() and _frac > 0.5 else
                        'steep cost surface - the corridor is genuinely tight')
            print(f'     verdict: {_verdict}  ({100*_frac:.1f}% of reachable cells '
                  f'are inside the tau {TAUS[1]:g} corridor)')
    del D                       # the forward field is finished with; free it now
    dC = np.asarray(dC, dtype=DT)
    rec = dict(site=name, ok=bool(ok), c_best=float(cbest))
    for t in TAUS:
        m = dC <= t * cbest
        vol = float((m * cell_km2[None, :, None]).sum() * dz)
        rec[f'cells_{t}'] = int(m.sum())
        rec[f'volume_{t}'] = vol
        # Width profile: equivalent diameter of the corridor's cross-section.
        # A shell the corridor does not reach has NO width, which is not a width
        # of zero. Writing 0.0 made an absent shell look like a perfectly
        # constrained one: the summary median below already excluded them, but
        # the per-shell profile did not, and plume_geometry.py took a median
        # straight over it. At the calibrated move set six corridors collapsed to
        # tens of cells and up to 99.6 per cent of their shells were empty, which
        # gave those sites a band width of exactly 0 km.
        w = []
        for k in range(k_seed, k0 + 1):
            a_ = float((m[k] * cell_km2[:, None]).sum())
            w.append(2.0 * np.sqrt(a_ / np.pi) if a_ > 0 else np.nan)
        w = np.array(w, float)
        ok_w = np.isfinite(w) & (w > 0)
        cov = float(ok_w.sum()) / max(len(w), 1)
        # A corridor that reaches only a few shells has no width to report. dC
        # describes the cells a route can DESCEND through, so a route that runs
        # sideways at one depth leaves the corridor empty below the shells it
        # descends into - not narrow, absent. Six RevealLO paths did exactly this
        # and covered 1 to 3 shells of 251 while the other 43 covered all 251; the
        # width statistic pooled all 49 as though each number meant the same thing.
        # The cut is at half the column, which the observed split does not come
        # near on either side, so the rule is not doing any of the separating.
        rec[f'shells_{t}'] = int(ok_w.sum())
        rec[f'shells_total_{t}'] = int(len(w))
        rec[f'coverage_{t}'] = cov
        rec[f'width_defined_{t}'] = bool(cov >= MIN_COVERAGE)
        if cov >= MIN_COVERAGE:
            rec[f'width_med_{t}'] = float(np.median(w[ok_w])) if ok_w.any() else np.nan
            rec[f'width_max_{t}'] = float(np.max(w[ok_w])) if ok_w.any() else np.nan
        else:
            rec[f'width_med_{t}'] = np.nan
            rec[f'width_max_{t}'] = np.nan
        if abs(t - A.mask_tau) < 1e-9:
            # The mask must span the same shells as the depth axis written
            # beside it. Slicing from 0 over the full depth range while the
            # axis was written as z[k_seed:k0+1:coarsen] left the two
            # describing different things, silently.
            masks[name] = m[k_seed:k0 + 1:A.coarsen,
                            ::A.coarsen, ::A.coarsen].astype(bool)
            for k, ww in zip(range(k_seed, k0 + 1), w):
                prof_rows.append(dict(site=name, depth=float(z[k]), width_km=float(ww)))
    del dC
    rows.append(rec)
    if (i + 1) % 5 == 0:
        print(f'  {i + 1}/{len(hs)} {name}', flush=True)

d = pd.DataFrame(rows)
# Say it out loud. A site silently carrying NaN into a median is how the fill-value
# group got into the paper in the first place.
_wd = f'width_defined_{A.mask_tau:g}'
if _wd in d.columns:
    _lost = d[~d[_wd].astype(bool)]
    print(f'\nwidth at tau {A.mask_tau:g}: defined for {len(d) - len(_lost)} of '
          f'{len(d)} sites')
    if len(_lost):
        print('  no width, the corridor does not span the column '
              '(the route ponds, so it has no descending corridor to measure):')
        for _, _r in _lost.sort_values(f'coverage_{A.mask_tau:g}').iterrows():
            print(f'    {_r.site:26s} {100 * _r[f"coverage_{A.mask_tau:g}"]:5.1f}% of '
                  f'{int(_r[f"shells_total_{A.mask_tau:g}"])} shells')
_cs = os.path.join(A.dir, f'corridor_summary_{A.tag}{A.suffix}.csv')
_cp = os.path.join(A.dir, f'corridor_profiles_{A.tag}{A.suffix}.csv')
d.to_csv(_cs, index=False)
pd.DataFrame(prof_rows).to_csv(_cp, index=False)
for _f in (_cs, _cp):
    provenance.stamp(_f, config=c, lateral=float(A.lateral), ncell=int(A.ncell),
                     taus=A.taus, mask_tau=A.mask_tau, coarsen=A.coarsen,
                     min_coverage=A.min_coverage, inputs=[A.file, A.hotspots])
mask_z = z[k_seed:k0 + 1:A.coarsen]
for _n, _m in masks.items():
    if _m.shape[0] != len(mask_z):
        raise SystemExit(f'mask depth axis mismatch for {_n}: mask has {_m.shape[0]} '
                         f'shells, depth axis has {len(mask_z)}. Refusing to write a '
                         f'file whose masks and coordinates disagree.')
np.savez_compressed(os.path.join(A.dir, f'corridor_masks_{A.tag}{A.suffix}.npz'),
                    coarsen=A.coarsen, tau=A.mask_tau,
                    depth=mask_z,
                    lat=np.asarray(lat)[::A.coarsen], lon=np.asarray(lon)[::A.coarsen],
                    **masks)
g = d[d.ok == True]
if not len(g) and len(d):
    print('\nNO corridor passed the self-check. If --float32 was used this is '
          'precision; otherwise the forward and backward fields are not halves of '
          'the same search and nothing below should be believed.')
print(f'\n{len(g)} of {len(d)} corridors computed, self-check passed for '
      f'{int(d.ok.sum())}')
t = A.mask_tau
print(f'\ncorridor width at tau = {t} (equivalent diameter, km):')
print(f'  median across hotspots {g[f"width_med_{t}"].median():.0f}, '
      f'range {g[f"width_med_{t}"].min():.0f}-{g[f"width_med_{t}"].max():.0f}')
print('\nnarrowest, the best constrained paths:')
for _, q in g.nsmallest(5, f'width_med_{t}').iterrows():
    print(f'  {q.site:26s} {q[f"width_med_{t}"]:6.0f} km')
print('broadest, the least constrained paths:')
for _, q in g.nlargest(5, f'width_med_{t}').iterrows():
    print(f'  {q.site:26s} {q[f"width_med_{t}"]:6.0f} km')
print(f'\nwrote corridor_summary, corridor_profiles and corridor_masks for {A.tag}')
