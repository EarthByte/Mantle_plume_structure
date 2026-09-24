"""Do hotspots overlie an axis, where comparable places in the mantle do not?

The statistic and every parameter it uses come from out/axis_config_frozen.json,
written by axis_coherence_test.py before this script was ever run on real sites.
This script has no parameters of its own for the measurement and will refuse to
run if that file is missing or has been edited, because the failure mode this
project keeps hitting is a configuration chosen, consciously or not, after seeing
which choice gives the desired answer.

Three comparisons, in increasing order of how much they are worth.

Uniform nulls answer almost nothing: they compare province interiors against
normal mantle and will be significant whatever is true of plumes.

Nulls matched nearest-neighbour on the regional field — the mean anomaly in an
annulus that excludes the column itself — hold "how slow is this part of the
mantle" constant. Band matching on a percentile range is not enough; it retains
four fifths of the nulls and is barely a restriction.

The rotation null is the strongest, because hotspots are a clustered sample and
rotating the whole set rigidly preserves every inter-hotspot distance while
randomising position. Anything that survives it is not the clustering.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon, check_shells
from axis_core import AxisTracker, coherence, disc_benchmark, R_E, DEG
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--var', default='voigt')
ap.add_argument('--config', default=None,
                help='defaults to out/axis_config_frozen_<tag>.json. The configuration '
                     'is per model: the search radius is a physical distance, and the '
                     'same distance is a different number of grid cells in each model, '
                     'so a configuration frozen on one does not transfer to another')
ap.add_argument('--ring-in', type=float, default=1000.0, dest='ring_in')
ap.add_argument('--ring-out', type=float, default=2500.0, dest='ring_out')
ap.add_argument('--n-null', type=int, default=300, dest='n_null')
ap.add_argument('--n-rot', type=int, default=200, dest='n_rot',
                help='rigid rotations of the hotspot set. Each one retracks all 49 '
                     'sites, so unlike the cheap rotation nulls elsewhere in this '
                     'study this costs about a track per site per rotation: 200 takes '
                     'minutes and resolves P to 0.005, 20000 would take most of a day')
ap.add_argument('--k-match', type=int, default=3, dest='k_match')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--dir', default='out')
ap.add_argument('--suffix', default='')
A = ap.parse_args()

A.config = A.config or os.path.join(A.dir, f'axis_config_frozen_{A.tag}.json')
if not os.path.exists(A.config):
    raise SystemExit(
        f'{A.config} does not exist. Run axis_coherence_test.py --tag {A.tag} --freeze '
        f'first, or, if that refused to freeze, this model cannot support the '
        f'measurement and should be reported as such.\n'
        'The measurement deliberately has no parameters of its own: a configuration\n'
        'chosen after seeing the answer is not a configuration, it is a result.')
cfg = json.load(open(A.config))
chk = cfg.pop('checksum', None)
if hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16] != chk:
    raise SystemExit(f'{A.config} has been edited since it was frozen (checksum '
                     f'mismatch). Refusing to run.')
if cfg['chosen_on']['model'] != A.tag:
    raise SystemExit(f'{A.config} was frozen on {cfg["chosen_on"]["model"]}, not '
                     f'{A.tag}. Configurations are not transferable between models.')
RAD, TOL = float(cfg['radius_km']), float(cfg['axial_tol_km'])
Z0, Z1 = float(cfg['z0']), float(cfg['z1'])
print(f'frozen config {chk} for {A.tag}: radius {RAD:.0f} km '
      f'({cfg.get("interior_cells", "?")} interior cells), axial tolerance {TOL:.0f} km, '
      f'window {Z0:.0f}-{Z1:.0f} km')
print(f'  chosen on {cfg["chosen_on"]["model"]} at {cfg["chosen_on"]["amp"]}% amplitude, '
      f'AUC {cfg["chosen_on"]["auc"]:.2f}, frozen {cfg["frozen_utc"]}Z', flush=True)

if A.ring_in <= RAD:
    raise SystemExit(f'--ring-in {A.ring_in:.0f} must exceed the frozen search radius '
                     f'{RAD:.0f}, or the matching covariate contains the column')

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=float(cfg['depth_max']),
                                    every=int(cfg['every']))
lon, arr = dedupe_lon(lon, arr)
check_shells(depth, arr, A.tag)
lat = np.asarray(lat, float)
lon = ((np.asarray(lon, float) + 180) % 360) - 180
o = np.argsort(lon); lon, arr = lon[o], arr[:, :, o]
kz = np.where((depth >= Z0) & (depth <= Z1))[0]
trk = AxisTracker(depth, lat, lon, arr, RAD, rim_frac=float(cfg['rim_frac']))
bench = disc_benchmark(RAD, len(kz), rim_frac=float(cfg['rim_frac']))
LO, LA = np.meshgrid(lon, lat)
win = np.nanmean(arr[kz], axis=0)
print(f'{A.tag}: {len(kz)} shells in window; no-axis benchmark scatter {bench:.0f} km',
      flush=True)


def ring_mean(hlat, hlon):
    d = R_E * np.arccos(np.clip(
        np.sin(hlat * DEG) * np.sin(LA * DEG) +
        np.cos(hlat * DEG) * np.cos(LA * DEG) * np.cos((LO - hlon) * DEG), -1, 1))
    m = (d >= A.ring_in) & (d <= A.ring_out)
    return float(np.nanmean(win[m])) if m.any() else np.nan


def measure(name, hlat, hlon, kind):
    c = coherence(trk.track(hlat, hlon, kz), len(kz), axial_tol_km=TOL)
    return dict(site=name, kind=kind, lat=hlat, lon=hlon,
                ring_anom=ring_mean(hlat, hlon), **c)


hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180'])
rows = [measure(str(r['hotspot']), float(r['lat']), float(r['lon_180']), 'hotspot')
        for _, r in hs.iterrows()]
print(f'  {len(rows)} hotspots done', flush=True)
rng = np.random.default_rng(51)
nlo = 360 * rng.random(A.n_null) - 180
nla = np.degrees(np.arcsin(2 * rng.random(A.n_null) - 1))
rows += [measure(f'null{i:03d}', float(a_), float(b_), 'null')
         for i, (a_, b_) in enumerate(zip(nla, nlo))]
print(f'  {A.n_null} nulls done', flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(A.dir, f'axis_coherence_{A.tag}{A.suffix}.csv'), index=False)
h, n = d[d.kind == 'hotspot'], d[d.kind == 'null']


def perm(a, b, n_=20000, seed=17):
    rg = np.random.default_rng(seed)
    obs = np.median(a) - np.median(b)
    pool = np.concatenate([a, b]); k = len(a); c = 0
    for _ in range(n_):
        q = rg.permutation(pool)
        if abs(np.median(q[:k]) - np.median(q[k:])) >= abs(obs) - 1e-12:
            c += 1
    return obs, (c + 1) / (n_ + 1)


def nn_match(col, k):
    pool = n[np.isfinite(n[col])]
    taken, keep = set(), []
    for _, hr in h.iterrows():
        if not np.isfinite(hr[col]):
            continue
        cand = pool[~pool.index.isin(taken)]
        if not len(cand):
            break
        for idx in (cand[col] - hr[col]).abs().nsmallest(min(k, len(cand))).index:
            taken.add(idx); keep.append(idx)
    sel = n.loc[keep]
    bal = float(np.mean([abs(n.loc[i, col] - h.iloc[0][col]) for i in keep])) if keep else np.nan
    return sel


print(f'\naxial fraction, the share of {Z0:.0f}-{Z1:.0f} km on a single axis:')
print(f'  hotspots              n={len(h):3d}   median {h.axial_frac.median():.3f}')
for lab, g in (('all nulls', n), ('regionally matched nulls', nn_match('ring_anom', A.k_match))):
    if len(g) < 20:
        print(f'  {lab:22s} n={len(g):3d}   too few'); continue
    dd, pp = perm(h.axial_frac.to_numpy(float), g.axial_frac.to_numpy(float))
    print(f'  {lab:22s} n={len(g):3d}   median {g.axial_frac.median():.3f}   '
          f'difference {dd:+.3f}   P = {pp:.4f}')

# Rotation null: the whole hotspot set turned rigidly, so clustering is preserved
# exactly and only position relative to mantle structure is randomised.
obs = float(h.axial_frac.median())
hl, hn = h.lat.to_numpy(float), h.lon.to_numpy(float)
x = np.stack([np.cos(hl * DEG) * np.cos(hn * DEG),
              np.cos(hl * DEG) * np.sin(hn * DEG), np.sin(hl * DEG)])
rg = np.random.default_rng(99)
rot = np.empty(A.n_rot)
for i in range(A.n_rot):
    q = rg.normal(size=4); q /= np.linalg.norm(q)
    w, xx, yy, zz = q
    R = np.array([[1 - 2 * (yy * yy + zz * zz), 2 * (xx * yy - zz * w), 2 * (xx * zz + yy * w)],
                  [2 * (xx * yy + zz * w), 1 - 2 * (xx * xx + zz * zz), 2 * (yy * zz - xx * w)],
                  [2 * (xx * zz - yy * w), 2 * (yy * zz + xx * w), 1 - 2 * (xx * xx + yy * yy)]])
    y = R @ x
    la = np.degrees(np.arcsin(np.clip(y[2], -1, 1)))
    lo = np.degrees(np.arctan2(y[1], y[0]))
    v = [coherence(trk.track(float(a_), float(b_), kz), len(kz), axial_tol_km=TOL)['axial_frac']
         for a_, b_ in zip(la, lo)]
    rot[i] = np.median(v)
    if (i + 1) % 200 == 0:
        print(f'    rotation {i + 1}/{A.n_rot}', flush=True)
p_rot = float((np.sum(rot >= obs) + 1) / (A.n_rot + 1))
print(f'\n  rotation null ({A.n_rot} rigid rotations of the whole set):')
print(f'    observed {obs:.3f}   rotated mean {rot.mean():.3f} '
      f'(5-95 {np.percentile(rot, 5):.3f}-{np.percentile(rot, 95):.3f})   P = {p_rot:.4f}')
np.save(os.path.join(A.dir, f'axis_rotation_{A.tag}{A.suffix}.npy'), rot)
print(f'\nwrote out/axis_coherence_{A.tag}{A.suffix}.csv', flush=True)
