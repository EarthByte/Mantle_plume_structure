"""Is the continuity of hotspot columns local to the column, or inherited from the region?

column_continuity.py compares hotspots against nulls matched on the anomaly at
2700 km. That holds the base constant and says nothing about the mid-mantle
material between, which is the material a 400 km search radius would pick up
whether or not a column exists. Hotspots sit over the largest slow structures in
the lower mantle, so their neighbourhoods are slow at every depth, not only at
the base — and the same ingredient, the minimum within 400 km at each depth, is
what destroyed the root-depth walk when it was finally calibrated at hotspot
locations rather than at random ones.

The control is a null matched on the REGION rather than on the base: the mean
anomaly over the same depth window, measured in an annulus that starts outside
the local search radius and so excludes the column itself. Matching on that holds
"how slow is this part of the mantle" constant while leaving the local column free
to differ. If hotspots still have more continuous columns than places with equally
slow surroundings, the continuity belongs to the column. If they do not, the
result was the neighbourhood all along.

The annulus deliberately excludes the inner region. Matching on anything that
includes the column would match on the quantity being measured and would destroy
a real signal along with a spurious one.
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon, check_shells
from root_core import Profiler
warnings.filterwarnings('ignore')

R_E, DEG = 6371.0, np.pi / 180.0

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=2)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--z0', type=float, default=800.0, help='top of the continuity window, km')
ap.add_argument('--z1', type=float, default=2850.0, help='base of the continuity window, km')
ap.add_argument('--radius', type=float, default=400.0, help='local search radius, km')
ap.add_argument('--threshold', type=float, default=-0.6,
                help='per cent; the level column_continuity.py uses, for comparability')
ap.add_argument('--ring-in', type=float, default=800.0, dest='ring_in',
                help='inner edge of the matching annulus, km; must exceed --radius')
ap.add_argument('--ring-out', type=float, default=2500.0, dest='ring_out')
ap.add_argument('--tol', type=float, default=0.10,
                help='matching tolerance on the annulus mean, per cent')
ap.add_argument('--n-null', type=int, default=300, dest='n_null')
ap.add_argument('--n-perm', type=int, default=20000, dest='n_perm')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--dir', default='out')
ap.add_argument('--suffix', default='')
A = ap.parse_args()

if A.ring_in <= A.radius:
    raise SystemExit(f'--ring-in {A.ring_in:.0f} must exceed --radius {A.radius:.0f}, '
                     f'or the annulus contains the column it is meant to exclude')

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
check_shells(depth, arr, A.tag)
lat = np.asarray(lat, float)
lon = ((np.asarray(lon, float) + 180) % 360) - 180
o = np.argsort(lon); lon, arr = lon[o], arr[:, :, o]
kz = np.where((depth >= A.z0) & (depth <= A.z1))[0]
if not len(kz):
    raise SystemExit(f'no shells between {A.z0:.0f} and {A.z1:.0f} km')
print(f'{A.tag}: window {depth[kz].min():.0f}-{depth[kz].max():.0f} km, '
      f'{len(kz)} shells, local radius {A.radius:.0f} km, '
      f'annulus {A.ring_in:.0f}-{A.ring_out:.0f} km', flush=True)

prof = Profiler(depth, lat, lon, arr, A.radius)
LO, LA = np.meshgrid(lon, lat)
win = np.nanmean(arr[kz], axis=0)          # one map: mean anomaly over the window


def ring_mean(hlat, hlon):
    d = R_E * np.arccos(np.clip(
        np.sin(hlat * DEG) * np.sin(LA * DEG) +
        np.cos(hlat * DEG) * np.cos(LA * DEG) * np.cos((LO - hlon) * DEG), -1, 1))
    m = (d >= A.ring_in) & (d <= A.ring_out)
    return float(np.nanmean(win[m])) if m.any() else np.nan


rows = []


def measure(name, hlat, hlon, kind):
    p = prof.profile(hlat, hlon)
    if p is None:
        return
    frac = float(np.mean(p[kz] < A.threshold))
    base = float(np.nanmean(p[(depth >= 2600) & (depth <= 2850)]))
    rows.append(dict(site=name, kind=kind, lat=hlat, lon=hlon,
                     frac_below=frac, base_anom=base, ring_anom=ring_mean(hlat, hlon)))


hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180'])
for _, r in hs.iterrows():
    measure(str(r['hotspot']), float(r['lat']), float(r['lon_180']), 'hotspot')
rng = np.random.default_rng(51)                 # the null the rest of the study uses
nlo = 360 * rng.random(A.n_null) - 180
nla = np.degrees(np.arcsin(2 * rng.random(A.n_null) - 1))
for i, (a_, b_) in enumerate(zip(nla, nlo)):
    measure(f'null{i:03d}', float(a_), float(b_), 'null')

d = pd.DataFrame(rows)
d.to_csv(os.path.join(A.dir, f'continuity_annulus_{A.tag}{A.suffix}.csv'), index=False)
h = d[d.kind == 'hotspot']
n = d[d.kind == 'null']


def perm(a, b, n_=A.n_perm, seed=13):
    rg = np.random.default_rng(seed)
    obs = np.median(a) - np.median(b)
    pool = np.concatenate([a, b]); k = len(a); c = 0
    for _ in range(n_):
        q = rg.permutation(pool)
        if abs(np.median(q[:k]) - np.median(q[k:])) >= abs(obs) - 1e-12:
            c += 1
    return obs, (c + 1) / (n_ + 1)


print(f'\nhotspots are slower in their surroundings than random sites, as expected:')
print(f'  annulus mean anomaly   hotspots {h.ring_anom.median():+.3f}%   '
      f'nulls {n.ring_anom.median():+.3f}%')

# Band matching on a 10th-90th percentile range is barely a restriction: it keeps
# 82 per cent of the nulls, which is why the basal-matched control in
# column_continuity.py moves the number so little. Nearest-neighbour matching
# without replacement gives each hotspot the k most similar nulls and nothing
# else, and the balance it achieves is reported rather than assumed.
def nn_match(col, k=3):
    pool = n[np.isfinite(n[col])].copy()
    taken, pairs = set(), []
    for _, hr in h.iterrows():
        if not np.isfinite(hr[col]):
            continue
        cand = pool[~pool.index.isin(taken)]
        if not len(cand):
            break
        dd = (cand[col] - hr[col]).abs().nsmallest(min(k, len(cand)))
        for idx in dd.index:
            taken.add(idx)
            pairs.append((hr.site, idx, abs(pool.loc[idx, col] - hr[col])))
    sel = n.loc[[i for _, i, _ in pairs]]
    bal = float(np.mean([g for _, _, g in pairs])) if pairs else np.nan
    return sel, bal


base_m, bal_b = nn_match('base_anom')
ring_m, bal_r = nn_match('ring_anom')
both_m, bal_x = ring_m, bal_r
print(f'\nnearest-neighbour matching, 3 nulls per hotspot, without replacement:')
print(f'  basal   covariate |difference| after matching {bal_b:.4f}%   '
      f'(hotspot spread {h.base_anom.std():.3f}%)')
print(f'  annulus covariate |difference| after matching {bal_r:.4f}%   '
      f'(hotspot spread {h.ring_anom.std():.3f}%)')

print(f'\ncontinuity, fraction of {A.z0:.0f}-{A.z1:.0f} km below {A.threshold:+.1f}%:')
print(f'  hotspots                       n={len(h):3d}   median {h.frac_below.median():.3f}')
for lab, g in (('all nulls', n), ('basal-matched nulls', base_m),
               ('ANNULUS-matched nulls', ring_m)):
    if len(g) < 20:
        print(f'  {lab:30s} n={len(g):3d}   too few to test')
        continue
    dd, pp = perm(h.frac_below.to_numpy(float), g.frac_below.to_numpy(float))
    print(f'  {lab:30s} n={len(g):3d}   median {g.frac_below.median():.3f}   '
          f'difference {dd:+.3f}   P = {pp:.4f}')
print(f'\nIf the annulus-matched comparison collapses while the basal-matched one holds,\n'
      f'the continuity was the neighbourhood rather than the column.', flush=True)
