"""How deep the continuous slow column beneath a site actually reaches.

column_continuity.py asks how much of a fixed lower-mantle window is slow, which
is the right question for "is this column exceptional" and the wrong one for
"where does it stop". This asks the second question: anchoring below the
lithosphere and walking down, at what depth does the column first break in a way
that cannot be bridged?

Two things make the answer interpretable rather than an artefact of amplitude.

The threshold is a per-depth percentile of each model's own anomaly distribution
rather than a fixed per cent, so it means the same thing in a model that renders
anomalies strongly and one that renders them weakly, and the same thing at 800 km
where the anomalies are large and at 2800 km where they are small. The fixed
level column_continuity.py uses is reported alongside it, because the two results
should be compared and the fixed level is not comparable across models.

A column that stops shallow is only a shallow root if there is nothing slow
beneath it. If slow material sits at the base with a gap in between, the site is
a deep root with an imaging gap, and calling it transition-zone rooted would be
the whole argument made on a hole in the model. Both are measured and reported
separately: root_km is where the connected column ends, deep_frac is how much of
the basal window is slow regardless of connection.
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon, check_shells
from root_core import Profiler, thresholds, walk as _walk
warnings.filterwarnings('ignore')

R_E, DEG = 6371.0, np.pi / 180.0

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=2, help='thin the depth axis; 2 gives 20 km on a 10 km file')
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--anchor', type=float, default=300.0,
                help='top of the walk, km; below the lithospheric and asthenospheric '
                     'low velocities every hotspot has and none of which is a plume root')
ap.add_argument('--radius', type=float, default=400.0, help='cap radius searched at each depth, km')
ap.add_argument('--pct', type=float, default=20.0,
                help='per-depth percentile defining slow; 20 keeps the slowest fifth of each shell')
ap.add_argument('--fixed', type=float, default=-0.6, help='the fixed per cent level, reported alongside')
ap.add_argument('--tol', type=float, default=200.0, help='longest gap the column may bridge, km')
ap.add_argument('--deep-z0', type=float, default=2400.0, dest='deep_z0')
ap.add_argument('--deep-z1', type=float, default=2850.0, dest='deep_z1')
ap.add_argument('--n-null', type=int, default=300, dest='n_null')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--dir', default='out')
ap.add_argument('--suffix', default='')
A = ap.parse_args()

os.makedirs(A.dir, exist_ok=True)
depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
# RevealLO's 2890 km shell is the outer core and sits at -97 per cent throughout.
# Left in, every column trivially reaches the base and the whole measurement is
# meaningless, so the screen the rest of the study uses is run here too rather
# than relying on the caller having passed the right --depth-max.
check_shells(depth, arr, A.tag)
lat = np.asarray(lat, float)
lon = ((np.asarray(lon, float) + 180) % 360) - 180
order = np.argsort(lon)
lon, arr = lon[order], arr[:, :, order]
print(f'{A.tag}: {len(depth)} shells, {depth.min():.0f}-{depth.max():.0f} km, '
      f'grid {len(lat)}x{len(lon)}', flush=True)



thr_pct = thresholds(arr, lat, lon, A.pct)
print(f'  per-depth {A.pct:.0f}th percentile runs '
      f'{np.nanmin(thr_pct):+.2f} to {np.nanmax(thr_pct):+.2f} per cent', flush=True)

_prof = Profiler(depth, lat, lon, arr, A.radius)


def profile(hlat, hlon):
    return _prof.profile(hlat, hlon)


def walk(prof, slow):
    return _walk(depth, prof, slow, A.anchor, A.tol)


rows = []


def measure(name, hlat, hlon, kind):
    prof = profile(hlat, hlon)
    if prof is None:
        return
    sp = prof < thr_pct
    sf = prof < A.fixed
    root_p, gap_p = walk(prof, sp)
    root_f, gap_f = walk(prof, sf)
    dk = np.where((depth >= A.deep_z0) & (depth <= A.deep_z1))[0]
    deep_p = float(np.mean(sp[dk])) if len(dk) else np.nan
    deep_f = float(np.mean(sf[dk])) if len(dk) else np.nan
    tz = np.where((depth >= 410) & (depth <= 660))[0]
    rows.append(dict(site=name, kind=kind, lat=hlat, lon=hlon,
                     root_km=root_p, gap_top_km=gap_p, deep_frac=deep_p,
                     root_km_fixed=root_f, deep_frac_fixed=deep_f,
                     tz_frac=float(np.mean(sp[tz])) if len(tz) else np.nan,
                     min_deep=float(np.nanmin(prof[dk])) if len(dk) else np.nan,
                     basal=float(np.nanmean(prof[dk])) if len(dk) else np.nan))


hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180'])
for _, r in hs.iterrows():
    measure(str(r['hotspot']), float(r['lat']), float(r['lon_180']), 'hotspot')
print(f'  {sum(1 for x in rows if x["kind"] == "hotspot")} hotspots done', flush=True)

rng = np.random.default_rng(51)                 # the null the rest of the study uses
nlo = 360 * rng.random(A.n_null) - 180
nla = np.degrees(np.arcsin(2 * rng.random(A.n_null) - 1))
for i, (a_, b_) in enumerate(zip(nla, nlo)):
    measure(f'null{i:03d}', float(a_), float(b_), 'null')
print(f'  {A.n_null} nulls done', flush=True)

df = pd.DataFrame(rows)
p = os.path.join(A.dir, f'root_depth_{A.tag}{A.suffix}.csv')
df.to_csv(p, index=False)
# Feeds mid_mantle_census.py, Figure 6a and Table S4; stamped since 22 Sep 2026.
import provenance
provenance.stamp(p, anchor=A.anchor, radius=A.radius, pct=A.pct, tol=A.tol,
                 deep_z0=A.deep_z0, deep_z1=A.deep_z1, n_null=A.n_null,
                 inputs=[A.file, A.hotspots])

h = df[df.kind == 'hotspot']
n = df[df.kind == 'null']


def band(v):
    if not np.isfinite(v):
        return 'none'
    if v >= A.deep_z0:
        return 'deep'
    if v >= 1000:
        return 'mid'
    return 'shallow'


for lab, d in (('hotspots', h), ('nulls', n)):
    b = d.root_km.map(band).value_counts()
    tot = len(d)
    print(f'{lab:9s} n={tot:4d}  median root '
          f'{np.nanmedian(d.root_km):7.0f} km   ' +
          '  '.join(f'{k} {b.get(k, 0):3d} ({100 * b.get(k, 0) / tot:4.1f}%)'
                    for k in ('deep', 'mid', 'shallow', 'none')), flush=True)

# A median printed beside another median is not a result. A site whose column
# never starts is given the anchor depth rather than dropped, because dropping it
# would remove the least-rooted sites from whichever group has more of them.
def perm(a, b, stat, n=20000, seed=11):
    rng = np.random.default_rng(seed)
    obs = stat(a) - stat(b)
    pool = np.concatenate([a, b])
    k = len(a)
    cnt = 0
    for _ in range(n):
        q = rng.permutation(pool)
        if abs(stat(q[:k]) - stat(q[k:])) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n + 1)


hr = h.root_km.fillna(A.anchor).to_numpy(float)
nr = n.root_km.fillna(A.anchor).to_numpy(float)
d_med, p_med = perm(hr, nr, np.median)
d_dp, p_dp = perm((hr >= A.deep_z0).astype(float), (nr >= A.deep_z0).astype(float), np.mean)
print(f'\nhotspots against nulls, {A.n_null} uniform sites, 20000 permutations:')
print(f'  median root depth   {np.median(hr):7.0f} vs {np.median(nr):7.0f} km   '
      f'difference {d_med:+.0f} km   P = {p_med:.4f}')
print(f'  fraction reaching {A.deep_z0:.0f} km   {100 * (hr >= A.deep_z0).mean():5.1f}% vs '
      f'{100 * (nr >= A.deep_z0).mean():5.1f}%      difference {100 * d_dp:+.1f} pts   P = {p_dp:.4f}')

# Hotspots overlying slow basal material will root deeper whether or not the
# column above it is organised, so the comparison is repeated against nulls whose
# basal anomaly lies inside the hotspot range. This is the same control the
# continuity result uses and the same reason for using it.
lo_, hi_ = np.nanpercentile(h.basal, [5, 95])
nm = n[(n.basal >= lo_) & (n.basal <= hi_)]
if len(nm) >= 30:
    mr = nm.root_km.fillna(A.anchor).to_numpy(float)
    d_m, p_m = perm(hr, mr, np.median)
    print(f'  against {len(nm)} basal-anomaly-matched nulls: '
          f'{np.median(hr):7.0f} vs {np.median(mr):7.0f} km   '
          f'difference {d_m:+.0f} km   P = {p_m:.4f}')
else:
    print(f'  only {len(nm)} basal-matched nulls, too few to test')

shal = h[(h.root_km < 1000) | (~np.isfinite(h.root_km))]
print(f'\nshallow-rooted hotspots ({len(shal)}), with what lies beneath them:')
for _, r in shal.sort_values('root_km').iterrows():
    print(f'  {r.site:16s} root {r.root_km:7.0f} km   gap opens {r.gap_top_km:7.0f} km   '
          f'basal window slow {100 * r.deep_frac:5.1f}%   mean basal anomaly {r.basal:+5.2f}%')
print(f'\nwrote {p}', flush=True)
