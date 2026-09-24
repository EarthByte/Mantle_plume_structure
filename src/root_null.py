"""Do least-cost paths from uniform starting points also finish deep inside a province?

Traced hotspot roots sit a median 613 km inside a province, which speaks against the
plume generation zone hypothesis placing origins at the margins and for the ridge
network of Hassan et al. (2015), whose ridges "extend into the interior". But a
least-cost path is drawn to slow material and will tend to finish inside a province for
reasons that have nothing to do with plumes. The inside/outside binary already has its
control - uniform starts finish inside 28.3 per cent of the time against 23.6 per cent
expected from area - but the DEPTH into the province has none, and that is the number
the claim rests on.

The machinery here is not a reimplementation. It reads the same calibrated
configuration from detection_<tag>.csv, builds the same contrast and cost fields with
the same parameters, and calls the same trace_path as paths.py. The seeds are the only
difference: uniform points on the sphere instead of hotspots.
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from contrast import contrast_field
from path_cost import cost_field, trace_path
import path_config
warnings.filterwarnings('ignore')

R_E, DEG = 6371.0, np.pi / 180.0

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--n-null', type=int, default=300, dest='n_null')
ap.add_argument('--seed', type=int, default=51, help='the null seed used elsewhere')
ap.add_argument('--province', default='out/province_vote.npz')
ap.add_argument('--min-families', type=int, default=3, dest='min_families')
ap.add_argument('--roots', default='out/plume_roots_RevealLO.csv')
# Defaulted to the CALIBRATED move set. These were 1 and 1.0, the superseded set,
# so every caller had to remember to override them and a runner that did not - mine -
# produced an hour of sweeps at a move set no result uses. A default that is never the
# right answer is a defect, not a neutral starting point.
ap.add_argument('--ncell', type=int, default=2,
                help='lateral cells a slanted move may cross; the calibrated value '
                     'is 2. The null MUST be traced with the same move set as the '
                     'hotspots or the comparison is between two different searches')
ap.add_argument('--lateral', type=float, default=0.60,
                help='weight on the lateral component of a move; calibrated 0.60')
ap.add_argument('--frozen-config', default=None, dest='frozen_config',
                help='unused; the configuration comes from path_config.py')
ap.add_argument('--dir', default='out')
A = ap.parse_args()


def _hmax(c):
    v = getattr(c, 'h_max', None)
    return None if v is None or (isinstance(v, float) and v != v) else float(v)


# The configuration is READ from out/path_config_<tag>.json, never re-derived.
# Re-deriving it as the median of the retained set makes it depend on which
# configurations happened to clear a 75 per cent floor on eighteen injections,
# and after the move-set retrace that median moved from s=0.4 z=2700 r=3.0 to
# s=0.3 z=2800 r=2.0. A null traced under one configuration and hotspots under
# another are two different searches and must not be compared.
c = path_config.load(A.tag, A.dir)

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
con = contrast_field(arr, lat, lon, 800.0)
C = cost_field(arr, con, depth, lat, lon, s=float(c.s), z_target=float(c.z_target),
               lateral=float(A.lateral), ncell=int(A.ncell),
               n_relax=6, channel=str(c.channel), h_max_km=_hmax(c))
_dcache = {}
print('cost field built', flush=True)

z = np.load(A.province)
vote, plat, plon = z['vote'], z['lat'], z['lon']
prov = vote >= A.min_families
PLO, PLA = np.meshgrid(plon, plat)
pin = np.stack([PLA[prov], ((PLO[prov] + 180) % 360) - 180])
pout = np.stack([PLA[~prov], ((PLO[~prov] + 180) % 360) - 180])


def gc(a, b, cc, d):
    return R_E * np.arccos(np.clip(np.sin(a * DEG) * np.sin(cc * DEG) +
                                   np.cos(a * DEG) * np.cos(cc * DEG) *
                                   np.cos((d - b) * DEG), -1, 1))


def inside(la, lo):
    j = int(np.argmin(np.abs(plat - la)))
    i = int(np.argmin(np.abs(((plon - lo + 180) % 360) - 180)))
    return bool(prov[j, i])


def margin(la, lo):
    return (-gc(la, lo, pout[0], pout[1]).min() if inside(la, lo)
            else gc(la, lo, pin[0], pin[1]).min())


rng = np.random.default_rng(A.seed)
rows = []
for i in range(A.n_null):
    nlo = float(360 * rng.random() - 180)
    nla = float(np.degrees(np.arcsin(2 * rng.random() - 1)))
    tr = trace_path(C, arr, con, depth, lat, lon, nla, nlo,
                    lateral=float(A.lateral), ncell=int(A.ncell),
                    s=float(c.s), radius_deg=float(c.radius), n_relax=6,
                    channel=str(c.channel), h_max_km=_hmax(c), cache=_dcache)
    if tr is None:
        continue
    zz, la, lo_ = tr
    k = int(np.argmax(np.asarray(zz, float)))
    rl, ro = float(np.asarray(la)[k]), float(np.asarray(lo_)[k])
    rows.append(dict(start_lat=nla, start_lon=nlo, root_lat=rl, root_lon=ro,
                     root_depth=float(np.asarray(zz)[k]),
                     root_in=inside(rl, ro), root_margin=margin(rl, ro)))
    if (i + 1) % 50 == 0:
        print(f'  {i + 1}/{A.n_null}', flush=True)

d = pd.DataFrame(rows)
d['lateral'] = float(A.lateral)
d['ncell'] = int(A.ncell)
d.to_csv(os.path.join(A.dir, f'root_null_{A.tag}.csv'), index=False)
h = pd.read_csv(A.roots)
hin, nin = h[h.root_in], d[d.root_in]
print(f'\n{len(d)} uniform starts traced, {len(h)} hotspots for comparison\n')
print(f'{"":>22s} {"hotspots":>10s} {"uniform":>10s}')
print(f'{"finish inside":>22s} {100 * h.root_in.mean():9.0f}% {100 * d.root_in.mean():9.0f}%')
print(f'{"median signed margin":>22s} {h.root_margin.median():+9.0f}  {d.root_margin.median():+9.0f}  km')
print(f'{"median depth inside":>22s} {-hin.root_margin.median():9.0f}  {-nin.root_margin.median():9.0f}  km'
      f'   (n {len(hin)} vs {len(nin)})')

if len(nin) >= 10 and len(hin) >= 10:
    a = (-hin.root_margin).to_numpy(float)
    b = (-nin.root_margin).to_numpy(float)
    rg = np.random.default_rng(3)
    obs = float(np.median(a) - np.median(b))
    pool = np.concatenate([a, b]); k = len(a); cnt = 0
    for _ in range(20000):
        q = rg.permutation(pool)
        if abs(np.median(q[:k]) - np.median(q[k:])) >= abs(obs) - 1e-12:
            cnt += 1
    P = (cnt + 1) / 20001
    print(f'\ndepth into the province, hotspots minus uniform: {obs:+.0f} km, P = {P:.4f}')
    print('If this is not significant the separation is the algorithm, not the hotspots,\nand no claim about rooting depth within a province can be made.')
