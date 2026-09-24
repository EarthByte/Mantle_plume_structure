"""The geometry of the connection beneath each hotspot, and how far down it holds.

Because the search recovers the path rather than prescribing it, the shape of a
connection is a measurement. Three questions can then be asked of every hotspot,
and this script answers all three from one pass.

  HOW DEEP does the connection remain distinguishable from the background? The
  search is repeated with the target surface at each depth in a ladder, and at
  each depth the hotspot is compared with the null obtained at that same depth.
  The tracked depth is the deepest level reached continuously from the top.

  HOW COMPLEX is it, where it exists? Tortuosity is the length of the recovered
  path divided by the vertical distance it spans, so a vertical cylinder scores
  1.0 and a conduit that ponds and steps sideways scores higher. The lateral
  offset and the depths at which the path moves sideways say where the departure
  from vertical happens - at the transition zone, near 1000 km, or deeper.

  WHICH HOTSPOTS have no deep connection at all? Those whose tracked depth stays
  in the upper mantle: not merely undetected at the base, but indistinguishable
  from an arbitrary location below shallow depths.

Nothing new is assumed. The comparison, the null and the path cost are those of
the classification; only the target depth varies.
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from contrast import contrast_field
from path_cost import cost_field, site_cost, trace_path
import path_config
warnings.filterwarnings('ignore')

def _hmax(c):
    """The horizontal reach of the chosen configuration, if the table records one."""
    v = getattr(c, 'h_max', None)
    return None if v is None or (isinstance(v, float) and v != v) else float(v)


ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--dir', default='out')
ap.add_argument('--ncell', type=int, default=1,
                help='lateral cells a single slanted move may cross; '
                     'slant_calibrate.py chose 2')
ap.add_argument('--lateral', type=float, default=1.0,
                help='weight on the lateral component of a move length; '
                     'slant_calibrate.py chose 0.60')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--ladder', default='400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800',
                help='target depths, km')
ap.add_argument('--n-null', type=int, default=400, dest='n_null')
ap.add_argument('--percentile-hotspots',
                default='Hawaii,Tahiti/Society,Macdonald (Cook-Austral),Iceland,Reunion,Kerguelen(Heard)',
                dest='pct_hotspots',
                help='comma-separated hotspot names to print a null-percentile trend for')
A = ap.parse_args()

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
print(f'{arr.shape[0]} shells, {arr.shape[1]}x{arr.shape[2]}', flush=True)

hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180']).reset_index(drop=True)
rng = np.random.default_rng(51)
nlo = 360 * rng.random(A.n_null) - 180
nla = np.degrees(np.arcsin(2 * rng.random(A.n_null) - 1))

ladder = [float(x) for x in A.ladder.split(',') if float(x) <= depth.max()]
below = {}                       # hotspot -> list of (depth, below_null?)
paths = {}
for zt in ladder:
    C = cost_field(arr, con, depth, lat, lon, s=float(c.s), z_target=zt,
                   n_relax=6, channel=str(c.channel), h_max_km=_hmax(c),
                   lateral=float(A.lateral), ncell=int(A.ncell))
    nul = np.array([site_cost(C, depth, lat, lon, la, lo, float(c.radius), 200.0)
                    for la, lo in zip(nla, nlo)], float)
    q = float(np.nanpercentile(nul, 5))
    for _, r in hs.iterrows():
        v = site_cost(C, depth, lat, lon, float(r.lat), float(r.lon_180),
                      float(c.radius), 200.0)
        pct = 100.0 * float(np.nanmean(nul < v)) if np.isfinite(v) else float('nan')
        below.setdefault(str(r.hotspot), []).append(
            (zt, bool(np.isfinite(v) and v < q), float(v), q, pct))
    print(f'  target {zt:5.0f} km: null p5 = {q:7.0f}, '
          f'{sum(1 for h in hs.hotspot if below[str(h)][-1][1]):2d} of {len(hs)} '
          f'below it', flush=True)
    del C

rows = []
pct_rows = []
for h, rec in below.items():
    zs = [zt for zt, _, _, _, _ in rec]
    ok = [o for _, o, _, _, _ in rec]
    vals = [v for _, _, v, _, _ in rec]
    qs = [q for _, _, _, q, _ in rec]
    pcs = [p for _, _, _, _, p in rec]
    deepest = max([z_ for z_, o in zip(zs, ok) if o], default=0.0)
    # The connection that matters is the one reaching the base, so the summary is
    # the shallowest depth from which the hotspot stays below its null all the way
    # down. Requiring the run to start at the top instead would fail every rooted
    # hotspot, because shallow slow structure is common everywhere and nothing is
    # exceptional in the upper few hundred kilometres.
    base_from = np.nan
    if ok[-1]:
        i = len(ok) - 1
        while i > 0 and ok[i - 1]:
            i -= 1
        base_from = zs[i]
    lost = np.nan
    if not ok[-1] and any(ok):
        lost = deepest
    rows.append(dict(hotspot=h, base_connected_from_km=base_from,
                     lost_below_km=lost, deepest_km=deepest,
                     n_levels=int(sum(ok)),
                     pattern=''.join('1' if o else '0' for o in ok),
                     **{f'z{int(z_)}': int(o) for z_, o in zip(zs, ok)}))
    for z_, v, q_, p in zip(zs, vals, qs, pcs):
        pct_rows.append(dict(hotspot=h, z_target_km=z_, value=v,
                             null_p5=q_, percentile=p))
out = pd.DataFrame(rows).sort_values(['deepest_km', 'n_levels'], ascending=False)
out.to_csv(os.path.join(A.dir, f'tracked_depth_{A.tag}.csv'), index=False)
pctdf = pd.DataFrame(pct_rows)
pctdf.to_csv(os.path.join(A.dir, f'depth_percentile_{A.tag}.csv'), index=False)

# the path each hotspot follows down to its own tracked depth
for _, r in out.iterrows():
    zt = float(r.deepest_km)
    if zt <= 0:
        continue
    if zt not in paths:
        paths[zt] = cost_field(arr, con, depth, lat, lon, s=float(c.s),
                               z_target=zt, n_relax=6, channel=str(c.channel),
                               h_max_km=_hmax(c),
                               lateral=float(A.lateral), ncell=int(A.ncell))
h2 = hs.set_index(hs.hotspot.astype(str))
geom = {}
for _, r in out.iterrows():
    zt = float(r.deepest_km)
    if zt <= 0 or zt not in paths:
        continue
    row = h2.loc[r.hotspot]
    tr = trace_path(paths[zt], arr, con, depth, lat, lon, float(row.lat),
                    float(row.lon_180), s=float(c.s), radius_deg=float(c.radius),
                    n_relax=6, channel=str(c.channel), h_max_km=_hmax(c))
    if tr is None:
        continue
    z, la_, lo_ = tr
    geom[str(r.hotspot)] = dict(depth=z.tolist(), lat=la_.tolist(),
                                lon=lo_.tolist(), deepest_km=zt,
                                base_from=(None if not np.isfinite(r.base_connected_from_km)
                                           else float(r.base_connected_from_km)))
json.dump(geom, open(os.path.join(A.dir, f'tracked_paths_{A.tag}.json'), 'w'))

# ---- shape of each recovered connection
R_E, DEG = 6371.0, np.pi / 180.0


def _gc(la1, lo1, la2, lo2):
    return R_E * np.arccos(np.clip(
        np.sin(la1 * DEG) * np.sin(la2 * DEG) +
        np.cos(la1 * DEG) * np.cos(la2 * DEG) * np.cos((lo2 - lo1) * DEG), -1, 1))


shape = []
for h, g in geom.items():
    z = np.asarray(g['depth'], float)
    la_ = np.asarray(g['lat'], float)
    lo_ = np.asarray(g['lon'], float)
    step_lat = _gc(la_[:-1], lo_[:-1], la_[1:], lo_[1:])
    step_ver = np.abs(np.diff(z))
    seg = np.sqrt(step_lat ** 2 + step_ver ** 2)
    span = float(z.max() - z.min())
    off = _gc(la_[0], lo_[0], la_, lo_)
    # where the lateral movement happens, by depth band
    bands = [(0, 410), (410, 660), (660, 1000), (1000, 1500), (1500, 2880)]
    zm = 0.5 * (z[:-1] + z[1:])
    lat_in = {f'lat_{lo:.0f}_{hi:.0f}': float(step_lat[(zm >= lo) & (zm < hi)].sum())
              for lo, hi in bands}
    # A path the search returns as vertical is vertical to the metre, and the
    # kilometre or so of travel left in it is the rounding of an arccos near 1,
    # not movement. Dividing that by itself yields a tidy-looking distribution
    # over depth - Tahiti and Cape Verde came out with the same 0.08/0.10/0.13/
    # 0.19/0.50 split, being the same noise, and it read as a real measurement.
    # Below a kilometre of total travel there is no distribution to report.
    tot = float(step_lat.sum())
    vertical = tot < 1.0
    shape.append(dict(
        hotspot=h, deepest_km=g['deepest_km'],
        tortuosity=float(seg.sum() / span) if span > 0 else np.nan,
        lateral_total_km=tot,
        max_offset_km=float(off.max()),
        max_offset_depth_km=float(z[int(np.argmax(off))]),
        mean_tilt_deg=float(np.degrees(np.arctan2(step_lat.sum(), span))),
        **lat_in,
        **{f'f{k[4:]}': (np.nan if vertical else round(v / tot, 3))
           for k, v in lat_in.items()}))
sh = pd.DataFrame(shape).sort_values('deepest_km', ascending=False)
sh.to_csv(os.path.join(A.dir, f'path_geometry_{A.tag}.csv'), index=False)

print()
print('geometry of each recovered connection: tortuosity 1.0 is a vertical')
print('cylinder; the last columns give the share of lateral movement by depth')
print(f"{'hotspot':26s} {'deepest':>8s} {'tort':>5s} {'offset':>7s} {'tilt':>5s}  "
      f"{'<410':>5s} {'410-660':>8s} {'660-1000':>9s} {'1000-1500':>10s} {'>1500':>6s}")
def _f(v, w):
    return ('-' if not np.isfinite(v) else f'{v:.2f}').rjust(w)


for _, r in sh.iterrows():
    print(f'  {str(r.hotspot)[:24]:26s} {r.deepest_km:6.0f}km {r.tortuosity:5.2f} '
          f'{r.max_offset_km:6.0f}km {r.mean_tilt_deg:4.0f}d  '
          f'{_f(r.f0_410, 5)} {_f(r.f410_660, 8)} {_f(r.f660_1000, 9)} '
          f'{_f(r.f1000_1500, 10)} {_f(r.f1500_2880, 6)}')

print()
print('Depth levels at which each hotspot is better connected than the null.')
print(f'ladder, km: ' + ' '.join(f'{int(z_)}' for z_ in ladder))
for _, r in out.iterrows():
    tail = (f'to the base from {r.base_connected_from_km:.0f} km'
            if np.isfinite(r.base_connected_from_km)
            else (f'lost below {r.lost_below_km:.0f} km'
                  if np.isfinite(r.lost_below_km) else 'no level'))
    print(f'  {str(r.hotspot)[:26]:28s} {r.pattern}  {tail}')
print(f'\nwrote tracked_depth_{A.tag}.csv, path_geometry_{A.tag}.csv, '
      f'tracked_paths_{A.tag}.json and depth_percentile_{A.tag}.csv')

ref = [x.strip() for x in A.pct_hotspots.split(',') if x.strip()]
print()
print('percentile within the null distribution at each target depth (lower means')
print('more anomalous than random locations there; the rooted threshold is 5).')
print(f"{'hotspot':26s} " + ' '.join(f'{int(z_):>5d}' for z_ in ladder))
for h in ref:
    rec = below.get(h)
    if rec is None:
        print(f'  {h}: not found (check --percentile-hotspots spelling)')
        continue
    pcs_ = [p for _, _, _, _, p in rec]
    print(f'  {h[:24]:26s} ' + ' '.join(f'{p:5.1f}' for p in pcs_))
