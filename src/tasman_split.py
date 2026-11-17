#!/usr/bin/env python3
"""Do the Tasmantid and Lord Howe chains share one deep conduit that splits above a slab?

Mather et al. (2026, Gondwana Research 160, 99-108) propose that the two parallel
seamount chains of the Tasman Sea are fed by one lower-mantle plume that is split in
two by a stagnant slab in the upper mantle, the two branches rising through windows
on either side of it. Three things follow that a tomographic model can be asked:

  ONE DEEP SOURCE   the descents traced from the two hotspots join at depth and share
                    one column below the split.
  A SLAB BETWEEN    material between the two branches at the depth of the split is
                    faster than the branches themselves.
  NOT GENERIC       two descents started this far apart in this region do not join
                    anyway. The least-cost operator funnels nearby starts into the
                    slowest deep column whether or not a plume is there, so a join
                    is evidence only if it is rarer from other start pairs.

The descents are traced exactly as paths.py traces every hotspot (the frozen
configuration of the model, the same move set), but on a regional crop so that a
grid of starting points can be traced on one cost field. The two hotspot descents
are traced again on the crop and compared with the global ones in
conduit_paths_all_<tag>.json; a crop that changed them would not be the same search.

  python3 tasman_split.py --file <model.nc> --tag RevealLO --var voigt \
      --every 1 --depth-max 2880 --lateral 0.60 --ncell 2
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from inject import gc_km
from path_cost import cost_field, trace_path
from contrast import contrast_field
import path_config
from paths import _hmax

HOT = {'Lord Howe': 'Lord Howe (Tasman East)', 'Tasmantid': 'Tasmanid (Tasman central)'}
MERGE_KM = 150.0          # two descents closer than this share a column
CENTRE = (-36.0, 158.0)

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--lateral', type=float, default=0.60)
ap.add_argument('--ncell', type=int, default=2)
ap.add_argument('--crop-lat', type=float, default=34.0, dest='crop_lat',
                help='half-height of the regional crop, degrees')
ap.add_argument('--crop-lon', type=float, default=46.0, dest='crop_lon',
                help='half-width of the regional crop, degrees of longitude')
ap.add_argument('--grid', type=float, default=1.5,
                help='spacing of the regional start grid, degrees')
ap.add_argument('--grid-box', default='-47,-15,145,173', dest='grid_box',
                help='lat0,lat1,lon0,lon1 of the start grid')
ap.add_argument('--dir', default='out')
A = ap.parse_args()
t0 = time.time()

c = path_config.load(A.tag, A.dir)
hm = _hmax(c)
depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
lat = np.asarray(lat, float); lon = np.asarray(lon, float); z = np.asarray(depth, float)
if lat[0] > lat[-1]:
    lat = lat[::-1]; arr = arr[:, ::-1, :]
jm = np.where(np.abs(lat - CENTRE[0]) <= A.crop_lat)[0]
dl = ((lon - CENTRE[1] + 180.0) % 360.0) - 180.0
im = np.where(np.abs(dl) <= A.crop_lon)[0]
im = im[np.argsort(dl[im])]
sub = np.ascontiguousarray(arr[:, jm[:, None], im[None, :]], dtype=np.float64)
sla, slo = lat[jm], ((lon[im] + 360.0) % 360.0)          # 0-360 across the dateline
del arr
print(f'{A.tag}: crop {sub.shape}, lat {sla[0]:.1f}..{sla[-1]:.1f}, '
      f'lon {slo[0]:.1f}..{slo[-1]:.1f}  ({time.time()-t0:.0f} s)', flush=True)

con = contrast_field(sub, sla, slo, 800.0)
C = cost_field(sub, con, z, sla, slo, s=float(c.s), z_target=float(c.z_target),
               n_relax=6, channel=str(c.channel), h_max_km=hm,
               lateral=A.lateral, ncell=A.ncell)
print(f'  cost field built ({time.time()-t0:.0f} s)', flush=True)
cache = {}


def trace(la, lo):
    tr = trace_path(C, sub, con, z, sla, slo, la, lo % 360.0, s=float(c.s),
                    radius_deg=float(c.radius), n_relax=6, channel=str(c.channel),
                    h_max_km=hm, lateral=A.lateral, ncell=A.ncell, cache=cache)
    if tr is None:
        return None
    pz, pla, plo = (np.asarray(x, float) for x in tr)
    return pz, pla, ((plo + 180.0) % 360.0) - 180.0


ZG = np.arange(200.0, float(c.z_target) + 1.0, 10.0)


def on_grid(p):
    """Position of a path at each grid depth: the last node at or above it."""
    pz, pla, plo = p
    k = np.searchsorted(pz, ZG, side='right') - 1
    k = np.clip(k, 0, len(pz) - 1)
    return pla[k], plo[k]


def separation(p, q):
    a, b = on_grid(p), on_grid(q)
    return gc_km(a[0], a[1], b[0], b[1])


def merge_depth(sep):
    """Shallowest depth from which the two stay within MERGE_KM to the target."""
    close = sep <= MERGE_KM
    if not close[-1]:
        return float('nan')
    k = len(close) - 1
    while k > 0 and close[k - 1]:
        k -= 1
    return float(ZG[k])


hs = pd.read_csv('hotspots_courtillot2003.csv').set_index('hotspot')
glob_paths = json.load(open(os.path.join(A.dir, f'conduit_paths_all_{A.tag}.json')))
P, check = {}, {}
for short, name in HOT.items():
    r = hs.loc[name]
    P[short] = trace(float(r.lat), float(r.lon_180))
    g = glob_paths[name]
    gp = (np.asarray(g['depth'], float), np.asarray(g['lat'], float),
          np.asarray(g['lon'], float))
    check[short] = float(np.max(separation(P[short], gp)))
    print(f'  {short}: regional trace departs from the global one by at most '
          f'{check[short]:.0f} km', flush=True)

sep_h = separation(P['Lord Howe'], P['Tasmantid'])
md_h = merge_depth(sep_h)
start_sep = float(gc_km(hs.loc[HOT['Lord Howe']].lat, hs.loc[HOT['Lord Howe']].lon_180,
                        hs.loc[HOT['Tasmantid']].lat, hs.loc[HOT['Tasmantid']].lon_180))

# regional start grid
la0, la1, lo0, lo1 = (float(x) for x in A.grid_box.split(','))
starts = [(a, b) for a in np.arange(la0, la1 + 1e-6, A.grid)
          for b in np.arange(lo0, lo1 + 1e-6, A.grid)]
G = {}
for (a, b) in starts:
    p = trace(a, b)
    if p is not None:
        G[(round(a, 3), round(b, 3))] = p
print(f'  {len(G)} of {len(starts)} grid starts traced ({time.time()-t0:.0f} s)',
      flush=True)

keys = list(G)
pos = {k: on_grid(G[k]) for k in keys}
rows, null_sep = [], []
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        d0 = float(gc_km(keys[i][0], keys[i][1], keys[j][0], keys[j][1]))
        if abs(d0 - start_sep) > 150.0:
            continue
        a, b = pos[keys[i]], pos[keys[j]]
        s_ = gc_km(a[0], a[1], b[0], b[1])
        null_sep.append(s_)
        rows.append(dict(lat1=keys[i][0], lon1=keys[i][1], lat2=keys[j][0],
                         lon2=keys[j][1], start_km=d0, merge_depth=merge_depth(s_),
                         sep_2700=float(s_[-1]), min_sep=float(np.min(s_))))
null = pd.DataFrame(rows)
null_pct = (np.percentile(np.array(null_sep), [10, 25, 50, 75, 90], axis=0)
            if null_sep else np.full((5, len(ZG)), np.nan))

# Near-optimal corridors, where corridor_all.py has written them: the share of the
# smaller corridor that the two hotspots have in common, by depth band, and the same
# for every other pair of hotspots within 1500 km as the reference. Corridors widen
# with depth, so nearby hotspots tend to share the lower mantle whatever feeds them.
BANDS = ((200, 400), (400, 660), (660, 1000), (1000, 1500), (1500, 2000),
         (2000, 2500), (2500, 2900))
corr = None
_mf = os.path.join(A.dir, f'corridor_masks_{A.tag}.npz')
if os.path.exists(_mf):
    _z = np.load(_mf, allow_pickle=True)
    _d = _z['depth']
    _hh = hs.dropna(subset=['lat', 'lon_180'])
    _nm = [k for k in _z.files if k in _hh.index]

    def _share(a, b, lo, hi):
        m = (_d >= lo) & (_d < hi)
        x, y = _z[a][m], _z[b][m]
        s = min(int(x.sum()), int(y.sum()))
        return float((x & y).sum() / s) if s else float('nan')

    pair = {f'{lo}-{hi}': _share(HOT['Lord Howe'], HOT['Tasmantid'], lo, hi)
            for lo, hi in BANDS} if all(h in _nm for h in HOT.values()) else None
    ref = []
    for i_, a_ in enumerate(_nm):
        for b_ in _nm[i_ + 1:]:
            if {a_, b_} == set(HOT.values()):
                continue
            d_ = float(gc_km(_hh.loc[a_].lat, _hh.loc[a_].lon_180,
                             _hh.loc[b_].lat, _hh.loc[b_].lon_180))
            if d_ <= 1500.0:
                ref.append(dict(a=a_, b=b_, km=d_, upper=_share(a_, b_, 200, 1000),
                                lower=_share(a_, b_, 1000, 2700)))
    corr = dict(pair=pair, pair_upper=_share(HOT['Lord Howe'], HOT['Tasmantid'], 200, 1000),
                pair_lower=_share(HOT['Lord Howe'], HOT['Tasmantid'], 1000, 2700),
                reference=ref, tau=float(_z['tau']))
f_merge = float(null.merge_depth.notna().mean()) if len(null) else float('nan')
f_as_shallow = (float((null.merge_depth <= md_h).mean())
                if len(null) and np.isfinite(md_h) else float('nan'))

# the anomaly at the two branches and between them, through the upper mantle and
# the mid mantle: the slab of the hypothesis is faster material between them
def val(zq, la, lo):
    k = int(np.argmin(np.abs(z - zq))); j = int(np.argmin(np.abs(sla - la)))
    i = int(np.argmin(np.abs(((slo - (lo % 360.0) + 180.0) % 360.0) - 180.0)))
    return float(sub[k, j, i])

prof = []
gl, gt = on_grid(P['Lord Howe']), on_grid(P['Tasmantid'])
for zq in (300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000, 2500):
    k = int(np.argmin(np.abs(ZG - zq)))
    al, ol, at, ot = gl[0][k], gl[1][k], gt[0][k], gt[1][k]
    # great-circle midpoint
    v1 = np.array([math.cos(math.radians(al)) * math.cos(math.radians(ol)),
                   math.cos(math.radians(al)) * math.sin(math.radians(ol)),
                   math.sin(math.radians(al))])
    v2 = np.array([math.cos(math.radians(at)) * math.cos(math.radians(ot)),
                   math.cos(math.radians(at)) * math.sin(math.radians(ot)),
                   math.sin(math.radians(at))])
    m = v1 + v2; m /= np.linalg.norm(m)
    am, om = math.degrees(math.asin(m[2])), math.degrees(math.atan2(m[1], m[0]))
    kk = int(np.argmin(np.abs(z - zq)))
    shell = sub[kk][np.abs(sla - CENTRE[0]) <= 20.0]
    vm = val(zq, am, om)
    prof.append(dict(depth=zq, sep_km=float(sep_h[k]),
                     lordhowe=val(zq, al, ol), tasmantid=val(zq, at, ot), between=vm,
                     between_pct=float((shell < vm).mean())))
prof = pd.DataFrame(prof)

# Which branch is the stronger feeder now? Mather et al. predict the Tasmantid branch
# strengthening and the Lord Howe branch waning. The tomographic proxy is the anomaly
# beneath each present hotspot, a 250 km cap, by depth band. Above about 400 km it is
# confounded by the lithosphere: Lord Howe sits on the stretched continental Lord Howe
# Rise and the Tasmantid hotspot on Tasman Sea oceanic lithosphere.
_LA, _LO = np.meshgrid(sla, slo, indexing='ij')
caps = {}
for short, name in HOT.items():
    r = hs.loc[name]
    _c = gc_km(float(r.lat), float(r.lon_180), _LA, ((_LO + 180.0) % 360.0) - 180.0) <= 250.0
    caps[short] = {f'{z0}-{z1}': float(np.nanmean(sub[(z >= z0) & (z < z1)][:, _c]))
                   for z0, z1 in ((200, 400), (400, 660), (660, 1000))}

out_json = os.path.join(A.dir, f'tasman_split_{A.tag}.json')
summary = dict(
    tag=A.tag, config=dict(s=float(c.s), z_target=float(c.z_target),
                           channel=str(c.channel), radius=float(c.radius), h_max=hm,
                           lateral=A.lateral, ncell=A.ncell),
    merge_km=MERGE_KM, start_separation_km=start_sep,
    regional_vs_global_max_km=check,
    merge_depth_km=md_h, separation_at_depth={f'{int(zq)}': float(sep_h[np.argmin(np.abs(ZG - zq))])
                                             for zq in (200, 400, 660, 1000, 1500, 2000, 2500, ZG[-1])},
    null_pairs=int(len(null)), null_sep_depth=ZG.tolist(),
    null_sep_pct={str(p): null_pct[i].tolist() for i, p in enumerate((10, 25, 50, 75, 90))},
    pair_sep=sep_h.tolist(), corridor=corr, null_fraction_merged=f_merge,
    null_fraction_merged_as_shallow=f_as_shallow,
    null_median_merge_depth=float(null.merge_depth.median()) if len(null) else float('nan'),
    profile=prof.to_dict(orient='records'), caps=caps,
    paths={k: dict(depth=v[0].tolist(), lat=v[1].tolist(), lon=v[2].tolist())
           for k, v in P.items()},
    grid_paths={f'{k[0]},{k[1]}': dict(depth=v[0].tolist(), lat=v[1].tolist(),
                                       lon=v[2].tolist()) for k, v in G.items()})
json.dump(summary, open(out_json, 'w'))
null.to_csv(os.path.join(A.dir, f'tasman_split_null_{A.tag}.csv'), index=False)

# The regional volume the figure is drawn from, so that drawing it does not mean
# reading the model file again: 55-5 S, 130-190 E, every shell, float32.
_j = (sla >= -55.0) & (sla <= -5.0)
_i = (slo >= 130.0) & (slo <= 190.0)
np.savez_compressed(os.path.join(A.dir, f'tasman_split_volume_{A.tag}.npz'),
                    depth=z, lat=sla[_j], lon=slo[_i],
                    anom=sub[:, _j][:, :, _i].astype(np.float32))

print(f'\n{A.tag}: hotspots {start_sep:.0f} km apart at the surface')
print('  separation of the two descents:  ' + '  '.join(
    f'{int(zq)} km {sep_h[np.argmin(np.abs(ZG - zq))]:.0f}' for zq in (200, 400, 660, 1000, 1500, 2000, 2500)))
print(f'  shared column (within {MERGE_KM:.0f} km) from: '
      + (f'{md_h:.0f} km' if np.isfinite(md_h) else 'never'))
print(f'  regional start pairs at the same separation: {len(null)}; joined: '
      f'{f_merge:.2f}; joined at or above the hotspot depth: {f_as_shallow:.2f}')
print(prof.round(3).to_string(index=False))
if corr is not None and corr['pair'] is not None:
    _r = pd.DataFrame(corr['reference'])
    print(f"  corridors shared, 200-1000 km {corr['pair_upper']:.2f}, 1000-2700 km "
          f"{corr['pair_lower']:.2f}; other hotspot pairs within 1500 km "
          f"(n={len(_r)}): median {_r.upper.median():.2f} and {_r.lower.median():.2f}")
print('  beneath each hotspot (250 km cap):  ' + '   '.join(
    f'{h} ' + ' '.join(f'{b} {v:+.2f}' for b, v in caps[h].items()) for h in caps))
print(f'wrote {out_json}  ({time.time()-t0:.0f} s)')
