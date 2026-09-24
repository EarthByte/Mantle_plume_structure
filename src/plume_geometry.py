#!/usr/bin/env python3
"""Rebuild the geometry tables the manuscript quotes, from the traced paths.

WHY THIS EXISTS

plume_parameters_<tag>.csv, plume_roots_<tag>.csv and plume_slant_<tag>.csv carry every
offset, tilt, corridor width, province rooting and network membership the paper
reports, and no script in the repository produced them: they were written once by code
that is gone. A retrace would therefore have updated the paths and left these tables at
their old values, so the paper's numbers would have silently belonged to a different
run from its figures. This makes them reproducible.

THE MEASUREMENT WINDOW, RECOVERED RATHER THAN CHOSEN

The stored offsets are reproduced to a median 0.17 km, and the stored spans exactly for
all 49, by measuring from 400 km to the target depth. That is the window the paper's
numbers already use: it trims the top 200 km, where the seed cap rather than the mantle
sets where the path starts, and runs to the target. It is stated here because it was
not stated anywhere before, and an offset without its window is not a number anyone can
check.

Offset is the great-circle distance between the path at the two ends of that window.
Tilt is its arctangent over the depth spanned, so tilt and offset are one quantity.

  python3 plume_geometry.py --tag RevealLO
  python3 plume_geometry.py --tag RevealLO --verify   # reproduce, do not overwrite
"""
from __future__ import annotations
import argparse, json, math, os, sys
import numpy as np, pandas as pd
import provenance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inject import gc_km

HERE = os.path.dirname(os.path.abspath(__file__))
Z0, Z1 = 400.0, 2700.0

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--dir', default='out')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--province', default='out/province_vote.npz')
ap.add_argument('--min-families', type=int, default=3, dest='min_families')
ap.add_argument('--link', type=float, default=0.25)
ap.add_argument('--verify', action='store_true',
                help='compare with the tables on disk and write nothing')
A = ap.parse_args()

D = os.path.join(HERE, A.dir)
paths = json.load(open(os.path.join(D, f'conduit_paths_all_{A.tag}.json')))
hs = pd.read_csv(os.path.join(HERE, A.hotspots)).dropna(
    subset=['lat', 'lon_180']).set_index('hotspot')
zz = np.load(os.path.join(HERE, A.province))
vote, plat, plon = zz['vote'], zz['lat'], zz['lon']
prov = vote >= A.min_families


def province(la, lo):
    j = int(np.argmin(np.abs(plat - la)))
    i = int(np.argmin(np.abs(((plon - lo + 180) % 360) - 180)))
    return bool(prov[j, i])


_J, _I = np.meshgrid(plat, plon, indexing='ij')


def margin_km(la, lo):
    """Signed distance to the nearest cell of opposite province membership.

    Negative inside. Measured to the nearest cell that differs rather than to a
    boundary traced between cells, which is what the stored table did and is the only
    definition the grid actually supports.
    """
    inside = province(la, lo)
    other = (~prov) if inside else prov
    if not other.any():
        return float('nan')
    d = float(np.min(gc_km(la, lo, _J[other], _I[other])))
    return -d if inside else d


def slant(q):
    z = np.asarray(q['depth'], float)
    la = np.asarray(q['lat'], float); lo = np.asarray(q['lon'], float)
    o = np.argsort(z); z, la, lo = z[o], la[o], lo[o]
    m = (z >= Z0) & (z <= Z1)
    if m.sum() < 2:
        return float('nan'), float('nan'), float('nan')
    off = float(gc_km(la[m][0], lo[m][0], la[m][-1], lo[m][-1]))
    span = float(z[m][-1] - z[m][0])
    return off, span, math.degrees(math.atan2(off, span))


# basin is a per-hotspot label that no script assigns and that tracing cannot change;
# it was recovered once from the table that had it and now lives in its own file.
basin = pd.read_csv(os.path.join(HERE, 'data', 'hotspot_basin.csv')
                    ).set_index('site')['basin'].to_dict()
# distance to the nearest spreading centre OF ANY AGE, active or extinct, from
# corridor_ridge_association.py; this is not a present-day ridge distance and the
# difference is large - 2728 km against 460 km for Tahiti.
_ra = os.path.join(D, f'corridor_ridge_association_{A.tag}.csv')
dring = (pd.read_csv(_ra).set_index('site')['d_any_ridge'].to_dict()
         if os.path.exists(_ra) else {})

cs = pd.read_csv(os.path.join(D, f'corridor_summary_{A.tag}.csv'))
prof = pd.read_csv(os.path.join(D, f'corridor_profiles_{A.tag}.csv'))
# One rule, decided where the corridors are built. A site whose corridor does not
# span the column has no width at any depth, and a band median taken over the one
# or two shells it does reach is not the same quantity as one taken over a hundred.
# Pooling them is what put a fill value into the paper as a result.
_wd = [c for c in cs.columns if c.startswith('width_defined_')]
NO_WIDTH = (set(cs.loc[~cs[_wd[0]].astype(bool), 'site'].astype(str)) if _wd
            else set())
if NO_WIDTH:
    print(f'{len(NO_WIDTH)} sites have no corridor to measure and carry no width: '
          f'{", ".join(sorted(NO_WIDTH))}')
ov = pd.read_csv(os.path.join(D, f'corridor_overlap_{A.tag}.csv'))
link = ov[ov.frac_small >= A.link]
deg = {}
for _, r in link.iterrows():
    deg[r.a] = deg.get(r.a, 0) + 1
    deg[r.b] = deg.get(r.b, 0) + 1

BANDS = (('w_shallow', 200.0, 660.0), ('w_mid', 660.0, 1500.0),
         ('w_deep', 1500.0, 2700.0))
rows, roots = [], []
for site, q in paths.items():
    if site not in hs.index:
        continue
    z = np.asarray(q['depth'], float)
    la = np.asarray(q['lat'], float); lo = np.asarray(q['lon'], float)
    o = np.argsort(z); z, la, lo = z[o], la[o], lo[o]
    off, span, tilt = slant(q)
    rlat, rlon, rdep = float(la[-1]), float(lo[-1]), float(z[-1])
    rin, rmar = province(rlat, rlon), margin_km(rlat, rlon)
    slat, slon = float(hs.at[site, 'lat']), float(hs.at[site, 'lon_180'])
    g = prof[prof.site == site] if 'site' in prof.columns else prof.iloc[0:0]
    w = {}
    _has_width = site not in NO_WIDTH
    for name, a, b in BANDS:
        sel = g[(g.depth >= a) & (g.depth < b)] if _has_width else g.iloc[0:0]
        # Only shells the corridor actually reaches carry a width. Older profile
        # files wrote 0.0 for an empty shell rather than NaN, and a median taken
        # over those returns 0 km for a corridor that is merely narrow - which is
        # how six sites acquired a deep corridor width of exactly zero. Both
        # spellings are excluded here, so this reads old and new files alike.
        v = pd.to_numeric(sel.width_km, errors='coerce')
        v = v[np.isfinite(v) & (v > 0)]
        w[name] = float(v.median()) if len(v) else float('nan')
        w[f'n_{name}'] = int(len(v))
    w['width_defined'] = _has_width
    rows.append(dict(site=site, **w, root_depth=rdep, root_in=rin,
                     root_margin=rmar, tilt=tilt, offset=off,
                     basin=basin.get(site, 'other'), degree=deg.get(site, 0),
                     linked=deg.get(site, 0) > 0,
                     d_any_ridge=dring.get(site, float('nan'))))
    roots.append(dict(site=site, root_depth=rdep, root_lat=rlat, root_lon=rlon,
                      root_in=rin, root_margin=rmar, surf_in=province(slat, slon),
                      surf_margin=margin_km(slat, slon)))

par = pd.DataFrame(rows).set_index('site').sort_index()
rt = pd.DataFrame(roots).set_index('site').sort_index()
sl = pd.DataFrame([dict(site=s, span=slant(q)[1], offset=slant(q)[0],
                        tilt=slant(q)[2]) for s, q in paths.items()
                   if s in hs.index]).set_index('site').sort_index()

if A.verify:
    old = pd.read_csv(os.path.join(D, f'plume_slant_{A.tag}.csv')).set_index('site')
    j = sl.join(old, rsuffix='_old', how='inner')
    for c in ('offset', 'tilt', 'span'):
        d = (j[c] - j[f'{c}_old']).abs()
        print(f'  {c:8s} max difference {d.max():.4f}, median {d.median():.4f}')
    oldp = pd.read_csv(os.path.join(D, f'plume_parameters_{A.tag}.csv')).set_index('site')
    jp = par.join(oldp, rsuffix='_old', how='inner')
    print(f'  basin agrees for '
          f'{int((jp.basin == jp.basin_old).sum())} of {len(jp)}')
    for c in ('w_shallow', 'w_mid', 'w_deep', 'root_margin', 'degree', 'd_any_ridge'):
        if f'{c}_old' in jp.columns:
            d = (pd.to_numeric(jp[c], errors='coerce')
                 - pd.to_numeric(jp[f'{c}_old'], errors='coerce')).abs()
            print(f'  {c:12s} max difference {d.max():.4f}')
    print(f'  root_in agrees for {int((jp.root_in == jp.root_in_old).sum())} of {len(jp)}')
    raise SystemExit(0)

for name, d in ((f'plume_slant_{A.tag}.csv', sl),
                (f'plume_roots_{A.tag}.csv', rt),
                (f'plume_parameters_{A.tag}.csv', par)):
    _p = os.path.join(D, name)
    d.to_csv(_p)
    # all three, not just the one a consumer happened to need: plume_roots_<tag>.csv
    # carried no provenance and the audit reads it for the province comparison
    provenance.stamp(_p, inputs=[os.path.join(D, f'conduit_paths_all_{A.tag}.json')])
    print(f'wrote {name}: {len(d)} rows')
