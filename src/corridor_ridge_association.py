"""Do the hotspots whose corridors link sit near spreading centres?

EXPLORATORY. This hypothesis was not pre-registered. It came from reading the list
of linked pairs and noticing that most of them sit on or beside a spreading centre:
Iceland and Jan Mayen and the Azores on the Mid-Atlantic Ridge, Bouvet, Discovery
and Meteor around the Bouvet triple junction, Crozet and Kerguelen on the Southwest
Indian Ridge, Bowie and Juan de Fuca on the Juan de Fuca system, and Lord Howe with
Tasmanid and Baja with Raton over extinct ones. A pattern found by looking cannot be
tested on the data it was found in, so nothing here is a confirmed result. Its
purpose is to fix the measurement and the decision rule so the claim can be put to
REVEAL and GLAD-M35, whose corridors have not been computed, as a genuine test.

Extinct ridge geometries are MacLeod et al. (2017), supplementary file S1: 129
primary-tier segments and 66 secondary-tier. The excluded tier is not used. Their
active-ridge file is NOT used: it holds the 93 segments they selected for their
morphological comparison and contains no Atlantic segment north of 40 degrees,
which places the Azores 7344 km from the nearest "active ridge" while it sits on
the Mid-Atlantic Ridge. Present-day ridges come from the Zahirovic 2022
topologies via ridges_presentday.py.

The measure carried forward for confirmation is distance to the nearest spreading
centre OF ANY AGE, active or extinct, one measure and one test, chosen because it
is what the visual pattern actually was and because choosing among four after
seeing four is how a P of 0.04 gets manufactured.

Two comparisons are reported. The first takes each hotspot as an observation, which
overstates the evidence because linked hotspots arrive in geographically coherent
clusters and are not independent draws. The second takes each connected component as
a single observation against the isolated hotspots individually, which is
conservative and is the number to believe.
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np, pandas as pd
import geopandas as gpd
from scipy.stats import mannwhitneyu

R_E, DEG = 6371.0, np.pi / 180.0
SHP = 'data/S2_Shapefiles_Extinct_and_active_ridge_segments'

ap = argparse.ArgumentParser()
ap.add_argument('--overlap', default=None)
ap.add_argument('--summary', default=None)
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--link', type=float, default=0.25)
ap.add_argument('--step-km', type=float, default=25.0, dest='step_km')
ap.add_argument('--tag', default='RevealLO',
                help='names the outputs so models do not overwrite each other')
ap.add_argument('--dir', default='out')
A = ap.parse_args()
A.overlap = A.overlap or os.path.join(A.dir, f'corridor_overlap_{A.tag}.csv')
A.summary = A.summary or os.path.join(A.dir, f'corridor_summary_{A.tag}.csv')


def densify(gdf, step_km):
    """Points along every segment, about step_km apart, as lat and lon arrays."""
    la, lo = [], []
    for geom in gdf.geometry:
        if geom is None:
            continue
        parts = geom.geoms if geom.geom_type.startswith('Multi') else [geom]
        for p in parts:
            xy = np.asarray(p.coords, float)
            if len(xy) < 2:
                la.append(xy[:, 1]); lo.append(xy[:, 0]); continue
            for i in range(len(xy) - 1):
                (x0, y0), (x1, y1) = xy[i], xy[i + 1]
                d = R_E * DEG * np.hypot((x1 - x0) * np.cos(0.5 * (y0 + y1) * DEG),
                                         y1 - y0)
                n = max(int(np.ceil(d / step_km)), 1)
                t = np.linspace(0, 1, n + 1)
                lo.append(x0 + t * (x1 - x0)); la.append(y0 + t * (y1 - y0))
    return np.concatenate(la), ((np.concatenate(lo) + 180) % 360) - 180


def gc(a, b, c, d):
    return R_E * np.arccos(np.clip(np.sin(a * DEG) * np.sin(c * DEG) +
                                   np.cos(a * DEG) * np.cos(c * DEG) *
                                   np.cos((d - b) * DEG), -1, 1))


# MacLeod's active-ridge shapefile is the 93 segments they selected for their
# morphological comparison, not a global ridge map: no Atlantic segment north of
# 40 degrees, which puts the Azores 7344 km from the nearest "active ridge" while
# it sits on the Mid-Atlantic Ridge. Present-day ridges come from the Zahirovic
# 2022 topologies instead, via ridges_presentday.py. The extinct catalogue is
# their actual product and is used as it stands.
sets = {}
_act = 'data/ridges_presentday_zahirovic2022.csv'
if not os.path.exists(_act):
    raise SystemExit(f'missing {_act}; run ridges_presentday.py first')
_a = pd.read_csv(_act)
sets['active'] = (_a.lat.to_numpy(float), _a.lon.to_numpy(float))
print(f'{"active":<18s} Zahirovic 2022 present-day -> {len(_a):6d} points')
for key, pat in (('extinct_primary', 'MacLeod_et_al_2017_Primary_Tier_extinct_ridges.shp'),
                 ('extinct_secondary', 'MacLeod_et_al_2017_Secondary_Tier_extinct_ridges.shp')):
    f = os.path.join(SHP, pat)
    if not os.path.exists(f):
        raise SystemExit(f'missing {f}; extract supplementary file S1 into data/')
    g = gpd.read_file(f)
    la, lo = densify(g, A.step_km)
    sets[key] = (la, lo)
    print(f'{key:<18s} {len(g):4d} segments -> {len(la):6d} points   '
          f'lat {la.min():+.1f} to {la.max():+.1f}, lon {lo.min():+.1f} to {lo.max():+.1f}')
if abs(sets['extinct_primary'][0]).max() > 90.5:
    raise SystemExit('shapefile coordinates are not longitude and latitude')
_az = float(gc(39.0, -28.0, *sets['active'])) if False else None
sets['extinct_any'] = (np.concatenate([sets['extinct_primary'][0], sets['extinct_secondary'][0]]),
                       np.concatenate([sets['extinct_primary'][1], sets['extinct_secondary'][1]]))
sets['any_ridge'] = (np.concatenate([sets['active'][0], sets['extinct_any'][0]]),
                     np.concatenate([sets['active'][1], sets['extinct_any'][1]]))

sm = pd.read_csv(A.summary)
sm = sm[sm.ok == True]
hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180'])
P = {str(r.hotspot): (float(r.lat), float(r.lon_180)) for _, r in hs.iterrows()}
names = [n for n in sm.site if n in P]

ov = pd.read_csv(A.overlap)
link = ov[ov.frac_small >= A.link]
deg = {n: 0 for n in names}
parent = {n: n for n in names}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for _, r in link.iterrows():
    if r.a in deg and r.b in deg:
        deg[r.a] += 1; deg[r.b] += 1
        ra, rb = find(r.a), find(r.b)
        if ra != rb:
            parent[ra] = rb
comp = {}
for n in names:
    comp.setdefault(find(n), []).append(n)

rows = []
for n in names:
    la, lo = P[n]
    rec = dict(site=n, lat=la, lon=lo, degree=deg[n],
               linked=deg[n] > 0, component=find(n),
               comp_size=len(comp[find(n)]))
    for k, (rla, rlo) in sets.items():
        rec[f'd_{k}'] = float(gc(la, lo, rla, rlo).min())
    rows.append(rec)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(A.dir, f'corridor_ridge_association_{A.tag}.csv'), index=False)

L, I = d[d.linked], d[~d.linked]
print(f'\n{len(L)} linked hotspots, {len(I)} isolated, '
      f'{len([v for v in comp.values() if len(v) > 1])} components\n')
print(f'{"distance to":<22s}{"linked":>10s}{"isolated":>11s}{"per hotspot":>14s}'
      f'{"per component":>16s}')
for k in ('active', 'extinct_primary', 'extinct_any', 'any_ridge'):
    col = f'd_{k}'
    u, p1 = mannwhitneyu(L[col], I[col], alternative='less')
    cm = [float(d[d.component == q][col].median()) for q, v in comp.items() if len(v) > 1]
    u2, p2 = mannwhitneyu(cm, I[col], alternative='less')
    print(f'{k:<22s}{L[col].median():9.0f}k{I[col].median():10.0f}k'
          f'{p1:14.4f}{p2:16.4f}')

print('\nper component, median distance to the nearest ridge of any age:')
for q, v in sorted(comp.items(), key=lambda t: -len(t[1])):
    if len(v) > 1:
        g = d[d.component == q]
        print(f'  {", ".join(sorted(v)):<58s}{g.d_any_ridge.median():6.0f} km')
print('\nisolated hotspots furthest from any ridge:')
for _, r in I.nlargest(6, 'd_any_ridge').iterrows():
    print(f'  {r.site:<30s}{r.d_any_ridge:6.0f} km')
print('\nEXPLORATORY. The hypothesis was read off this list, so these numbers cannot')
print('confirm it. The test is REVEAL and GLAD-M35, with this rule fixed beforehand:')
print('linked hotspots sit closer to spreading centres than isolated ones, judged')
print('per component, one-sided, at P < 0.05.')
