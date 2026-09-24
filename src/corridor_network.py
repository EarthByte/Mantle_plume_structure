"""Isolated conduits or interconnected networks: the end-member question, measured.

Bao et al. emphasise networks of interconnected plume-like structures. They worked
East Africa and the Indian Ocean, where the African province dominates and
interconnection is most likely, so a global survey should recover the whole spectrum
with their region at one end of it rather than as the general case.

Two hotspots are connected when their near-optimal corridors share mantle: the set of
routes the tomography cannot distinguish from optimal for one overlaps the set for the
other, so the data do not separate the two structures. That is a statement about what
the model can resolve, not an assumed physical link, which is the right footing for a
claim about whether distinct conduits exist.

Overlap is reported as the fraction of the SMALLER corridor that is shared. Jaccard
would penalise a small conduit sitting inside a large one, which is exactly the case
of a distinct plume embedded in a province-wide structure - the configuration the
argument is about.
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd

R_E, DEG = 6371.0, np.pi / 180.0

ap = argparse.ArgumentParser()
ap.add_argument('--masks', default=None)
ap.add_argument('--summary', default=None)
ap.add_argument('--province', default='out/province_vote.npz')
ap.add_argument('--min-families', type=int, default=3, dest='min_families')
ap.add_argument('--link', type=float, default=0.25,
                help='fraction of the smaller corridor shared, above which two '
                     'hotspots count as connected')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--tag', default='RevealLO',
                help='names the outputs so models do not overwrite each other')
ap.add_argument('--dir', default='out')
A = ap.parse_args()
A.masks = A.masks or os.path.join(A.dir, f'corridor_masks_{A.tag}.npz')
A.summary = A.summary or os.path.join(A.dir, f'corridor_summary_{A.tag}.csv')

z = np.load(A.masks, allow_pickle=True)
meta = {'coarsen', 'tau', 'depth', 'lat', 'lon'}
names = [k for k in z.files if k not in meta]
print(f'{len(names)} corridors, coarsened by {int(z["coarsen"])}, tau = {float(z["tau"])}')

# Masks written before the depth-slice fix carry more shells than their own depth
# axis. Overlap is a boolean intersection and so survives that, but nothing else
# would, so refuse the file rather than let a later depth-resolved use read it.
_bad = [n for n in names if z[n].shape[0] != len(z['depth'])]
if _bad:
    raise SystemExit(f'{len(_bad)} masks have {z[_bad[0]].shape[0]} depth shells against '
                     f'{len(z["depth"])} on the axis. Run corridor_masks_repair.py '
                     'or regenerate with the fixed corridor_all.py.')

summ = pd.read_csv(A.summary)
ok = set(summ[summ.ok == True].site) if 'ok' in summ else set(names)
names = [n for n in names if n in ok]
print(f'{len(names)} passed the self-check and are used')

M = {n: z[n] for n in names}
sizes = {n: int(M[n].sum()) for n in names}

rows = []
for i, a in enumerate(names):
    for b in names[i + 1:]:
        inter = int(np.logical_and(M[a], M[b]).sum())
        if inter == 0:
            continue
        small = min(sizes[a], sizes[b])
        rows.append(dict(a=a, b=b, shared=inter,
                         frac_small=inter / max(small, 1),
                         jaccard=inter / max(int(np.logical_or(M[a], M[b]).sum()), 1)))
ov = pd.DataFrame(rows)
ov.to_csv(os.path.join(A.dir, f'corridor_overlap_{A.tag}.csv'), index=False)
print(f'\n{len(ov)} hotspot pairs share any corridor cell at all')

link = ov[ov.frac_small >= A.link] if len(ov) else ov
parent = {n: n for n in names}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for _, r in link.iterrows():
    ra, rb = find(r.a), find(r.b)
    if ra != rb:
        parent[ra] = rb
comp = {}
for n in names:
    comp.setdefault(find(n), []).append(n)
deg = {n: 0 for n in names}
for _, r in link.iterrows():
    deg[r.a] += 1
    deg[r.b] += 1

print(f'at a linking threshold of {A.link:.2f}, {len(link)} pairs are connected')
print(f'{len(comp)} components; {sum(1 for v in comp.values() if len(v) == 1)} '
      f'hotspots are isolated\n')
for k, v in sorted(comp.items(), key=lambda t: -len(t[1])):
    if len(v) > 1:
        print(f'  component of {len(v)}: ' + ', '.join(sorted(v)))

zz = np.load(A.province)
vote, plat, plon = zz['vote'], zz['lat'], zz['lon']
prov = vote >= A.min_families
hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180']).set_index('hotspot')


def inside(n):
    if n not in hs.index:
        return None
    la, lo = float(hs.loc[n, 'lat']), float(hs.loc[n, 'lon_180'])
    j = int(np.argmin(np.abs(plat - la)))
    i = int(np.argmin(np.abs(((plon - lo + 180) % 360) - 180)))
    return bool(prov[j, i])


d = pd.DataFrame([dict(site=n, degree=deg[n], cells=sizes[n],
                       component=len(comp[find(n)]), inside=inside(n))
                  for n in names])
d = d.merge(summ, on='site', how='left')
d.to_csv(os.path.join(A.dir, f'corridor_network_{A.tag}.csv'), index=False)

print(f'\n{"":>18s} {"n":>4s} {"median degree":>14s} {"median cells":>13s}')
for lab, g in (('inside a province', d[d.inside == True]),
               ('outside', d[d.inside == False])):
    if len(g):
        print(f'{lab:>18s} {len(g):4d} {g.degree.median():14.1f} {g.cells.median():13.0f}')
if (d.inside == True).sum() > 3 and (d.inside == False).sum() > 3:
    a = d[d.inside == True].degree.to_numpy(float)
    b = d[d.inside == False].degree.to_numpy(float)
    rg = np.random.default_rng(19)
    obs = float(np.median(a) - np.median(b))
    pool = np.concatenate([a, b]); k, c = len(a), 0
    for _ in range(20000):
        q = rg.permutation(pool)
        if abs(np.median(q[:k]) - np.median(q[k:])) >= abs(obs) - 1e-12:
            c += 1
    print(f'\nconnectivity inside minus outside: {obs:+.1f} links, '
          f'P = {(c + 1) / 20001:.4f}')
print('\nmost networked:')
for _, r in d.nlargest(6, 'degree').iterrows():
    print(f'  {r.site:26s} degree {int(r.degree):3d}  component {int(r.component):3d}'
          f'  {"in a province" if r.inside else "outside"}')
print('most isolated (degree zero):')
for _, r in d[d.degree == 0].head(8).iterrows():
    print(f'  {r.site:26s}  {"in a province" if r.inside else "outside"}')
