"""Pool root depths across inversion families and ask whether the classes are real.

A root-depth class is worth naming only if independent inversions agree on which
hotspots belong to it. They will agree somewhat by chance, because each model
puts most hotspots in whichever class is commonest in that model, so the test
permutes class labels independently within each model. That preserves every
model's own marginal distribution and destroys only the correspondence between
models, which is the thing being claimed.

RevealLO and REVEAL share an inversion lineage, so they are one family and one
vote, not two.

The classes are then compared against two attributes fixed before any of this
was measured: the number of Courtillot et al. (2003) criteria a hotspot
satisfies, and whether its helium is enriched. A deep root that samples the base
of the mantle should carry primordial helium; a root that stops in or near the
transition zone has no access to it. That is a prediction the measurement can
fail.
"""
from __future__ import annotations
import argparse, itertools, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--suffix', default='')
ap.add_argument('--deep-z0', type=float, default=2400.0, dest='deep_z0')
ap.add_argument('--mid-z0', type=float, default=1000.0, dest='mid_z0')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--n-perm', type=int, default=20000, dest='n_perm')
A = ap.parse_args()

FAMILY = {'RevealLO': 'REVEAL-family', 'REVEAL': 'REVEAL-family',
          'GLADM35': 'GLAD', 'SPiRaL': 'SPiRaL', 'SEMUCB-WM1': 'SEMUCB'}
PREFER = ['RevealLO', 'REVEAL']


def band(v):
    if not np.isfinite(v):
        return 'none'          # no slow column at all, which is not the same as a shallow one
    if v >= A.deep_z0:
        return 'deep'
    if v >= A.mid_z0:
        return 'mid'
    return 'shallow'


frames = {}
for tag in FAMILY:
    p = os.path.join(A.dir, f'root_depth_{tag}{A.suffix}.csv')
    if os.path.exists(p):
        frames[tag] = pd.read_csv(p)
if not frames:
    raise SystemExit(f'no root_depth_*{A.suffix}.csv in {A.dir}; run root_depth.py first')
print('found:', ', '.join(sorted(frames)), flush=True)

fam_tag = {}
for tag in frames:
    f = FAMILY[tag]
    if f not in fam_tag or (tag in PREFER and PREFER.index(tag) <
                            (PREFER.index(fam_tag[f]) if fam_tag[f] in PREFER else 99)):
        fam_tag[f] = tag
print('one vote per family:', ', '.join(f'{f}={t}' for f, t in sorted(fam_tag.items())), flush=True)

votes = {}
for f, tag in fam_tag.items():
    d = frames[tag]
    d = d[d.kind == 'hotspot']
    votes[f] = dict(zip(d.site, d.root_km.map(band)))
sites = sorted(set().union(*[set(v) for v in votes.values()]))
fams = sorted(votes)
M = pd.DataFrame({f: [votes[f].get(s) for s in sites] for f in fams}, index=sites)
M = M.dropna()
print(f'{len(M)} hotspots classified in all {len(fams)} families\n', flush=True)

print('per-family class counts:')
for f in fams:
    c = M[f].value_counts()
    print(f'  {f:14s} ' + '  '.join(f'{k} {c.get(k, 0):2d}' for k in ('deep', 'mid', 'shallow', 'none')))


def consensus(mat):
    out, need = [], int(np.ceil(0.75 * mat.shape[1]))
    for i in range(mat.shape[0]):
        vals, cnt = np.unique(mat[i], return_counts=True)
        j = int(np.argmax(cnt))
        out.append(vals[j] if cnt[j] >= need else None)
    return out


X = M.values
cons = consensus(X)
n_agree = sum(c is not None for c in cons)
rng = np.random.default_rng(7)
null = np.empty(A.n_perm, int)
for k in range(A.n_perm):
    Y = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    null[k] = sum(c is not None for c in consensus(Y))
p = float((np.sum(null >= n_agree) + 1) / (A.n_perm + 1))
print(f'\nhotspots with a {int(np.ceil(0.75 * len(fams)))}-of-{len(fams)} consensus: '
      f'{n_agree} of {len(M)}; label-permuted null {null.mean():.1f} '
      f'(5-95 {np.percentile(null, 5):.0f}-{np.percentile(null, 95):.0f}), P = {p:.4f}', flush=True)

pairs = list(itertools.combinations(range(len(fams)), 2))
pa = [float(np.mean(X[:, i] == X[:, j])) for i, j in pairs]
print('pairwise agreement: ' + '  '.join(
    f'{fams[i][:6]}/{fams[j][:6]} {v:.2f}' for (i, j), v in zip(pairs, pa)))

lab = sorted(set(X.ravel()))
n_i = np.array([[np.sum(X[i] == l) for l in lab] for i in range(len(M))], float)
nn, k_ = X.shape[1], len(lab)
Pi = (np.sum(n_i ** 2, axis=1) - nn) / (nn * (nn - 1))
pj = n_i.sum(axis=0) / (len(M) * nn)
Pe = np.sum(pj ** 2)
kappa = (Pi.mean() - Pe) / (1 - Pe) if Pe < 1 else np.nan
print(f"Fleiss' kappa = {kappa:.3f}")
if kappa < 0.4:
    print(f'  Kappa below 0.4 is slight-to-fair agreement. The families do not agree on\n'
          f'  which hotspot belongs to which class, so per-hotspot classes below are not\n'
          f'  a reproducible result and must not be reported as one. The distributional\n'
          f'  comparison in root_depth.py can still hold when this fails: they are\n'
          f'  different claims.')
print(f'median disagreement between the deepest and shallowest family, per hotspot: '
      f'{C.root_spread.median():.0f} km')

C = pd.DataFrame({'consensus': cons}, index=M.index)
hs = pd.read_csv(A.hotspots).set_index('hotspot')
C['criteria'] = hs['count'].reindex(C.index)
C['he'] = hs['he_ratio'].reindex(C.index)
# The class label is a vote across families, so the quantities it is compared
# against must be too. Taking them from one model cross-tabulates a four-model
# label against a one-model measurement, and when the families agree as weakly as
# they do here the two are often not describing the same answer.
def across(col):
    v = pd.DataFrame({f: frames[t][frames[t].kind == 'hotspot']
                      .set_index('site')[col].reindex(C.index)
                      for f, t in fam_tag.items()})
    return v.median(axis=1)


C['root_km'] = across('root_km')
C['tz_frac'] = across('tz_frac')
C['deep_frac'] = across('deep_frac')
C['root_spread'] = (pd.DataFrame({f: frames[t][frames[t].kind == 'hotspot']
                                  .set_index('site')['root_km'].reindex(C.index)
                                  for f, t in fam_tag.items()}).max(axis=1)
                    - pd.DataFrame({f: frames[t][frames[t].kind == 'hotspot']
                                    .set_index('site')['root_km'].reindex(C.index)
                                    for f, t in fam_tag.items()}).min(axis=1))

print('\nconsensus classes:')
for cl in ('deep', 'mid', 'shallow', 'none', None):
    s = C[C.consensus.isna()] if cl is None else C[C.consensus == cl]
    lbl = 'no cons' if cl is None else cl
    print(f'  {lbl:8s} {len(s):2d}   ' + ', '.join(sorted(s.index))[:150])


def fisher(a, b, c, d):
    from math import comb
    n = a + b + c + d
    obs = comb(a + b, a) * comb(c + d, c) / comb(n, a + c)
    tot = 0.0
    for i in range(max(0, (a + c) - (c + d)), min(a + c, a + b) + 1):
        pr = comb(a + b, i) * comb(c + d, a + c - i) / comb(n, a + c)
        if pr <= obs * (1 + 1e-9):
            tot += pr
    return min(1.0, tot)


D = C.dropna(subset=['consensus'])
D = D[D.consensus.isin(['deep', 'shallow', 'mid', 'none'])]
if len(D):
    deep = D.consensus == 'deep'
    prim = D.criteria.fillna(0) >= 3
    a, b = int((deep & prim).sum()), int((deep & ~prim).sum())
    c, d = int((~deep & prim).sum()), int((~deep & ~prim).sum())
    print(f'\ndeep root against >=3 Courtillot criteria: '
          f'deep {a}/{a + b} primary, not-deep {c}/{c + d}, Fisher P = {fisher(a, b, c, d):.4f}')
    he = D.he.astype(str).str.lower().eq('high')
    kn = D.he.notna() & D.he.astype(str).str.len().gt(0)
    if kn.sum() >= 4:
        E = D[kn]
        dd = E.consensus == 'deep'
        hh = E.he.astype(str).str.lower().eq('high')
        a, b = int((dd & hh).sum()), int((dd & ~hh).sum())
        c, d = int((~dd & hh).sum()), int((~dd & ~hh).sum())
        print(f'deep root against enriched helium:          '
              f'deep {a}/{a + b} high, not-deep {c}/{c + d}, Fisher P = {fisher(a, b, c, d):.4f}'
              f'   (n = {int(kn.sum())} with helium)')

# A column that stops near the transition zone means one thing if the transition
# zone beneath it is slow and another if it is not. The first is a candidate
# upwelling out of the transition-zone reservoir; the second has no deep feed at
# all and belongs with the lithospheric and plate-boundary explanations. The
# distinction is not an interpretation applied afterwards, it is a measurement,
# so it is made here rather than in the text.
print('\nwhat lies beneath each class (transition zone 410-660 km, basal window):')
for cl in ('deep', 'mid', 'shallow', 'none'):
    s_ = C[C.consensus == cl]
    if not len(s_):
        continue
    print(f'  {cl:8s} n={len(s_):2d}   transition zone slow {100 * s_.tz_frac.median():5.1f}%   '
          f'basal window slow {100 * s_.deep_frac.median():5.1f}%')
sh = C[C.consensus == 'shallow']
if len(sh):
    wet = sh[sh.tz_frac >= 0.5]
    dry = sh[sh.tz_frac < 0.5]
    print(f'\n  shallow-rooted with a slow transition zone ({len(wet)}): ' + ', '.join(sorted(wet.index)))
    print(f'  shallow-rooted without one      ({len(dry)}): ' + ', '.join(sorted(dry.index)))

C.to_csv(os.path.join(A.dir, f'root_classes{A.suffix}.csv'))
print(f'\nwrote {os.path.join(A.dir, f"root_classes{A.suffix}.csv")}', flush=True)
