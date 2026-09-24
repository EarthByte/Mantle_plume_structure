"""Do the hotspots fall into groups, or is it a continuum?

The parameters that survived calibration are corridor width in three depth bands,
whether the conduit terminates inside a province, its lateral offset and tilt,
network membership and distance to a spreading centre. Root depth is not among them:
every trace terminates at the 2700 km target by construction, so it is a constant.

Grouping is only worth reporting if the data actually cluster, and the obvious test
does not show that. Permuting each column independently destroys all correlation
between parameters, so any correlated continuum beats it at every k - which is exactly
what these data do, at k = 2 through 6 with near-identical silhouettes. That is
evidence the parameters covary, not that hotspots fall into classes.

The null that answers the question preserves the covariance and removes only the
clustering: draw from a multivariate normal with the observed mean and covariance,
which is a single elongated cloud with no groups in it. If the observed silhouette
exceeds that, there are discrete groups; if not, the hotspots lie along a continuum
and must be described by their parameters rather than sorted into classes.
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument('--table', default='out/plume_parameters_RevealLO.csv')
ap.add_argument('--n-null', type=int, default=500, dest='n_null')
ap.add_argument('--dir', default='out')
A = ap.parse_args()

d0 = pd.read_csv(A.table, index_col=0)
cols = ['w_shallow', 'w_mid', 'w_deep', 'root_margin', 'offset', 'tilt', 'd_any_ridge']

# COMPLETE CASES ONLY, and the reason matters.
#
# A corridor that never reaches a depth band has no width there. Until this was
# fixed the profile wrote 0.0 for a shell the corridor does not occupy and the band
# median came back as exactly 0 km, so six sites entered the clustering carrying a
# literal zero in two of the seven parameters. They then clustered together on that
# shared zero, and made up six of the seven members of the group the manuscript
# described. The grouping was keying on a fill value.
#
# Imputing would put the artefact back. The honest test is whether structure exists
# among the hotspots that HAVE all seven parameters, so incomplete cases are dropped
# and counted.
d = d0.dropna(subset=cols)
drop = d0.index.difference(d.index)
if len(drop):
    print(f'dropped {len(drop)} of {len(d0)} hotspots for missing parameters: '
          f'{", ".join(map(str, drop))}')
    miss = d0.loc[drop, cols].isna().sum()
    print('  missing by parameter: '
          + ', '.join(f'{c} {int(n)}' for c, n in miss.items() if n))
    print('  a corridor with no width in a band is narrow, not wide; excluding these\n'
          '  removes them from the clustering but not from the paper\n')
X = StandardScaler().fit_transform(d[cols].to_numpy(float))
print(f'{len(d)} hotspots on {len(cols)} calibrated parameters: {", ".join(cols)}\n')

rng = np.random.default_rng(17)
print(f'{"k":>3s}{"silhouette":>13s}{"permuted":>14s}{"gaussian":>15s}'
      f'{"gauss 95th":>12s}{"P":>9s}')
print('  the permuted column is shown only to make the point that it is the wrong null')
best = None
for k in range(2, 7):
    lab = KMeans(k, n_init=25, random_state=0).fit_predict(X)
    s = silhouette_score(X, lab)
    perm, gauss = [], []
    for _ in range(A.n_null):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        perm.append(silhouette_score(Xp, KMeans(k, n_init=5, random_state=0).fit_predict(Xp)))
        Xg = rng.multivariate_normal(X.mean(0), np.cov(X, rowvar=False), size=len(X))
        gauss.append(silhouette_score(Xg, KMeans(k, n_init=5, random_state=0).fit_predict(Xg)))
    perm, gauss = np.array(perm), np.array(gauss)
    P = float((gauss >= s).mean())
    print(f'{k:3d}{s:13.3f}{np.median(perm):14.3f}{np.median(gauss):15.3f}'
          f'{np.percentile(gauss, 95):12.3f}{P:9.3f}')
    if best is None or s > best[1]:
        best = (k, s, P, lab)

k, s, P, lab = best
print(f'\nbest k = {k}, silhouette {s:.3f}, P = {P:.3f}')
if P >= 0.05:
    print('The hotspots do NOT cluster more than a single correlated cloud of the same')
    print('covariance does. They lie on a continuum and are to be described by their')
    print('parameters, not sorted into classes.')
else:
    d['group'] = lab
    d.to_csv(os.path.join(A.dir, 'plume_groups_RevealLO.csv'))
    print('Clustering beats the null; groups written.')
    for g in sorted(set(lab)):
        m = d[d.group == g]
        print(f'\n  group {g}, n = {len(m)}: {", ".join(sorted(m.index)[:8])}'
              f'{" ..." if len(m) > 8 else ""}')
        print(f'    width {m.w_shallow.median():.0f}/{m.w_mid.median():.0f}/'
              f'{m.w_deep.median():.0f} km, offset {m.offset.median():.0f} km, '
              f'inside a province {100 * m.root_in.mean():.0f}%, '
              f'networked {100 * m.linked.mean():.0f}%')
