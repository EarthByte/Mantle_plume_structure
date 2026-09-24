#!/usr/bin/env python3
"""Which traced paths are stable enough for anything site-specific to be said.

This is NOT track_support.py. That measures the evidence a path has for a deep
origin - how much slower than ambient the material along it is. This measures
whether the path is a reproducible object at all. A path can run through very slow
material and still be one of several nearly tied routes, in which case its geometry
is not a property of the mantle and no statement about ITS tilt, offset or endpoint
can be defended.

THE CRITERION. A path is called unstable if it moves 500 km or more when the depth
sampling is changed from 10 km to 30 km, which is a change in the numerics and not
in the model: same file, same configuration, same move set. Reproducibility under a
choice that has nothing to do with the mantle is the lowest bar a site-specific
claim has to clear, and it is the criterion because it answers the question directly.

TWO MECHANISMS ARE REPORTED BESIDE IT, because they explain which paths fail and
they are free to compute:

  ponding fraction   the share of a path's nodes that step sideways within one
                     depth shell. A route that ponds is one where two routes are
                     nearly tied, which is what lets a path flip. It predicts the
                     criterion at rho = 0.57.
  corridor coverage  the share of the depth column the near-optimal corridor
                     reaches. Before the forward-field pricing was fixed this
                     collapsed for exactly the ponding sites.

  python3 path_reliability.py --tag RevealLO --control RevealLO_30km
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import provenance
import pandas as pd

R_E, DEG = 6371.0, np.pi / 180.0

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--control', default='RevealLO_30km',
                help='the tag to measure reproducibility against: the same model at '
                     'coarser depth sampling, so only the numerics differ')
ap.add_argument('--dir', default='out')
ap.add_argument('--unstable-km', type=float, default=500.0, dest='unstable_km')
ap.add_argument('--tau', default='0.02')
A = ap.parse_args()


def paths(tag):
    p = os.path.join(A.dir, f'conduit_paths_all_{tag}.json')
    if not os.path.exists(p):
        raise SystemExit(f'no traced paths for {tag}')
    out = {}
    for n, q in json.load(open(p)).items():
        z = np.asarray(q['depth'], float)
        o = np.argsort(z)
        out[n] = (z[o], np.asarray(q['lat'], float)[o],
                  np.asarray(q['lon'], float)[o])
    return out


def gc_km(la1, lo1, la2, lo2):
    p1, p2 = la1 * DEG, la2 * DEG
    dl = (lo2 - lo1) * DEG
    h = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2)
    return 2 * R_E * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


PA, PB = paths(A.tag), paths(A.control)
sm = pd.read_csv(os.path.join(A.dir, f'corridor_summary_{A.tag}.csv')).set_index('site')
_ts = os.path.join(A.dir, f'track_support_{A.tag}.csv')
TS = (pd.read_csv(_ts).set_index('site')['group'].to_dict()
      if os.path.exists(_ts) else {})

rows = []
for s, (za, laa, loa) in PA.items():
    pond = float(np.mean(np.diff(za) == 0.0)) if len(za) > 1 else np.nan
    cov = float(sm.at[s, f'coverage_{A.tau}']) if s in sm.index \
        and f'coverage_{A.tau}' in sm.columns else np.nan
    sep = np.nan
    if s in PB:
        zb, lab, lob = PB[s]
        lo_, hi_ = max(za.min(), zb.min()), min(za.max(), zb.max())
        zz = za[(za >= lo_) & (za <= hi_)]
        if len(zz) >= 4:
            # unwrap before interpolating, or a path over the date line averages
            # to the wrong side of the planet
            sep = float(np.max(gc_km(
                np.interp(zz, za, laa), np.interp(zz, za, np.unwrap(loa * DEG) / DEG),
                np.interp(zz, zb, lab), np.interp(zz, zb, np.unwrap(lob * DEG) / DEG))))
    rows.append(dict(site=s, ponding=100 * pond, coverage=100 * cov,
                     separation_km=sep,
                     stable=bool(np.isfinite(sep) and sep < A.unstable_km),
                     support=TS.get(s, '')))

# Two measured criteria, not one, and neither is a judgement. A path is set aside
# NO BINARY VERDICT IS WRITTEN HERE, and the attempt is recorded because it failed
# instructively. "Not reproducible OR no coherent material support" set aside 33 of
# 49 sites, Reunion and Kerguelen among them, whose published tilts this paper
# discusses: "weak throughout" is the MAJORITY condition of the sample and a finding
# in its own right, not a disqualification. Every other single measure mis-sorts a
# different case - distance at 660 km leaves Discovery mid-pack at 572 km, and the
# tightest continuity level at which any route exists puts Afar in the best group at
# p5 while its path runs 1857 km from the volcano. The measures are reported per
# site; the classification is not made here.
d = pd.DataFrame(rows)
d['no_support'] = d.support.astype(str).str.contains('weak throughout|disjointed')
d = d.sort_values('separation_km', ascending=False)
out = os.path.join(A.dir, f'path_reliability_{A.tag}.csv')
d.to_csv(out, index=False)
provenance.stamp(out, control=A.control, unstable_km=A.unstable_km, tau=A.tau,
                 inputs=[os.path.join(A.dir, f'conduit_paths_all_{A.tag}.json'),
                         os.path.join(A.dir, f'conduit_paths_all_{A.control}.json')])

n_un = int((~d.stable).sum())
print(f'{A.tag} against {A.control}: a path is unstable if it moves '
      f'{A.unstable_km:.0f} km or more\n')
print(f'{n_un} of {len(d)} paths are unstable\n')
print(f'{"site":26s} {"separation":>10s} {"ponding":>8s} {"coverage":>9s}  support')
for _, r in d.iterrows():
    if r.stable:
        continue
    print(f'{r.site:26s} {r.separation_km:9.0f}k {r.ponding:7.1f}% {r.coverage:8.1f}%'
          f'  {r.support}')
st = d[d.stable]
print(f'\nthe {len(st)} stable paths: separation median {st.separation_km.median():.0f} km, '
      f'largest {st.separation_km.max():.0f} km, ponding median {st.ponding.median():.1f}%')
print(f'\nreproducibility and material support are independent, and neither alone is',
      f' a verdict:\n  {int((~d.stable).sum())} paths are not reproducible, '
      f'{int(d.no_support.sum())} lack coherent support, '
      f'{int(((~d.stable) & d.no_support).sum())} are both.')
print(f'wrote {out}')
