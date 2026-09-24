#!/usr/bin/env python3
"""Does the traced root classify hotspots differently from their surface position,
and does the compositional association reported on surface position hold better on
the traced one?

Homrighausen et al. (2023) report that oceanic hotspots with enriched-mantle
compositions sit geographically near the low-velocity provinces while those far from
them do not. That association is made on surface position. A traced root is a
different classification: the descent moves a hotspot several hundred kilometres
before it reaches province depth, and it can move one across a province boundary.

This measures three things and writes them where the manuscript audit can read them:
how many of the hotspots with isotopic data the two classifications disagree about,
how many each places inside a province, and whether strontium and neodymium separate
more sharply on one than on the other.

The comparison is one-sided because Homrighausen et al. predict a direction:
enriched compositions - higher 87Sr/86Sr, lower 143Nd/144Nd - inside the provinces.
A two-sided test would answer a question nobody asked. Mann-Whitney is used because
these are small, skewed samples and no distributional form is safe to assume.

Neither probability is corrected for the number of isotopes examined, and strontium
and neodymium are anti-correlated in ocean island basalts, so the two are not
independent tests. The manuscript says so; this script does not pretend otherwise.

  python3 province_flip.py
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--dir', default='out')
ap.add_argument('--province', default='out/province_vote.npz')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--min-families', type=int, default=3, dest='min_families')
A = ap.parse_args()

z = np.load(os.path.join(HERE, A.province))
vote, plat, plon = z['vote'], z['lat'], z['lon']
prov = vote >= A.min_families


def inside(la, lo):
    j = int(np.argmin(np.abs(plat - la)))
    i = int(np.argmin(np.abs(((plon - lo + 180) % 360) - 180)))
    return bool(prov[j, i])


hs = pd.read_csv(os.path.join(HERE, A.hotspots)).set_index('hotspot')
par = pd.read_csv(os.path.join(HERE, A.dir, f'plume_parameters_{A.tag}.csv'), index_col=0)
geo = pd.read_csv(os.path.join(HERE, A.dir, f'geochem_link_{A.tag}.csv')).set_index('hotspot')

rows = []
for n in par.index:
    if n not in geo.index or n not in hs.index:
        continue
    sr, nd = geo.at[n, 'med_87Sr/86Sr'], geo.at[n, 'med_143Nd/144Nd']
    if not np.isfinite(sr) and not np.isfinite(nd):
        continue
    rows.append(dict(hotspot=n,
                     surface_in=inside(float(hs.at[n, 'lat']), float(hs.at[n, 'lon_180'])),
                     traced_in=bool(par.at[n, 'root_in']),
                     sr=sr, nd=nd))
d = pd.DataFrame(rows).set_index('hotspot')
d['flipped'] = d.surface_in != d.traced_in

# enriched inside: Sr higher inside, Nd lower inside
def test(col, how, greater):
    m = d[col].notna()
    a = d.loc[m & d[how], col]
    b = d.loc[m & ~d[how], col]
    if len(a) < 3 or len(b) < 3:
        return np.nan, len(a), len(b)
    p = mannwhitneyu(a, b, alternative='greater' if greater else 'less').pvalue
    return float(p), len(a), len(b)


out = []
for col, greater in (('sr', True), ('nd', False)):
    for how in ('traced_in', 'surface_in'):
        p, na, nb = test(col, how, greater)
        out.append(dict(isotope={'sr': '87Sr/86Sr', 'nd': '143Nd/144Nd'}[col],
                        classification={'traced_in': 'traced root',
                                        'surface_in': 'surface position'}[how],
                        n_inside=na, n_outside=nb, p=p))
r = pd.DataFrame(out)

summary = dict(n_hotspots=len(d), n_flipped=int(d.flipped.sum()),
               n_traced_inside=int(d.traced_in.sum()),
               n_surface_inside=int(d.surface_in.sum()),
               flipped_to_inside=int((d.flipped & d.traced_in).sum()),
               flipped_to_outside=int((d.flipped & ~d.traced_in).sum()))

os.makedirs(os.path.join(HERE, A.dir), exist_ok=True)
r.to_csv(os.path.join(HERE, A.dir, f'province_flip_tests_{A.tag}.csv'), index=False)
pd.Series(summary).to_csv(os.path.join(HERE, A.dir, f'province_flip_summary_{A.tag}.csv'),
                          header=['value'])
d.to_csv(os.path.join(HERE, A.dir, f'province_flip_{A.tag}.csv'))

print(f'hotspots with isotopic data       {summary["n_hotspots"]}')
print(f'classifications disagree          {summary["n_flipped"]}'
      f'  ({summary["flipped_to_inside"]} moved inside, '
      f'{summary["flipped_to_outside"]} moved outside)')
print(f'inside a province, traced root    {summary["n_traced_inside"]}')
print(f'inside a province, surface        {summary["n_surface_inside"]}')
print()
print(r.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print('\nflipped hotspots:', ', '.join(sorted(d.index[d.flipped])))
