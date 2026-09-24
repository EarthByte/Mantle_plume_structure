#!/usr/bin/env python3
"""Does the province association survive being given the other root?

competing_routes.py establishes that every hotspot has a second deep root in reach,
at a median cost margin of a couple of per cent - a difference in the quantity the
criterion integrates of under a tenth of one per cent, which no model resolves. The
root the paper reports is therefore one of two that the data do not separate, and the
honest question is whether anything built on it depends on which one was returned.

So: take the SECOND root at every hotspot, the one the search would have chosen had
the first been unavailable, and redo the province comparison with it. The null is left
alone, which makes the test conservative in the right direction - every hotspot is
moved and none of the comparison set is.

If the association holds, it does not rest on a choice the data cannot make, and that
is worth a paragraph. If it does not, the paper needs to say so before a reviewer does.

  python3 root_robustness.py --tag RevealLO
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--province', default='out/province_vote.npz')
    ap.add_argument('--min-families', type=int, default=3, dest='min_families')
    A = ap.parse_args()

    D = os.path.join(HERE, A.dir)
    zz = np.load(os.path.join(HERE, A.province))
    vote, plat, plon = zz['vote'], zz['lat'], zz['lon']
    prov = vote >= A.min_families

    def inside(la, lo):
        j = int(np.argmin(np.abs(plat - la)))
        i = int(np.argmin(np.abs(((plon - lo + 180) % 360) - 180)))
        return bool(prov[j, i])

    cr = pd.read_csv(os.path.join(D, f'competing_routes_{A.tag}.csv'))
    cr = cr[cr.alt_reached == True]
    # The first root comes from competing_routes itself, not from plume_roots, so the
    # two sides of the comparison are the same traced paths under the same
    # configuration. plume_roots is used only to confirm they agree where it exists.
    rt = cr.rename(columns={'end_lat': 'root_lat', 'end_lon': 'root_lon'})
    pr = os.path.join(D, f'plume_roots_{A.tag}.csv')
    if os.path.exists(pr):
        q = pd.read_csv(pr).set_index('site')
        j0 = rt.set_index('site').join(q[['root_lat', 'root_lon']], rsuffix='_pub',
                                       how='inner')
        agree = int((np.abs(j0.root_lat - j0.root_lat_pub) < 1e-6).sum())
        print(f'  first root agrees with plume_roots_{A.tag}.csv at {agree} of '
              f'{len(j0)} sites')

    nlp = os.path.join(D, f'root_null_{A.tag}.csv')
    if os.path.exists(nlp):
        nl = pd.read_csv(nlp)
        n_in, n_n = int(nl.root_in.astype(bool).sum()), len(nl)
    else:
        n_in = n_n = 0
        print(f'  no root_null_{A.tag}.csv: reporting the fractions without a test')

    rows = []
    for lab, la, lo, src in (
            ('reported root', rt.root_lat, rt.root_lon, rt.site),
            ('second root', cr.alt_end_lat, cr.alt_end_lon, cr.site)):
        ins = [inside(float(a), float(o)) for a, o in zip(la, lo)]
        h_in, h_n = int(np.sum(ins)), len(ins)
        if n_n:
            odds, P = fisher_exact([[h_in, h_n - h_in], [n_in, n_n - n_in]],
                                   alternative='greater')
            rows.append((lab, h_in, h_n, odds, P))
            print(f'  {lab:14s} inside a province {h_in:2d} of {h_n:2d} '
                  f'({100 * h_in / h_n:4.1f} %)   against the null {n_in} of {n_n} '
                  f'({100 * n_in / n_n:4.1f} %)   odds {odds:5.2f}   '
                  f'Fisher P = {P:.5f}')
        else:
            rows.append((lab, h_in, h_n, float('nan'), float('nan')))
            print(f'  {lab:14s} inside a province {h_in:2d} of {h_n:2d} '
                  f'({100 * h_in / h_n:4.1f} %)')

    # how many individual hotspots change membership at all
    j = cr.set_index('site').copy()
    j['first_in'] = [inside(float(a), float(o))
                     for a, o in zip(j.end_lat, j.end_lon)]
    j['alt_in'] = [inside(float(a), float(o))
                   for a, o in zip(j.alt_end_lat, j.alt_end_lon)]
    flip = j[j.first_in != j.alt_in]
    print(f'\n  {len(flip)} of {len(j)} hotspots change province membership when given '
          f'their second root')
    if len(flip):
        print(flip.reset_index()[['site', 'first_in', 'alt_in', 'margin_pct',
                                  'separation_km']]
              .to_string(index=False, float_format=lambda v: f'{v:.1f}'))
    # per-site membership, so Table S1 can carry the second root beside the first
    ps = os.path.join(D, f'root_robustness_sites_{A.tag}.csv')
    j.reset_index()[['site', 'first_in', 'alt_in', 'margin_pct', 'separation_km',
                     'split_depth_km']].to_csv(ps, index=False)
    print(f'wrote {ps}')
    out = os.path.join(D, f'root_robustness_{A.tag}.csv')
    pd.DataFrame(rows, columns=['root', 'in_province', 'n', 'odds', 'fisher_p']
                 ).to_csv(out, index=False)
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
