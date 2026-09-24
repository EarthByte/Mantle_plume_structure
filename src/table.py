"""Stage 25 - the modernised classification table.

Courtillot et al. (2003) sorted hotspots into primary, secondary and tertiary
types by counting how many of five criteria each satisfied. The counting was the
weak part: the five criteria are not independent, not equally reliable, and a
count of three can be assembled from different combinations that mean different
things.

What replaces the count here is a single measured quantity with a stated
uncertainty - the fraction of calibrated configurations, in each of four
tomographic models, under which the hotspot is underlain by slow structure
extending deeper than beneath random locations. Hotspots are then ranked by how
many models find that, and by the mean fraction within each rank. The original
criteria count is carried alongside, unchanged, as the external comparison it is.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

TIERS = [
    ('I',   'deep root in a majority of the independent models'),
    ('II',  'deep root in at least one'),
    ('III', 'decisive in none, evidence leaning positive'),
    ('IV',  'no evidence of a deep root'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comparison',
                    default='out/comparison_all_models.csv')
    ap.add_argument('--courtillot', default='hotspots_courtillot2003.csv')
    ap.add_argument('--out', default='out')
    a = ap.parse_args()

    d = pd.read_csv(a.comparison)
    c = pd.read_csv(a.courtillot)[['hotspot', 'count', 'he_ratio']]
    d = d.merge(c, on='hotspot', how='left', suffixes=('', '_c'))
    if 'count' not in d.columns:
        d['count'] = d['count_c']

    # RevealLO at two lateral samplings is one model and is counted once, so
    # n_deep runs over the independent models only; n_models records how many
    # that is, which differs if a model fails calibration.
    nmod = int(d.n_models.max()) if 'n_models' in d.columns else 5

    # The root fraction reported is the mean over the independent models, not
    # over the runs: RevealLO appears twice in the run list, at two lateral
    # samplings, and averaging the runs would weight it double. The two differ
    # by at most 0.10 and by less than 0.01 for most hotspots, but the pooled
    # number should not depend on how many times one model was run.
    if 'f_ind_mean' not in d.columns:
        d['f_ind_mean'] = d.f_mean

    def tier(r):
        if r.n_deep >= (nmod + 1) // 2:
            return 'I'
        if r.n_deep >= 1:
            return 'II'
        return 'III' if r.f_ind_mean >= 0.5 else 'IV'
    d['tier'] = d.apply(tier, axis=1)
    # Sorting on the number of models and then the root fraction keeps the tiers
    # contiguous and puts the table in the same order as the cross-sections, so
    # a reader moving between them does not have to re-find a hotspot.
    d = d.sort_values(['n_deep', 'f_ind_mean'], ascending=False)

    d['root_fraction'] = d.f_ind_mean.round(2)
    d['spread'] = (d.f_max - d.f_min).round(2)
    d['helium'] = d.he_ratio.fillna('-')
    d['n_deep_models'] = d.n_deep.astype(int)

    cols = ['tier', 'hotspot', 'lat', 'lon_180', 'root_fraction', 'spread',
            'n_deep_models', 'count', 'helium']
    tab = d[cols].rename(columns={'lon_180': 'lon', 'count': 'criteria',
                                  'n_deep_models': 'n_models'})
    tab.to_csv(os.path.join(a.out, 'table1_classification.csv'), index=False)

    print('=' * 100)
    print('TABLE 1  Hotspot classification by connectivity of slow structure')
    print('=' * 100)
    for t, label in TIERS:
        s = tab[tab.tier == t]
        if not len(s):
            continue
        print(f'\nTIER {t} - {label}  (n = {len(s)}, '
              f'mean criteria count {s.criteria.mean():.2f})')
        print(s.drop(columns=['tier']).to_string(
            index=False, float_format=lambda x: f'{x:.2f}'))

    print('\n' + '=' * 100)
    print('mean Courtillot criteria count by tier')
    print(tab.groupby('tier').criteria.agg(['mean', 'count']).round(2).to_string())
    from scipy.stats import spearmanr
    # on the unrounded column: rounding to two places creates ties and moves
    # the coefficient, and the paper quotes the unrounded value
    m = d.dropna(subset=['count', 'f_ind_mean'])
    rho, p = spearmanr(m['count'], m.f_ind_mean)
    print(f'\nrho(criteria count, root fraction) = {rho:+.3f}, p = {p:.4f}, '
          f'n = {len(m)}')
    # Kruskal-Wallis across tiers, since the tiers are ordered but the counts
    # are small integers and far from normal
    from scipy.stats import kruskal
    groups = [g.criteria.dropna().values for _, g in tab.groupby('tier')]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) > 1:
        h, pk = kruskal(*groups)
        print(f'Kruskal-Wallis across tiers: H = {h:.2f}, p = {pk:.4f}')
    print(f'\nwrote table1_classification.csv')


if __name__ == '__main__':
    main()
