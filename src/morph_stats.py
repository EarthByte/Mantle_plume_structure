#!/usr/bin/env python3
"""The two comparisons the reframed paper rests on, tested exactly where possible.

  THE PARADOX. Whether a hotspot's independently established track class predicts
  whether the least-cost search roots it. The classification is built from the
  surface volcanic record and from non-seismic evidence of a deep source, so a
  null result here is a statement about tomography rather than about the
  classification: geological persistence and tomographic distinctness would then
  be different observables, which is the premise the morphology analysis exists
  to explore.

  THE MORPHOLOGY CONTRAST. Whether the long-track hotspots the search does not
  root differ in the shape of their corridors from the hotspots it does. This is
  the test that decides whether the reframing has content: if the corridors
  beneath rooted and unrooted long-track hotspots are the same shape, nothing has
  been explained.

Permutations are enumerated whenever the number of arrangements allows it, and a
Monte-Carlo count is used only when it does not, with the number of arrangements
stated either way. Every comparison resting on fewer than ten observations is
reported with its leave-one-out range, because a difference that does not survive
the loss of one hotspot is not a difference this data can support.
"""
from __future__ import annotations

import argparse, itertools, math, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_ENUM = 3_000_000


def perm_diff(a, b, n_mc=200_000, seed=11):
    """Two-sided permutation test on the difference of means.

    Enumerated when the number of arrangements is small enough that an exact p
    can be given; a reviewer cannot argue with an exact p and will ask why a
    Monte-Carlo one was used where enumeration was possible.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 1 or len(b) < 1:
        return np.nan, np.nan, 0, 'none'
    obs = a.mean() - b.mean()
    allv = np.concatenate([a, b])
    n, n1 = len(allv), len(a)
    total = math.comb(n, n1)
    if total <= MAX_ENUM:
        hits = 0
        idx = np.arange(n)
        for c in itertools.combinations(idx, n1):
            m = np.zeros(n, bool); m[list(c)] = True
            if abs(allv[m].mean() - allv[~m].mean()) >= abs(obs) - 1e-12:
                hits += 1
        return obs, hits / total, total, 'exact'
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_mc):
        p = rng.permutation(allv)
        if abs(p[:n1].mean() - p[n1:].mean()) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n_mc + 1), n_mc, 'monte carlo'


def leave_one_out(a, b):
    """The range of the difference of means when any single member is dropped."""
    a, b = list(np.asarray(a, float)), list(np.asarray(b, float))
    vals = []
    for q in range(len(a)):
        vals.append(np.mean(a[:q] + a[q + 1:]) - np.mean(b))
    for q in range(len(b)):
        vals.append(np.mean(a) - np.mean(b[:q] + b[q + 1:]))
    return (float(np.min(vals)), float(np.max(vals))) if vals else (np.nan, np.nan)


def paradox(cls, models, dirname):
    signs = []
    print('=' * 78)
    print('THE PARADOX: does the independent track class predict a tomographic root?')
    print('=' * 78)
    from scipy.stats import spearmanr
    rank = {'P1': 3, 'P2': 2, 'P3': 1}
    for tag in models:
        f = os.path.join(dirname, f'classification_{tag}.csv')
        if not os.path.exists(f):
            continue
        m = pd.read_csv(f).merge(cls[['hotspot', 'track_class', 'duration_myr']],
                                 on='hotspot', how='inner')
        p1 = m[m.track_class == 'P1'].root_fraction.values
        rest = m[m.track_class != 'P1'].root_fraction.values
        obs, p, tot, how = perm_diff(p1, rest)
        d = m.dropna(subset=['duration_myr'])
        rho, pr = spearmanr(d.duration_myr, d.root_fraction)
        lo, hi = leave_one_out(p1, rest)
        print(f'\n{tag}')
        print(f'  root fraction, P1 (n={len(p1)}) mean {p1.mean():.3f} against '
              f'the other {len(rest)} mean {rest.mean():.3f}')
        print(f'    difference {obs:+.3f}, {how} p = {p:.4f} over {tot:,} '
              f'arrangements; leave-one-out {lo:+.3f} to {hi:+.3f}')
        print(f'  dated duration against root fraction: rho = {rho:+.3f}, '
              f'p = {pr:.4f}, n = {len(d)}')
        det = m[m.root_fraction >= 0.8]
        signs.append((tag, obs, rho, pr))
        print(f'  the {len(det)} hotspots the search roots have a median dated '
              f'duration of '
              f'{det.duration_myr.median():.0f} Myr against '
              f'{m[m.root_fraction < 0.8].duration_myr.median():.0f} Myr '
              f'for the rest')
    if len(signs) > 1:
        neg = sum(1 for _, _, r, _ in signs if r < 0)
        sig = [t for t, _, r, p in signs if r < 0 and p < 0.05]
        print(f'\nacross {len(signs)} model runs the duration-against-root-'
              f'fraction correlation is negative in {neg} and reaches p < 0.05 '
              f'in {len(sig)}' + (f' ({", ".join(sig)})' if sig else ''))
        print('  the runs are not independent: REVEAL, RevealLO and its '
              'decimation share an inversion lineage, so the effective number '
              'of independent families is smaller than the number of rows above')


def morphology(summ, cls, field, dirname, tag):
    print('\n' + '=' * 78)
    print(f'THE MORPHOLOGY CONTRAST, on {field}')
    print('=' * 78)
    s = summ[(summ.field == field) & (~summ.site.str.startswith('null'))].copy()
    if not len(s):
        print('  no sites'); return
    key = {h.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', ''): h
           for h in cls.hotspot}
    s['hotspot'] = s.site.map(key)
    cl = pd.read_csv(os.path.join(dirname, f'classification_{tag}.csv'))
    s = s.merge(cls[['hotspot', 'track_class', 'duration_myr']], on='hotspot',
                how='left').merge(cl[['hotspot', 'root_fraction']], on='hotspot',
                                  how='left')
    s['rooted'] = s.root_fraction >= 0.8
    longt = s[s.track_class.isin(('P1', 'P2'))]
    a = longt[~longt.rooted]
    b = longt[longt.rooted]
    print(f'long-track hotspots the search does not root: '
          f'{", ".join(a.hotspot.astype(str))}')
    print(f'long-track hotspots it does root:            '
          f'{", ".join(b.hotspot.astype(str))}')
    # gap_fraction is not in this list and must not be put back. A descending
    # path passes every shell between the seed and the target, so the corridor
    # carries weight at every depth by construction and the metric is
    # identically zero at every site and every null. It measures nothing. A
    # real bottleneck test needs the anomaly along the corridor, which the
    # volumes do not carry.
    METRICS = [c for c in ('r50_km_lower', 'r90_km_lower', 'anisotropy_lower',
                           'ncomp_max_lower', 'turning_total_deg',
                           'turning_net_deg', 'pooled_jaccard',
                           'tortuosity', 'endpoint_spread_km',
                           'r50_km_mid', 'anisotropy_mid', 'ncomp_max_mid')
               if c in s.columns]
    rows = []
    for c in METRICS:
        obs, p, tot, how = perm_diff(a[c].values, b[c].values)
        lo, hi = leave_one_out(a[c].dropna().values, b[c].dropna().values)
        rows.append(dict(metric=c, unrooted_mean=a[c].mean(), rooted_mean=b[c].mean(),
                         difference=obs, p=p, arrangements=tot, how=how,
                         loo_lo=lo, loo_hi=hi))
    r = pd.DataFrame(rows)
    print()
    print(r.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
    pcols = [f'pct_{c}' for c in METRICS if f'pct_{c}' in s.columns]
    if pcols:
        print('\npercentile within the matched null locations '
              '(50 is the median random location):')
        show = ['hotspot', 'track_class', 'root_fraction', 'n_matched_nulls'] + pcols
        print(s[[c for c in show if c in s.columns]].to_string(
            index=False, float_format=lambda x: f'{x:.3g}'))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--field', default='corridor0.02')
    ap.add_argument('--models', nargs='+',
                    default=['RevealLO', 'REVEAL', 'GLADM35', 'SPiRaL',
                             'SEMUCB-WM1', 'RevealLO_30km'])
    a = ap.parse_args()
    cls = pd.read_csv(os.path.join(a.dir, 'track_classes.csv'))
    h = open(os.path.join(a.dir, 'track_classes.sha256')).read().strip() \
        if os.path.exists(os.path.join(a.dir, 'track_classes.sha256')) else '?'
    print(f'track classification sha256 {h}: '
          + ', '.join(f'{k} {v}' for k, v in
                      cls.track_class.value_counts().sort_index().items()))
    paradox(cls, a.models, a.dir)
    f = os.path.join(a.dir, f'morph_summary_{a.tag}{a.suffix}.csv')
    if os.path.exists(f):
        r = morphology(pd.read_csv(f), cls, a.field, a.dir, a.tag)
        if r is not None:
            r.to_csv(os.path.join(a.dir,
                                  f'morph_contrast_{a.tag}{a.suffix}.csv'),
                     index=False)
            print(f'\nwrote morph_contrast_{a.tag}{a.suffix}.csv')
    else:
        print(f'\n{os.path.basename(f)} not present; run morph_run.py and '
              f'morph_metrics.py for the morphology contrast')


if __name__ == '__main__':
    main()
