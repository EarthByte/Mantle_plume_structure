"""Stage 26 - does depth of root go with longevity of the surface record?

The plume hypothesis suggests it should. A hotspot fed from a deep, long-lived
thermal boundary layer ought to leave a longer volcanic record than one fed by a
shallow, transient source, so record longevity is an external variable with a
predicted sign, and the scan never sees it.

Two measures of longevity are tested, because no single compilation is both
complete and unambiguous: the catalogue hotspot ages of Steinberger (2000) as
tabulated by Torsvik et al. (2006), which cover 44 of the 49 but whose definition
is not restated there; and the age of the oldest volcanism actually dated and
attributed to the hotspot, which is defensible case by case but exists for only
19. Both are tested against the root fraction, per model and pooled, and against
the criteria count of Courtillot et al. (2003) for comparison.

Because the dated-age sample is small and is not a random subset of the 49 - it
exists for hotspots with a flood-basalt link or a drilled Pacific chain - the
stability of any coefficient it produces is tested by leave-one-out, and the
association is also taken with membership of the lowermost-mantle slow provinces
held fixed, the control used elsewhere in this work.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr, t as tdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longevity import CATALOGUE, DOCUMENTED, NOT_FOUND


# Where a single number is quoted across models it is the mean over the
# independent models, not over the runs: RevealLO appears twice in the run list,
# at two lateral samplings, and averaging the runs would weight it double. This
# must match the pooling used by table.py and manuscript_stats.py, or the log
# and the manuscript quote different values for the same statistic.
POOL = 'f_ind_mean'


# The per-model columns are whatever the comparison table holds; listing them
# here by hand went stale as soon as a tag changed.
def model_columns(d):
    return [c for c in d.columns
            if c.startswith('f_') and c not in
            ('f_mean', 'f_min', 'f_max', 'f_ind_mean')]


def partial(x, y, z):
    """Spearman correlation of x and y with z held fixed.

    Ranks first, then removes z from each of x and y by least squares and
    correlates what is left, which is the rank partial correlation.
    """
    x, y, z = (pd.Series(v).rank().values for v in (x, y, z))
    A = np.column_stack([np.ones_like(z), z])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]
    n = len(x)
    if n <= 4 or abs(r) >= 1:
        return r, np.nan
    ts = r * np.sqrt((n - 3) / (1 - r * r))
    return r, 2 * tdist.sf(abs(ts), n - 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comparison', default='out/comparison_all_models.csv')
    ap.add_argument('--table', default='out/table1_classification.csv')
    ap.add_argument('--province', default='out/province_RevealLO.csv')
    ap.add_argument('--paper-model', default='f_RevealLO')
    ap.add_argument('--out', default='out')
    a = ap.parse_args()

    d = pd.read_csv(a.comparison)
    d['age_cat'] = d.hotspot.map(CATALOGUE)
    d['age_doc'] = d.hotspot.map(lambda h: DOCUMENTED.get(h, (np.nan,))[0])
    d['age_doc_kind'] = d.hotspot.map(lambda h: DOCUMENTED.get(h, (0, ''))[1])

    print('=' * 92)
    print('LONGEVITY OF THE SURFACE RECORD AGAINST DEPTH OF ROOT')
    print('=' * 92)
    print(f'catalogue ages for {d.age_cat.notna().sum()} of {len(d)} hotspots; '
          f'documented oldest volcanism for {d.age_doc.notna().sum()}')
    print(f'no age found: {", ".join(NOT_FOUND)}\n')

    for col, lab in (('age_cat', 'Steinberger catalogue age'),
                     ('age_doc', 'documented oldest volcanism')):
        m = d.dropna(subset=[col, POOL])
        r, p = spearmanr(m[col], m[POOL])
        print(f'{lab}  (n = {len(m)})')
        print(f'   vs mean root fraction     rho = {r:+.3f}, p = {p:.3f}')
        for mod in model_columns(d):
            mm = d.dropna(subset=[col, mod])
            rr, pp = spearmanr(mm[col], mm[mod])
            print(f'      {mod[2:]:16s} rho = {rr:+.3f}, p = {pp:.3f}')
        hi, lo = m[m.n_deep >= 1][col], m[m.n_deep == 0][col]
        if len(hi) > 2 and len(lo) > 2:
            _, pu = mannwhitneyu(hi, lo, alternative='greater')
            print(f'   median age where a root is found in at least one model '
                  f'{np.median(hi):.0f} Ma (n = {len(hi)}), where none '
                  f'{np.median(lo):.0f} Ma (n = {len(lo)}); '
                  f'one-sided p = {pu:.3f}')
        mc = d.dropna(subset=[col, 'count'])
        rc, pc = spearmanr(mc[col], mc['count'])
        print(f'   vs Courtillot criteria count  rho = {rc:+.3f}, p = {pc:.3f}')
        ties = (m[POOL] == 0).sum()
        print(f'   {ties} of {len(m)} share a root fraction of exactly zero\n')

    # Stability of the dated-age coefficient.
    m = d.dropna(subset=['age_doc', POOL])
    print('LEAVE-ONE-OUT ON THE DATED AGES')
    out = []
    for h in m.hotspot:
        s = m[m.hotspot != h]
        rr, pp = spearmanr(s.age_doc, s[POOL])
        out.append((pp, rr, h))
    lost = [o for o in out if o[0] >= 0.05]
    print(f'   dropping any one of {len(lost)} of the {len(m)} hotspots takes '
          f'p above 0.05:')
    for pp, rr, h in sorted(lost, reverse=True):
        print(f'      without {h:28s} rho = {rr:+.3f}, p = {pp:.3f}')
    lo, hi = min(o[1] for o in out), max(o[1] for o in out)
    print(f'   coefficient ranges {lo:+.3f} to {hi:+.3f} over the deletions\n')

    # Province membership held fixed, the control used elsewhere in this work.
    if os.path.exists(a.province):
        pv = pd.read_csv(a.province)[['hotspot', 'pct_vs_null']]
        j = d.merge(pv, on='hotspot', how='left')
        print('WITH LOWERMOST-MANTLE PROVINCE MEMBERSHIP HELD FIXED')
        for col, lab in (('age_cat', 'catalogue age'),
                         ('age_doc', 'dated age')):
            for yc, yl in ((POOL, 'five-model mean'),
                           (a.paper_model, a.paper_model[2:])):
                s = j.dropna(subset=[col, yc, 'pct_vs_null'])
                r0, p0 = spearmanr(s[col], s[yc])
                r1, p1 = partial(s[col], s[yc], s.pct_vs_null)
                print(f'   {lab:14s} vs {yl:12s} n = {len(s):2d}   '
                      f'rho = {r0:+.3f} (p = {p0:.3f})   '
                      f'partial = {r1:+.3f} (p = {p1:.3f})')
        print()

    t = pd.read_csv(a.table)
    t['age_cat'] = t.hotspot.map(CATALOGUE)
    t['age_doc'] = t.hotspot.map(lambda h: DOCUMENTED.get(h, (np.nan,))[0])
    print('median longevity by tier (Ma)')
    med = t.groupby('tier')[['age_cat', 'age_doc']].median().round(0)
    cnt = t.groupby('tier')[['age_cat', 'age_doc']].count()
    print(med.join(cnt, rsuffix='_n').to_string())
    g = [x.dropna().values for _, x in t.groupby('tier').age_cat]
    g = [x for x in g if len(x) > 1]
    if len(g) > 1:
        h, pk = kruskal(*g)
        print(f'\nKruskal-Wallis on catalogue age across tiers: '
              f'H = {h:.2f}, p = {pk:.3f}')

    t['longevity_Ma'] = t.age_cat
    t['longevity_doc_Ma'] = t.age_doc
    t.to_csv(os.path.join(a.out, 'table1_classification.csv'), index=False)
    print('\nThe predicted positive association is absent on both measures. The '
          'catalogue ages,\nwhich cover 44 of the 49, give a coefficient '
          'indistinguishable from zero. The dated\nages give a negative '
          'coefficient, but on a sample of 19 in which many share a root\n'
          'fraction of exactly zero, and one that no single hotspot can be '
          'removed from without\nchanging the answer.')
    print('\nwrote longevity columns into table1_classification.csv')


if __name__ == '__main__':
    main()
