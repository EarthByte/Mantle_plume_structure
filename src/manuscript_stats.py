"""Every number the manuscript quotes, computed from the current run.

The manuscript build holds its statistics as literals, so a re-run invalidates
them silently and nothing errors. This writes them all to one JSON file, so the
text can be checked against a single current source, and a stale figure has one
place to be caught.

Numbers that need the tomographic model files themselves - the root-mean-square
anomaly of each model, and the anomaly beneath a named hotspot as a function of
depth - are not here; site_profile.py produces those from the models directly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr, t as tdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longevity import CATALOGUE, DOCUMENTED, NOT_FOUND

MODELS = ['REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1', 'RevealLO',
          'RevealLO_30km']
# The pooled root fraction averages the independent models, not the runs:
# RevealLO appears twice in the run list and averaging the runs would weight it
# double. compare.py writes both; f_mean is kept only for the spread.
POOL = 'f_ind_mean'
INDEPENDENT = MODELS[:5]
AMPS = [0.5, 0.75, 1.0, 1.5]
# the six sites section 4.2 compares
COST_SITES = ['Macdonald (Cook-Austral)', 'Tahiti/Society', 'Iceland',
              'Reunion', 'Kerguelen(Heard)', 'Hawaii']


def rho(x, y):
    r, p = spearmanr(x, y)
    # p is kept to six places because several of these are below 0.0001 and the
    # text distinguishes them.
    return {'rho': round(float(r), 3), 'p': round(float(p), 6), 'n': int(len(x))}


def _pair(g, a, b):
    k = g[a].notna() & g[b].notna()
    return g.loc[k, a], g.loc[k, b]


def partial(x, y, z):
    """Spearman correlation of x and y with z held fixed."""
    x, y, z = (pd.Series(v).rank().values for v in (x, y, z))
    A = np.column_stack([np.ones_like(z), z])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    p = float(2 * tdist.sf(abs(r * np.sqrt((n - 3) / (1 - r * r))), n - 3))
    return {'rho': round(r, 3), 'p': round(p, 6), 'n': int(n)}


def half_recovery(kept):
    """Amplitude at which the recovered fraction crosses one half, interpolated.

    The mean, not the median. Figure 1a plots the fraction of injected conduits
    recovered and its caption defines it that way, so a median of the per
    configuration rates is a different quantity from the one the panel draws and
    the one the text describes. It took a median here and a mean in the figure,
    and the two annotated the same sentence with different numbers.
    """
    f = [float(kept[f'detect_{a:g}'].mean()) for a in AMPS]
    if f[0] >= 0.5:
        return AMPS[0]
    for i in range(len(AMPS) - 1):
        if f[i] < 0.5 <= f[i + 1]:
            return round(AMPS[i] + (0.5 - f[i]) / (f[i + 1] - f[i])
                         * (AMPS[i + 1] - AMPS[i]), 3)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out')
    ap.add_argument('--paper-model', default='RevealLO')
    a = ap.parse_args()
    O = lambda *p: os.path.join(a.out, *p)

    d = pd.read_csv(O('comparison_all_models.csv'))
    t = pd.read_csv(O('table1_classification.csv'))
    fcols = [f'f_{m}' for m in MODELS if f'f_{m}' in d.columns]
    S = {}

    # ---- the classification itself
    S['classification'] = {
        'n_hotspots': int(len(d)),
        'n_models_independent': len(INDEPENDENT),
        'tier_counts': {k: int(v) for k, v in
                        t.tier.value_counts().sort_index().items()},
        'tier_members': {k: list(g.hotspot) for k, g in t.groupby('tier')},
        'rooted_in_at_least_one': int((d.n_deep >= 1).sum()),
        'rooted_in_majority': int((d.n_deep >= 3).sum()),
        'rooted_in_none': int((d.n_deep == 0).sum()),
    }

    # The paper's own model, taken alone: its three-way class, which is what the
    # alternative to the six-model consensus rests on.
    pm = pd.read_csv(O(f'classification_{a.paper_model}.csv'))
    counts = pm.ensemble_class.value_counts()
    cons = t.set_index('hotspot').tier
    alone = pm[pm.ensemble_class == 'deeper than null'].sort_values(
        'root_fraction', ascending=False)
    S['paper_model_alone'] = {
        'model': a.paper_model,
        'class_counts': {k: int(v) for k, v in counts.items()},
        'deeper_than_null': list(alone.hotspot),
        'ambiguous': list(pm[pm.ensemble_class == 'ambiguous'].hotspot),
        'deeper_here_but_minority_in_consensus': [
            h for h in alone.hotspot if cons.get(h) != 'I'],
        'majority_in_consensus_but_not_deeper_here': [
            h for h in cons[cons == 'I'].index
            if h not in set(alone.hotspot)],
    }

    # ---- calibration
    cal = {'grid': {}, 'per_model': {}}
    det = {m: pd.read_csv(O(f'detection_{m}.csv')) for m in MODELS}
    g = det[a.paper_model]
    for c in ('s', 'z_target', 'channel', 'h_max', 'radius'):
        cal['grid'][c] = sorted(g[c].unique().tolist())
    cal['n_configurations'] = int(len(g))
    halves = []
    for m in MODELS:
        kept = det[m][det[m].detect_cal >= 0.75]
        h = half_recovery(kept)
        halves.append(h)
        cal['per_model'][m] = {
            'n_configurations': int(len(det[m])),
            'n_retained': int(len(kept)),
            'recovery': {f'{x:g}': round(float(kept[f'detect_{x:g}'].mean()), 3)
                         for x in AMPS},
            'median_recovery': {f'{x:g}': round(float(kept[f'detect_{x:g}']
                                                     .median()), 3)
                                for x in AMPS},
            'half_recovery_percent': h,
            'recovery_by_geometry': {
                k: round(float(kept[f'detect_{k}'].median()), 3)
                for k in ('vertical', 'transition', 'uniform')},
        }
    cal['half_recovery_range_percent'] = [min(halves), max(halves)]
    S['calibration'] = cal

    # ---- agreement between models
    cls = {m: pd.read_csv(O(f'classification_{m}.csv')).set_index('hotspot')
           for m in MODELS}
    ag = {}
    for i, m1 in enumerate(MODELS):
        for m2 in MODELS[i + 1:]:
            j = cls[m1][['ensemble_class']].join(
                cls[m2][['ensemble_class']], rsuffix='_2', how='inner')
            ag[f'{m1} vs {m2}'] = round(
                float((j.ensemble_class == j.ensemble_class_2).mean()), 3)
    diff = (d[f'f_{a.paper_model}'] - d.f_RevealLO_30km).abs()
    S['agreement'] = {
        'class_agreement': ag,
        'two_samplings': {
            **rho(d[f'f_{a.paper_model}'], d.f_RevealLO_30km),
            # to five places, because this one rounds to 1.000 at three and
            # should not be read as an identity
            'rho_precise': round(float(spearmanr(
                d[f'f_{a.paper_model}'], d.f_RevealLO_30km)[0]), 5),
            'mean_abs_difference': round(float(diff.mean()), 3),
            'max_abs_difference': round(float(diff.max()), 3),
            'max_at': str(d.hotspot[diff.idxmax()]),
        },
    }

    # ---- the criteria of Courtillot et al., which the search never sees
    cr = {'pooled': rho(d['count'], d[POOL]),
          'paper_model': rho(d['count'], d[f'f_{a.paper_model}']),
          'per_model': {m: rho(d['count'], d[f'f_{m}']) for m in MODELS},
          'tier_mean_criteria': {k: round(float(v), 2) for k, v in
                                 t.groupby('tier').criteria.mean().items()}}
    grp = [x.values for _, x in t.groupby('tier').criteria]
    H, p = kruskal(*grp)
    cr['kruskal'] = {'H': round(float(H), 2), 'p': round(float(p), 3)}
    out = []
    for h in d.hotspot:
        s = d[d.hotspot != h]
        out.append(spearmanr(s['count'], s[POOL]))
    cr['leave_one_out'] = {
        'n_deletions_losing_significance': int(sum(o[1] >= 0.05 for o in out)),
        'rho_range': [round(min(o[0] for o in out), 3),
                      round(max(o[0] for o in out), 3)]}
    s = d[~d.hotspot.isin(t[t.tier == 'I'].hotspot)]
    cr['without_tier_I'] = rho(s['count'], s[POOL])
    S['criteria'] = cr

    # ---- longevity of the surface record
    d['age_cat'] = d.hotspot.map(CATALOGUE)
    d['age_doc'] = d.hotspot.map(lambda h: DOCUMENTED.get(h, (np.nan,))[0])
    lg = {'no_age_found': NOT_FOUND}
    pv = (pd.read_csv(O(f'province_{a.paper_model}.csv'))[
              ['hotspot', 'pct_vs_null']]
          if os.path.exists(O(f'province_{a.paper_model}.csv')) else None)
    for col, key in (('age_cat', 'catalogue'), ('age_doc', 'dated')):
        m = d.dropna(subset=[col, POOL])
        e = {'pooled': rho(m[col], m[POOL]),
             'paper_model': rho(m[col], m[f'f_{a.paper_model}']),
             'per_model': {mm: rho(m[col], m[f'f_{mm}']) for mm in MODELS},
             'vs_criteria_count': rho(m[col], m['count']),
             'n_tied_at_zero': int((m[POOL] == 0).sum())}
        oo = []
        for h in m.hotspot:
            s = m[m.hotspot != h]
            oo.append(spearmanr(s[col], s[POOL]))
        e['leave_one_out'] = {
            'n_deletions_losing_significance': int(sum(o[1] >= 0.05
                                                       for o in oo)),
            'rho_range': [round(min(o[0] for o in oo), 3),
                          round(max(o[0] for o in oo), 3)]}
        if pv is not None:
            j = m.merge(pv, on='hotspot').dropna(subset=['pct_vs_null'])
            e['province_held_fixed'] = {
                'pooled': partial(j[col], j[POOL], j.pct_vs_null),
                'paper_model': partial(j[col], j[f'f_{a.paper_model}'],
                                       j.pct_vs_null)}
        lg[key] = e
    S['longevity'] = lg

    # ---- cost of the cheapest path, for the sites section 4.2 compares
    cost = {}
    for m in (a.paper_model, 'REVEAL'):
        c = cls[m].reindex([x for x in COST_SITES if x in cls[m].index])
        ref = float(c.cost_median.min())
        cost[m] = {h: {'cost': round(float(r.cost_median), 1),
                       'ratio_to_cheapest': round(float(r.cost_median) / ref, 3),
                       'root_fraction': round(float(r.root_fraction), 3)}
                   for h, r in c.iterrows()}
    S['path_cost_by_site'] = cost

    # ---- individual hotspots the text names
    S['hotspots'] = {
        h: {**{m: round(float(d[d.hotspot == h][f'f_{m}'].iloc[0]), 3)
               for m in MODELS},
            'mean': round(float(d[d.hotspot == h][POOL].iloc[0]), 3),
            'n_deep': int(d[d.hotspot == h].n_deep.iloc[0]),
            'criteria': int(d[d.hotspot == h]['count'].iloc[0])}
        for h in ('Hawaii', 'Afar', 'Tahiti/Society', 'Pitcairn',
                  'Macdonald (Cook-Austral)', 'Iceland', 'Reunion', 'Canary',
                  'Jan Mayen', 'Louisville', 'Easter', 'Kerguelen(Heard)')
        if (d.hotspot == h).any()}

    # ---- province membership and the geochemical comparison
    if pv is not None:
        j = d.merge(pv, on='hotspot')
        S['province'] = {
            'vs_root_fraction': rho(j.pct_vs_null, j[POOL]),
            'vs_paper_model': rho(j.pct_vs_null, j[f'f_{a.paper_model}']),
            'slowest': list(pv.sort_values('pct_vs_null').hotspot[:6]),
        }
    # ---- agreement with the ordering of Boschi et al. (2007)
    bpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'bbs2007_figure14_ranking.csv')
    if os.path.exists(bpath):
        b = pd.read_csv(bpath, comment='#')
        # Their rank 1 is the hotspot they place most likely to have a deep
        # origin, so it is reversed here to run the same way as a root fraction:
        # larger means deeper on both scales, and a positive coefficient means
        # the two orderings agree.
        b['bscore'] = b['rank'].max() + 1 - b['rank']
        j = d.merge(b, on='hotspot', how='inner')
        S['boschi_2007'] = {
            'n_ranked_by_them': int(len(b)),
            'n_matched': int(len(j)),
            'not_matched': [h for h in b.hotspot if h not in set(d.hotspot)],
            'vs_root_fraction': rho(j.bscore, j[POOL]),
            'vs_paper_model': rho(j.bscore, j[f'f_{a.paper_model}']),
            'vs_criteria_count': rho(j.bscore, j['count']),
            'per_model': {m: rho(j.bscore, j[f'f_{m}']) for m in MODELS},
        }
        if pv is not None:
            k = j.merge(pv, on='hotspot').dropna(subset=['pct_vs_null'])
            S['boschi_2007']['vs_province'] = rho(k.bscore, k.pct_vs_null)
            S['boschi_2007']['ours_vs_province'] = rho(k[POOL], k.pct_vs_null)
            S['boschi_2007']['province_held_fixed'] = partial(
                k.bscore, k[POOL], k.pct_vs_null)

    # ---- where each named hotspot falls in the null, under the single
    # configuration the null-snapshot figure is drawn on
    npz = O(f'null_{a.paper_model}.npz')
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        null, cst = z['null'], z['cost']
        names = np.array([str(x) for x in z['hotspot']])
        S['null_snapshot'] = {
            'configuration': {'s': float(z['s']), 'z_target': float(z['z_target']),
                              'channel': str(z['channel']),
                              'radius_deg': float(z['radius'])},
            'n_random': int(len(null)),
            'p5': round(float(z['p5']), 1),
            'null_percentiles': {str(q): round(float(np.percentile(null, q)), 1)
                                 for q in (1, 5, 10, 25, 50, 75, 95)},
            # the percentile of the null each hotspot's path cost falls at;
            # rooting under this configuration means falling below the 5th
            'hotspot_percentile': {
                h: round(100.0 * float((null < cst[np.where(names == h)[0][0]])
                                       .mean()), 1)
                for h in COST_SITES if (names == h).any()},
        }

    # ---- anomaly beneath the compared sites, and each model's shell r.m.s.
    # site_profile.py writes these from the model cubes; reading them here keeps
    # every number the text quotes in one file.
    sp = {}
    for m in MODELS:
        f = O(f'site_profiles_{m}.csv')
        if not os.path.exists(f):
            continue
        p_ = pd.read_csv(f)
        shell = p_[p_.hotspot == '(shell)']
        e = {'shell_rms': {}}
        for z in (1000, 1500, 2000, 2400, 2800):
            r = shell.iloc[(shell.depth_km - z).abs().argmin()]
            e['shell_rms'][str(z)] = round(float(r.rms_dvs), 3)
        e['sites'] = {}
        for h in p_[p_.hotspot != '(shell)'].hotspot.unique():
            x = p_[p_.hotspot == h]
            at = lambda z, c: round(float(
                x.iloc[(x.depth_km - z).abs().argmin()][c]), 2)
            band = lambda z0, z1, c: round(float(
                x[(x.depth_km >= z0) & (x.depth_km <= z1)][c].mean()), 2)
            e['sites'][h] = {
                # The quantity the discussion quotes is the slowest material
                # available in the cap, averaged over depth - not the cap mean,
                # which is much weaker because the cap is mostly background.
                'slowest_at_660': at(660, 'min_dvs'),
                'slowest_at_100': at(100, 'min_dvs'),
                'contrast_at_100': at(100, 'min_contrast'),
                'slowest_mean_1000_2000': band(1000, 2000, 'min_dvs'),
                'slowest_mean_2000_2880': band(2000, 2880, 'min_dvs'),
                'cap_mean_1000_2000': band(1000, 2000, 'mean_dvs'),
                'cap_mean_2000_2880': band(2000, 2880, 'mean_dvs'),
            }
        sp[m] = e
    if sp:
        S['site_profiles'] = sp

    # ---- ocean island basalt geochemistry against the two depth measures
    gpath = O(f'geochem_link_{a.paper_model}.csv')
    if os.path.exists(gpath):
        g = pd.read_csv(gpath)
        g['deepest_km'] = g.deepest_km.fillna(0.0)
        e = {'source': os.path.basename(gpath), 'n_hotspots': int(len(g)),
             'n_samples': int(g.n_samples.sum()), 'measures': {}}
        e['their_ranking_vs_province'] = rho(*_pair(g, 'bbs_score', 'dVs_deep'))
        e['depth_here_vs_province'] = rho(*_pair(g, 'deepest_km', 'dVs_deep'))
        e['root_fraction_vs_province'] = rho(*_pair(g, 'f_ind_mean', 'dVs_deep'))
        e['their_ranking_vs_depth_here'] = rho(*_pair(g, 'bbs_score', 'deepest_km'))
        e['their_ranking_vs_root_fraction'] = rho(*_pair(g, 'bbs_score', 'f_ind_mean'))
        for c, lab in (('max_3He/4He (R/RA)', 'max_He3_He4'),
                       ('max_87Sr/86Sr', 'max_Sr87_Sr86'),
                       ('max_206Pb/204Pb', 'max_Pb206_Pb204'),
                       ('max_208Pb/204Pb', 'max_Pb208_Pb204'),
                       ('max_187Os/188Os', 'max_Os187_Os188')):
            k = g[c].notna() & g.bbs_score.notna()
            k2 = g[c].notna()
            k3 = k & g.dVs_deep.notna()
            e['measures'][lab] = {
                'vs_their_ranking': rho(g.loc[k, c], g.loc[k, 'bbs_score']),
                'vs_depth_here': rho(g.loc[k2, c], g.loc[k2, 'deepest_km']),
                'vs_root_fraction_here': rho(g.loc[k2, c], g.loc[k2, 'f_ind_mean']),
                'their_ranking_province_held_fixed': partial(
                    g.loc[k3, c], g.loc[k3, 'bbs_score'], g.loc[k3, 'dVs_deep']),
            }
        S['geochemistry'] = e

    with open(O('manuscript_stats.json'), 'w') as f:
        json.dump(S, f, indent=1, sort_keys=False)
    print(json.dumps(S, indent=1)[:1])
    print(f'wrote {O("manuscript_stats.json")}: '
          f'{len(json.dumps(S))} bytes, {len(S)} sections')
    for k in S:
        print(f'   {k}')


if __name__ == '__main__':
    main()
