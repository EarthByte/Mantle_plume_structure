"""Stage 46 - compare the calibrated path classification across tomographic models.

Each model is calibrated on itself, because the amplitude of its own
heterogeneity, and therefore the sensitivity of the search and the height of the
bar the null sets, are properties of that model. What is compared is the outcome
of the same procedure, not the outcome of the same numbers.

Six models are used. Two are the same inversion at two lateral samplings, which
measures the effect of sampling alone; the remaining four are independently
constructed, and two of those - SPiRaL and SEMUCB-WM1 - do not share REVEAL's
data or method.

A hotspot whose class survives across independently constructed models is a
statement about the mantle. One that does not is a statement about the model.
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import provenance

CM = 1 / 2.54
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.linewidth': 1.0,
    'pdf.fonttype': 42, 'savefig.dpi': 400,
    'axes.spines.top': False, 'axes.spines.right': False})
INK, ACC, BLU, GRY = '#1a1a1a', '#B4442E', '#2E5E8E', '#9a9a9a'
ORDER = ['RevealLO_05deg', 'RevealLO_1deg', 'REVEAL', 'GLADM35', 'SPiRaL',
         'SEMUCB-WM1']
NICE = {'RevealLO_30km': 'RevealLO (30 km)', 'RevealLO': 'RevealLO',
        'RevealLO_1deg': 'RevealLO (1$\\degree$)',
        'RevealLO_05deg': 'RevealLO (0.5$\\degree$)', 'REVEAL': 'REVEAL',
        'GLADM35': 'GLAD-M35', 'SPiRaL': 'SPiRaL', 'SEMUCB-WM1': 'SEMUCB-WM1'}
MARK = {'RevealLO_1deg': 'o', 'RevealLO_05deg': 's', 'REVEAL': '^',
        'GLADM35': 'D', 'SPiRaL': 'v', 'SEMUCB-WM1': 'P'}
# RevealLO at two samplings is one model. Counting it twice would inflate any
# consensus statement, so the independent-model count excludes the duplicate.
def base_model(tag):
    """The model a run belongs to.

    A tag is either a model name or a model name and a variant, separated by an
    underscore: RevealLO and RevealLO_30km are the same model at two depth
    samplings, and must count once in any consensus statement. Deriving this
    from the tag rather than listing the runs by hand is deliberate - the list
    silently went stale when the tags changed, and dropped the paper's primary
    model out of the count without erroring.
    """
    return str(tag).split('_')[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='out')
    ap.add_argument('--out', default='figures')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tabs = {}
    for f in sorted(glob.glob(os.path.join(a.dir, 'classification_*.csv'))):
        tag = os.path.basename(f)[len('classification_'):-4]
        tabs[tag] = pd.read_csv(f)
    tags = [t for t in ORDER if t in tabs] + [t for t in tabs if t not in ORDER]
    if not tags:
        raise SystemExit('no classification tables found')
    print('models: ' + ', '.join(tags))

    base = tabs[tags[0]][['hotspot', 'lat', 'lon_180']].copy()
    if 'count' in tabs[tags[0]].columns:
        base['count'] = tabs[tags[0]]['count']
    for t in tags:
        d = tabs[t][['hotspot', 'root_fraction', 'ensemble_class', 'cost_median']]
        base = base.merge(d.rename(columns={
            'root_fraction': f'f_{t}', 'ensemble_class': f'c_{t}',
            'cost_median': f'w_{t}'}), on='hotspot', how='left')

    fcols = [f'f_{t}' for t in tags]
    base['f_mean'] = base[fcols].mean(axis=1)
    base['f_min'] = base[fcols].min(axis=1)
    base['f_max'] = base[fcols].max(axis=1)
    # one run per model: the first listed, ORDER having put the preferred
    # variant first
    seen, ind = set(), []
    for t_ in tags:
        b = base_model(t_)
        if b not in seen:
            seen.add(b)
            ind.append(t_)
    base['n_deep'] = sum((base[f'c_{t}'] == 'deeper than null').astype(int)
                         for t in ind)
    base['n_models'] = len(ind)
    base['f_ind_mean'] = base[[f'f_{t}' for t in ind]].mean(axis=1)
    base['consensus'] = np.where(
        base.n_deep == len(ind), 'deep in every model',
        np.where(base.n_deep == 0, 'deep in none',
                 np.where(base.n_deep >= (len(ind) + 1) // 2,
                          'deep in a majority', 'deep in a minority')))
    base = base.sort_values('f_mean', ascending=False)
    _cm = os.path.join(a.dir, 'comparison_all_models.csv')
    base.to_csv(_cm, index=False)
    provenance.stamp(_cm, n_rows=len(base))

    print('\n' + '=' * 92)
    print('CROSS-MODEL COMPARISON: fraction of calibrated configurations in which')
    print('the path beneath the hotspot is cheaper than that model\'s own null')
    print('=' * 92)
    print(f'independent models counted ({len(ind)}): ' + ', '.join(ind))
    dup = [t_ for t_ in tags if t_ not in ind]
    if dup:
        print('additional runs of those same models, not counted again: '
              + ', '.join(dup))
    show = ['hotspot'] + fcols + ['n_deep', 'consensus']
    if 'count' in base.columns:
        show.append('count')
    print(base[show].head(22).to_string(index=False,
                                        float_format=lambda x: f'{x:.2f}'))
    print('\nconsensus summary')
    print(base.consensus.value_counts().to_string())

    # pairwise agreement on the three-way class
    print('\npairwise agreement on class:')
    for i, t1 in enumerate(tags):
        for t2 in tags[i + 1:]:
            m = base[[f'c_{t1}', f'c_{t2}']].dropna()
            if len(m):
                print(f'  {NICE.get(t1, t1)} vs {NICE.get(t2, t2)}: '
                      f'{(m[f"c_{t1}"] == m[f"c_{t2}"]).mean():.0%} '
                      f'({len(m)} hotspots)')

    if 'count' in base.columns:
        from scipy.stats import spearmanr
        m = base.dropna(subset=['count', 'f_mean'])
        rho, p = spearmanr(m['count'], m.f_mean)
        print(f'\nCourtillot criteria count vs mean root fraction: '
              f'rho = {rho:+.3f}, p = {p:.4f}, n = {len(m)}')
        print('per model:')
        for t in tags:
            mm = base.dropna(subset=['count', f'f_{t}'])
            r_, p_ = spearmanr(mm['count'], mm[f'f_{t}'])
            print(f'  {NICE.get(t, t):20s} rho = {r_:+.3f}  p = {p_:.4f}')

    # ---------------------------------------------------------------- figure
    top = base.head(24).iloc[::-1]
    fig, ax = plt.subplots(figsize=(14.0 * CM, 13.5 * CM), constrained_layout=True)
    y = np.arange(len(top))
    ax.axvspan(0.8, 1.0, color='#f3e2de', zorder=0)
    ax.axvspan(0.0, 0.2, color='#e5eaf0', zorder=0)
    for j, row in enumerate(top.itertuples()):
        ax.plot([getattr(row, 'f_min'), getattr(row, 'f_max')], [j, j],
                color='#cccccc', lw=3.0, solid_capstyle='round', zorder=1)
    for t in tags:
        ax.scatter(top[f'f_{t}'], y, s=22, marker=MARK.get(t, 'o'),
                   facecolor='white', edgecolor=INK, linewidth=0.9,
                   zorder=3, label=NICE.get(t, str(t).replace('_', ' ')))
    ax.set_yticks(y)
    ax.set_yticklabels([str(h).split('(')[0].strip() for h in top.hotspot])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel('fraction of calibrated configurations in which the path\nis cheaper than the null')
    ax.axvline(0.8, color=ACC, lw=1.0, ls='--')
    ax.axvline(0.2, color=BLU, lw=1.0, ls='--')
    t1 = ax.text(0.985, len(top) - 0.4, 'rooted', ha='right',
                 va='top', fontsize=7, color=ACC)
    t2 = ax.text(0.015, len(top) - 0.4, 'not rooted', ha='left', va='top',
                 fontsize=7, color=BLU)
    for t_ in (t1, t2):
        t_.set_path_effects([pe.withStroke(linewidth=2.2, foreground='white')])
    ax.legend(frameon=False, loc='lower right', handletextpad=0.3)
    ax.tick_params(axis='y', length=0)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, f'fig_models.{ext}'),
                    bbox_inches='tight', facecolor='white')
    print('\nwrote comparison_all_models.csv and fig_models.pdf')


if __name__ == '__main__':
    main()
