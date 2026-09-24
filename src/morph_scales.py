#!/usr/bin/env python3
"""What the fine model adds, measured by taking it away.

RevealLO is smoothed laterally to a ladder of length scales and the whole
analysis is repeated on each copy. Everything else - the data, the inversion, the
calibration, the configurations, the search - is held fixed, so the only thing
that changes is how much spatial detail the model is allowed to carry. Any
difference between the ladder rungs is therefore attributable to resolution and
to nothing else, which is what a comparison against an independently constructed
model can never claim.

Two readings come out of it.

  WHAT SURVIVES. A structure still present after smoothing to 800 km was never
  the fine model's contribution, and could have been found in a coarse model. It
  is the safest thing to interpret and the least interesting.

  WHAT DISAPPEARS. A structure that vanishes under smoothing is exactly what
  RevealLO adds. Whether it is worth believing is then decided by whether the
  analysis can recover an injected structure of that size and amplitude in the
  same model - which synth_morph.py measures - and not by whether a
  coarser model happens to carry it. A model with an order of magnitude larger
  parameterisation cannot represent a 200 km conduit at all, so its silence about
  one is not evidence against it.

The second reading is the point. On a controlled test the displacement of a
corridor from its hotspot fell from 563 km to 20 km when the model was smoothed
to 900 km: the coarse model does not disagree about the displacement, it cannot
express it.
"""
from __future__ import annotations

import argparse, os, re, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

METRICS = ['offset_km_lower', 'r90_km_lower', 'r50_km_lower', 'anisotropy_lower',
           'ncomp_max_lower', 'pooled_jaccard', 'offset_km_mid', 'r90_km_mid']
NICE = {'offset_km_lower': 'displacement, lower mantle (km)',
        'r90_km_lower': 'corridor width r90, lower mantle (km)',
        'r50_km_lower': 'corridor width r50, lower mantle (km)',
        'anisotropy_lower': 'anisotropy, lower mantle',
        'ncomp_max_lower': 'components, lower mantle',
        'pooled_jaccard': 'corridor stability',
        'offset_km_mid': 'displacement, mid mantle (km)',
        'r90_km_mid': 'corridor width r90, mid mantle (km)'}


def scale_of(tag, base):
    if tag == base:
        return 0.0
    m = re.search(r'_s(\d+)$', tag)
    return float(m.group(1)) if m else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='RevealLO')
    ap.add_argument('--tags', nargs='+', default=None,
                    help='the ladder; default is the base plus every '
                         '<base>_s<km> summary present')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--field', default='corridor0.02')
    ap.add_argument('--out', default='figures/fig_scales.pdf')
    a = ap.parse_args()

    if a.tags is None:
        a.tags = [a.base] + sorted(
            {re.sub(r'^morph_summary_|_pilot\.csv$|\.csv$', '', f)
             for f in os.listdir(a.dir)
             if f.startswith(f'morph_summary_{a.base}_s')},
            key=lambda t: scale_of(t, a.base))
    rows = []
    for t in a.tags:
        f = None
        for sfx in ('', '_pilot'):
            p = os.path.join(a.dir, f'morph_summary_{t}{sfx}.csv')
            if os.path.exists(p):
                f = p
                break
        if f is None:
            print(f'  {t}: no summary, skipped')
            continue
        d = pd.read_csv(f)
        d = d[(~d.site.str.startswith('null')) & (d.field == a.field)]
        d['smoothing_km'] = scale_of(t, a.base)
        d['tag'] = t
        rows.append(d)
    if len(rows) < 2:
        raise SystemExit('need the base model and at least one smoothed copy')
    d = pd.concat(rows, ignore_index=True).sort_values(['site', 'smoothing_km'])
    keep = ['site', 'tag', 'smoothing_km'] + [m for m in METRICS if m in d.columns]
    d[keep].to_csv(os.path.join(a.dir, 'morph_scales.csv'), index=False)

    print(f'ladder: ' + ', '.join(f'{t} ({scale_of(t, a.base):.0f} km)'
                                  for t in d.tag.unique()))
    for m in [q for q in METRICS if q in d.columns]:
        piv = d.pivot_table(index='site', columns='smoothing_km', values=m)
        print(f'\n{NICE.get(m, m)}')
        print(piv.to_string(float_format=lambda x: f'{x:.0f}' if abs(x) > 10
                            else f'{x:.2f}'))
        base = piv[0.0] if 0.0 in piv.columns else None
        if base is not None and len(piv.columns) > 1:
            far = piv[max(piv.columns)]
            with np.errstate(invalid='ignore', divide='ignore'):
                frac = (far - base) / base.replace(0, np.nan)
            print(f'  median change from native to {max(piv.columns):.0f} km '
                  f'smoothing: {100 * frac.median():+.0f} per cent')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import figstyle
        from figstyle import CM, INK, ACC, BLU, GRY
        figstyle.apply(10.0)
        show = [m for m in ('offset_km_lower', 'r90_km_lower', 'ncomp_max_lower',
                            'pooled_jaccard') if m in d.columns]
        fig, axes = plt.subplots(1, len(show), figsize=(17.5 * CM, 8.4 * CM))
        axes = np.atleast_1d(axes)
        fig.subplots_adjust(left=0.085, right=0.99, bottom=0.38, top=0.94,
                            wspace=0.32)
        sites = sorted(d.site.unique())
        for ax, m in zip(axes, show):
            for s in sites:
                g = d[d.site == s].sort_values('smoothing_km')
                ax.plot(g.smoothing_km.values, g[m].values, '-o', ms=3,
                        lw=1.0, color=GRY, alpha=0.75, zorder=2)
            med = d.groupby('smoothing_km')[m].median()
            ax.plot(med.index.values, med.values, '-o', ms=5, lw=2.2,
                    color=ACC, zorder=4, label='median')
            ax.set_xlabel('lateral smoothing (km)', fontsize=9.5)
            ax.set_title(NICE.get(m, m).split(',')[0], fontsize=9.5, pad=4)
            ax.grid(axis='y', color=figstyle.GRID, lw=0.6)
            ax.set_axisbelow(True)
        axes[0].legend(frameon=False, fontsize=9, loc='upper right')
        fig.text(0.085, 0.045, 'grey: one hotspot.  Zero smoothing is RevealLO '
                 'at its native sampling.\nEverything but the spatial detail is '
                 'held fixed, so a change along an axis is resolution and '
                 'nothing else.', fontsize=9, color=INK, linespacing=1.35)
        figstyle.check(fig, placed_cm=17.5)
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        fig.savefig(a.out, bbox_inches='tight')
        fig.savefig(a.out.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
        print(f'\nwrote {a.out}')
    except Exception as e:
        print(f'\nfigure not drawn: {e}')
    print(f'wrote morph_scales.csv')


if __name__ == '__main__':
    main()
