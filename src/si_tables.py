"""Generate the SI data tables from the files in out/, as markdown.

A table typed by hand goes stale the moment anything upstream is re-run, and it
goes stale silently: Table S2 carried R and P from a superseded move set while the
rest of the supplement had moved on, and a check that looked for those digits
anywhere in the text found them in Table S1 and passed. Numbers that come out of a
file cannot drift from it. Everything here is derived; nothing is typed.

    python3 si_tables.py --table S2 > /tmp/S2.md

--check compares what is in the supplement against what this script generates and
exits non-zero if they differ, so the drift is caught rather than discovered.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--si', default='../manuscript/G3_SI_v7.md')
ap.add_argument('--table', default='S2', choices=['S2', 'S4', 'S5', 'S6'])
ap.add_argument('--check', action='store_true',
                help='compare the supplement against the generated table and exit '
                     'non-zero on any difference')
A = ap.parse_args()

# The order the supplement presents them in: the paper model first, its resolution
# control beside it, then the independent models. Ordering by the file would let a
# re-run reorder the table.
ORDER = ['RevealLO', 'RevealLO_30km', 'REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1']
import model_names
LABEL = model_names.DISPLAY
WINDOWS = ['300-660', '660-1500']


def table_s2(d):
    cm = pd.read_csv(os.path.join(d, 'crossmodel_lean.csv'))
    ref = cm[(cm.model == 'RevealLO') & (cm.window == '300-660')]
    r_ref = float(ref.R.iloc[0]) if len(ref) and np.isfinite(ref.R.iloc[0]) else np.nan

    out = []
    # R and P in italics, as statistics are everywhere else in the paper.
    out.append(f'| Model | Window (km) | Paths tested | Paths excluded | *R* | *P* | '
               f'Power vs {r_ref:.3f} |')
    out.append('|---|---|---|---|---|---|---|')
    for w in WINDOWS:
        for m in ORDER:
            r = cm[(cm.model == m) & (cm.window == w)]
            if not len(r):
                continue
            r = r.iloc[0]
            testable = np.isfinite(r.R)
            name = LABEL[m] + ('' if testable else ' (not testable)')
            _ex = r.excluded if 'excluded' in r.index else float('nan')
            vert = '—' if not np.isfinite(_ex) else f'{int(_ex)}'
            RR = '—' if not testable else f'{float(r.R):.3f}'
            PP = '—' if not testable else f'{float(r.P):.4f}'
            # Power is reported only where the test returned a null. Quoting it
            # beside a significant result invites reading it as the power of a
            # positive finding, which is not a quantity.
            pw = ('—' if not (testable and np.isfinite(r.power) and float(r.P) >= 0.05)
                  else f'{float(r.power):.2f}')
            out.append(f'| {name} | {w.replace("-", "–")} | {int(r.n)} | {vert} | '
                       f'{RR} | {PP} | {pw} |')
    return '\n'.join(out)


BANDS5 = ['400-660', '660-1000', '1000-1500', '1500-2200', '2200-2700']


def _summary(d):
    return pd.read_csv(os.path.join(d, 'mid_mantle_summary.csv'))


def table_s4(d):
    """Connected slow columns, hotspots against random sites, by model."""
    S = _summary(d)
    out = ['| Model | Column reaches 2600 km, hotspots | Column reaches 2600 km, random sites | '
           'Median root, hotspots (km) | Median root, random sites (km) | '
           'Basal window slow, hotspots | Basal window slow, random sites |',
           '|---|---|---|---|---|---|---|']
    for m in ORDER:
        r = S[(S.tag == m) & (S.quantity == 'root_ge_2600_share')]
        if not len(r):
            continue
        r = r.iloc[0]
        rk = S[(S.tag == m) & (S.quantity == 'root_km_median')].iloc[0]
        df = S[(S.tag == m) & (S.quantity == 'deep_frac_median')].iloc[0]
        out.append(f'| {LABEL[m]} | {100 * r.hotspot:.1f}% ({int(r.n_hotspot)}) | '
                   f'{100 * r.ambient:.1f}% ({int(r.n_ambient)}) | {rk.hotspot:.0f} | '
                   f'{rk.ambient:.0f} | {df.hotspot:.2f} | {df.ambient:.2f} |')
    return '\n'.join(out)


def table_s5(d):
    """The depth census: every quantity by band, hotspots against the control."""
    S = _summary(d)
    QUANT = [('shell_percentile', 'shell percentile', '{:.3f}'),
             ('prominence', 'prominence of the minimum (%)', '{:.2f}'),
             ('tilt_deg', 'apparent tilt (°)', '{:.1f}'),
             ('deflection_share', 'share of deflections (%)', '{:.1f}', 100.0),
             ('half_width_km', 'half-width of the minimum (km)', '{:.0f}'),
             ('corridor_width_km', 'corridor width (km)', '{:.0f}')]
    out = ['| Quantity | Model | Population | ' + ' | '.join(b.replace('-', '–') + ' km' for b in BANDS5) + ' |',
           '|---|---|---|' + '---|' * len(BANDS5)]
    for q in QUANT:
        key, name, fmt = q[:3]
        scale = q[3] if len(q) > 3 else 1.0
        for m in ORDER:
            r = S[(S.tag == m) & (S.quantity == key)].set_index('band')
            if not len(r):
                continue
            for pop, col in (('hotspots', 'hotspot'), ('control', 'ambient')):
                vals = [r[col].get(b, np.nan) for b in BANDS5]
                if not np.isfinite(vals).any():
                    continue
                cells = ['—' if not np.isfinite(v) else fmt.format(scale * v) for v in vals]
                if col == 'hotspot' and key in ('tilt_deg', 'shell_percentile') and np.isfinite(r['P']).any():
                    def _pp(v):
                        return '*P* < 0.001' if v < 0.001 else f'*P* = {v:.3f}'
                    cells = [c + (f' ({_pp(r["P"].get(b))})' if np.isfinite(r['P'].get(b, np.nan)) and r['P'].get(b) < 0.05 else '')
                             for c, b in zip(cells, BANDS5)]
                out.append(f'| {name} | {LABEL[m]} | {pop} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def table_s6(d):
    """The deflection horizon test: the share of 660-2200 km deflections that fall in
    the 800-1200 km range of the viscosity increase, against the 26.0 per cent its
    thickness would give it under a uniform spread."""
    S = _summary(d)
    out = ['| Model | Population | Deflections at 800–1200 km, share of 660–2200 km | Ratio to uniform (26.0%) |',
           '|---|---|---|---|']
    for m in ORDER:
        r = S[(S.tag == m) & (S.quantity == 'horizon_ratio_800_1200')]
        if not len(r):
            continue
        r = r.iloc[0]
        for pop, col in (('hotspots', 'hotspot'), ('ambient paths', 'ambient')):
            v = r[col]
            if not np.isfinite(v):
                continue
            out.append(f'| {LABEL[m]} | {pop} | {100 * v * 400 / 1540:.1f}% | {v:.2f} |')
    return '\n'.join(out)


GEN = {'S2': table_s2, 'S4': table_s4, 'S5': table_s5, 'S6': table_s6}[A.table](A.dir)
HEAD = {'S2': r'^\| Model \| Window', 'S4': r'^\| Model \| Column reaches', 'S5': r'^\| Quantity \| Model',
        'S6': r'^\| Model \| Population \| Deflections'}[A.table]

if not A.check:
    print(GEN)
    sys.exit(0)

si = open(A.si).read() if os.path.exists(A.si) else ''
m = re.search(HEAD + r'.*?(?=\n\s*\n|\Z)', si, re.M | re.S)
if not m:
    sys.exit(f'no Table {A.table} found in {A.si}')
have = [x.rstrip() for x in m.group(0).strip().splitlines()]
want = [x.rstrip() for x in GEN.strip().splitlines()]
if have == want:
    print(f'Table {A.table} matches the outputs on disk')
    sys.exit(0)
print(f'Table {A.table} DIFFERS from the outputs on disk\n')
for i in range(max(len(have), len(want))):
    a = have[i] if i < len(have) else '(missing)'
    b = want[i] if i < len(want) else '(missing)'
    if a != b:
        print(f'  supplement: {a}\n  computed  : {b}\n')
sys.exit(1)
