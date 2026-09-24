#!/usr/bin/env python3
"""Apply the single-axis rule to the cost channel, for every model, from the screen.

THE RULE, WHICH IS FIXED BEFORE THE ANSWER IS LOOKED AT. A model's configuration is the
one its own synthetic-recovery screen selects. Where the frozen configuration no longer
clears the 75 per cent floor that defines its retained set, the single axis that restores
it is changed and no other; re-deriving the median moves several axes at once on
differences of one injection in eighteen. Where a channel that was never screened is now
screened, the channel is decided the same way the rest of the configuration was: by
injection recovery at the model's own frozen axes, holding every other axis fixed.

THE CHANGE NEEDS A MARGIN, OR THE RULE REPRODUCES THE DEFECT IT REPLACES. Recovery here
is a proportion of eighteen or twenty-four injections, where one trial is 0.056 or 0.042.
Taking simply the highest would move SPiRaL on a quarter of a trial and SEMUCB-WM1 on one,
which is the single-trial instability that made the median-of-the-retained-set selector
unusable in the first place. A channel is therefore changed only when it wins by at least
MARGIN_TRIALS whole injections, or when the frozen channel is below the floor and the
alternative clears it.

This reports that comparison and nothing else. It changes no frozen file. The decision it
supports is about the CALIBRATION, not about how the traced paths look, which is what
keeps it a test rather than a search - the path geometry is reported separately by
moveset_check.py and is not an input here.

    python3 channel_decision.py
"""
from __future__ import annotations
import json
import os

import pandas as pd

FLOOR, MIN_CASES = 0.75, 4
MARGIN_TRIALS = 3        # whole injections a challenger must win by
TAGS = ('RevealLO', 'RevealLO_30km', 'REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1')
AMPS = ['detect_0.5', 'detect_0.75', 'detect_1', 'detect_1.5']
GEOM = ['detect_vertical', 'detect_transition', 'detect_uniform']
D = 'out'


def row_at(tab, c, channel):
    q = tab[(tab.s == float(c['s'])) & (tab.z_target == float(c['z_target']))
            & (tab.channel == channel) & (tab.h_max == float(c['h_max']))
            & (tab.radius == float(c['radius']))]
    return q.iloc[0] if len(q) else None


print('injection recovery at each model\'s own frozen axes, by channel.')
print('Every axis but the channel is held at the frozen value.\n')
print(f'{"model":15s} {"channel":9s} {"n_cal":>5s} {"cal":>6s}  '
      + ' '.join(f'{a.replace("detect_",""):>6s}' for a in AMPS)
      + '   vert  trans  unif   verdict')
decisions = []
for tag in TAGS:
    fp = os.path.join(D, f'path_config_{tag}.json')
    base = os.path.join(D, f'detection_{tag}.csv')
    if not (os.path.exists(fp) and os.path.exists(base)):
        continue
    c = json.load(open(fp))['config']
    tabs = {'base': pd.read_csv(base)}
    cf = os.path.join(D, f'detection_{tag}_contrast.csv')
    if os.path.exists(cf):
        tabs['contrast'] = pd.read_csv(cf)
    best, seen = None, []
    for ch in ('anom', 'min', 'contrast'):
        tab = tabs.get('contrast') if ch == 'contrast' else tabs['base']
        if tab is None:
            continue
        r = row_at(tab, c, ch)
        if r is None:
            if ch == 'contrast':
                print(f'{tag:15s} contrast  not screened yet')
            continue
        ok = r.n_cal >= MIN_CASES and r.detect_cal >= FLOOR
        mark = ' <- frozen' if ch == str(c['channel']) else ''
        print(f'{tag:15s} {ch:9s} {int(r.n_cal):5d} {r.detect_cal:6.3f}  '
              + ' '.join(f'{r[a]:6.3f}' for a in AMPS) + '  '
              + ' '.join(f'{r[g]:6.3f}' for g in GEOM)
              + f'   {"clears" if ok else "below"}{mark}')
        # The margin is counted in WHOLE INJECTIONS, from the integer number recovered,
        # not from a difference of two floats times n. (1.0 - 15/18) * 18 is
        # 2.9999999999999996, so REVEAL and SPiRaL - both winning by exactly three
        # injections - fell on opposite sides of a three-injection margin. A rule meant
        # to stop configurations flipping on one trial must not itself flip on the last
        # bit of a float.
        seen.append((ch, float(r.detect_cal), ok, int(r.n_cal),
                     int(round(float(r.detect_cal) * int(r.n_cal)))))
        if ok and (best is None or r.detect_cal > best[1]):
            best = (ch, float(r.detect_cal), int(r.n_cal),
                    int(round(float(r.detect_cal) * int(r.n_cal))))
    cur = str(c['channel'])
    cur_rec = next((d for ch, d, _, _, _ in seen if ch == cur), None)
    cur_ok = next((o for ch, _, o, _, _ in seen if ch == cur), False)
    cur_k = next((k for ch, _, _, _, k in seen if ch == cur), 0)
    sel, why = cur, 'frozen channel stands'
    gain = 0
    if best and best[0] != cur:
        gain = best[3] - cur_k
        if not cur_ok:
            sel, why = best[0], f'frozen channel is below the floor; {best[0]} clears'
        elif gain >= MARGIN_TRIALS:
            sel, why = best[0], f'wins by {gain} injections, past the {MARGIN_TRIALS}-trial margin'
        else:
            why = f'best alternative wins by only {gain} injections, inside the margin'
    decisions.append((tag, cur, sel, best[1] if best else float('nan'), cur_rec, why))
    print()

print(f'what the rule selects, with a {MARGIN_TRIALS}-injection margin\n')
print(f'{"model":15s} {"frozen":9s} {"rec":>6s}   {"selected":9s} {"rec":>6s}   why')
for tag, cur, sel, dsel, dcur, why in decisions:
    print(f'{tag:15s} {cur:9s} {"  n/a" if dcur is None else f"{dcur:6.3f}"}   '
          f'{sel:9s} {dsel:6.3f}   {why}')
n_change = sum(1 for d in decisions if d[1] != d[2])
print(f'\n{n_change} of {len(decisions)} models change channel')
print('\nA change here is a calibration result. It still costs a full retrace of the model '
      'it applies to,\nand every number derived from that model moves with it.')
