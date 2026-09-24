#!/usr/bin/env python3
"""Which injected vertical occupancies are excluded by the observed corridors.

Section 3.7 states that the model runs reject occupancies of 50 per cent or more.
That claim had no script behind it: the numbers were produced once, by hand, and
there was no way to reproduce them or to re-run them after the move set changed.
This is that test, written down.

THE TEST. For one model, take the corridor widths measured at its hotspots and the
corridor widths measured at ambient sites into which a conduit of known occupancy was
injected. If the real structure were a conduit occupying that fraction of the column,
the observed widths should look like the injected ones. A one-sided Mann-Whitney asks
whether the observed widths are wider than the injected: a small P says the observed
corridors are too broad for that occupancy, and the occupancy is excluded.

Each model is compared only with its own injection curve, so smearing that widens
both sides cancels. Comparing across models would not, which is why the bound is a
statement per run and not a pooled one.

Only sweeps carrying the current move set are used. Three of the four sweeps behind
the published claim were run at the old one, which is why the files now record
lateral and ncell and why this script refuses to mix them.

  python3 occupancy_bound.py --lateral 0.60 --ncell 2
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--tags', default='RevealLO,RevealLO_30km,GLADM35,SEMUCB-WM1')
ap.add_argument('--lateral', type=float, default=0.60,
                help='require sweeps recorded at this lateral weight')
ap.add_argument('--ncell', type=int, default=2,
                help='require sweeps recorded at this many lateral cells per move')
ap.add_argument('--alpha', type=float, default=0.05)
ap.add_argument('--tau', default='0.02', help='the corridor tolerance to compare at')
ap.add_argument('--assume', default='', metavar='TAG=LATERAL:NCELL',
                help='declare the move set of a sweep written before the files '
                     'recorded it, e.g. RevealLO=0.6:2. Use it only where the run is '
                     'known from its command, never inferred from its date, and it is '
                     'written into the result so the assumption travels with it. '
                     'Comma-separate several.')
ap.add_argument('--allow-unstamped', action='store_true', dest='allow_unstamped',
                help='use every unstamped sweep regardless. They predate the stamping '
                     'and are almost certainly the old move set; the bound they give '
                     'is not comparable with an observed corridor traced at the new '
                     'one. Only for reproducing what the old text said.')
A = ap.parse_args()

ASSUMED = {}
for _a in filter(None, (x.strip() for x in A.assume.split(','))):
    _t, _v = _a.split('=')
    _l, _n = _v.split(':')
    ASSUMED[_t.strip()] = (float(_l), int(_n))

rows, skipped = [], []
for tag in A.tags.split(','):
    f = sorted(glob.glob(os.path.join(
        A.dir, f'corridor_width_calibration_{tag}_dutysweep*.csv')))
    obs_f = os.path.join(A.dir, f'corridor_summary_{tag}.csv')
    if not f or not os.path.exists(obs_f):
        skipped.append((tag, 'no sweep or no corridor summary'))
        continue
    sw = pd.read_csv(f[-1])

    if 'lateral' in sw.columns and 'ncell' in sw.columns:
        got = (float(sw.lateral.dropna().iloc[0]), int(sw.ncell.dropna().iloc[0]))
        if got != (A.lateral, A.ncell):
            skipped.append((tag, f'sweep is at lateral {got[0]}, ncell {got[1]}'))
            continue
    elif tag in ASSUMED:
        if ASSUMED[tag] != (A.lateral, A.ncell):
            skipped.append((tag, f'declared at lateral {ASSUMED[tag][0]}, '
                                 f'ncell {ASSUMED[tag][1]}'))
            continue
        print(f'  {tag}: move set not recorded in the file; taking the declared '
              f'lateral {A.lateral}, ncell {A.ncell}')
    elif not A.allow_unstamped:
        skipped.append((tag, f'{os.path.basename(f[-1])} records no move set '
                             f'(predates stamping; re-run it, or --assume it)'))
        continue

    obs = pd.read_csv(obs_f)
    col = f'width_med_{A.tau}'
    if col not in obs.columns:
        skipped.append((tag, f'no {col} in the corridor summary'))
        continue
    o = pd.to_numeric(obs[obs.ok == True][col], errors='coerce')
    o = o[np.isfinite(o) & (o > 0)].to_numpy(float)

    inj = sw[sw.injected_radius > 0]
    amb = pd.to_numeric(sw[sw.injected_radius == 0].width, errors='coerce').dropna()
    for occ in sorted(inj.duty.dropna().unique(), reverse=True):
        w = pd.to_numeric(inj[inj.duty == occ].width, errors='coerce').dropna()
        if len(w) < 5 or len(o) < 5:
            continue
        P = float(mannwhitneyu(o, w.to_numpy(float), alternative='greater').pvalue)
        rows.append(dict(model=tag, occupancy=float(occ), n_injected=len(w),
                         n_observed=len(o), injected_median=float(w.median()),
                         observed_median=float(np.median(o)),
                         P=P, excluded=bool(P < A.alpha),
                         move_set_declared=tag in ASSUMED,
                         uninjected_median=float(amb.median()) if len(amb) else np.nan))

d = pd.DataFrame(rows)
if skipped:
    print('NOT TESTED')
    for t, why in skipped:
        print(f'  {t:14s} {why}')
    print()
if not len(d):
    raise SystemExit('no model run could be tested; nothing is written')

print(f'observed corridor widths at tau {A.tau} against injected conduits of known '
      f'vertical occupancy\none-sided Mann-Whitney, observed wider than injected; '
      f'excluded at P < {A.alpha}\n')
print(f'{"model":<14s} {"occupancy":>9s} {"injected":>9s} {"observed":>9s} '
      f'{"P":>9s}  {"n inj":>5s} {"n obs":>5s}  verdict')
for _, r in d.iterrows():
    print(f'{r.model:<14s} {100*r.occupancy:8.0f}% {r.injected_median:8.0f}km '
          f'{r.observed_median:8.0f}km {r.P:9.4f}  {r.n_injected:5d} {r.n_observed:5d}'
          f'  {"EXCLUDED" if r.excluded else "not excluded"}')

print()
tested = sorted(d.model.unique())
for occ in sorted(d.occupancy.unique(), reverse=True):
    sub = d[d.occupancy == occ]
    yes = sorted(sub[sub.excluded].model)
    print(f'{100*occ:3.0f}% occupancy: excluded by {len(yes)} of {len(sub)} runs'
          + (f' ({", ".join(yes)})' if yes else ''))

# The bound is the smallest occupancy every tested run excludes. Quoting a bound that
# only some runs support is how a four-model claim turns into a one-model claim.
allrun = [occ for occ in sorted(d.occupancy.unique())
          if d[d.occupancy == occ].excluded.all()]
print()
if allrun:
    print(f'Every tested run ({", ".join(tested)}) excludes occupancies of '
          f'{100*min(allrun):.0f} per cent and above.')
else:
    print(f'No occupancy is excluded by every tested run ({", ".join(tested)}).')

out = os.path.join(A.dir, 'occupancy_bound.csv')
d.to_csv(out, index=False)
print(f'\nwrote {out}')
