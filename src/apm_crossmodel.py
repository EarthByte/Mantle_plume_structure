#!/usr/bin/env python3
"""Does the upper-mantle lean against plate motion hold in models other than RevealLO?

The lower-mantle result was tested in three models from two inversion families. The
upper-mantle result was not, and that asymmetry is the obvious thing for a reviewer to
press: a signal seen in one model only is a property of that model until shown
otherwise.

This repeats the 300-660 km test in every model that has adequate shallow coverage.

COVERAGE CRITERION, FIXED BEFORE ANY TEST STATISTIC WAS COMPUTED. A model qualifies if
at least 20 of its traced paths carry 4 or more samples inside 300-660 km AND deviate
laterally over that window, so that a lean is defined rather than extrapolated or
undefined. Twenty is the floor because the lower-mantle test ran on 34 to 48 paths, and
a materially smaller sample would not be a comparable test. Models below the floor are
reported as untested rather than as disagreeing, and no model is dropped after its
result is seen.

A path with exactly zero lateral displacement across the window has no lean to test. It
is not a lean of zero degrees, and averaging it in as one would pull every model toward
the predicted direction for a numerical reason. These are counted and reported, because
their number is itself the clearest resolution diagnostic available here: a descent that
takes no lateral step between 300 and 660 km is one for which the model offers no
lateral cost gradient to follow at the grid scale of the tracing.

POWER. Where a model returns a null, the power of that test against an effect the size
of the RevealLO result is computed and reported, so an absence of evidence is not read
as evidence of absence. The V-test statistic is R * sqrt(2n) against a one-sided
critical value of 1.645.

PREDICTED DIRECTION: opposite absolute plate motion, the same 50 Myr averaging window
the RevealLO result uses, taken from plume_slant_apm.py without modification. The
statistic is the circular V-test against that direction, as in the lower-mantle test.

RevealLO_30km is the same model at coarser depth sampling and is a resolution control,
not an independent test; it is reported separately and not counted as replication.

  python3 apm_crossmodel.py
"""
from __future__ import annotations
import argparse, json, math, os
import numpy as np, pandas as pd

R_E, DEG = 6371.0, np.pi / 180.0
Z0, Z1 = 300.0, 660.0
MIN_SAMPLES, MIN_PATHS = 4, 20

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--tags', default='RevealLO,REVEAL,GLADM35,SPiRaL,SEMUCB-WM1,RevealLO_30km')
ap.add_argument('--apm', default='out/plume_slant_apm.csv')
ap.add_argument('--window', type=float, default=50.0)
ap.add_argument('--dir', default='out')
A = ap.parse_args()

apm = pd.read_csv(os.path.join(HERE, A.apm))
apm = apm[apm.window == A.window].set_index('site')['apm_az']
if not len(apm):
    raise SystemExit(f'no absolute plate motion at the {A.window} Myr window')


def lean(q):
    """Bearing across the window, with the reason when there is none."""
    la = np.asarray(q['lat'], float); lo = np.asarray(q['lon'], float)
    z = np.asarray(q['depth'], float)
    o = np.argsort(z); la, lo, z = la[o], lo[o], z[o]
    m = (z >= Z0) & (z <= Z1)
    if m.sum() < MIN_SAMPLES:
        return None, 'thin'
    la, lo = la[m], lo[m]
    e = R_E * DEG * (((lo[-1] - lo[0] + 180) % 360) - 180) * math.cos(la[0] * DEG)
    n = R_E * DEG * (la[-1] - la[0])
    if math.hypot(e, n) < 1e-6:
        return None, 'vertical'
    return math.degrees(math.atan2(e, n)) % 360.0, ''


def power(n, R_alt, alpha=0.05):
    """Power of the V-test at n samples against a true resultant length R_alt."""
    if not n:
        return float('nan')
    crit = 1.6449
    return 0.5 * math.erfc(-(R_alt * math.sqrt(2.0 * n) - crit) / math.sqrt(2.0))


def vtest(diff_deg):
    """Circular V-test against zero misfit. Returns R, P, n."""
    a = np.radians(np.asarray(diff_deg, float))
    n = len(a)
    C, S = np.cos(a).sum(), np.sin(a).sum()
    R = math.hypot(C, S) / n
    V = C / n
    u = V * math.sqrt(2.0 * n)
    P = 0.5 * math.erfc(u / math.sqrt(2.0))
    return R, P, n


rows = []
for tag in [t.strip() for t in A.tags.split(',') if t.strip()]:
    f = os.path.join(HERE, A.dir, f'conduit_paths_all_{tag}.json')
    if not os.path.exists(f):
        rows.append(dict(model=tag, n=0, R=np.nan, P=np.nan, tested=False,
                         note='no traced paths'))
        continue
    paths = json.load(open(f))
    diffs, vertical, thin = [], 0, 0
    for site, q in paths.items():
        if site not in apm.index:
            continue
        b, why = lean(q)
        if b is None:
            thin += why == 'thin'
            vertical += why == 'vertical'
            continue
        # predicted: opposite absolute plate motion
        pred = (float(apm.loc[site]) + 180.0) % 360.0
        diffs.append(((b - pred + 180.0) % 360.0) - 180.0)
    n = len(diffs)
    if n < MIN_PATHS:
        rows.append(dict(model=tag, n=n, vertical=vertical, thin=thin,
                         R=np.nan, P=np.nan, power=np.nan, tested=False,
                         note=f'below the {MIN_PATHS}-path floor'))
        continue
    R, P, _ = vtest(diffs)
    rows.append(dict(model=tag, n=n, vertical=vertical, thin=thin, R=R, P=P,
                     power=np.nan, tested=True,
                     note='resolution control, not independent'
                          if tag.endswith('_30km') else ''))

r = pd.DataFrame(rows)
# power of each null against the effect the reference model shows
_ref = r[(r.model == 'RevealLO') & r.tested]
if len(_ref):
    R_ref = float(_ref.R.iloc[0])
    r['power'] = [power(int(x.n), R_ref) if (x.tested and x.P >= 0.05) else np.nan
                  for _, x in r.iterrows()]
r.to_csv(os.path.join(HERE, A.dir, 'apm_crossmodel.csv'), index=False)
print(f'300-660 km lean against the reverse of absolute plate motion, {A.window} Myr\n')
print(f'{"model":<16s} {"n":>4s} {"vert":>5s} {"thin":>5s} {"R":>7s} {"P":>9s} '
      f'{"power":>6s}  note')
for _, x in r.iterrows():
    R = f'{x.R:.3f}' if np.isfinite(x.R) else '-'
    P = f'{x.P:.4f}' if np.isfinite(x.P) else '-'
    pw = f'{x.power:.2f}' if np.isfinite(x.power) else '-'
    print(f'{x.model:<16s} {int(x.n):4d} {int(x.vertical):5d} {int(x.thin):5d} '
          f'{R:>7s} {P:>9s} {pw:>6s}  {x.note}')
ind = r[(r.tested) & (~r.model.str.endswith('_30km'))]
print(f'\n{len(ind)} models tested independently; '
      f'{int((ind.P < 0.05).sum())} reach P < 0.05')
