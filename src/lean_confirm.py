"""Confirmatory test: does deep plume lean point AWAY from subduction, in independent models?

PRE-REGISTRATION. Everything below was fixed before this script was run on any model
other than RevealLO, and nothing in it may be tuned to the outcome.

HYPOTHESIS. Hassan et al. (2016, Nature) model plume tilt as controlled primarily by
"lateral advection of plume sources rooted on chemical ridges in the deep lower
mantle", with the ridge network "adjusting to slab push", and find tilt "starting at
mid-mantle". Slab push displaces a plume's source away from the subduction zone, so
the conduit's deep end should sit further from the trench than its shallow end.

PREDICTED DIRECTION: AWAY, i.e. a misfit of 180 degrees between the lean bearing and
the weighted bearing to nearby trenches. This direction comes from the published model,
NOT from the RevealLO data, in which it was found post hoc.

DEPTH WINDOW: 660-1500 km. Fixed from Hassan's "starting at mid-mantle" and from the
RevealLO result, and not varied here.

TEST: circular V-test against the predicted 180 degrees, one-sided, since the direction
is now specified in advance.

TRENCH WEIGHTING: arc length divided by the square of distance, segments within 6000 km,
from tessellate_subduction_zones at 0 Ma on Zahirovic et al. (2022). Unchanged from the
RevealLO run.

DECISION RULE, FIXED NOW. Both REVEAL and GLAD-M35 concentrating near 180 degrees with
P < 0.05 confirms the result. Either failing means the RevealLO signal is not robust and
the lower-mantle lean claim is dropped. REVEAL shares an inversion lineage with RevealLO
and is therefore weak evidence on its own; GLAD-M35 is the independent test.
"""
from __future__ import annotations
import argparse, json, math, os
import numpy as np, pandas as pd
from math import erfc, sqrt

R_E, DEG = 6371.0, np.pi / 180.0
Z0, Z1, CUTOFF, PREDICTED = 660.0, 1500.0, 6000.0, 180.0

ap = argparse.ArgumentParser()
ap.add_argument('--tags', default='RevealLO,REVEAL,GLADM35')
ap.add_argument('--hinge', default='out/hinge_migration.csv')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--dir', default='out')
A = ap.parse_args()

t = pd.read_csv(A.hinge)
tlon, tlat, al = t.lon.to_numpy(), t.lat.to_numpy(), t.arc_length.to_numpy()
hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180']).set_index('hotspot')


def gc(a, b, c, d):
    return R_E * np.arccos(np.clip(np.sin(a * DEG) * np.sin(c * DEG) +
                                   np.cos(a * DEG) * np.cos(c * DEG) *
                                   np.cos((d - b) * DEG), -1, 1))


def bearing(a, b, c, d):
    y = np.sin((d - b) * DEG) * np.cos(c * DEG)
    x = np.cos(a * DEG) * np.sin(c * DEG) - np.sin(a * DEG) * np.cos(c * DEG) * np.cos((d - b) * DEG)
    return np.degrees(np.arctan2(y, x)) % 360.0


def misfits(tag):
    p = os.path.join(A.dir, f'conduit_paths_all_{tag}.json')
    if not os.path.exists(p):
        return None
    paths = json.load(open(p))
    # Three rules drop a path here, and until now all three dropped it without
    # trace, so the deep window reported a sample size with no account of what it
    # excluded. They are counted and carried out. Note the second is NOT the
    # "identically zero displacement" rule the upper-mantle test uses: it drops
    # anything under 20 km, which is a different and stricter thing, and the
    # supplement has to say so rather than call both columns "vertical".
    out = []
    excl = dict(thin=0, short=0, few_trench=0)
    for n, q in paths.items():
        if n not in hs.index:
            continue
        hla, hlo = float(hs.loc[n, 'lat']), float(hs.loc[n, 'lon_180'])
        la = np.asarray(q['lat'], float); lo = np.asarray(q['lon'], float)
        zz = np.asarray(q['depth'], float)
        o = np.argsort(zz); la, lo, zz = la[o], lo[o], zz[o]
        m = (zz >= Z0) & (zz <= Z1)
        if m.sum() < 4:
            excl['thin'] += 1
            continue
        la2, lo2 = la[m], lo[m]
        e = R_E * DEG * (((lo2 - lo2[0] + 180) % 360) - 180) * np.cos(la2[0] * DEG)
        nn = R_E * DEG * (la2 - la2[0])
        if float(np.hypot(e[-1], nn[-1])) < 20:
            excl['short'] += 1
            continue
        lean = math.degrees(math.atan2(e[-1], nn[-1])) % 360.0
        d = gc(hla, hlo, tlat, tlon); sel = d <= CUTOFF
        if sel.sum() < 20:
            excl['few_trench'] += 1
            continue
        w = al[sel] / np.maximum(d[sel], 200.0) ** 2
        br = np.radians(bearing(hla, hlo, tlat[sel], tlon[sel]))
        tb = math.degrees(math.atan2((np.sin(br) * w).sum(), (np.cos(br) * w).sum())) % 360.0
        df = (lean - tb) % 360.0
        out.append(df - 360 if df > 180 else df)
    return out, excl


print(f'PRE-REGISTERED: away from subduction (misfit {PREDICTED:.0f} deg), '
      f'{Z0:.0f}-{Z1:.0f} km, V-test one-sided, P < 0.05 in both to confirm\n')
print(f'{"model":>12s} {"n":>4s} {"R":>7s} {"mean misfit":>12s} {"V":>7s} {"P":>9s}  verdict')
res = {}
_rows = []
for tag in A.tags.split(','):
    _got = misfits(tag)
    mis, _ex = (_got if _got else (None, dict(thin=0, short=0, few_trench=0)))
    if not mis or len(mis) < 8:
        print(f'{tag:>12s}  no usable paths')
        continue
    a = np.radians(np.asarray(mis, float) - PREDICTED)   # rotate so prediction is zero
    n = len(a)
    C, S = np.cos(a).mean(), np.sin(a).mean()
    Rr = math.hypot(C, S)
    V = C                       # projection on the predicted direction
    u = V * math.sqrt(2 * n)
    P = 0.5 * erfc(u / sqrt(2))
    mu = (math.degrees(math.atan2(S, C)) + PREDICTED) % 360.0
    res[tag] = P
    _rows.append(dict(model=tag, window='660-1500', n=n, R=Rr, P=P,
                      mean_misfit=mu if mu <= 180 else mu - 360,
                      short=_ex['short'], thin=_ex['thin'],
                      few_trench=_ex['few_trench']))
    print(f'{tag:>12s} {n:4d} {Rr:7.3f} {mu if mu <= 180 else mu - 360:+11.0f} '
          f'{V:7.3f} {P:9.4f}  {"pass" if P < 0.05 else "FAIL"}')

ind = [k for k in res if k not in ('RevealLO', 'REVEAL')]
print()
if 'GLADM35' in res:
    print(f'GLAD-M35 is the independent family: '
          f'{"CONFIRMS" if res["GLADM35"] < 0.05 else "DOES NOT CONFIRM"} (P = {res["GLADM35"]:.4f})')
if 'REVEAL' in res:
    print(f'REVEAL shares a lineage with RevealLO, so it is weak evidence: '
          f'P = {res["REVEAL"]:.4f}')
print('\nBy the rule fixed in advance, the lower-mantle lean claim stands only if '
      'both pass.')

# Power of each null against the effect the reference model shows. The supplement
# states that power is reported wherever a test returns a null; without this the
# deep window quietly exempted itself from its own rule.
def _power(n, R_alt, crit=1.6449):
    return (float('nan') if not n else
            0.5 * erfc(-(R_alt * sqrt(2.0 * n) - crit) / sqrt(2.0)))


if _rows:
    _d = pd.DataFrame(_rows)
    _ref = _d[_d.model == 'RevealLO']
    _R = float(_ref.R.iloc[0]) if len(_ref) else float('nan')
    _d['power'] = [_power(int(r.n), _R) if r.P >= 0.05 else float('nan')
                   for _, r in _d.iterrows()]
    _rows = _d.to_dict('records')

# written so the figure and the manuscript audit read the same numbers the run printed
if _rows:
    pd.DataFrame(_rows).to_csv(os.path.join(A.dir, 'lean_confirm_crossmodel.csv'),
                               index=False)
    print(f"\nwrote {os.path.join(A.dir, 'lean_confirm_crossmodel.csv')}")
