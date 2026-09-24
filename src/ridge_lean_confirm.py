"""Do upper-mantle conduits lean TOWARD the nearest mid-ocean ridge?

PRE-REGISTRATION. Written before this test was run on any model, including RevealLO.
Nothing in it may be tuned to the outcome.

HYPOTHESIS. Whittaker et al. (2015, Nature Geoscience) describe plume capture, in
which "plumes bend and flow towards migrating ridges, possibly induced by ridge
suction, from distances of up to ~1,400 km", and argue that upwelling at slower
ridges is "strong enough to focus plumes towards the ridge but insufficiently strong
to capture them entirely". If ridge suction bends conduits, an upper-mantle conduit
within the interaction distance should lean toward the ridge that is drawing it.

PREDICTED DIRECTION: TOWARD, a misfit of 0 degrees between the lean bearing and the
bearing to the nearest present-day ridge. The direction is Whittaker's, not this
data set's, and no version of this measurement has been made in any model.

DEPTH WINDOW: 300-660 km, the window already established here as the one in which
lean tracks surface plate motion rather than deep processes. Ridge suction is an
upper-mantle flow. Whittaker's own thermal signature sits at 100-175 km, shallower
than any path here reaches, since traces are seeded at 200 km.

INTERACTION DISTANCE: 1400 km, the upper limit Whittaker gives for plume-ridge
interaction, from their references 16 and 26 which bracket it at 1000-1400 km. Not
tuned: the far group below uses the same number as its boundary.

RIDGE BEARING: to the nearest point on the present-day ridge system, Zahirovic et
al. (2022) topologies, unweighted. Plume capture is a statement about the particular
ridge drawing a particular plume, so the nearest ridge is the relevant one, and a
distance-weighted alternative is NOT computed.

TEST: circular V-test against 0 degrees, one-sided, on hotspots within 1400 km.

CONTROL, WHICH IS PART OF THE RULE: the same test on hotspots beyond 1400 km, where
Whittaker's mechanism predicts nothing. A significant result there would mean the
test responds to something other than ridge suction and the near-group result could
not be attributed to it.

DECISION RULE, FIXED NOW. In RevealLO, the near group must reach P < 0.05 AND the far
group must not. That is the result. REVEAL and GLAD-M35 are then run as replication,
subject to the rule this project adopted after the ridge-network episode: a null in a
smeared model withdraws nothing unless the power calculation shows it could have seen
the effect, and that calculation is printed here rather than left to be asked for.

CONFOUND ALREADY CHECKED. Plates diverge from ridges, so leaning toward a ridge and
leaning against absolute plate motion could be one measurement. They are not: across
the 49 hotspots the two predicted bearings differ by a median of 69 degrees, and by
66 degrees within the 1400 km group, against the 90 expected of unrelated directions.
Only 30 per cent agree to within 45 degrees. The two tests are largely independent.
"""
from __future__ import annotations
import argparse, json, math, os
import numpy as np, pandas as pd
from math import erfc, sqrt

R_E, DEG = 6371.0, np.pi / 180.0
Z0, Z1, INTERACT, PREDICTED = 300.0, 660.0, 1400.0, 0.0

ap = argparse.ArgumentParser()
ap.add_argument('--tags', default='RevealLO,REVEAL,GLADM35')
ap.add_argument('--ridges', default='data/ridges_presentday_zahirovic2022.csv')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--n-boot', type=int, default=4000, dest='n_boot')
ap.add_argument('--dir', default='out')
A = ap.parse_args()

rg = pd.read_csv(A.ridges)
RLA, RLO = rg.lat.to_numpy(float), rg.lon.to_numpy(float)
hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180']).set_index('hotspot')


def gc(a, b, c, d):
    return R_E * np.arccos(np.clip(np.sin(a * DEG) * np.sin(c * DEG) +
                                   np.cos(a * DEG) * np.cos(c * DEG) *
                                   np.cos((d - b) * DEG), -1, 1))


def bearing(a, b, c, d):
    y = np.sin((d - b) * DEG) * np.cos(c * DEG)
    x = np.cos(a * DEG) * np.sin(c * DEG) - np.sin(a * DEG) * np.cos(c * DEG) * np.cos((d - b) * DEG)
    return np.degrees(np.arctan2(y, x)) % 360.0


def vtest(mis, predicted=PREDICTED):
    a = np.radians(np.asarray(mis, float) - predicted)
    n = len(a)
    C, S = np.cos(a).mean(), np.sin(a).mean()
    Rr = math.hypot(C, S)
    P = 0.5 * erfc((C * math.sqrt(2 * n)) / sqrt(2))
    mu = (math.degrees(math.atan2(S, C)) + predicted) % 360.0
    return n, Rr, C, P, (mu if mu <= 180 else mu - 360)


def misfits(tag):
    p = os.path.join(A.dir, f'conduit_paths_all_{tag}.json')
    if not os.path.exists(p):
        return None
    paths = json.load(open(p))
    near, far = [], []
    for n, q in paths.items():
        if n not in hs.index:
            continue
        hla, hlo = float(hs.loc[n, 'lat']), float(hs.loc[n, 'lon_180'])
        la = np.asarray(q['lat'], float); lo = np.asarray(q['lon'], float)
        zz = np.asarray(q['depth'], float)
        o = np.argsort(zz); la, lo, zz = la[o], lo[o], zz[o]
        m = (zz >= Z0) & (zz <= Z1)
        if m.sum() < 4:
            continue
        la2, lo2 = la[m], lo[m]
        e = R_E * DEG * (((lo2 - lo2[0] + 180) % 360) - 180) * np.cos(la2[0] * DEG)
        nn = R_E * DEG * (la2 - la2[0])
        if float(np.hypot(e[-1], nn[-1])) < 20:
            continue
        lean = math.degrees(math.atan2(e[-1], nn[-1])) % 360.0
        d = gc(hla, hlo, RLA, RLO)
        k = int(np.argmin(d))
        rb = float(bearing(hla, hlo, RLA[k], RLO[k]))
        df = (lean - rb) % 360.0
        df = df - 360 if df > 180 else df
        (near if float(d[k]) <= INTERACT else far).append(df)
    return near, far


print(f'PRE-REGISTERED: toward the nearest ridge (misfit {PREDICTED:.0f} deg), '
      f'{Z0:.0f}-{Z1:.0f} km, within {INTERACT:.0f} km,\nV-test one-sided; the far '
      f'group must NOT be significant\n')
print(f'{"model":>12s} {"group":>7s} {"n":>4s} {"R":>7s} {"mean":>7s} {"V":>7s} {"P":>9s}')
res = {}
rows = []   # written to out/ridge_lean_<tag>.csv: Text S3 and Figure 4c quote these
for tag in A.tags.split(','):
    got = misfits(tag)
    if got is None:
        print(f'{tag:>12s}  no conduit paths on disk')
        continue
    near, far = got
    row = {}
    for label, mis in (('near', near), ('far', far)):
        if len(mis) < 8:
            print(f'{tag:>12s} {label:>7s} {len(mis):4d}  too few')
            continue
        n, Rr, V, P, mu = vtest(mis)
        row[label] = P
        rows.append(dict(model=tag, group=label, n=n, R=Rr, mean_misfit_deg=mu, V=V, P=P))
        print(f'{tag:>12s} {label:>7s} {n:4d} {Rr:7.3f} {mu:+6.0f} {V:7.3f} {P:9.4f}')
    if 'near' in row and 'far' in row:
        ok = row['near'] < 0.05 <= row['far']
        res[tag] = (row['near'], row['far'], ok)
        print(f'{"":>12s} {"":>7s} -> {"PASSES the rule" if ok else "does not pass"}')

for tag in A.tags.split(','):
    sub = [r for r in rows if r['model'] == tag]
    if sub:
        pd.DataFrame(sub).to_csv(os.path.join(A.dir, f'ridge_lean_{tag}.csv'), index=False)
        print(f'wrote {A.dir}/ridge_lean_{tag}.csv')

if 'RevealLO' in res and res['RevealLO'][2]:
    rng = np.random.default_rng(11)
    base = misfits('RevealLO')[0]
    print(f'\nPOWER, so that any null below can be read: how often a model finding')
    print(f'only k usable conduits would reach P < 0.05 given RevealLO\'s effect')
    for k in (10, 15, 20, 25, len(base)):
        if k > len(base):
            continue
        hits = sum(vtest(list(rng.choice(base, size=k, replace=False)))[3] < 0.05
                   for _ in range(A.n_boot))
        print(f'  {k:3d} conduits   {100 * hits / A.n_boot:5.1f}%')
    print('\nA model below about 50 per cent here cannot withdraw anything by failing.')
