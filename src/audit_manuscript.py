"""Check every load-bearing number in the manuscript against the outputs on disk.

The manuscript's statistics are typed rather than generated, which is how a paper
and the run behind it drift apart. This recomputes each one and reports agreement,
so the drift is caught here rather than by a reviewer. It checks values, not prose:
a claim it cannot express as a number is listed as unchecked rather than passed.
"""
from __future__ import annotations
import argparse, glob, json, math, os, re, subprocess, sys
import numpy as np, pandas as pd
import provenance
import freshness
from math import erfc, sqrt
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

R_E, DEG = 6371.0, np.pi / 180.0
ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--tag', default='RevealLO',
                help='the model whose outputs are checked. The audit was written for '
                     'the paper model and had no way to say so, which is also why it '
                     'could not check its own inputs.')
ap.add_argument('--ms', default='../manuscript/G3_MS_v11.md')
ap.add_argument('--si', default='../manuscript/G3_SI_v7.md')
ap.add_argument('--expect', default='manuscript_expect.json',
                help='label -> value the current manuscript quotes; a label mapped '
                     'to null is a claim the paper no longer makes and its check is '
                     'retired. Anything absent keeps the literal in this script')
A = ap.parse_args()
# Before the first read. provenance_report answers this for a STAMPED file; most of what
# the audit opens carries no stamp, and every staleness that has reached the manuscript
# so far was in an unstamped file.
freshness.record()
MS = open(A.ms).read()
SI = open(A.si).read() if os.path.exists(A.si) else ''
BOTH = MS + '\n' + SI
checks = []
retired = []
missing = []   # inputs the audit needs and did not find


EXPECT = {}
if os.path.exists(A.expect):
    EXPECT = json.load(open(A.expect))


# The audit is the one tool here that has to be believed, so it reports the standing of
# its own inputs before it reports anything else. A stale input does not make the audit
# fail - it makes the audit lie, agreeing about a number that describes a configuration
# nobody is using. That happened: bao_pdf_<tag>.csv survived a retrace untouched and the
# four deflection checks reported the previous run's values as current.
PROV_INPUTS = [
    'conduit_paths_all_{t}.json', 'ambient_null_{t}.csv', 'corridor_summary_{t}.csv',
    'corridor_profiles_{t}.csv', 'bao_pdf_{t}.csv', 'path_reliability_{t}.csv',
    'track_support_{t}.csv', 'detection_{t}.csv', 'classification_{t}.csv',
    'province_null_{t}.csv', 'geochem_geometry_{t}.csv', 'plume_roots_{t}.csv',
]


TAGS = ('RevealLO', 'RevealLO_30km', 'REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1')


def provenance_report(d, tag):
    # Every model, not just the paper model. A comparison model traced under a
    # configuration that has since been revised is exactly as wrong as the paper model
    # would be, and it reaches the reader through the cross-model table.
    import path_config
    stale, unstamped = [], []
    for tg in TAGS:
        try:
            frozen = path_config.load(tg, d)
        except Exception:
            frozen = None
        for pat in PROV_INPUTS:
            p = os.path.join(d, pat.format(t=tg))
            if not os.path.exists(p):
                continue
            v = provenance.verdict(p, frozen)
            if v == ['unstamped']:
                unstamped.append(os.path.basename(p))
            elif v:
                stale.append((os.path.basename(p), v))
    if stale:
        print('INPUTS THAT DISAGREE WITH THE FROZEN CONFIGURATION OR THEIR OWN INPUTS.')
        print('Every number below that rests on one of these describes a configuration')
        print('nobody is running. Regenerate them before reading any of it.')
        for n, v in stale:
            print(f'  {n}')
            for m in v:
                print(f'      {m}')
        print()
    if unstamped:
        print(f'{len(unstamped)} of the audit\'s inputs carry no provenance and were not')
        print('checked: ' + ', '.join(unstamped))
        print()
    if not stale and not unstamped:
        print('every audit input is stamped and consistent with the frozen configuration\n')
    return len(stale)


def configs_clearing_floor(d):
    """How many frozen configurations still clear the floor that admitted them.

    Text S1 states that configurations are retained by a 75 per cent injection floor. A
    frozen configuration can fall below it afterwards without anything noticing, because
    the floor is applied when the configuration is chosen and never again: the paper
    model, its 30 km resolution control and REVEAL had all drifted below it and stayed
    frozen, and the paths behind every figure were traced under them. This asks the
    question at every run instead of once.
    """
    import json as _j
    import path_config as _pc
    n_ok, n_seen, below = 0, 0, []
    for tag in ('RevealLO', 'RevealLO_30km', 'REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1'):
        fp = os.path.join(d, f'path_config_{tag}.json')
        dp = os.path.join(d, f'detection_{tag}.csv')
        if not (os.path.exists(fp) and os.path.exists(dp)):
            continue
        c = _j.load(open(fp))['config']
        t_ = pd.read_csv(dp)
        q = t_[(t_.s == float(c['s'])) & (t_.z_target == float(c['z_target']))
               & (t_.channel == str(c['channel'])) & (t_.h_max == float(c['h_max']))
               & (t_.radius == float(c['radius']))]
        n_seen += 1
        if len(q) and q.iloc[0].n_cal >= _pc.MIN_CASES and q.iloc[0].detect_cal >= _pc.FLOOR:
            n_ok += 1
        else:
            below.append(f'{tag} at '
                         + (f'{q.iloc[0].detect_cal:.3f}' if len(q) else 'not in its table'))
    if below:
        print('FROZEN CONFIGURATIONS BELOW THE FLOOR THAT ADMITTED THEM: '
              + ', '.join(below))
        print('The paths behind every figure for those models were traced under a '
              'configuration the calibration does not retain.\n')
    return n_ok, n_seen


def chk(label, computed, quoted, tol, unit=''):
    # A check whose input has gone missing is a failure to report, not a crash.
    # The audit exists to list everything that disagrees; stopping at the first
    # absent key hides the other hundred checks, which is the opposite of its job.
    try:
        c = float(computed)
    except (TypeError, ValueError):
        c = float('nan')
    # The value the manuscript quotes is held here as a literal, which means the audit
    # goes stale whenever the manuscript is rewritten and then reports its own
    # obsolescence as a disagreement. A run of 85 mismatches, most of them the audit's
    # fault, teaches a reader to ignore the audit, which is the worst outcome for a tool
    # whose only job is to be believed. An override file lets a rewritten claim be
    # re-keyed in one place, and any label it does not mention keeps the literal below.
    q = EXPECT.get(label, quoted)
    if q is None:                 # explicitly retired: the claim is gone from the paper
        retired.append(label)
        return True
    ok = (c == c) and abs(c - q) <= tol
    checks.append((ok, label, c, q, unit))
    return ok


class Missing(dict):
    """A lookup that yields NaN for an absent key and records what was asked for.

    Analyses change shape - the grouping went from two clusters to six and its
    per-group keys went with it - and the audit has to survive that and say which
    inputs vanished, rather than raising KeyError on the first one.
    """

    absent = []

    def __missing__(self, key):
        if key not in Missing.absent:
            Missing.absent.append(key)
        return float('nan')


def in_ms(s):
    return s in MS


# corridor widths at the three tolerances
sm = pd.read_csv(os.path.join(A.dir, 'corridor_summary_RevealLO.csv'))
sm = sm[sm.ok == True]
for t, q in (('0.01', 271), ('0.02', 406), ('0.05', 757)):
    chk(f'median corridor width at tau {t}', float(sm[f'width_med_{t}'].median()), q, 1, 'km')
chk('hotspots passing the self-check', len(sm), 49, 0)

# corridor width by depth band
pr = pd.read_csv(os.path.join(A.dir, 'corridor_profiles_RevealLO.csv'))
# Section 2.3 quotes the per-depth profile read off at four places, not a median
# over five bands. Checking bands meant the audit was checking a quantity the paper
# never states, and its five literals could never be reconciled with the text.
_w = pd.to_numeric(pr.width_km, errors='coerce')
_pr = pr[np.isfinite(_w) & (_w > 0)]
_prof = _pr.groupby('depth').width_km.median()


def _band(lo, hi):
    v = _prof[(_prof.index >= lo) & (_prof.index < hi)]
    return float(v.median()) if len(v) else float('nan')


# Section 2.3 in v8 quotes BAND MEDIANS again - "the medians for individual paths are
# 377 km at 200-660 km, 452 km at 660-1500 km, and 620 km at 1500-2700 km" - after an
# earlier version quoted read-offs, which is what the four checks below were built for.
# The text moved and the checks did not, so the audit was testing three quantities the
# paper does not state while its three stated band medians went unchecked. Both are here
# now; the read-offs stay because they are the diagnostic the corridor work is steered by,
# but only the one the manuscript quotes carries an expected value.
def _path_band(lo, hi):
    b = _pr[(_pr.depth >= lo) & (_pr.depth < hi)]
    per = b.groupby('site').width_km.median()
    return float(per.median()) if len(per) else float('nan')


chk('band median width, 200-660 km', _path_band(200, 660), 377, 2, 'km')
chk('band median width, 660-1500 km', _path_band(660, 1500), 452, 2, 'km')
chk('band median width, 1500-2700 km', _path_band(1500, 2700), 620, 2, 'km')

chk('profile width, upper mantle', _band(200, 410), 414, 2, 'km')
chk('profile width, transition zone minimum',
    float(_prof.min()) if len(_prof) else float('nan'), 356, 2, 'km')
chk('profile width, lowermost mantle', _band(2000, 2700), 693, 2, 'km')
chk('profile width at the target depth',
    float(_prof.loc[_prof.index.max()]) if len(_prof) else float('nan'), 925, 2, 'km')

# Province rooting. The four rows of the table in section 3.2, read from the file
# province_null_test.py writes. The matching is implemented once, there; an audit
# that re-derived it here would be checking its own arithmetic, and an earlier
# version of this block did exactly that while its comment claimed otherwise - it
# compared against the unmatched null under the label "matched".
rt = pd.read_csv(os.path.join(A.dir, 'plume_roots_RevealLO.csv'))
_pn = os.path.join(A.dir, 'province_null_RevealLO.csv')
if os.path.exists(_pn):
    pn = pd.read_csv(_pn).set_index('null')
    chk('hotspots inside a province (%)', float(pn.hot_pct.iloc[0]), 75.5, 0.2)
    for _key, _lab, _np_, _fp, _or in (
            ('unmatched',  'uniform',     55.3, 0.0054, 2.49),
            ('matched x1', 'matched x1',  59.2, 0.0655, 2.13),
            ('matched x3', 'matched x3',  55.8, 0.0102, 2.44),
            ('matched x5', 'matched x5',  58.4, 0.0167, 2.20)):
        if _key not in pn.index:
            continue
        r_ = pn.loc[_key]
        chk(f'province null {_lab}, null inside (%)', float(r_.null_pct), _np_, 0.2)
        chk(f'province null {_lab}, Fisher P', float(r_.fisher_p), _fp, 0.0005)
        chk(f'province null {_lab}, odds ratio', float(r_.odds), _or, 0.05)
    if 'matched x3' in pn.index:
        r3 = pn.loc['matched x3']
        chk('median margin, hotspots', float(r3.margin_med_hot), -567, 2, 'km')
        chk('median margin, matched null', float(r3.margin_med_null), -364, 2, 'km')
        chk('margin P, matched x3', float(r3.margin_p), 0.036, 0.002)
        chk('depth inside, hotspots', float(r3.depth_in_hot), 818, 2, 'km')
        chk('depth inside, matched null', float(r3.depth_in_null), 833, 2, 'km')
        chk('depth inside, difference', float(r3.depth_diff), -15, 2, 'km')
        chk('depth inside, P', float(r3.depth_p), 0.86, 0.02)
else:
    missing.append('province_null_RevealLO.csv (run province_null_test.py)')

# the calibration
cal = pd.read_csv(os.path.join(A.dir, 'corridor_width_calibration_RevealLO.csv'))
ci = cal[cal.injected_radius > 0].dropna(subset=['width'])
r1, p1 = spearmanr(ci.injected_fwhm, ci.width)
r2, _ = spearmanr(ci.ambient_width, ci.width)
chk('calibration rho against injected', r1, 0.885, 0.005)
chk('calibration rho against ambient', r2, -0.032, 0.005)
chk('injected corridors measured', len(ci), 60, 0)

# the continuity bound
for tag, q in (('RevealLO', 406), ('GLADM35', 458), ('SEMUCB-WM1', 418)):
    s = pd.read_csv(os.path.join(A.dir, f'corridor_summary_{tag}.csv'))
    chk(f'observed median width, {tag}',
        float(s[s.ok == True]['width_med_0.02'].median()), q, 1, 'km')
f = [x for x in glob.glob(os.path.join(A.dir,
     'corridor_width_calibration_RevealLO_dutysweep*.csv'))]
duty = pd.read_csv(sorted(f)[-1])
dg = duty[duty.injected_radius > 0].groupby('duty').width.median()
for occ, q in ((1.0, 250), (0.75, 277), (0.5, 326), (0.25, 365), (0.1, 378)):
    if occ in dg.index:
        chk(f'continuity, {100*occ:.0f} per cent occupancy', float(dg.loc[occ]), q, 2, 'km')

# grouping and geochemistry
gg = os.path.join(A.dir, 'geochem_geometry_RevealLO.csv')
if os.path.exists(gg):
    g = pd.read_csv(gg)
    chk('isotope-by-geometry pairs tested', len(g), 35, 0)
    chk('pairs surviving FDR 0.1', int((g.q < 0.1).sum()), 0, 0)

# The population test, which is the paper's headline and was checked by nothing. The
# table in section 3.1 - median shell percentile per depth band, the Mann-Whitney P and
# the area under the curve, hotspots against the 300 ambient paths - is the first result
# in the abstract and the first Key Point, and the audit ran 73 checks without touching
# it. It went unchecked long enough for its ambient side to stand five days stale behind
# two changes of configuration, with the hotspot side moving in every band and the
# ambient side identical to the published value in every band, and nothing said so.
#
# The P values span five orders of magnitude, so the tolerance is relative to the
# manuscript's own figure rather than absolute: the paper quotes two significant digits
# and a fixed tolerance would either pass everything at 1e-3 or fail everything at 1e-6.
_tsp = os.path.join(A.dir, f'track_support_population_{A.tag}.csv')
if os.path.exists(_tsp):
    _tp = pd.read_csv(_tsp).set_index('band')
    for _b, _hm, _am, _p, _auc in (('200-660', 0.134, 0.206, 3.6e-3, 0.620),
                                   ('660-1500', 0.088, 0.233, 3.9e-6, 0.699),
                                   ('1500-2700', 0.049, 0.114, 3.7e-4, 0.650)):
        if _b not in _tp.index:
            continue
        _r = _tp.loc[_b]
        chk(f'population median, hotspots, {_b} km', float(_r.hotspot_median), _hm, 0.002)
        chk(f'population median, ambient, {_b} km', float(_r.ambient_median), _am, 0.002)
        chk(f'population P, {_b} km', float(_r.P),
            _p, 0.06 * EXPECT.get(f'population P, {_b} km', _p))
        chk(f'population AUC, {_b} km', float(_r.auc), _auc, 0.002)
    _r0 = _tp.iloc[0]
    chk('population test, hotspot paths', float(_r0.n_hotspot), 49, 0, '')
    chk('population test, ambient paths', float(_r0.n_ambient), 300, 0, '')
else:
    missing.append(f'track_support_population_{A.tag}.csv (run track_support.py)')

# deflection distribution
bp = pd.read_csv(os.path.join(A.dir, 'bao_pdf_RevealLO.csv'))
edges = np.arange(260.0, 2900.0 + 50.0, 50.0)
centres = 0.5 * (edges[:-1] + edges[1:])
pdf = np.zeros(len(centres))
for _, gg_ in bp.groupby('site'):
    h = np.histogram(gg_.depth.to_numpy(float), bins=edges)[0].astype(float)
    if h.sum() > 0:
        pdf += h / h.sum()
pdf /= pdf.sum()
chk('deflection peak', float(centres[int(np.argmax(pdf))]), 935, 1, 'km')
# Band shares with exact edges, each path at unit weight, as fig_deflection.py labels
# them and mid_mantle_census.py tabulates them. Summing 50 km bins put 1000 km inside
# a bin and gave 8.0/24.9/57.6 where the text now says 7.7/24.9/56.4.
_bs = {}
for _, gg_ in bp.groupby('site'):
    z_ = gg_.depth.to_numpy(float)
    for lo, hi in ((660, 1000), (1000, 1500), (1500, 2200)):
        _bs[(lo, hi)] = _bs.get((lo, hi), 0.0) + ((z_ >= lo) & (z_ < hi)).mean()
for (lo, hi), q in zip(((660, 1000), (1000, 1500), (1500, 2200)), (7.7, 24.9, 56.4)):
    chk(f'deflection band {lo}-{hi}', 100 * _bs[(lo, hi)] / bp.site.nunique(), q, 0.1, '%')

# The validation section: these were the numbers the audit first missed, because
# it only checked what had already been trusted. An audit written from the claims
# one believes will pass on exactly the claims that are wrong.
import json as _json
_hs = pd.read_csv('hotspots_courtillot2003.csv').dropna(subset=['lat', 'lon_180']).set_index('hotspot')
_paths = _json.load(open(os.path.join(A.dir, 'conduit_paths_all_RevealLO.json')))


def _offsets(z0, z1):
    out = {}
    for n, q in _paths.items():
        if n not in _hs.index:
            continue
        la = np.asarray(q['lat'], float); lo = np.asarray(q['lon'], float)
        zz = np.asarray(q['depth'], float)
        o = np.argsort(zz); la, lo, zz = la[o], lo[o], zz[o]
        m = (zz >= z0) & (zz <= z1)
        if m.sum() < 4:
            continue
        la2, lo2 = la[m], lo[m]
        e = R_E * DEG * (((lo2 - lo2[0] + 180) % 360) - 180) * np.cos(la2[0] * DEG)
        nn = R_E * DEG * (la2 - la2[0])
        out[n] = float(np.hypot(e[-1], nn[-1]))
    return pd.Series(out)


_deep = _offsets(660.0, 1500.0)
chk('Louisville deep offset', float(_deep['Louisville']), 93, 1, 'km')
chk('Louisville deep offset rank', int((_deep < _deep['Louisville']).sum()) + 1, 4, 0)
# Hawaii's quoted azimuth is measured over the pre-registered deep window, not
# over the whole traced span, so it is checked against that window.
_dl = {}
for n, q in _paths.items():
    if n not in _hs.index:
        continue
    la = np.asarray(q['lat'], float); lo = np.asarray(q['lon'], float)
    zz = np.asarray(q['depth'], float)
    o = np.argsort(zz); la, lo, zz = la[o], lo[o], zz[o]
    m = (zz >= 660.0) & (zz <= 1500.0)
    if m.sum() < 4:
        continue
    la2, lo2 = la[m], lo[m]
    e = R_E * DEG * (((lo2 - lo2[0] + 180) % 360) - 180) * np.cos(la2[0] * DEG)
    nn = R_E * DEG * (la2 - la2[0])
    _dl[n] = math.degrees(math.atan2(e[-1], nn[-1])) % 360.0
chk('Hawaii lean azimuth, 660-1500 km', _dl['Hawaii'], 242, 1, 'deg')
# Section 2.3 and 3.2: the second descent. Every number the manuscript states about it
# is read here from the file that produced it - competing_routes.py, root_robustness.py
# and route_anomaly_spread.py - so a retrace that moves any of them shows up as a
# disagreement rather than as a paragraph quietly describing a previous run.
_cr = os.path.join(A.dir, 'competing_routes_RevealLO.csv')
_rr = os.path.join(A.dir, 'root_robustness_RevealLO.csv')
_rs = os.path.join(A.dir, 'root_robustness_sites_RevealLO.csv')
_sp = os.path.join(A.dir, 'route_anomaly_spread.csv')
_c30 = os.path.join(A.dir, 'competing_routes_RevealLO_30km.csv')
if all(os.path.exists(x) for x in (_cr, _rr, _rs, _sp, _c30)):
    _c = pd.read_csv(_cr); _c = _c[_c.alt_reached == True]
    chk('second descent, median cost margin (%)', _c.margin_pct.median(), 2.4, 0.05, '%')
    chk('second descent, within the 3.0% margin', float((_c.margin_pct <= 3.0).sum()),
        33, 0, '')
    chk('second descent, median endpoint separation (km)', _c.separation_km.median(),
        590, 5, 'km')
    _c33 = _c[_c.margin_pct <= 3.0]
    chk('second descent within margin, median endpoint separation (km)',
        _c33.separation_km.median(), 960, 5, 'km')
    chk('second descent within margin, all above 500 km', float((_c33.separation_km > 500).all()), 1, 0, '')
    chk('second descent within margin, above 1000 km', float((_c33.separation_km > 1000).sum()), 15, 0, '')
    _gap = (_c.mean_a - _c.alt_mean_a).abs()
    chk('second descent, median gap in path-averaged contrast (%)', _gap.median(),
        0.08, 0.005, '%')
    _c3 = pd.read_csv(_c30).set_index('site')
    _j = _c.set_index('site')[['mean_a']].join(_c3[['mean_a']], rsuffix='_30',
                                               how='inner').dropna()
    chk('path-averaged contrast, 10 against 30 km (%)',
        (_j.mean_a - _j.mean_a_30).abs().median(), 0.02, 0.005, '%')
    _w = pd.read_csv(_sp).pivot(index='site', columns='model', values='mean_a')
    _sd = _w.std(axis=1, ddof=1)
    chk('path-averaged contrast, cross-model s.d. (%)', _sd.median(), 0.31, 0.005, '%')
    chk('second-descent gap below the cross-model s.d.',
        float((_gap.values < _sd.reindex(_c.site).values).sum()), 48, 0, '')
    _r = pd.read_csv(_rr).set_index('root')
    _sr = _r.loc['second root']
    chk('second root inside a province (%)', 100.0 * _sr.in_province / _sr.n, 61.2, 0.1, '%')
    chk('second root province, Fisher P', _sr.fisher_p, 0.00035, 0.00005, '')
    chk('second root province, odds ratio', _sr.odds, 3.02, 0.005, '')
    _s = pd.read_csv(_rs)
    chk('hotspots changing province on the second root',
        float((_s.first_in.astype(bool) != _s.alt_in.astype(bool)).sum()), 10, 0, '')
else:
    missing.append('second-descent outputs (run competing_routes.py, root_robustness.py '
                   'and route_anomaly_spread.py)')

# Section 4.5: what a continuity constraint would buy and why it is not used. Each number
# comes from the file that measured it - bottleneck_report, frontier, tilt_recovery with
# the level axis, and detection_veto - so none of them is a figure typed from memory.
_bn = os.path.join(A.dir, 'bottleneck_report_RevealLO.csv')
_fr = os.path.join(A.dir, 'frontier_RevealLO.csv')
_tl = os.path.join(A.dir, 'tilt_recovery_RevealLO_levels.csv')
_dv = os.path.join(A.dir, 'detection_veto_RevealLO.csv')
if all(os.path.exists(x) for x in (_bn, _fr, _tl, _dv)):
    from scipy.stats import wilcoxon as _wx
    _b = pd.read_csv(_bn)
    chk('routes crossing a cell within 0.1% of its surroundings',
        float((_b.accepted > -0.10).sum()), 35, 0, '')
    _f = pd.read_csv(_fr); _f = _f[_f.reached == True]
    _fw = _f.pivot(index='site', columns='level')
    _m = _fw[('cap_anom', 35.0)].notna() & _fw[('cap_anom', 100.0)].notna()
    _a, _bb = _fw.loc[_m, ('cap_anom', 35.0)], _fw.loc[_m, ('cap_anom', 100.0)]
    _ca, _cb = _fw.loc[_m, ('cost', 35.0)], _fw.loc[_m, ('cost', 100.0)]
    chk('p35 median cost increase (%)', 100.0 * float(np.median(_ca / _cb - 1.0)),
        0.8, 0.05, '%')
    chk('p35 endpoints on slower deep material', float((_a < _bb - 0.05).sum()), 16, 0, '')
    chk('p35 endpoints on faster deep material', float((_a > _bb + 0.05).sum()), 1, 0, '')
    chk('p35 deep material, Wilcoxon P', float(_wx(_a, _bb).pvalue), 3.8e-5, 0.05e-5, '')
    _t = pd.read_csv(_tl); _t = _t[_t.tilt_deg == 20.0]
    chk('offset error at 20 deg tilt, p100 (km)',
        float(_t[_t.level == 100.0].error_km.abs().median()), 581, 1, 'km')
    chk('offset error at 20 deg tilt, p35 (km)',
        float(_t[_t.level == 35.0].error_km.abs().median()), 257, 1, 'km')
    _d = pd.read_csv(_dv); _d = _d[(_d.kind == 'injection') & _d.hit.notna()]
    _h = _d.hit.astype(str).str.lower().isin(['true', '1', '1.0'])
    chk('injection recovery, p100 (of 18)', float(_h[_d.level == 100.0].sum()), 18, 0, '')
    chk('injection recovery, p35 (of 18)', float(_h[_d.level == 35.0].sum()), 11, 0, '')
else:
    missing.append('continuity outputs (bottleneck_report, frontier, tilt_recovery '
                   '--levels, detection_veto)')

# Section 4.4: the second descents share a shallow column no more often, and part no
# deeper, than descents from random starts. The test was fixed before the paper-model run.
_hn = os.path.join(A.dir, 'competing_routes_RevealLO_null.csv')
if os.path.exists(_hn) and os.path.exists(os.path.join(A.dir, 'competing_routes_RevealLO.csv')):
    from scipy.stats import mannwhitneyu as _mw
    _hh = pd.read_csv(os.path.join(A.dir, 'competing_routes_RevealLO.csv'))
    _hh = _hh[_hh.alt_reached == True].split_depth_km.dropna()
    _nn = pd.read_csv(_hn); _nn = _nn[_nn.alt_reached == True].split_depth_km.dropna()
    chk('split depth, hotspots (km)', float(_hh.median()), 800, 1, 'km')
    chk('split depth, random starts (km)', float(_nn.median()), 700, 1, 'km')
    chk('split depth, one-sided Mann-Whitney P',
        float(_mw(_hh, _nn, alternative='greater').pvalue), 0.46, 0.005, '')
else:
    missing.append('competing_routes_RevealLO_null.csv (competing_routes.py --suffix _null)')

# Section 5.3: these numbers lived only in the typed manuscript until province_flip.py
# was written to compute them. An audit that checks only what someone remembered to
# compute is an audit of the author's memory.
_fs = pd.read_csv(os.path.join(A.dir, 'province_flip_summary_RevealLO.csv'),
                  index_col=0)['value']
chk('hotspots with isotopic data', float(_fs['n_hotspots']), 39, 0, '')
chk('classifications disagree', float(_fs['n_flipped']), 12, 0, '')
chk('inside a province, traced root', float(_fs['n_traced_inside']), 29, 0, '')
chk('inside a province, surface position', float(_fs['n_surface_inside']), 17, 0, '')
chk('flips that move a hotspot outward', float(_fs['flipped_to_outside']), 0, 0, '')
_ft = pd.read_csv(os.path.join(A.dir, 'province_flip_tests_RevealLO.csv'))
for _iso, _cls, _want in (('87Sr/86Sr', 'traced root', 0.043),
                          ('87Sr/86Sr', 'surface position', 0.089),
                          ('143Nd/144Nd', 'traced root', 0.043),
                          ('143Nd/144Nd', 'surface position', 0.126)):
    _r = _ft[(_ft.isotope == _iso) & (_ft.classification == _cls)]
    chk(f'{_iso} on {_cls}', float(_r.p.iloc[0]), _want, 0.0006, '')

# Section 3.2: the ambient control. These are the numbers that decide what the paper
# may say about displacement, so they are checked like any other.
_an = os.path.join(A.dir, 'ambient_null_summary_RevealLO.csv')
if os.path.exists(_an):
    _a = pd.read_csv(_an).set_index('what')['value']
    chk('hotspot median offset', float(_a['hotspot median offset km']), 496, 1, 'km')
    chk('ambient median offset', float(_a['ambient median offset km']), 480, 1, 'km')
    chk('P offset hotspots vs ambient',
        float(_a['P magnitude hotspots vs ambient']), 0.87, 0.006, '')
    chk('ambient directional n', float(_a['ambient n direction']), 28, 0, '')
    chk('ambient directional R', float(_a['ambient R direction']), 0.074, 0.0006, '')
    chk('ambient directional P', float(_a['ambient P direction']), 0.52, 0.006, '')
else:
    print('  ambient_null_summary not on disk; run ambient_null.py')

# Section 2: the cross-model directional tests. Both windows in all six runs, from
# crossmodel_lean.csv, so the figure panel and the text cannot drift apart.
_cm = pd.read_csv(os.path.join(A.dir, 'crossmodel_lean.csv'))


def _cmv(model, window, col):
    _r = _cm[(_cm.model == model) & (_cm.window == window)]
    return float(_r[col].iloc[0]) if len(_r) else float('nan')


def _tol(v):
    """Tolerance from the precision the manuscript quotes.

    A number written as 0.42 is not claiming 0.4200, so checking it to four decimals
    reports a disagreement that is only rounding. Half a unit in the last quoted place
    is the tolerance the text actually asserts.
    """
    t = f'{v!r}'
    d = len(t.split('.')[1]) if '.' in t else 0
    return 0.5 * 10 ** (-d) + 1e-12


# Table S2 is generated from crossmodel_lean.csv by si_tables.py, so there is nothing
# here to check row by row: 25 literals restating the file were how the supplement came
# to carry R and P from a superseded move set while every digit in them matched
# something else in the document. One structural check replaces them, and it compares
# the supplement against the file rather than against a copy of the file.
_s2 = subprocess.run([sys.executable, 'si_tables.py', '--table', 'S2', '--check',
                      '--dir', A.dir, '--si', A.si],
                     cwd=os.path.dirname(os.path.abspath(__file__)),
                     capture_output=True, text=True)
chk('Table S2 matches crossmodel_lean.csv', 0 if _s2.returncode == 0 else 1, 0, 0, '')
if _s2.returncode != 0:
    print(_s2.stdout)

# Tables S1 and S3 are pasted from supplement_table.py's output in the same way, and are
# checked the same way.
_s13 = subprocess.run([sys.executable, 'supplement_table.py', '--dir', A.dir,
                       '--tag', A.tag, '--check-si', A.si],
                      cwd=os.path.dirname(os.path.abspath(__file__)),
                      capture_output=True, text=True)
chk('Tables S1 and S3 match supplement_table.py', 0 if _s13.returncode == 0 else 1, 0, 0, '')
if _s13.returncode != 0:
    print(_s13.stdout)

# Text S3 counts, model by model, the paths each window excludes, beside Table S2 and
# from the same file. The counts were typed and had gone stale - no RevealLO path
# vertical, 39 paths in the plate-motion test - while the generated table next to them
# was current, and no check covered them.
_clp = os.path.join(A.dir, 'crossmodel_lean.csv')
if os.path.exists(_clp):
    _cx = pd.read_csv(_clp).set_index(['model', 'window'])

    def _cxv(m, w, col):
        try:
            return float(_cx.loc[(m, w), col])
        except KeyError:
            return float('nan')
    chk('Text S3 plate-motion test, RevealLO paths tested', _cxv('RevealLO', '300-660', 'n'),
        31, 0, '')
    for _m, _q in (('RevealLO', 3), ('RevealLO_30km', 1), ('REVEAL', 3),
                   ('SEMUCB-WM1', 6), ('SPiRaL', 9), ('GLADM35', 19)):
        chk(f'Text S3 vertical over 300-660 km, {_m}', _cxv(_m, '300-660', 'vertical'),
            _q, 0, '')
    for _m, _q in (('RevealLO', 0), ('RevealLO_30km', 0), ('REVEAL', 1), ('SPiRaL', 1),
                   ('GLADM35', 5), ('SEMUCB-WM1', 5)):
        chk(f'Text S3 displaced under 20 km over 660-1500 km, {_m}',
            _cxv(_m, '660-1500', 'short'), _q, 0, '')
    chk('Text S3 GLAD-M35 paths tested over 660-1500 km', _cxv('GLADM35', '660-1500', 'n'),
        44, 0, '')
else:
    missing.append('crossmodel_lean.csv')

# Section 3.4: which paths are reproducible. path_reliability.py measures this from
# the traced paths and the resolution control, so the section quotes a file rather
# than a judgement.
_pr = os.path.join(A.dir, 'path_reliability_RevealLO.csv')
if os.path.exists(_pr):
    rel = pd.read_csv(_pr)
    _st = rel[rel.stable.astype(bool)]
    _un = rel[~rel.stable.astype(bool)]
    chk('stable paths', len(_st), 36, 0, '')
    chk('unstable paths', len(_un), 13, 0, '')
    chk('stable separation, median', float(_st.separation_km.median()), 120, 2, 'km')
    chk('stable separation, largest', float(_st.separation_km.max()), 492, 2, 'km')
    chk('unstable separation, smallest', float(_un.separation_km.min()), 751, 2, 'km')
    chk('unstable separation, largest', float(_un.separation_km.max()), 3339, 2, 'km')
    # The manuscript states a RANGE over three named sites, not four per-site values, and
    # never mentions Louisville here. Checking per-site numbers the paper does not quote
    # leaves the sentence it does quote unchecked, which is how 35-84 km went stale
    # unnoticed. The range is what the reader is given, so the range is what is checked.
    _s = rel.set_index('site')
    sep_trio = [n for n in ('Kerguelen(Heard)', 'Reunion', 'Hawaii') if n in _s.index]
    if len(sep_trio) == 3:
        _sep = _s.loc[sep_trio, 'separation_km']
        chk('separation range over Kerguelen, Reunion and Hawaii, low',
            float(_sep.min()), 35, 1, 'km')
        chk('separation range over Kerguelen, Reunion and Hawaii, high',
            float(_sep.max()), 84, 1, 'km')

    # The four tilt angles quoted in the same paragraph are, since v9, those of Table S1:
    # measured over 400-2700 km by plume_geometry.py and read from plume_slant_RevealLO.csv,
    # the file supplement_table.py builds the table from, so text and table cannot differ.
    # (v8 quoted tilts from the hotspot's surface position over the whole path, 13.5,
    # 14.2, 25.7 and 17.5 degrees, beside a table saying 12, 10, 20 and 15.)
    tlt_sl = os.path.join(A.dir, 'plume_slant_RevealLO.csv')
    if os.path.exists(tlt_sl):
        tlt_s = pd.read_csv(tlt_sl).set_index('site')
        for tlt_name, tlt_q in (('Kerguelen(Heard)', 11.7), ('Reunion', 10.0),
                                ('Marion', 20.3), ('Hawaii', 15.0)):
            if tlt_name in tlt_s.index:
                chk(f'tilt, {tlt_name}', float(tlt_s.loc[tlt_name, 'tilt']), tlt_q, 0.06, ' deg')
    else:
        missing.append('plume_slant_RevealLO.csv (the four tilts of section 3.4)')
    # The ridge test of Text S3 and Figure 4c, written by ridge_lean_confirm.py.
    tlt_rl = os.path.join(A.dir, 'ridge_lean_RevealLO.csv')
    if os.path.exists(tlt_rl):
        tlt_r = pd.read_csv(tlt_rl).set_index('group')
        chk('ridge test, near paths', int(tlt_r.loc['near', 'n']), 20, 0, '')
        chk('ridge test, far paths', int(tlt_r.loc['far', 'n']), 11, 0, '')
        chk('ridge test, near P', float(tlt_r.loc['near', 'P']), 0.48, 0.006, '')
        chk('ridge test, far P', float(tlt_r.loc['far', 'P']), 0.27, 0.006, '')
    else:
        missing.append('ridge_lean_RevealLO.csv (Text S3: run ridge_lean_confirm.py)')

    for _n, _q in (('Hawaii', 53), ('Kerguelen(Heard)', 35), ('Reunion', 84),
                   ('Louisville', 68)):
        if _n in _s.index:
            chk(f'separation, {_n}', float(_s.at[_n, 'separation_km']), _q, 2, 'km')
    for _n, _q in (('Crozet/Pr. Edward', 39), ('Darfur', 38), ('Baja/Guadalupe', 32)):
        if _n in _s.index:
            chk(f'ponding, {_n}', float(_s.at[_n, 'ponding']), _q, 0.6, '%')
    _r, _ = spearmanr(rel.ponding, rel.separation_km)
    chk('ponding against instability, rho', float(_r), 0.57, 0.006, '')
else:
    missing.append('path_reliability_RevealLO.csv (run path_reliability.py)')

# Section 3: the deformation-driver numbers, computed by deformation_drivers.py.
_dd = pd.read_csv(os.path.join(A.dir, 'deformation_drivers_RevealLO.csv')
                  ).set_index('what')['value']
_dd = Missing(_dd.to_dict())
chk('tilt is offset, max reconstruction error', float(_dd['tilt reconstructed from '
    'offset, max abs error (deg)']), 0.0, 1e-6, 'deg')
chk('rho offset vs tilt', float(_dd['rho offset vs tilt']), 0.998, 0.001, '')
chk('rho offset vs plate speed', float(_dd['rho offset vs plate speed']), 0.07, 0.006, '')
chk('P offset vs plate speed', float(_dd['P offset vs plate speed']), 0.65, 0.006, '')
chk('rho offset vs spreading centre of any age', float(_dd['rho offset vs spreading centre of any age']),
    -0.17, 0.006, '')
chk('P offset vs spreading centre of any age', float(_dd['P offset vs spreading centre of any age']), 0.25, 0.006, '')
chk('rho offset vs depth into province',
    float(_dd['rho offset vs depth into province']), -0.11, 0.006, '')
chk('P offset vs depth into province',
    float(_dd['P offset vs depth into province']), 0.47, 0.006, '')
chk('narrowest width-band correlation', float(_dd['rho w_shallow vs w_deep']), 0.40, 0.006, '')
chk('widest width-band correlation', float(_dd['rho w_mid vs w_deep']), 0.73, 0.006, '')
chk('rho deep width vs offset', float(_dd['rho deep width vs offset']), -0.13, 0.006, '')
chk('P group split on plate speed', float(_dd['P group split on plate speed']), 0.12, 0.006, '')
chk('P group split on spreading centre of any age',
    float(_dd['P group split on spreading centre of any age']), 0.30, 0.006, '')
chk('P group split on depth into province',
    float(_dd['P group split on depth into province']), 0.75, 0.006, '')
chk('P group split on province rooting', float(_dd['P group split on root_in']), 0.70, 0.006, '')
chk('P group split on network membership', float(_dd['P group split on linked']), 1.00, 0.006, '')
chk('median offset, deformed group', float(_dd['median offset group 1']), 669, 1, 'km')
chk('median offset, rest', float(_dd['median offset group 0']), 334, 1, 'km')
chk('median deep corridor, deformed group', float(_dd['median w_deep group 1']), 565, 1, 'km')
chk('median deep corridor, rest', float(_dd['median w_deep group 0']), 464, 1, 'km')
chk('size of the deformed group', float(_dd['n group 1']), 10, 0, '')

chk('Hawaii agreement with Zhang and Hu',
    abs((_dl['Hawaii'] - 225 + 180) % 360 - 180), 17, 1, 'deg')

if Missing.absent:
    print('INPUTS THAT NO LONGER EXIST, so their checks could not run:')
    for _k in Missing.absent:
        print(f'  {_k}')
    print('An analysis has changed shape. Decide whether the claim resting on it\n'
          'survives before restoring the check or removing it, and do not read the\n'
          'nan rows below as disagreements - they are absences.\n')

if missing:
    print(f'{len(missing)} inputs the audit could not read, so their checks did not run:')
    for _m in missing:
        print(f'  {_m}')
    print()

# ---------------------------------------------------------------- the depth census
# Sections 3.1, 3.2 and 3.6, Figure 6, Tables S4-S6: everything read from
# out/mid_mantle_summary.csv, which mid_mantle_census.py writes from the paths, the
# ambient paths, root_depth_<tag>.csv, conduit_profile_<tag>[_xm].csv and the
# corridor profiles. Values are quoted as the text quotes them.
_mm = os.path.join(A.dir, 'mid_mantle_summary.csv')
if os.path.exists(_mm):
    _S = pd.read_csv(_mm)

    def _mmv(tag, q, band, col='hotspot'):
        r = _S[(_S.tag == tag) & (_S.quantity == q) & (_S.band == band)]
        return float(r[col].iloc[0]) if len(r) else float('nan')

    for tag, qh, qa in (('RevealLO', 22.4, 4.3), ('REVEAL', 26.5, 7.3), ('GLADM35', 20.4, 6.3),
                        ('SEMUCB-WM1', 24.5, 8.7), ('SPiRaL', 4.1, 6.0)):
        chk(f'connected column to 2600 km, hotspots, {tag} (%)', 100 * _mmv(tag, 'root_ge_2600_share', 'column'), qh, 0.05, '%')
        chk(f'connected column to 2600 km, random sites, {tag} (%)', 100 * _mmv(tag, 'root_ge_2600_share', 'column', 'ambient'), qa, 0.05, '%')
    for tag, q in (('RevealLO', 5.2), ('REVEAL', 3.6), ('GLADM35', 3.2), ('SEMUCB-WM1', 2.8)):
        chk(f'connected column, hotspot-to-random factor, {tag}', _mmv(tag, 'root_ge_2600_share', 'column') / _mmv(tag, 'root_ge_2600_share', 'column', 'ambient'), q, 0.05, '')
    for tag, q in (('RevealLO', 23), ('REVEAL', 5)):
        chk(f'basal slow fraction, hotspot-to-random factor, {tag}', _mmv(tag, 'deep_frac_median', 'column') / _mmv(tag, 'deep_frac_median', 'column', 'ambient'), q, 0.5, '')
    _rh = [_mmv(t, 'root_km_median', 'column') for t in ('RevealLO', 'REVEAL', 'GLADM35', 'SEMUCB-WM1')]
    _ra = [_mmv(t, 'root_km_median', 'column', 'ambient') for t in ('RevealLO', 'REVEAL', 'GLADM35', 'SEMUCB-WM1')]
    chk('median root, hotspots, four models, low (km)', min(_rh), 1141, 1, 'km')
    chk('median root, hotspots, four models, high (km)', max(_rh), 1530, 1, 'km')
    chk('median root, random sites, four models, low (km)', min(_ra), 855, 1, 'km')
    chk('median root, hotspots, RevealLO (km)', _mmv('RevealLO', 'root_km_median', 'column'), 1350, 1, 'km')
    chk('median root, hotspots, REVEAL (km)', _mmv('REVEAL', 'root_km_median', 'column'), 1450, 1, 'km')
    chk('median root, random sites, four models, high (km)', max(_ra), 1030, 1, 'km')
    _dh = [_mmv(t, 'deep_frac_median', 'column') for t in ('RevealLO', 'REVEAL', 'GLADM35', 'SEMUCB-WM1')]
    _da = [_mmv(t, 'deep_frac_median', 'column', 'ambient') for t in ('RevealLO', 'REVEAL', 'GLADM35', 'SEMUCB-WM1')]
    chk('basal window slow, hotspots, low', min(_dh), 0.26, 0.005, '')
    chk('basal window slow, hotspots, high', max(_dh), 0.50, 0.005, '')
    for tag, qh, qa in (('RevealLO', 0.50, 0.02), ('REVEAL', 0.50, 0.10), ('GLADM35', 0.26, 0.00), ('SEMUCB-WM1', 0.44, 0.00)):
        chk(f'basal window slow, hotspots, {tag}', _mmv(tag, 'deep_frac_median', 'column'), qh, 0.005, '')
        chk(f'basal window slow, random sites, {tag}', _mmv(tag, 'deep_frac_median', 'column', 'ambient'), qa, 0.005, '')
    chk('basal window slow, random sites, low', min(_da), 0.00, 0.005, '')
    chk('basal window slow, random sites, high', max(_da), 0.10, 0.005, '')
    for tag, qa, qb in (('RevealLO', 0.67, 0.71), ('REVEAL', 0.70, 0.73), ('GLADM35', 0.34, 0.39), ('SPiRaL', 0.29, 0.42), ('SEMUCB-WM1', 0.38, 0.41)):
        chk(f'prominence at 660-1000 km, hotspots, {tag} (%)', _mmv(tag, 'prominence', '660-1000'), qa, 0.005, '%')
        chk(f'prominence at 1000-1500 km, hotspots, {tag} (%)', _mmv(tag, 'prominence', '1000-1500'), qb, 0.005, '%')
        chk(f'prominence rises across the increase, {tag}', float(_mmv(tag, 'prominence', '1000-1500') > _mmv(tag, 'prominence', '660-1000')), 1, 0, '')
        _pk = max(_mmv(tag, 'prominence', b) for b in ('660-1000', '1000-1500', '1500-2200', '2200-2700'))
        chk(f'prominence highest at 1000-1500 km, {tag}', float(_mmv(tag, 'prominence', '1000-1500') == _pk),
            0 if tag == 'GLADM35' else 1, 0, '')
    for b_ in ('1500-2200', '2200-2700'):
        chk(f'prominence, GLADM35, {b_} km (%)', _mmv('GLADM35', 'prominence', b_), 0.39, 0.005, '%')
    for tag, qs in (('RevealLO', (0.25, 0.19, 0.11, 0.15)), ('REVEAL', (0.28, 0.21, 0.09, 0.11)), ('GLADM35', (0.14, 0.11, 0.08, 0.08)),
                    ('SEMUCB-WM1', (0.13, 0.07, 0.02, 0.00))):
        for b, q in zip(('660-1000', '1000-1500', '1500-2200', '2200-2700'), qs):
            chk(f'prominence excess over null sites, {tag}, {b} km', _mmv(tag, 'prominence', b) - _mmv(tag, 'prominence', b, 'ambient'), q, 0.005, '')
    for b, qh, qa, qp in (('400-660', 24.7, 24.9, None), ('660-1000', 24.5, 21.8, 0.49), ('1000-1500', 19.3, 14.3, 0.005),
                          ('1500-2200', 13.8, 12.7, 0.086), ('2200-2700', 4.2, 4.1, None)):
        chk(f'apparent tilt, hotspots, RevealLO, {b} km', _mmv('RevealLO', 'tilt_deg', b), qh, 0.05, 'deg')
        chk(f'apparent tilt, ambient, RevealLO, {b} km', _mmv('RevealLO', 'tilt_deg', b, 'ambient'), qa, 0.05, 'deg')
        if qp is not None:
            chk(f'apparent tilt, permutation P, RevealLO, {b} km', _mmv('RevealLO', 'tilt_deg', b, 'P'), qp, 0.0015 if qp < 0.01 else 0.01, '')
    for tag, q in (('GLADM35', (20.6, 20.8, 0.94)), ('SEMUCB-WM1', (10.4, 11.2, 0.63))):
        chk(f'apparent tilt, hotspots, {tag}, 1000-1500 km', _mmv(tag, 'tilt_deg', '1000-1500'), q[0], 0.05, 'deg')
        chk(f'apparent tilt, ambient, {tag}, 1000-1500 km', _mmv(tag, 'tilt_deg', '1000-1500', 'ambient'), q[1], 0.05, 'deg')
        chk(f'apparent tilt, permutation P, {tag}, 1000-1500 km', _mmv(tag, 'tilt_deg', '1000-1500', 'P'), q[2], 0.01, '')
    chk('shell percentile, hotspots, RevealLO, 400-660 km', _mmv('RevealLO', 'shell_percentile', '400-660'), 0.292, 0.0005, '')
    chk('shell percentile, ambient, RevealLO, 400-660 km', _mmv('RevealLO', 'shell_percentile', '400-660', 'ambient'), 0.295, 0.0005, '')
    chk('shell percentile, hotspots, RevealLO, 1000-1500 km', _mmv('RevealLO', 'shell_percentile', '1000-1500'), 0.145, 0.0005, '')
    chk('shell percentile, ambient, RevealLO, 1000-1500 km', _mmv('RevealLO', 'shell_percentile', '1000-1500', 'ambient'), 0.324, 0.0005, '')
    chk('shell percentile, permutation P below 1e-4, RevealLO, 1000-1500 km', float(_mmv('RevealLO', 'shell_percentile', '1000-1500', 'P') < 1e-4), 1, 0, '')
    for tag, band in (('RevealLO', '1000-1500'), ('GLADM35', '1000-1500'), ('SEMUCB-WM1', '1500-2200')):
        _sep = {b: _mmv(tag, 'shell_percentile', b, 'ambient') - _mmv(tag, 'shell_percentile', b) for b in ('400-660', '660-1000', '1000-1500', '1500-2200', '2200-2700')}
        chk(f'shell percentile, largest separation in {band} km, {tag}', float(max(_sep, key=_sep.get) == band), 1, 0, '')
    chk('corridor width, RevealLO, 400-660 km', _mmv('RevealLO', 'corridor_width_km', '400-660'), 457, 1, 'km')
    chk('corridor width, RevealLO, 2200-2700 km', _mmv('RevealLO', 'corridor_width_km', '2200-2700'), 730, 1, 'km')
    _ow = [_mmv(t, 'corridor_width_km', '400-660') for t in ('REVEAL', 'GLADM35', 'SEMUCB-WM1')]
    _od = [_mmv(t, 'corridor_width_km', '2200-2700') for t in ('REVEAL', 'GLADM35', 'SEMUCB-WM1')]
    chk('corridor width, other models, 400-660 km, low', min(_ow), 322, 1, 'km')
    chk('corridor width, other models, 400-660 km, high', max(_ow), 484, 1, 'km')
    chk('corridor width, other models, 2200-2700 km, low', min(_od), 656, 1, 'km')
    chk('corridor width, other models, 2200-2700 km, high', max(_od), 742, 1, 'km')
    for b, qh, qa in (('660-1000', 7.7, 8.6), ('1000-1500', 24.9, 37.1), ('1500-2200', 56.4, 47.3)):
        chk(f'deflection share, hotspots, RevealLO, {b} km (%)', 100 * _mmv('RevealLO', 'deflection_share', b), qh, 0.05, '%')
        chk(f'deflection share, ambient, RevealLO, {b} km (%)', 100 * _mmv('RevealLO', 'deflection_share', b, 'ambient'), qa, 0.05, '%')
    chk('per-path median deflection depth, hotspots (km)', _mmv('RevealLO', 'deflection_depth_median_km', 'column'), 1598, 1, 'km')
    chk('per-path median deflection depth, ambient (km)', _mmv('RevealLO', 'deflection_depth_median_km', 'column', 'ambient'), 1475, 1, 'km')
    chk('per-path median deflection depth, permutation P', _mmv('RevealLO', 'deflection_depth_median_km', 'column', 'P'), 0.20, 0.01, '')
    for tag, qh, qa, qp in (('GLADM35', 1440, 1530, 0.24), ('SEMUCB-WM1', 804, 841, 0.73)):
        chk(f'per-path median deflection depth, hotspots, {tag} (km)', _mmv(tag, 'deflection_depth_median_km', 'column'), qh, 1, 'km')
        chk(f'per-path median deflection depth, ambient, {tag} (km)', _mmv(tag, 'deflection_depth_median_km', 'column', 'ambient'), qa, 1, 'km')
        chk(f'per-path median deflection depth, permutation P, {tag}', _mmv(tag, 'deflection_depth_median_km', 'column', 'P'), qp, 0.01, '')
    for tag, q in (('RevealLO', 18), ('GLADM35', 22), ('SEMUCB-WM1', 37)):
        chk(f'deflections at 800-1200 km, ambient, {tag} (%)', 100 * _mmv(tag, 'horizon_ratio_800_1200', 'column', 'ambient') * 400 / 1540, q, 0.5, '%')
    for tag, q in (('RevealLO', 15), ('GLADM35', 29), ('SPiRaL', 31), ('REVEAL', 37), ('SEMUCB-WM1', 38)):
        chk(f'deflections at 800-1200 km, share of 660-2200 km, {tag} (%)', 100 * _mmv(tag, 'horizon_ratio_800_1200', 'column') * 400 / 1540, q, 0.5, '%')
    chk('deflection horizon ratio, largest of five models', max(_mmv(t, 'horizon_ratio_800_1200', 'column') for t in ('RevealLO', 'REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1')), 1.48, 0.02, '')
else:
    missing.append('mid_mantle_summary.csv (run mid_mantle_census.py)')

if retired:
    print(f'{len(retired)} checks retired by manuscript_expect.json (claim no longer made):')
    for lab in retired:
        print(f'  {lab}')
    print()

# Standing of the inputs first. A reader who sees the mismatch list before knowing
# whether the files behind it are current will act on the wrong half of it.
def si_config_table(si_path, d):
    """Does Text S1's configuration table say what the frozen files say?

    The table is the reader's only statement of what produced every number in the paper,
    and it is typed. It had RevealLO at s = 0.4 on the anomaly channel for a day after the
    frozen file said otherwise, and nothing reported it, because no check compared the two.
    """
    import json as _sj
    NAME = {'RevealLO': 'RevealLO', 'RevealLO, 30 km sampling': 'RevealLO_30km',
            'REVEAL': 'REVEAL', 'GLAD-M35': 'GLADM35', 'SPiRaL': 'SPiRaL',
            'SEMUCB-WM1': 'SEMUCB-WM1'}
    CH = {'local contrast': 'contrast', 'anomaly': 'anom',
          'minimum of anomaly and local contrast': 'min'}
    if not os.path.exists(si_path):
        return 0, 0
    agree = seen = 0
    for line in open(si_path).read().splitlines():
        if not line.startswith('| '):
            continue
        p = [x.strip() for x in line.strip('|').split('|')]
        if len(p) != 6 or p[0] not in NAME:
            continue
        fp = os.path.join(d, f'path_config_{NAME[p[0]]}.json')
        if not os.path.exists(fp):
            continue
        c = _sj.load(open(fp))['config']
        seen += 1
        try:
            same = (float(p[1]) == float(c['s'])
                    and float(p[2]) == float(c['z_target'])
                    and CH.get(p[3], p[3]) == str(c['channel'])
                    and float(p[4]) == float(c['radius'])
                    and float(p[5]) == float(c['h_max']))
        except ValueError:
            same = False
        if same:
            agree += 1
        else:
            print(f'  Text S1 row {p[0]!r} disagrees with {os.path.basename(fp)}: '
                  f'table says s={p[1]} z={p[2]} {p[3]} r={p[4]} reach={p[5]}, '
                  f'frozen says s={c["s"]:g} z={c["z_target"]:g} {c["channel"]} '
                  f'r={c["radius"]:g} reach={c["h_max"]:g}')
    return agree, seen


_si_ok, _si_n = si_config_table(A.si, A.dir)
if _si_n:
    chk('Text S1 configuration rows matching the frozen files', _si_ok, _si_n, 0)

_n_ok, _n_seen = configs_clearing_floor(A.dir)
chk('frozen configurations clearing the detection floor', _n_ok, _n_seen, 0)

_n_stale = provenance_report(A.dir, A.tag)
_n_stale += freshness.report(A.dir, A.tag)

bad = [c for c in checks if not c[0]]
print(f'{len(checks)} numbers checked, {len(checks) - len(bad)} agree\n')
for ok, lab, comp, quoted, unit in checks:
    if not ok:
        print(f'  MISMATCH  {lab}: computed {comp:.4g}{unit}, manuscript says '
              f'{quoted:g}{unit}')
if not bad and not _n_stale:
    print('  every checked number in the manuscript matches the outputs on disk')
elif not bad:
    raise SystemExit(f'\nno mismatches, but {_n_stale} input(s) are stale, so the '
                     f'agreement above is not evidence of anything')
else:
    raise SystemExit(f'\n{len(bad)} mismatches; fix the manuscript or the claim'
                     + (f'. {_n_stale} input(s) are also stale - regenerate those first, '
                        f'because their checks are describing an old configuration'
                        if _n_stale else ''))
