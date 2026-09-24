#!/usr/bin/env python3
"""Is the track evidence, or only the route the search was obliged to return?

The search is asked for the cheapest descending path to 2700 km and it returns one for
every hotspot, whether or not there is anything there to follow. It cannot decline. So
a track reaching the core-mantle boundary is not by itself a claim about the mantle,
and the paper needs a measurement that separates the tracks that are evidence from the
tracks that are merely output.

Two things are measured, and they turn out to be independent.

MATERIAL. At every point on the path, where does the anomaly sit within the whole
shell at that depth? A path running through the slowest two per cent of the mantle at
its depth is in the kind of material a conduit occupies; one sitting near the middle of
its shell is in ordinary mantle. Reported per depth band as the median percentile.

This does not decide which material belongs to the plume, which is why the earlier
root-depth measurements were withdrawn and why this one is not the same thing. It asks
only where the route the search chose sits in the distribution of the shell it passes
through. No segmentation, no ownership.

GEOMETRY. Total lateral travel divided by net lateral displacement. A conduit followed
cleanly gives a number near one. A path that doubles back, overshoots and returns has
travelled far to arrive nowhere, and the ratio grows without any anomaly being weak.
A path can be strong in material and incoherent in geometry at the same time, and two
of the hardest cases in this study are exactly that, so a single quality score would
hide them.

THE NULL. Both are measured identically on paths traced from ambient sites with nothing
injected, which is the only scale on which either number means anything. Thresholds
come from that distribution and not from the hotspots.

DECISION RULE, FIXED BEFORE THIS WAS RUN ON THE 49.

  A depth band is SUPPORTED if the path's median shell percentile in that band is below
  the 5th percentile of the ambient paths' median percentile over the same band. Five
  per cent because that is the threshold the classification already uses against its
  own null, and changing it here to something else would be choosing a number to suit
  an answer.

  A path is COHERENT if its wander is below the 95th percentile of ambient wander.

  THE VERDICT IS ABOUT THE DEEP BAND, and connection to the surface is reported
  beside it rather than folded into it. The first version of this rule made rooting
  depth the base of the deepest supported band with every shallower band supported
  too, which sounds careful and is wrong: it makes the shallowest band gate
  everything. Run on the 49 it called Hoggar, Meteor, Easter and Tristan weak - the
  second, third, fourth and fifth strongest DEEP signatures in the whole set - because
  their upper mantle is unremarkable. A hotspot whose deep segment sits in the slowest
  half per cent of the lower mantle is the best deep-rooting candidate there is, and
  the upper mantle, where ridges and plate-scale structure dominate, is the band least
  able to speak to depth of origin.

  So three facts are reported, each independently:

    DEEP        1500-2700 supported.
    CONNECTED   every band supported, so the deep body and the hotspot are joined by
                material that is slow all the way. A deep signature that is not
                connected is the interesting case, not a failure to be hidden: it is
                what a search obliged to reach 2700 km does when the two ends are real
                and the middle is not.
    COHERENT    wander below the ambient threshold. Reported as a flag rather than as
                a verdict of its own, because a path can be incoherent AND have a
                strong deep signature, and the first version of this rule threw the
                second fact away.

  The label combines them: 'deep, connected'; 'deep, not connected'; 'shallow only'
  where some band is supported but not the deep one; 'no support' where none is. The
  coherence flag is appended to whichever of those applies.

  This correction was made after the rule's first run, which is exactly what
  pre-registration is meant to prevent, so it is recorded plainly: the reason is that
  the rule did not measure what it claimed, demonstrated on the four sites above, and
  not that the answer was unwelcome. It has not been applied to any result yet - the
  ambient null it needs is being rebuilt at 300 sites, because a 5th percentile of 30
  is the second-smallest value in the sample and the deep threshold it produced,
  0.012, came out TIGHTER than the shallow one, which has no physical reading.

Root depth was dropped from the parameter set because every trace terminates at the
target by construction and is therefore a constant. This gives it back as something
measured rather than imposed.

    python3 track_support.py --file <model>.nc --tag RevealLO

Needs ambient_paths_<tag>.json, written by
    python3 tilt_recovery.py --file <model>.nc --amp 0 --tilts 0 --sites 30 \
        --lateral 0.60 --ncell 2 --no-resume --save-paths ambient_paths_RevealLO.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import provenance
import freshness
import pandas as pd
from scipy.stats import mannwhitneyu

from tomo_io import ModelSpec, load_anomaly, dedupe_lon

R_E, DEG = 6371.0, np.pi / 180.0
BANDS = (('200-660', 200.0, 660.0), ('660-1500', 660.0, 1500.0),
         ('1500-2700', 1500.0, 2701.0))
SUPPORT_PCT = 5.0        # ambient percentile a band must beat
COHERENT_PCT = 95.0      # ambient percentile wander must stay below


def gc_km(a, b, c, d):
    return R_E * np.arccos(np.clip(
        np.sin(a * DEG) * np.sin(c * DEG)
        + np.cos(a * DEG) * np.cos(c * DEG) * np.cos((d - b) * DEG), -1, 1))


def measure(paths, z, lat, lon, order):
    """Band percentiles and wander for every path in a dict of traces."""
    out = {}
    for name, q in paths.items():
        pz = np.asarray(q['depth'], float)
        pla = np.asarray(q['lat'], float)
        plo = np.asarray(q['lon'], float)
        if len(pz) < 3:
            continue
        o = np.argsort(pz)
        pz, pla, plo = pz[o], pla[o], plo[o]

        pct = np.empty(len(pz))
        for n, (zz, la, lo_) in enumerate(zip(pz, pla, plo)):
            k = int(np.argmin(np.abs(z - zz)))
            j = int(np.argmin(np.abs(lat - la)))
            i = int(np.argmin(np.abs(lon - lo_)))
            pct[n] = np.searchsorted(order[k], float(_anom[k, j, i])) / order.shape[1]

        rec = {}
        for nm, a, b in BANDS:
            m = (pz >= a) & (pz < b)
            rec[nm] = float(np.median(pct[m])) if m.any() else np.nan
        seg = gc_km(pla[:-1], plo[:-1], pla[1:], plo[1:])
        net = float(gc_km(pla[0], plo[0], pla[-1], plo[-1]))
        rec['wander'] = float(seg.sum()) / max(net, 1.0)
        rec['depth_max'] = float(pz.max())
        out[name] = rec
    return pd.DataFrame(out).T


def _report(H, A_, a):
    names = [nm for nm, _, _ in BANDS]

    # ---- THE RESULT: do hotspot paths run through slower material than paths that
    # have nothing to follow? A rank test on the whole distribution, one per band.
    print('=' * 74)
    print('POPULATION TEST, hotspot paths against ambient paths')
    print(f"{'band':12s}{'hotspot':>10s}{'ambient':>10s}{'Mann-Whitney P':>18s}{'AUC':>8s}")
    pop = []
    for nm in names:
        hv = H[nm].dropna().to_numpy(float)
        av = A_[nm].dropna().to_numpy(float)
        P = float(mannwhitneyu(hv, av, alternative='less').pvalue)
        auc = float(np.mean(hv[:, None] < av[None, :])
                    + 0.5 * np.mean(hv[:, None] == av[None, :]))
        print(f'{nm:12s}{np.median(hv):10.3f}{np.median(av):10.3f}{P:18.2e}{auc:8.3f}')
        pop.append(dict(band=nm, hotspot_median=float(np.median(hv)),
                        ambient_median=float(np.median(av)), P=P, auc=auc,
                        n_hotspot=len(hv), n_ambient=len(av)))
    wP = float(mannwhitneyu(H['wander'].dropna(), A_['wander'].dropna(),
                            alternative='greater').pvalue)
    print(f"{'wander':12s}{float(np.median(H.wander)):10.3f}"
          f"{float(np.median(A_.wander)):10.3f}{wP:18.2e}")
    pop.append(dict(band='wander', hotspot_median=float(np.median(H.wander)),
                    ambient_median=float(np.median(A_.wander)), P=wP, auc=np.nan,
                    n_hotspot=len(H), n_ambient=len(A_)))
    pd.DataFrame(pop).to_csv(
        os.path.join(a.dir, f'track_support_population_{a.tag}.csv'), index=False)

    # ---- the per-hotspot description. Where each hotspot sits in the ambient
    # distribution, as a fraction: 0.02 means only 2 per cent of paths with nothing to
    # follow found material this slow in this band. Continuous, and no threshold.
    #
    # The thresholded verdict this replaced is not reported. A cut at the ambient 5th
    # percentile admits 5 per cent of ambient paths by construction, so it can only
    # detect hotspots if far more than 5 per cent clear it; in the deepest band the
    # ambient distribution has a long slow tail, because every descending path ends in
    # a low-velocity province, so only 4 of 49 hotspots beat it against 2.5 expected
    # (binomial P = 0.23) while the two distributions differ at P = 8e-06. The
    # threshold has almost no power exactly where the null is long-tailed.
    for nm in names:
        H['amb_' + nm] = [float(np.nanmean(A_[nm] < v)) if v == v else np.nan
                          for v in H[nm]]
    H['amb_wander'] = [float(np.nanmean(A_['wander'] > v)) for v in H['wander']]

    # A DESCRIPTIVE grouping, for ordering the section atlas and for discussion. It is
    # not a test and nothing rests on it: the cut at 0.10 is a presentational choice,
    # stated here so it cannot be mistaken for a result.
    CUT, WCUT = 0.10, float(np.nanpercentile(A_['wander'], COHERENT_PCT))
    def group(r):
        if r['wander'] >= WCUT:
            return '4 disjointed'
        if r['amb_' + names[-1]] < CUT:
            return '1 deep signature'
        if any(r['amb_' + nm] < CUT for nm in names[:-1]):
            return '2 upper or mid only'
        return '3 weak throughout'
    H['group'] = H.apply(group, axis=1)
    # The thresholded columns are dropped, not merely left unreported: a column named
    # 'verdict' sitting in the output file is an invitation to quote it, and it has no
    # power in the band that matters.
    H = H.drop(columns=[c for c in H.columns
                        if c.startswith('sup_') or c in ('coherent', 'deep',
                                                         'connected', 'verdict')])
    H = H.sort_values(['group', 'amb_' + names[-1]])
    cols = ['amb_' + nm for nm in names] + ['wander', 'group']
    print('\n' + '=' * 74)
    print('PER HOTSPOT: the fraction of ambient paths that found SLOWER material')
    print('low means the hotspot is slower than almost every path with nothing to follow')
    print(H[cols].to_string(float_format=lambda x: f'{x:.3f}'))
    print('\n' + H.group.value_counts().sort_index().to_string())
    print(f'\nbelow {CUT} in the deepest band: '
          f'{int((H["amb_" + names[-1]] < CUT).sum())} of {len(H)}')
    print(f'below {CUT} in all three bands:  '
          f'{int((H[["amb_" + n for n in names]] < CUT).all(axis=1).sum())} of {len(H)}')

    return H


def main():
    global _anom
    ap = argparse.ArgumentParser()
    ap.add_argument('--file')
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--paths', default=None)
    ap.add_argument('--ambient', default=None)
    ap.add_argument('--report-only', action='store_true', dest='report_only',
                    help='re-print from the measured tables without loading the model')
    a = ap.parse_args()

    pth = a.paths or os.path.join(a.dir, f'conduit_paths_all_{a.tag}.json')
    amb = a.ambient or os.path.join(a.dir, f'ambient_paths_{a.tag}.json')
    if not os.path.exists(amb):
        raise SystemExit(
            f'{amb} not found. The null is not optional: without it there is no scale '
            'on which either number means anything. See this script\'s docstring for '
            'the tilt_recovery.py command that writes it.')
    # Nor is its AGE optional. The ambient step of retrace.sh was not passing
    # --save-paths, so this file stood at its 13 September version through a change of
    # cost channel and of s while the hotspot paths beside it were retraced. Every
    # ambient number in the population table then matched the published value to three
    # decimals while every hotspot number moved, and the table read as a result.
    freshness.require_newer(
        amb, pth, 'the ambient path set',
        'The population test and the atlas grouping are both measured against it.')

    if a.report_only:
        H = pd.read_csv(os.path.join(a.dir, f'track_support_{a.tag}.csv'), index_col=0)
        A_ = pd.read_csv(os.path.join(a.dir, f'track_support_ambient_{a.tag}.csv'),
                         index_col=0)
        print(f'{len(H)} hotspot paths against {len(A_)} ambient paths\n')
        H = _report(H, A_, a)
        H.index.name = 'site'
        _ts = os.path.join(a.dir, f'track_support_{a.tag}.csv')
        H.to_csv(_ts)
        provenance.stamp(_ts, report_only=a.report_only,
                         inputs=[a.paths or os.path.join(
                             a.dir, f'conduit_paths_all_{a.tag}.json'),
                             a.ambient or os.path.join(
                                 a.dir, f'ambient_null_{a.tag}.csv')])
        print(f'\nrewrote track_support_{a.tag}.csv with the ambient scores and groups')
        return
    if not a.file:
        raise SystemExit('--file is required unless --report-only')
    depth, lat, lon, arr = load_anomaly(a.file, ModelSpec(a.tag, a.var),
                                        depth_max=a.depth_max, every=a.every)
    lon, arr = dedupe_lon(lon, arr)
    _anom = arr
    z = np.asarray(depth, float)
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    order = np.sort(arr.reshape(arr.shape[0], -1), axis=1)
    print(f'{a.tag}: {arr.shape[0]} shells', flush=True)

    H = measure(json.load(open(pth)), z, lat, lon, order)
    A_ = measure(json.load(open(amb)), z, lat, lon, order)
    print(f'{len(H)} hotspot paths against {len(A_)} ambient paths\n')

    H = _report(H, A_, a)

    out = os.path.join(a.dir, f'track_support_{a.tag}.csv')
    H.index.name = 'site'
    H.to_csv(out)
    # stamped like its ambient counterpart beside it: the audit reported this one as
    # carrying no provenance, so it was the one table in the pair nothing could check
    provenance.stamp(out, inputs=[pth, amb])
    A_.index.name = 'site'
    _ta = os.path.join(a.dir, f'track_support_ambient_{a.tag}.csv')
    A_.to_csv(_ta)
    provenance.stamp(_ta, inputs=[a.ambient or os.path.join(
        a.dir, f'ambient_null_{a.tag}.csv')])
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
