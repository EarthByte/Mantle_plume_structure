#!/usr/bin/env python3
"""The one configuration every path and corridor is traced under, held explicitly.

It used to be derived on the fly, in two places, as the median row of the set of
configurations that clear the 75 per cent injection detection floor:

    keep = det[(det.n_cal >= 4) & (det.detect_cal >= 0.75)].sort_values(...)
    c = keep.iloc[len(keep) // 2]

That is a selector with no stability. The floor is applied to a point estimate
from eighteen injections, where one trial is 0.056 and the 95 per cent interval
around the floor itself runs from 0.55 to 0.91, so which configurations clear it
is decided well inside the noise. Taking the median of the survivors then makes
the chosen configuration depend on the whole composition of that noisy set.

Freeing the move set to two lateral cells demonstrated the consequence. In the
paper model the retained set fell from 102 configurations to 49, but 34 of the 53
that left moved by a single trial and 46 by two or fewer; and the configuration
the manuscript reports - s = 0.4, target 2700 km, anomaly channel, 3-degree seed,
900 km reach - went from 14/18 to 13/18 and so from retained to dropped. Had the
retrace re-derived the median it would have traced every path at s = 0.3, target
2800 km, 2-degree seed and 300 km reach: four of five dimensions changed, on the
strength of one injection in eighteen, silently.

So the configuration is frozen to a file and read from it. It was fixed before the
geometry and geochemistry were analysed, which is what makes those results a test
rather than a search, and it does not get re-derived because a threshold moved
underneath it. Whether it still clears the floor is reported at every run, because
that is a fact about the configuration worth knowing, not a reason to swap it.

    freeze(tag)            write the frozen file from a detection table
    load(tag)              read it, and say whether it still clears the floor
"""
from __future__ import annotations

import json
import os

import pandas as pd

FLOOR, MIN_CASES = 0.75, 4
SORT = ['s', 'z_target', 'channel', 'radius']
FIELDS = ['s', 'z_target', 'channel', 'h_max', 'radius']


def _path(tag, d='out'):
    return os.path.join(d, f'path_config_{tag}.json')


def retained(det, channel=None):
    """Configurations clearing the injection floor, optionally on one cost channel.

    THE ENSEMBLE IS TAKEN OVER ONE CHANNEL, decided 2026-09-16. The classification's
    root_fraction is a mean over the retained set, so what that set contains decides what
    the number means. Screening a third channel widened it from 49 configurations to 145
    for the paper model, 96 of them contrast, because the floor admits every contrast
    configuration while admitting a minority of anom and min. That weighting is an
    accident of how permissive the floor happens to be per channel, not a statement about
    the mantle, and it blends sub-ensembles that disagree outright - Tahiti/Society roots
    in 96 per cent of anom and min configurations and 0 per cent of contrast ones, and the
    blended 0.32 describes neither. Restricted to the chosen channel, root_fraction means
    what it is described as meaning: how robust the rooting call is to the target depth,
    the reach, the seed radius and the cost exponent, at the channel the paper uses.
    """
    k = det[(det.n_cal >= MIN_CASES) & (det.detect_cal >= FLOOR)]
    if channel is not None:
        k = k[k.channel.astype(str) == str(channel)]
    return k.sort_values(SORT).reset_index(drop=True)


def ensemble_channel(tag, d='out'):
    """The channel the ensemble is taken over: the frozen one, or None if not frozen."""
    p = _path(tag, d)
    if not os.path.exists(p):
        return None
    try:
        return str(json.load(open(p))['config']['channel'])
    except Exception:
        return None


def median_config(det):
    k = retained(det)
    if not len(k):
        return None
    r = k.iloc[len(k) // 2]
    return {f: (str(r[f]) if f == 'channel' else float(r[f])) for f in FIELDS}


def freeze(tag, d='out', detection=None, note=''):
    """Write the frozen configuration from a detection table. Never overwrites."""
    p = _path(tag, d)
    if os.path.exists(p):
        return json.load(open(p))
    det = pd.read_csv(detection or os.path.join(d, f'detection_{tag}.csv'))
    c = median_config(det)
    if c is None:
        raise SystemExit(f'{tag}: no configuration clears the detection floor')
    rec = dict(tag=tag, config=c, source='median of the retained set', note=note)
    json.dump(rec, open(p, 'w'), indent=1)
    print(f'froze {p}: ' + ' '.join(f'{k}={v}' for k, v in c.items()))
    return rec


def load(tag, d='out', detection=None, refreeze=False):
    """The configuration to use, as a Series, with its standing reported.

    Reading the frozen file is the normal path. `refreeze` deliberately re-derives
    it from the current detection table and is the only way the configuration ever
    changes; it says so loudly, because every number in the paper moves with it.
    """
    det = pd.read_csv(detection or os.path.join(d, f'detection_{tag}.csv'))
    p = _path(tag, d)

    if refreeze or not os.path.exists(p):
        c = median_config(det)
        if c is None:
            raise SystemExit(f'{tag}: no configuration clears the detection floor')
        if refreeze and os.path.exists(p):
            old = json.load(open(p))['config']
            if old != c:
                print('!! REFREEZING THE CONFIGURATION. Every path, corridor, offset,')
                print('   tilt and corridor width in the paper is computed under it.')
                print(f'   was: ' + ' '.join(f'{k}={v}' for k, v in old.items()))
                print(f'   now: ' + ' '.join(f'{k}={v}' for k, v in c.items()))
        json.dump(dict(tag=tag, config=c, source='median of the retained set'),
                  open(p, 'w'), indent=1)
    else:
        c = json.load(open(p))['config']

    m = det
    for f in FIELDS:
        m = m[m[f] == c[f]]
    if len(m):
        r = m.iloc[0]
        k, n = int(round(r.detect_cal * r.n_cal)), int(r.n_cal)
        ok = (r.n_cal >= MIN_CASES) and (r.detect_cal >= FLOOR)
        print(f'configuration (frozen): ' + ' '.join(f'{a}={b}' for a, b in c.items()))
        print(f'  injection recovery {k}/{n} = {r.detect_cal:.3f} against a '
              f'{FLOOR:.0%} floor: {"clears it" if ok else "BELOW IT"}')
        if not ok:
            need = int(-(-FLOOR * n // 1)) - k
            print(f'  short by {need} trial{"s" if need != 1 else ""} of {n}, which is '
                  f'{need / n:.3f} - report it, do not swap the configuration for it')
    else:
        print(f'configuration (frozen): ' + ' '.join(f'{a}={b}' for a, b in c.items()))
        print('  not present in this detection table at all')

    row = pd.Series(c)
    row['h_max'] = float(c['h_max'])
    return row


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tags', default='RevealLO,RevealLO_30km,REVEAL,GLADM35,'
                                      'SPiRaL,SEMUCB-WM1')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--from-dir', default=None, dest='from_dir',
                    help='detection tables to freeze FROM, e.g. out/lateral_1.0 to '
                         'freeze the configuration the manuscript reports')
    a = ap.parse_args()
    for t in a.tags.split(','):
        src = (os.path.join(a.from_dir, f'detection_{t}.csv') if a.from_dir else None)
        freeze(t, a.dir, detection=src,
               note=('frozen from ' + (a.from_dir or a.dir)))
