#!/usr/bin/env python3
"""Which scripts write into out/ and are in no runner?

An inventory, not a guard. Staleness is caught at runtime by freshness.py, which
records what a script actually opens and dates it, and by require_newer at the point
of use; a static scan cannot attribute an output to one author when six scripts share
an f-string prefix, and a report that names six is worse than no report. What this
gives is the list of scripts nothing calls, with whether their products in out/
predate the current paths - the inventory to read before believing any number that
came out of one of them.

This is the defect that has cost this study the most. A script that no runner calls
still leaves its output in out/, and the output looks exactly like a current one: it
has a name, a date nobody reads, and numbers that go on being used. Three separate
occasions this month, each found by accident rather than by a check -

  ambient_null.py        ran a step too early and compared a current hotspot
                         population against a null from a superseded configuration,
                         so both headline results read as lost for five days
  corridor_width_calibrate.py   three of four occupancy sweeps stood at the move set
                         and channel of three different earlier days
  frontier.py            its single run closed the continuity question on the anom
                         channel, and the conclusion outlived that channel by five
                         days and a whole retrace

None of the three was wrong when it ran. Each became wrong when something upstream
moved and nothing re-ran it. So the test is not whether a script is correct; it is
whether anything would notice if the ground moved under its output.

WHAT IS REPORTED. A script is flagged when it is named in none of the runners and an
output in out/ that carries its stem is older than the traced paths. Being in no
runner is not by itself a fault - plenty of these are one-off explorations and are
meant to be - but an orphan with a STALE OUTPUT IN out/ is a fact waiting to be read
by something that does not know how old it is.

  python3 orphan_scripts.py
"""
from __future__ import annotations
import argparse, os, re, time

RUNNERS = ('retrace.sh', 'run_all.sh', 'refigure.sh', 'run_calibration_all.sh',
           'run_width_calibration.sh')
# Outputs whose age carries no meaning, for a stated reason.
EXEMPT = (
    ('path_config_', 'the frozen configuration, which the tracing reads'),
    ('hotspots_', 'an input list, not a product'),
    ('hinge_migration', 'a plate-model product, independent of the tomography'),
)


_PRE_CACHE = {}


def _prefixes(path):
    """The output names a script writes, as literal prefixes.

    Attributing an output to a script by its name is what a first attempt did, and
    it matched every axis_* file to every axis_*.py. So the source is read instead
    and every string literal that names a product is taken, an f-string contributing
    the fixed text before its first placeholder. A prefix shorter than six characters
    is dropped: it would match half the directory and turn the report into noise.
    """
    if path in _PRE_CACHE:
        return _PRE_CACHE[path]
    try:
        src = open(path, errors='ignore').read()
    except OSError:
        return ()
    out = set()
    for lit in re.findall(r"""['"]([^'"\n]*\.(?:csv|json|npz|nc|pdf|png))['"]""", src):
        head = lit.split('{')[0].split('/')[-1]
        if len(head) >= 6:
            out.add(head)
    _PRE_CACHE[path] = tuple(sorted(out))
    return _PRE_CACHE[path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='out')
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--strict', action='store_true',
                    help='exit non-zero when anything is flagged')
    A = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    ref = os.path.join(here, A.dir, f'conduit_paths_all_{A.tag}.json')
    if not os.path.exists(ref):
        raise SystemExit(f'no reference to date against: {ref}')
    tref = os.path.getmtime(ref)

    called = ''
    for r in RUNNERS:
        p = os.path.join(here, r)
        if os.path.exists(p):
            called += open(p, errors='ignore').read()
    # a script is also covered if another script the runners call imports it
    for f in sorted(os.listdir(here)):
        if f.endswith('.py') and re.search(rf'\b{re.escape(f)}\b', called):
            called += open(os.path.join(here, f), errors='ignore').read()

    outs = {}
    d = os.path.join(here, A.dir)
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isfile(p) and not f.endswith(('.log', '.prov.json')):
            outs[f] = os.path.getmtime(p)

    flagged, quiet = [], []
    for f in sorted(os.listdir(here)):
        if not f.endswith('.py') or f == os.path.basename(__file__):
            continue
        stem = f[:-3]
        if re.search(rf'\b{re.escape(f)}\b', called) or re.search(
                rf'\bimport\s+{re.escape(stem)}\b', called):
            continue
        mine = [(n, t) for n, t in outs.items()
                if any(n.startswith(pre) for pre in _prefixes(os.path.join(here, f)))
                and not any(n.startswith(e) for e, _ in EXEMPT)]
        stale = [(n, t) for n, t in mine if t < tref]
        if stale:
            flagged.append((f, sorted(stale, key=lambda x: x[1])))
        else:
            quiet.append(f)

    fmt = '%Y-%m-%d %H:%M'
    print(f'reference: conduit_paths_all_{A.tag}.json  '
          f'{time.strftime(fmt, time.localtime(tref))}\n')
    print(f'{len(flagged)} script(s) in no runner with output in {A.dir}/ older than '
          f'the paths:')
    for f, st in flagged:
        oldest = time.strftime(fmt, time.localtime(st[0][1]))
        print(f'  {f:34s} {len(st):3d} file(s), oldest {oldest}  e.g. {st[0][0]}')
    print(f'\n{len(quiet)} script(s) in no runner whose output is current or absent.')
    print('\nStaleness itself is caught at runtime: freshness.py records what each '
          'script\nopens and dates it against the paths, and require_newer refuses at '
          'the point of use.')
    if flagged and A.strict:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
