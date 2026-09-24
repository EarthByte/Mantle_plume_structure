"""Which of a script's inputs were produced before the paths they are read beside.

Provenance answers this for a stamped file. Most files here are not stamped, and the
unstamped ones are exactly where staleness has hidden. plume_slant_apm.csv stood eight
days old in no runner while its consumers read it. The corridor width calibration was
five days older than the configuration whose corridors it was compared against, so
section 3.7 set injections made under one cost channel beside corridors traced under
another. ambient_paths_<tag>.json, which sets the ambient level the whole atlas grouping
is judged against, was from before two changes of channel. None of the three carried a
stamp, so the provenance sweep reported them as "not checked" and the run carried on.

This needs no stamp. It records every file a script opens under the output directory and
compares each modification time against the traced paths - the output of the step that
every path-derived number descends from. A file older than the paths is not necessarily
wrong: the frozen configuration and the detection screen are older BY CONSTRUCTION,
because they precede the tracing that reads them. Those are named and excluded here with
that reason, rather than passed silently, so the exclusion is a statement someone can
disagree with rather than a gap.

    import freshness
    freshness.record()          # before the first read
    ...
    n = freshness.report('out', 'RevealLO')   # after the last

report() returns the number of suspicious inputs, for a caller that fails on it.
"""
from __future__ import annotations
import builtins, os, time

# Older than the paths by construction: the tracing reads these, so they cannot postdate
# it. Matched as a prefix against the file's base name.
BY_CONSTRUCTION = (
    ('path_config_', 'the frozen configuration, which the tracing reads'),
    ('detection_', 'the detection screen, which the configuration is chosen from'),
    ('classification_', 'the screen output the ensemble is taken over'),
    ('hotspots_', 'an input list, not a product'),
    ('hinge_migration', 'a plate-model product, independent of the tomography'),
)

_READ: set[str] = set()
_ORIG = None


def record():
    """Start recording. Patches open, pandas.read_csv and numpy.load."""
    global _ORIG
    if _ORIG is not None:
        return
    _ORIG = builtins.open

    def _note(f):
        try:
            _READ.add(os.path.abspath(str(f)))
        except Exception:
            pass

    def _open(file, mode='r', *a, **k):
        if 'w' not in mode and 'a' not in mode and 'x' not in mode:
            _note(file)
        return _ORIG(file, mode, *a, **k)

    builtins.open = _open
    try:
        import pandas as pd
        import numpy as np
        _rc, _nl = pd.read_csv, np.load

        def rc(f, *a, **k):
            _note(f)
            return _rc(f, *a, **k)

        def nl(f, *a, **k):
            _note(f)
            return _nl(f, *a, **k)

        pd.read_csv, np.load = rc, nl
    except Exception:
        pass


def _excuse(name):
    for pre, why in BY_CONSTRUCTION:
        if name.startswith(pre):
            return why
    return None


def report(d, tag, ref=None, out=print):
    """Name every recorded input older than the traced paths. Returns how many."""
    d = os.path.abspath(d)
    ref = ref or os.path.join(d, f'conduit_paths_all_{tag}.json')
    if not os.path.exists(ref):
        out(f'\nno {os.path.basename(ref)} to date inputs against, so freshness was '
            f'not checked')
        return 0
    t_ref = os.path.getmtime(ref)
    older, excused = [], []
    for p in sorted(_READ):
        if os.path.dirname(p) != d or p.endswith('.prov.json') or p == ref:
            continue
        if not os.path.exists(p) or os.path.getmtime(p) >= t_ref:
            continue
        name = os.path.basename(p)
        why = _excuse(name)
        (excused if why else older).append((name, os.path.getmtime(p), why))
    if excused:
        out(f'\n{len(excused)} input(s) older than the paths by construction, '
            f'which is correct:')
        for n, m, why in excused:
            out(f'  {time.strftime("%Y-%m-%d %H:%M", time.localtime(m))}  {n}  - {why}')
    if not older:
        out('\nevery other input was written after the paths it is read beside')
        return 0
    out(f'\n{len(older)} INPUT(S) OLDER THAN THE TRACED PATHS, so the numbers taken '
        f'from them describe an earlier configuration:')
    for n, m, _ in older:
        out(f'  {time.strftime("%Y-%m-%d %H:%M", time.localtime(m))}  {n}')
    out(f'  (the paths were written '
        f'{time.strftime("%Y-%m-%d %H:%M", time.localtime(t_ref))})')
    out('Regenerate these before reading any check that rests on them: a check can '
        'agree\nbecause both sides are stale, which is not agreement.')
    return len(older)


def require_newer(path, ref, what, why=''):
    """Stop unless `path` is at least as new as `ref`. For a null beside its sample.

    A comparison between a current population and a null traced under a superseded
    configuration is not a weak result, it is not a result: the two sides come from
    different searches. Both of this paper's headline comparisons were made that way for
    five days without anything saying so, because the null files carried no stamp and sat
    in no runner. This is the cheap, local form of that check - one file against one
    reference, at the point of use, where the script that is about to read it can refuse.
    """
    import os
    if not os.path.exists(path):
        raise SystemExit(f'{what} is missing: {path}')
    if not os.path.exists(ref):
        return
    if os.path.getmtime(path) >= os.path.getmtime(ref):
        return
    import time
    fmt = '%Y-%m-%d %H:%M'
    raise SystemExit(
        f'\n{what} is older than the paths it would be compared against:\n'
        f'  {time.strftime(fmt, time.localtime(os.path.getmtime(path)))}  '
        f'{os.path.basename(path)}\n'
        f'  {time.strftime(fmt, time.localtime(os.path.getmtime(ref)))}  '
        f'{os.path.basename(ref)}\n'
        + (f'{why}\n' if why else '')
        + 'Regenerate it under the current configuration. A null from a different search\n'
          'is not a weaker comparison, it is not a comparison.')
