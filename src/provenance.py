#!/usr/bin/env python3
"""What a generated file was made from, recorded beside it.

WHY THIS EXISTS. Every expensive mistake in this project has the same shape: a file
was read as current when it was not, and the failure presented as agreement rather
than as an error. ambient_null_<tag>.csv does not say it was written with --amp 0, so
its recovered_km was read as spurious drift when no conduit had been injected at all.
classification_<tag>.csv stamps lateral and ncell but not s, so its staleness guard
could not see a change to s. track_support_<tag>.csv, bao_pdf_<tag>.csv and
path_reliability_<tag>.csv carried no configuration at all and sat two days stale while
the section atlas was grouped by one of them and drawn from newer paths. In each case
nothing raised an error; the numbers simply meant something other than what they were
taken to mean.

The fix is not more care. It is that a generated file should say what produced it, and
that a reader should be able to ask.

WHAT IS RECORDED. A sidecar <file>.prov.json holding the time, the script and its
arguments, the configuration in force, the move set, and every input the script read
WITH the size and modification time each input had at the time. That last part is what
makes staleness detectable rather than merely suspectable: an output whose recorded
input is now newer than the output itself is stale, and that is a fact, not a judgement.

A sidecar rather than a column or a JSON key, because the outputs here are CSV, JSON and
NPZ, and every reader of them would otherwise have to change. Nothing reads the sidecar
except the checker, so stamping cannot break an existing script. A file with no sidecar
is reported as unstamped rather than assumed fine - an audit that passes on the files it
does not know about is the failure mode this replaces.

    import provenance
    provenance.stamp(out_path, config=c, lateral=A.lateral, ncell=A.ncell,
                     inputs=[paths_file, ambient_file])

    python3 provenance.py --check              sweep out/ and report
    python3 provenance.py --check --strict     exit non-zero if anything is wrong
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys

SUFFIX = '.prov.json'
CONFIG_FIELDS = ('s', 'z_target', 'channel', 'h_max', 'radius')


def _asdict(c):
    """Accept a pandas Series, a dict, or None for the configuration."""
    if c is None:
        return None
    if hasattr(c, 'to_dict'):
        c = c.to_dict()
    out = {}
    for k in CONFIG_FIELDS:
        if k in c:
            v = c[k]
            out[k] = str(v) if k == 'channel' else (
                None if v is None else float(v))
    return out or None


def _fstat(p):
    try:
        st = os.stat(p)
        return dict(path=p, bytes=int(st.st_size), mtime=round(st.st_mtime, 3))
    except OSError:
        return dict(path=p, missing=True)


def _git():
    try:
        r = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                           cwd=os.path.dirname(os.path.abspath(__file__)),
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def stamp(path, config=None, inputs=(), **extra):
    """Write <path>.prov.json beside a file that has just been written.

    config may be the frozen configuration as a Series or dict; inputs is every file
    the script read to produce this one. Anything else worth recording - a move set, an
    amplitude, a site count, a channel override - goes in as a keyword and is stored
    verbatim. Failing to stamp never fails the run: a missing sidecar is reported by the
    checker, which is better than a script dying at the last line after an hour of work.
    """
    try:
        rec = dict(
            written=_dt.datetime.now().astimezone().isoformat(timespec='seconds'),
            script=os.path.basename(sys.argv[0]) or None,
            argv=list(sys.argv[1:]),
            git=_git(),
            output=_fstat(path),
            config=_asdict(config),
            inputs=[_fstat(p) for p in inputs if p],
        )
        for k, v in extra.items():
            if hasattr(v, 'item'):
                try:
                    v = v.item()
                except Exception:
                    v = str(v)
            rec[k] = v
        with open(path + SUFFIX, 'w') as fh:
            json.dump(rec, fh, indent=1, default=str)
    except Exception as e:                       # never fail the caller
        print(f'  (provenance: could not stamp {path}: {e})', file=sys.stderr)


def read(path):
    p = path + SUFFIX
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def verdict(path, frozen=None):
    """Say what is wrong with one file, as a list of strings. Empty means nothing is.

    Three questions, in the order that matters. Is there a stamp at all? Was it made
    under the configuration in force now? And is it older than something it was made
    from? The third needs no configuration and catches the case that has bitten hardest.
    """
    out = []
    if not os.path.exists(path):
        return ['missing']
    rec = read(path)
    if rec is None:
        return ['unstamped']
    try:
        mt = os.stat(path).st_mtime
    except OSError:
        return ['missing']
    rec_out = rec.get('output') or {}
    if rec_out.get('mtime') and abs(rec_out['mtime'] - mt) > 2:
        out.append('rewritten since it was stamped')
    f = _asdict(frozen)
    c = rec.get('config')
    if f and c:
        diff = [f'{k}: made at {c.get(k)}, frozen is {f.get(k)}'
                for k in CONFIG_FIELDS
                if k in f and k in c and str(c.get(k)) != str(f.get(k))]
        out += diff
    for i in rec.get('inputs') or []:
        ip = i.get('path')
        if not ip or not os.path.exists(ip):
            continue
        if os.stat(ip).st_mtime > mt + 2:
            out.append(f'older than its input {os.path.basename(ip)}')
    return out


def require(path, frozen=None, hard=True):
    """Read-side guard. Call before trusting a file that decides a number."""
    bad = verdict(path, frozen)
    if not bad:
        return True
    msg = f'{path}: ' + '; '.join(bad)
    if hard and any(b != 'unstamped' for b in bad):
        raise SystemExit('stale input: ' + msg)
    print(f'  WARNING {msg}', file=sys.stderr)
    return False


def _selftest():
    """Prove the guard fires. A check that has never been seen to fail is not a check.

    Two regression tests in this project passed with the fault reintroduced, so the
    three failure modes this module exists to catch are exercised here against real
    files in a scratch directory: an unstamped file, an output older than its input,
    and an output written under a configuration other than the frozen one.
    """
    import tempfile
    import time
    ok = True

    def expect(name, got, want):
        nonlocal ok
        good = (want in got) if isinstance(want, str) else (got == want)
        print(f'  {"PASS" if good else "FAIL"}  {name}: {got}')
        ok = ok and good

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'input.csv')
        out = os.path.join(d, 'derived.csv')
        open(src, 'w').write('a\n1\n')
        open(out, 'w').write('b\n2\n')

        expect('an unstamped file is reported', verdict(out), ['unstamped'])

        stamp(out, config={'s': 0.3, 'z_target': 2700.0, 'channel': 'anom',
                           'h_max': 900.0, 'radius': 3.0}, inputs=[src])
        expect('a fresh stamp is clean', verdict(out), [])

        time.sleep(1.1)
        open(src, 'w').write('a\n1\n2\n')          # the input changes afterwards
        os.utime(src, (time.time() + 5, time.time() + 5))
        expect('an output older than its input is caught',
               '; '.join(verdict(out)), 'older than its input')

        os.utime(src, None)
        expect('a configuration that disagrees is caught',
               '; '.join(verdict(out, {'s': 0.4, 'z_target': 2700.0,
                                       'channel': 'anom', 'h_max': 900.0,
                                       'radius': 3.0})),
               'made at 0.3, frozen is 0.4')

        os.remove(out + SUFFIX)
        expect('removing the sidecar returns it to unstamped', verdict(out),
               ['unstamped'])
    return ok


if __name__ == '__main__':
    import argparse
    import glob as _glob

    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true',
                    help='prove the three failure modes are detected')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--strict', action='store_true',
                    help='exit non-zero if anything is stale or unstamped')
    ap.add_argument('--pattern', default='*',
                    help='restrict the sweep, e.g. "*RevealLO*"')
    A = ap.parse_args()

    if A.selftest:
        print('provenance self-test')
        sys.exit(0 if _selftest() else 1)

    # A pattern can span models, and comparing one model's output against another
    # model's frozen configuration reports a disagreement that is not there: the 30 km
    # control's paths were flagged for a target depth and reach that are correct for the
    # control and simply differ from the paper model's. The tag is taken from the file
    # name, longest match first so RevealLO_30km is not read as RevealLO.
    KNOWN = ('RevealLO_30km', 'SEMUCB-WM1', 'RevealLO', 'GLADM35', 'SPiRaL', 'REVEAL')
    _frozen_cache = {}

    def frozen_for(path):
        name = os.path.basename(path)
        tag = next((k for k in KNOWN if k in name), A.tag)
        if tag not in _frozen_cache:
            try:
                import path_config
                _frozen_cache[tag] = path_config.load(tag, A.dir)
            except Exception:
                _frozen_cache[tag] = None
        return _frozen_cache[tag], tag

    files = [f for f in sorted(_glob.glob(os.path.join(A.dir, A.pattern)))
             if os.path.isfile(f) and not f.endswith(SUFFIX)]
    stale, unstamped, ok = [], [], []
    for f in files:
        _fz, _tg = frozen_for(f)
        v = verdict(f, _fz)
        if not v:
            ok.append(f)
        elif v == ['unstamped']:
            unstamped.append(f)
        else:
            stale.append((f, v))

    print(f'{len(files)} files in {A.dir}/ matching {A.pattern!r}\n')
    if stale:
        print(f'STALE OR INCONSISTENT ({len(stale)}):')
        for f, v in stale:
            print(f'  {os.path.basename(f)}')
            for m in v:
                print(f'      {m}')
        print()
    print(f'stamped and consistent: {len(ok)}')
    if unstamped:
        print(f'unstamped: {len(unstamped)} - not checked, and not therefore correct')
        for f in unstamped[:40]:
            print(f'  {os.path.basename(f)}')
        if len(unstamped) > 40:
            print(f'  ... and {len(unstamped) - 40} more')
    if A.strict and (stale or unstamped):
        sys.exit(1)
