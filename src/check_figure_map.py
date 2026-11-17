#!/usr/bin/env python3
"""Every figure caption in the manuscripts resolves to a file that exists.

The supplement once went to a co-author with seven of its eleven figures missing. The
build reported it, in a line among a hundred others, and nobody read it. This fails
loudly instead, at the end of the figure build, before anything is sent anywhere.

  python3 check_figure_map.py
"""
from __future__ import annotations
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, '..', 'figures')
# The version matters. This pointed at v5 and SI v2 for three and two revisions after
# they were superseded, and both files still exist, so the check ran and passed against
# manuscripts nobody is submitting - a caption check that could not see the captions.
DOCS = ('../manuscript/G3_MS_v17.md', '../manuscript/G3_SI_v17.md')
FIG = re.compile(r'^\*\*(Fig(?:ure)?\.?\s*(S?\d+[a-z]?)\.)', re.I)

m = {}
mp = os.path.join(FIGDIR, 'figure_map.txt')
if os.path.exists(mp):
    for line in open(mp):
        line = line.split('#')[0].strip()
        if line:
            k, v = line.split(None, 1)
            m[k.lower()] = v.strip()


def resolve(num):
    n = num.lower()
    if n in m:
        p = os.path.join(FIGDIR, m[n])
        return m[n] if os.path.exists(p) else None
    for f in sorted(os.listdir(FIGDIR)):
        if f.lower().startswith((f'fig{n}_', f'fig{n}.')) and f.endswith('.png'):
            return f
    return None


bad = []
for doc in DOCS:
    p = os.path.join(HERE, doc)
    if not os.path.exists(p):
        continue
    want = [FIG.match(l).group(2) for l in open(p) if FIG.match(l)]
    got = {w: resolve(w) for w in want}
    miss = [w for w, g in got.items() if g is None]
    print(f'{os.path.basename(doc):18s} {len(want):2d} captions, '
          f'{len(want) - len(miss):2d} resolve')
    for w in want:
        print(f'    {w:>4s}  {got[w] or "MISSING"}')
    bad += [(os.path.basename(doc), w) for w in miss]

if bad:
    print()
    raise SystemExit('figures named in a caption with no file to place:\n'
                     + '\n'.join(f'  {d}  Figure {w}' for d, w in bad))
print('\nevery caption in both documents resolves to a file that exists')
