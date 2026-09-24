"""The per-hotspot table for the supplement: every measured quantity, one row each.

Written from the run outputs rather than typed, so the table cannot drift away from
the analysis behind it. Quantities that were withdrawn do not appear, and root depth
does not appear because every path terminates at the target by construction and it
carries no information.
"""
from __future__ import annotations
import argparse, os, re, sys
import numpy as np, pandas as pd
import model_names

R_E, DEG = 6371.0, np.pi / 180.0
ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--link', type=float, default=0.25)
ap.add_argument('--out', default='out/supplement_table_S1')
ap.add_argument('--check-si', dest='check_si', metavar='SI_MD',
                help='write nothing; compare Tables S1 and S3 in this supplement with what '
                     'the outputs on disk give, and exit non-zero if they differ')
A = ap.parse_args()

pr = pd.read_csv(os.path.join(A.dir, f'corridor_profiles_{A.tag}.csv'))
sm = pd.read_csv(os.path.join(A.dir, f'corridor_summary_{A.tag}.csv'))
rt = pd.read_csv(os.path.join(A.dir, f'plume_roots_{A.tag}.csv')).set_index('site')
sl = pd.read_csv(os.path.join(A.dir, f'plume_slant_{A.tag}.csv')).set_index('site')
ov = pd.read_csv(os.path.join(A.dir, f'corridor_overlap_{A.tag}.csv'))
ra = os.path.join(A.dir, f'corridor_ridge_association_{A.tag}.csv')
ra = pd.read_csv(ra).set_index('site') if os.path.exists(ra) else None

BANDS = ((200, 660, 'w_um_tz'), (660, 1500, 'w_ulm'), (1500, 2700, 'w_dlm'))
W = {lab: pr[(pr.depth >= lo) & (pr.depth < hi)].groupby('site').width_km.median()
     for lo, hi, lab in BANDS}

# network components at the linking threshold
link = ov[ov.frac_small >= A.link]
parent = {s: s for s in sm.site}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for _, r in link.iterrows():
    if r.a in parent and r.b in parent:
        ra_, rb_ = find(r.a), find(r.b)
        if ra_ != rb_:
            parent[ra_] = rb_
comp, deg = {}, {s: 0 for s in sm.site}
for _, r in link.iterrows():
    if r.a in deg:
        deg[r.a] += 1
    if r.b in deg:
        deg[r.b] += 1
for s in sm.site:
    comp.setdefault(find(s), []).append(s)
cid = {}
for i, (k, v) in enumerate(sorted(((k, v) for k, v in comp.items() if len(v) > 1),
                                  key=lambda t: -len(t[1])), start=1):
    for s in v:
        cid[s] = i

_cp = os.path.join(A.dir, f'competing_routes_{A.tag}.csv')
_rp = os.path.join(A.dir, f'root_robustness_sites_{A.tag}.csv')
cr = pd.read_csv(_cp).set_index('site') if os.path.exists(_cp) else None
rr = pd.read_csv(_rp).set_index('site') if os.path.exists(_rp) else None
if cr is None:
    print(f'  NOTE: {_cp} absent, so the table gives one root per hotspot without '
          f'saying whether the data fix it. Run competing_routes.py.')

rows = []
for _, r in sm.sort_values('site').iterrows():
    s = str(r.site)
    rec = {'hotspot': s, 'self_check': 'pass' if bool(r.ok) else 'FAIL'}
    for _, _, lab in BANDS:
        rec[lab] = round(float(W[lab].get(s, np.nan)), 0) if s in W[lab] else np.nan
    rec['width_all'] = round(float(r['width_med_0.02']), 0)
    if s in rt.index:
        rec['ends_in_province'] = 'yes' if bool(rt.loc[s, 'root_in']) else 'no'
        rec['margin_km'] = round(float(rt.loc[s, 'root_margin']), 0)
    if s in sl.index:
        rec['offset_km'] = round(float(sl.loc[s, 'offset']), 0)
        rec['tilt_deg'] = round(float(sl.loc[s, 'tilt']), 1)
        # Newer tracing outputs separate the directional tests from the scalar
        # slant table.  Keep this table generator compatible with both schemas:
        # include the optional fields when an older output supplies them, and
        # omit them otherwise instead of failing before the SI can be rebuilt.
        if 'lean_az' in sl.columns:
            rec['lean_azimuth'] = round(float(sl.loc[s, 'lean_az']), 0)
        if 'basin' in sl.columns:
            rec['basin'] = str(sl.loc[s, 'basin'])
    # The second descent, where the cost difference between the two is below what
    # the model resolves. A table that gave one root per hotspot without this would
    # state a position the data do not fix.
    if cr is not None and s in cr.index:
        rec['second_margin_pct'] = round(float(cr.loc[s, 'margin_pct']), 1)
        rec['second_sep_km'] = round(float(cr.loc[s, 'separation_km']), 0)
        sd = float(cr.loc[s, 'split_depth_km'])
        rec['split_depth_km'] = round(sd, 0) if sd == sd else np.nan
    if rr is not None and s in rr.index:
        rec['second_in_province'] = 'yes' if bool(rr.loc[s, 'alt_in']) else 'no'
    rec['network_degree'] = int(deg.get(s, 0))
    rec['network_component'] = cid.get(s, np.nan)
    if ra is not None and s in ra.index:
        rec['nearest_ridge_km'] = round(float(ra.loc[s, 'd_any_ridge']), 0)
    rows.append(rec)

d = pd.DataFrame(rows)

# Two tables, not one. The second-descent columns made Table S1 seventeen columns wide,
# which does not fit a portrait page at reading size: hotspot names and headers broke
# mid-word. They are a coherent set of their own and go to Table S3.
S3 = ['second_margin_pct', 'second_sep_km', 'split_depth_km', 'second_in_province']
s1 = d[[c for c in d.columns if c not in S3]]

# Headers short enough that no word has to break inside a column thirteen columns wide
# on a portrait page; the caption carries the units and says which widths these are.
HEAD = {'hotspot': 'Hotspot', 'self_check': 'Check',
        'w_um_tz': '200–660', 'w_ulm': '660–1500', 'w_dlm': '1500–2700',
        'width_all': 'All', 'ends_in_province': 'Province',
        'margin_km': 'Margin', 'offset_km': 'Offset',
        'tilt_deg': 'Tilt', 'lean_azimuth': 'Lean azimuth',
        'basin': 'Basin', 'network_degree': 'Links',
        'network_component': 'Network', 'nearest_ridge_km': 'Ridge'}
HEAD3 = {'hotspot': 'Hotspot', 'ends_in_province': 'First endpoint in province',
         'second_margin_pct': 'Extra cost (%)',
         'second_sep_km': 'Distance between endpoints (km)',
         'split_depth_km': 'Routes part at (km)',
         'second_in_province': 'Second endpoint in province'}
# Decimal places per column. Everything else is an integer; the extra cost of the
# second descent is a few per cent and rounding it to a whole number hides the only
# thing the column is for.
DEC = {'second_margin_pct': 1}
WIDTH_COLS = set(list(W.keys()) + ['width_all'])
_nw = int(d['width_all'].isna().sum()) if 'width_all' in d.columns else 0


def _number(v, nd):
    """A value as printed: a true minus sign, and no negative zero."""
    t = f'{v:.{nd}f}'
    if float(t) == 0:
        t = t.lstrip('-')
    return t.replace('-', '\u2212')


def _table_md(frame, head):
    cols = [c for c in head if c in frame.columns]
    lines = ['| ' + ' | '.join(head[c] for c in cols) + ' |']
    # The separator's dashes give the relative column widths to anything that converts
    # the markdown directly. build_docx.py measures the columns instead, which is what
    # the Word files use, so these only need to be reasonable, not exact.
    def _w(c):
        body = max((len(str(x)) for x in frame[c].tolist()), default=1)
        head_word = max(len(t) for t in str(head[c]).split())
        return min(16, max(5, body, head_word))
    lines.append('|' + '|'.join('-' * _w(c) for c in cols) + '|')
    for _, r in frame.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            # An empty cell reads as a measurement that went missing, and none of these
            # is. Where the least-cost route ponds there is no corridor below the shells
            # it descends into, so no width exists; a path whose corridor overlaps no
            # other belongs to no network. An em dash says so, and the caption says what
            # it means in each column.
            if pd.isna(v):
                cells.append('\u2014' if c in DASH_COLS else '')
            elif c == 'hotspot':
                cells.append(model_names.hotspot(v))
            elif isinstance(v, float):
                cells.append(_number(v, DEC.get(c, 0)))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return cols, '\n'.join(lines) + '\n'


DASH_COLS = WIDTH_COLS | {'network_component'}
_s3 = A.out.replace('S1', 'S3')
_has3 = all(c in d.columns for c in S3)
cols, _md1 = _table_md(s1, HEAD)
_c3, _md3 = _table_md(d, HEAD3) if _has3 else ([], '')

if A.check_si:
    # The supplement carries these tables pasted from the files this script writes, as
    # Table S2 carries si_tables.py's. A paste is a copy, and a copy goes stale without
    # a sound; this says whether it has.
    si = open(A.check_si).read().splitlines()
    bad = 0
    for label, md in (('S1', _md1), ('S3', _md3)):
        if not md:
            continue
        want = md.strip().splitlines()
        first = want[0]
        at = [i for i, l in enumerate(si) if l.rstrip() == first]
        if not at:
            print(f'Table {label}: its header row is not in {A.check_si}')
            bad += 1
            continue
        have = []
        for l in si[at[0]:]:
            if not l.startswith('|'):
                break
            have.append(l.rstrip())
        if have == want:
            print(f'Table {label} matches the outputs on disk ({len(want) - 2} rows)')
            continue
        bad += 1
        print(f'Table {label} DIFFERS from the outputs on disk')
        for i in range(max(len(have), len(want))):
            a_ = have[i] if i < len(have) else '(missing)'
            b_ = want[i] if i < len(want) else '(missing)'
            if a_ != b_:
                print(f'  supplement: {a_}\n  computed  : {b_}')
    sys.exit(1 if bad else 0)

s1.to_csv(f'{A.out}.csv', index=False)
with open(f'{A.out}.md', 'w') as fh:
    fh.write(_md1)
if _has3:
    d[['hotspot', 'ends_in_province'] + S3].to_csv(f'{_s3}.csv', index=False)
    with open(f'{_s3}.md', 'w') as fh:
        fh.write(_md3)
    print(f'wrote {_s3}.csv and {_s3}.md: {len(d)} hotspots, {len(_c3)} columns')
else:
    print(f'  no second-descent columns, so {_s3} is not written: run competing_routes.py')

print(f'wrote {A.out}.csv and {A.out}.md: {len(d)} hotspots, {len(cols)} columns')
print(f'  self-check passes: {int((d.self_check == "pass").sum())} of {len(d)}')
print(f'  ends inside a province: {int((d.ends_in_province == "yes").sum())}')
print(f'  in a network component: {int((d.network_degree > 0).sum())}')
if _nw:
    print(f'  {_nw} hotspots have an em dash for every width: their least-cost route '
          f'runs\n  laterally within a depth shell, so no corridor exists below the '
          f'shells it\n  descends into and the width is undefined rather than narrow. '
          f'The caption\n  must say so.')
print('  corridor width is an error bar on the path, not a conduit diameter, and')
print('  root depth is omitted: every path ends at the target by construction.')
