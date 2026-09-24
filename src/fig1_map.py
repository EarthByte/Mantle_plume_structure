"""Figure 1: where the conduits go, over the tomography they were traced through.

Replaces the earlier map, whose symbols counted how many of five models called a
hotspot deep-rooted. That is the cross-model consensus this study does not use, and
it cannot be repaired by relabelling: the quantity itself is the wrong one.

What is drawn instead are three results of a single model. The background is the
depth slice at which the traced conduits stand out most clearly against their own
shell, chosen by conduit_depth_slice.py rather than by eye, at high transparency
because it is context. Filled symbols mark hotspots whose traced conduit terminates
inside a large low-shear-velocity province and open symbols those that do not. Thin
lines join hotspots whose corridors overlap, which is the network measurement.

No title. Every point size is given at the size the map is printed rather than the
size it is drawn, so nothing falls below the readable floor on the page.
"""
from __future__ import annotations
import argparse, math, os, sys
import sys
import numpy as np, pandas as pd, pygmt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import PLACED_CM, FLOOR_PT

# Drawn at the width it is placed at, so the scale factor between drawn and printed
# point size is exactly one and every size below is the size that reaches the page.
# Drawing at 24 cm and placing at 16 cm silently multiplied every font by 0.65, which
# is how a map whose own check passed arrived at 4.2 pt annotations.
MAP_CM = PLACED_CM
S = MAP_CM / 24.0          # every physical dimension below was tuned at 24 cm

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--slice', default=None,
                help='depth slice for the background; by default whichever one '
                     'conduit_depth_slice.py wrote')
ap.add_argument('--transparency', type=float, default=62.0)
ap.add_argument('--lim', type=float, default=1.8,
                help='per cent, the same saturation as the cross-sections')
ap.add_argument('--link', type=float, default=0.25)
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--out', default='../figures/fig1_map')
ap.add_argument('--label', default=(
    'Hawaii,Louisville,Afar,Comores,Bouvet,Meteor,Iceland,Jan Mayen,Marion,'
    'Australia E,Tahiti/Society,Pitcairn'),
    help='hotspots to name on the map, comma separated. The default is the twelve '
         'drawn as cross-sections, which are the ones a reader needs to locate. '
         'The printed map is 16 by 8 cm and a name set at the 8 pt floor is about '
         '1.7 cm long, so eighteen names do not fit however they are arranged; '
         'that is geometry, not a placement failure.')
A = ap.parse_args()


def pt(p):
    """A point size given at the printed size, not the drawn size."""
    q = p * MAP_CM / PLACED_CM
    if q < FLOOR_PT:
        raise SystemExit(f'{p} pt drawn is {q:.1f} pt printed, under the {FLOOR_PT} '
                         'pt floor; enlarge it or move the text to the caption')
    return f'{math.ceil(q * 10) / 10:.1f}p'


hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180']).set_index('hotspot')
roots = pd.read_csv(os.path.join(A.dir, f'plume_roots_{A.tag}.csv')).set_index('site')
ov = pd.read_csv(os.path.join(A.dir, f'corridor_overlap_{A.tag}.csv'))
sm = pd.read_csv(os.path.join(A.dir, f'corridor_summary_{A.tag}.csv'))
ok = set(sm[sm.ok == True].site)

sl = A.slice
if sl is None:
    cand = [f for f in os.listdir(A.dir) if f.startswith(f'slice_{A.tag}_')
            and f.endswith('.nc')]
    sl = os.path.join(A.dir, sorted(cand)[0]) if cand else None

fig = pygmt.Figure()
pygmt.config(FONT_ANNOT_PRIMARY=pt(9.0), FONT_LABEL=pt(9.0),
             MAP_FRAME_PEN='0.7p')
proj = f'W0/{MAP_CM}c'
fig.basemap(region='d', projection=proj, frame=['af', 'WSne'])

if sl and os.path.exists(sl):
    pygmt.makecpt(cmap='roma', series=[-A.lim, A.lim, 0.02], continuous=True)
    fig.grdimage(grid=sl, projection=proj, cmap=True,
                 transparency=A.transparency, nan_transparent=True)
    depth = ''.join(ch for ch in os.path.basename(sl) if ch.isdigit())
    # No quotes around the label: pygmt passes them through and GMT draws them.
    # GMT scales colourbar text toward the bar height rather than honouring the
    # font size given. Measured on the rendered PDF the factor is 0.67, so the
    # size below is the one that lands above the floor, not the one it looks like.
    with pygmt.config(FONT_ANNOT_PRIMARY=pt(14.0), FONT_LABEL=pt(14.0)):
        fig.colorbar(position=f'JBC+w{MAP_CM * 0.42}c/{0.42 * S}c+h+o0/{0.9 * S}c',
                     frame=[f'xaf+lshear-velocity anomaly at {depth} km, per cent'])
else:
    print('no depth slice found; run conduit_depth_slice.py for the background')
fig.coast(projection=proj, shorelines='0.25p,gray45', area_thresh=10000)

# the network, drawn under the symbols so a line never hides a hotspot
link = ov[ov.frac_small >= A.link]
nlink = 0
for _, r in link.iterrows():
    if r.a not in hs.index or r.b not in hs.index:
        continue
    la1, lo1 = float(hs.loc[r.a, 'lat']), float(hs.loc[r.a, 'lon_180'])
    la2, lo2 = float(hs.loc[r.b, 'lat']), float(hs.loc[r.b, 'lon_180'])
    if abs(lo2 - lo1) > 180:          # never draw a link across the date line seam
        continue
    fig.plot(x=[lo1, lo2], y=[la1, la2], pen='0.9p,gray25', projection=proj)
    nlink += 1

inside, outside = [], []
for n in sorted(ok):
    if n not in hs.index or n not in roots.index:
        continue
    rec = (float(hs.loc[n, 'lon_180']), float(hs.loc[n, 'lat']))
    (inside if bool(roots.loc[n, 'root_in']) else outside).append(rec)
if outside:
    fig.plot(x=[p[0] for p in outside], y=[p[1] for p in outside],
             style=f'c{0.34 * S}c', fill='white', pen='1.1p,black', projection=proj)
if inside:
    fig.plot(x=[p[0] for p in inside], y=[p[1] for p in inside],
             style=f'c{0.34 * S}c', fill='black', pen='1.1p,white', projection=proj)

# Hotspot names. Mollweide is analytic, so label boxes can be laid out in
# centimetres on the drawn figure and checked against each other before anything
# is sent to GMT. Eight candidate directions per label, first clear one wins;
# anything that cannot be placed is named on stderr rather than drawn on top of
# its neighbour.
# Mollweide spans 2R*sqrt(2) in x for a half-width of MAP_CM/2, so R is set from
# the half-width, not the width. Getting this wrong by a factor of two puts every
# label off the sheet, where the collision test drops it without complaint.
R_MOLL = (MAP_CM / 2.0) / (2.0 * math.sqrt(2.0))


def _moll(lon, lat):
    ph = math.radians(lat)
    if abs(abs(ph) - math.pi / 2) < 1e-9:
        th = math.copysign(math.pi / 2, ph)
    else:
        th = ph
        for _ in range(60):
            f = 2 * th + math.sin(2 * th) - math.pi * math.sin(ph)
            d = 2 + 2 * math.cos(2 * th)
            if abs(d) < 1e-12:
                break
            st = f / d
            th -= st
            if abs(st) < 1e-12:
                break
    _l = ((lon + 180) % 360) - 180
    if _l == -180 and lon > 0:      # +180 and -180 are the same seam; keep the side given
        _l = 180.0
    la = math.radians(_l)
    return ((2 * math.sqrt(2) / math.pi) * R_MOLL * la * math.cos(th),
            math.sqrt(2) * R_MOLL * math.sin(th))


LAB_PT = 8.5                                        # margin over the 8.0 pt floor
_ch = LAB_PT / 72.0 * 2.54                          # cm per point; drawn size is printed size
_LH = _ch * 1.25
_SYM = (0.34 * S) / 2 + 0.05 * S                    # symbol radius plus its pen


def _box(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _hits(b, boxes):
    return any(not (b[2] <= o[0] or b[0] >= o[2] or b[3] <= o[1] or b[1] >= o[3])
               for o in boxes)


_occupied = [_box(*_moll(lo, la), 2 * _SYM, 2 * _SYM)
             for n in sorted(ok) if n in hs.index and n in roots.index
             for lo, la in [(float(hs.loc[n, 'lon_180']), float(hs.loc[n, 'lat']))]]
# the legend box at the bottom left, and the graticule annotations along the left
# and bottom edges of the frame: a label overlapping either is a collision the eye
# sees even though neither is a hotspot symbol
_LEG_H = 3 * _LH + 0.25 * S
_LEG_R, _LEG_T = MAP_CM / 2 - 0.15 * S, MAP_CM / 4 - 0.15 * S
_occupied.append((_LEG_R - MAP_CM * 0.34, _LEG_T - _LEG_H, _LEG_R, _LEG_T))
_EDGE = 0.9 * _ch                                   # width of the annotation gutter
_occupied.append((-MAP_CM / 2, -MAP_CM / 4, -MAP_CM / 2 + _EDGE, MAP_CM / 4))
_occupied.append((-MAP_CM / 2, -MAP_CM / 4, MAP_CM / 2, -MAP_CM / 4 + _EDGE))

_want = [x.strip() for x in A.label.split(',') if x.strip()]
_miss = [n for n in _want if n not in hs.index]
# Two rings of candidate positions. The inner ring keeps a name against its own
# symbol, which is what a reader wants; the outer is tried only when the inner is
# blocked, and trades a little ambiguity for a name that appears at all.
_cands = []
for _k in (1.0, 1.7):
    _o1, _o2, _o3 = 0.30 * S * _k, 0.42 * S * _k, 0.26 * S * _k
    _cands += [('ML', _o1, 0.0), ('MR', -_o1, 0.0), ('MC', 0.0, _o2), ('MC', 0.0, -_o2),
               ('ML', _o3, _o1), ('MR', -_o3, _o1), ('ML', _o3, -_o1), ('MR', -_o3, -_o1)]
_lx, _ly, _lt, _lj, _unplaced = [], [], [], [], []
for n in _want:
    if n not in hs.index:
        continue
    lo, la = float(hs.loc[n, 'lon_180']), float(hs.loc[n, 'lat'])
    cx, cy = _moll(lo, la)
    w, h = _ch * 0.60 * len(n) + 0.10 * S, _LH
    for just, dx, dy in _cands:
        bx = cx + dx + (w / 2 if just == 'ML' else -w / 2 if just == 'MR' else 0)
        b = _box(bx, cy + dy, w, h)
        if abs(b[0]) > MAP_CM / 2 or abs(b[2]) > MAP_CM / 2:
            continue
        if _hits(b, _occupied):
            continue
        _occupied.append(b)
        _lx.append(lo); _ly.append(la); _lt.append(n); _lj.append((just, dx, dy))
        break
    else:
        _unplaced.append(n)

for (just, dx, dy), lo, la, txt in zip(_lj, _lx, _ly, _lt):
    fig.text(x=lo, y=la, text=txt, justify=just,
             offset=f'{dx}c/{dy}c', font=f'{pt(LAB_PT)},Helvetica,black',
             fill='white@25', clearance=f'{0.04*S}c/{0.03*S}c', projection=proj)

if _miss:
    print('  not in the hotspot table, so not labelled: ' + ', '.join(_miss))
if _unplaced:
    print('  no clear position, left unlabelled: ' + ', '.join(_unplaced))
print(f'  labelled {len(_lt)} of {len(_want)} requested hotspots')

leg = os.path.join(A.dir, f'_legend_{A.tag}.txt')
with open(leg, 'w') as fh:
    # Terse: the caption carries the explanation, and a legend wide enough to
    # restate it reaches across the map into the north Atlantic hotspots.
    fh.write(f'S {0.25*S}c c {0.34*S}c black 1.1p,white {0.75*S}c '
             f'endpoint within a LLSVP ({len(inside)})\n')
    fh.write(f'S {0.25*S}c c {0.34*S}c white 1.1p,black {0.75*S}c outside ({len(outside)})\n')
    fh.write(f'S {0.25*S}c - {0.7*S}c - 0.9p,gray25 {0.75*S}c corridors overlap ({nlink})\n')
fig.legend(spec=leg, position=f'JTR+jTR+o{0.15*S}c/{0.15*S}c+w{MAP_CM*0.34}c',
           box='+gwhite+p0.6p,gray40')

for ext in ('pdf', 'png'):
    fig.savefig(f'{A.out}.{ext}', dpi=400)
print(f'wrote {A.out}.pdf and .png')
print(f'  {len(inside)} conduits end inside a province, {len(outside)} outside')
print(f'  {nlink} corridor-overlap pairs drawn at a threshold of {A.link:g}')
print(f'  background: {os.path.basename(sl) if sl else "none"}')
