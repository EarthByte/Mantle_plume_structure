#!/usr/bin/env python3
"""What does the search do where there is no conduit?

Every offset and tilt in this paper is the output of a search that will return SOME
descending path from any starting point, whether or not a conduit is there. Until now
nothing measured what that path looks like at a location with no hotspot and nothing
injected, so there was no scale against which a hotspot offset of 500 km meant
anything.

This traces from ambient sites, at least 1500 km from any hotspot, with nothing
injected, and compares two things with the hotspots:

  MAGNITUDE  the lateral offset accumulated between 200 and 2700 km.
  DIRECTION  the lean between 660 and 1500 km against the distance-weighted bearing
             to trenches, tested exactly as the hotspots are tested.

The two answers differ, and the difference is the point. Reading it requires both:
a magnitude that matches the null says the amount of displacement is uninformative; a
direction that does not match the null says the orientation still is.

  python3 ambient_null.py
"""
from __future__ import annotations
import argparse, json, math, os, sys
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inject import gc_km

R_E, DEG, CUTOFF, PREDICTED = 6371.0, math.pi / 180.0, 6000.0, 180.0
HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--dir', default='out')
ap.add_argument('--ambient', default=None,
                help='ambient traces from tilt_recovery.py --amp 0 --tilts 0')
ap.add_argument('--hinge', default='out/hinge_migration.csv')
A = ap.parse_args()
D = os.path.join(HERE, A.dir)
amb = pd.read_csv(A.ambient or os.path.join(D, f'ambient_null_{A.tag}.csv'))

P = json.load(open(os.path.join(D, f'conduit_paths_all_{A.tag}.json')))
off = []
for s, q in P.items():
    z = np.asarray(q['depth'], float)
    la = np.asarray(q['lat'], float); lo = np.asarray(q['lon'], float)
    o = np.argsort(z); z, la, lo = z[o], la[o], lo[o]
    m = (z >= 200) & (z <= 2700)
    if m.sum() >= 2:
        off.append(float(gc_km(la[m][0], lo[m][0], la[m][-1], lo[m][-1])))
off = np.asarray(off)
a_off = amb.recovered_km.to_numpy(float)
a_off = a_off[np.isfinite(a_off)]

print('MAGNITUDE, lateral offset from 200 to 2700 km')
print(f'  {len(off)} hotspots      median {np.median(off):5.0f} km  '
      f'IQR {np.percentile(off, 25):.0f}-{np.percentile(off, 75):.0f}')
print(f'  {len(a_off)} ambient sites median {np.median(a_off):5.0f} km  '
      f'IQR {np.percentile(a_off, 25):.0f}-{np.percentile(a_off, 75):.0f}')
pm = float(mannwhitneyu(off, a_off).pvalue)
print(f'  Mann-Whitney P = {pm:.3f}'
      + ('   the two are not distinguishable' if pm > 0.05 else ''))

# ---- the same magnitude question, over the window the named cases use
#
# Section 3.1 supports the validation with Louisville "displaced by only 93 km over
# 660-1500 km, the fourth-smallest in the sample". A rank within the sample is a true
# fact about the sample; 93 km is evidence that the conduit is STABLE only if it is
# small against paths that have no conduit in them. That is this comparison, and it
# could not be made until the ambient run recorded an offset over the same window.
if 'offset_660_1500' in amb.columns and amb.offset_660_1500.notna().any():
    w = []
    for s_, q in P.items():
        z = np.asarray(q['depth'], float)
        la = np.asarray(q['lat'], float); lo = np.asarray(q['lon'], float)
        o = np.argsort(z); z, la, lo = z[o], la[o], lo[o]
        m = (z >= 660) & (z <= 1500)
        if m.sum() >= 2:
            w.append((s_, float(gc_km(la[m][0], lo[m][0], la[m][-1], lo[m][-1]))))
    hw = pd.DataFrame(w, columns=['hotspot', 'offset_km']).sort_values('offset_km')
    aw = amb.offset_660_1500.to_numpy(float)
    aw = aw[np.isfinite(aw)]
    print('\nMAGNITUDE, lateral offset from 660 to 1500 km, the named-case window')
    print(f'  {len(hw)} hotspots      median {hw.offset_km.median():5.0f} km  '
          f'IQR {hw.offset_km.quantile(.25):.0f}-{hw.offset_km.quantile(.75):.0f}')
    print(f'  {len(aw)} ambient sites median {np.median(aw):5.0f} km  '
          f'IQR {np.percentile(aw, 25):.0f}-{np.percentile(aw, 75):.0f}')
    pw = float(mannwhitneyu(hw.offset_km.to_numpy(float), aw).pvalue)
    print(f'  Mann-Whitney P = {pw:.3f}'
          + ('   the two are not distinguishable' if pw > 0.05 else ''))
    for name in ('Louisville', 'Afar'):
        r = hw[hw.hotspot.str.contains(name, case=False, na=False)]
        if not len(r):
            continue
        v = float(r.offset_km.iloc[0])
        rank = int((hw.offset_km < v).sum()) + 1
        pct = float((aw < v).mean())
        print(f'  {name}: {v:.0f} km, rank {rank} of {len(hw)} in the sample; '
              f'{pct:.0%} of ambient paths move less')
        if pct > 0.10:
            print(f'    {pct:.0%} of paths with NO conduit are quieter than this, so it '
                  'is a rank\n    within the sample and not a demonstration of '
                  'stability. Say that.')
else:
    print('\nambient file carries no offset_660_1500 column; re-run tilt_recovery.py '
          'at the\ncurrent move set to make the named-case comparison')

t = pd.read_csv(os.path.join(HERE, A.hinge))
tlat, tlon = t.lat.to_numpy(float), t.lon.to_numpy(float)
al = t.arc_km.to_numpy(float) if 'arc_km' in t.columns else np.ones(len(t))


def bearing(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    return np.degrees(np.arctan2(np.sin(d - b) * np.cos(c),
                                 np.cos(a) * np.sin(c)
                                 - np.sin(a) * np.cos(c) * np.cos(d - b))) % 360.0


mis = []
for _, r in amb[amb.lean_deep.notna()].iterrows():
    dd = gc_km(r.lat, r.lon, tlat, tlon)
    sel = dd <= CUTOFF
    if sel.sum() < 20:
        continue
    w = al[sel] / np.maximum(dd[sel], 200.0) ** 2
    br = np.radians(bearing(r.lat, r.lon, tlat[sel], tlon[sel]))
    tb = math.degrees(math.atan2((np.sin(br) * w).sum(),
                                 (np.cos(br) * w).sum())) % 360.0
    f = (float(r.lean_deep) - tb) % 360.0
    mis.append(f - 360 if f > 180 else f)
a = np.radians(np.asarray(mis, float) - PREDICTED)
n = len(a)
C, S = np.cos(a).mean(), np.sin(a).mean()
R = math.hypot(C, S)
pv = 0.5 * math.erfc(C * math.sqrt(2 * n) / math.sqrt(2))
print('\nDIRECTION, lean from 660 to 1500 km against the bearing away from subduction')
print(f'  ambient  n = {n:3d}   R = {R:.3f}   P = {pv:.4f}')
# The hotspot side was a literal here - 48, 0.303, 0.0034 - and stayed at its old
# value through a retrace that changed it, so it is read from the file that computes it.
# Reading the right file is not enough on its own: this script once ran in an earlier
# step than lean_confirm.py, so it read the PREVIOUS run's hotspot statistic and printed
# it beside a current ambient null. The two were from different cost channels and the
# comparison was meaningless. Hence the freshness check - a stale hotspot side is a
# stop, not a warning, because the sentence this script ends on is a scientific claim.
_pf = os.path.join(D, f'conduit_paths_all_{A.tag}.json')
_lc = os.path.join(D, 'lean_confirm_crossmodel.csv')
if not os.path.exists(_lc):
    raise SystemExit('lean_confirm_crossmodel.csv not found. Run lean_confirm.py first: '
                     'the hotspot side of this comparison comes from it.')
if os.path.getmtime(_lc) < os.path.getmtime(_pf):
    raise SystemExit(
        f'lean_confirm_crossmodel.csv is older than {os.path.basename(_pf)}, so its\n'
        'hotspot statistic was computed from superseded paths. Run lean_confirm.py\n'
        'before this script; comparing a current null against a stale hotspot result\n'
        'is how a withdrawn claim gets reported as confirmed.')
_t = pd.read_csv(_lc)
_r = _t[(_t.model == A.tag) & (_t.window == '660-1500')]
if not len(_r):
    raise SystemExit(f'{A.tag} has no 660-1500 row in lean_confirm_crossmodel.csv')
_r = _r.iloc[0]
_hn, _hR, _hP = int(_r.n), float(_r.R), float(_r.P)
print(f'  hotspots n = {_hn:3d}   R = {_hR:.3f}   P = {_hP:.4f}   (lean_confirm.py)')
# What the pair means depends on BOTH sides. The old text asserted that the hotspot
# direction survives the null whenever the null itself is flat, which is only half the
# statement: if the hotspot test does not reach its own threshold there is no direction
# for a wandering path to fail to reproduce, and a flat null says nothing either way.
if pv <= 0.05:
    print('  WARNING: ambient also shows the predicted direction, so the hotspot '
          'result is not specific to conduits')
elif _hP < 0.05:
    print('  the hotspot direction is not reproduced by a wandering path')
else:
    print(f'  neither side reaches P < 0.05. The null is flat, but with the hotspot '
          f'test at P = {_hP:.4f}\n  there is no hotspot direction for it to fail to '
          f'reproduce, so this control\n  neither supports nor refutes a directional '
          f'claim.')

pd.DataFrame([dict(what='hotspot median offset km', value=float(np.median(off))),
              dict(what='ambient median offset km', value=float(np.median(a_off))),
              dict(what='P magnitude hotspots vs ambient', value=pm),
              dict(what='ambient n direction', value=float(n)),
              dict(what='ambient R direction', value=float(R)),
              dict(what='ambient P direction', value=float(pv)),
              dict(what='hotspot n direction', value=float(_hn)),
              dict(what='hotspot R direction', value=float(_hR)),
              dict(what='hotspot P direction', value=float(_hP))]
             ).to_csv(os.path.join(D, f'ambient_null_summary_{A.tag}.csv'), index=False)
