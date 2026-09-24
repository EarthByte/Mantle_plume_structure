"""Figure: what corridor width measures, and the bound it places on continuity.

Four panels, each with its own axis, because the quantities swept are not
commensurable and putting amplitude and tilt on one axis would invite a reader to
compare per cent with kilometres. The first panel is the calibration that licenses
the measure. The second and third are the two explanations for the observed widths
that fail. The fourth is the one that bounds them, with each model's observed
hotspot median drawn against its own injection curve, since occupancy is read off
that curve and never off an absolute width.

No titles; the description belongs in the caption. Text size and overlap are
checked before the file is written.
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figstyle as FS

R_E, DEG = 6371.0, np.pi / 180.0
ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--out', default='../figures/fig_calibration')
A = ap.parse_args()
FS.apply(10.0)

MODELS = [('RevealLO', FS.ACC), ('RevealLO_30km', FS.GRY),
          ('GLADM35', FS.BLU), ('SEMUCB-WM1', '#4a7c1f')]


def sweep(tag, kind):
    """Both naming conventions: sweeps written before and after the values were
    put into the filename."""
    f = sorted(glob.glob(os.path.join(A.dir,
               f'corridor_width_calibration_{tag}_{kind}sweep*.csv')))
    f = [x for x in f if 'coarse' not in x]
    if not f:
        return None
    c = pd.read_csv(f[-1])
    return (c[c.injected_radius > 0].dropna(subset=['width']),
            c[c.injected_radius == 0].dropna(subset=['width']))


def observed(tag):
    p = os.path.join(A.dir, f'corridor_summary_{tag}.csv')
    if not os.path.exists(p):
        return np.nan
    s = pd.read_csv(p)
    return float(s[s.ok == True]['width_med_0.02'].median())


fig, axes = plt.subplots(2, 2, figsize=(FS.PLACED_CM * FS.CM, 12.6 * FS.CM))
(a, b), (c, d) = axes
OBS = observed('RevealLO')


def hotspot_line(ax, val=OBS, where=0.03):
    ax.axhline(val, color=FS.INK, lw=1.2, ls=(0, (5, 2)), zorder=1)
    ax.text(where, val, 'hotspots', transform=ax.get_yaxis_transform(),
            va='bottom', ha='left', fontsize=9.0, color=FS.INK, bbox=FS.mask())


# (a) the radius calibration that licenses the measure
cal = os.path.join(A.dir, 'corridor_width_calibration_RevealLO.csv')
if (not os.path.exists(cal)) or os.path.getsize(cal) < 60:
    raise SystemExit(f'{cal} is missing or empty; panel a is the calibration that '
                     'licenses the whole measure, so the figure is not drawn '
                     'without it')
cd = pd.read_csv(cal)
inj = cd[cd.injected_radius > 0].dropna(subset=['width'])
a.plot(inj.injected_fwhm, inj.width, 'o', ms=3.2, mfc='none', mec=FS.GRY,
       mew=0.8, zorder=2)
ga = inj.groupby('injected_fwhm').width.median()
a.plot(ga.index, ga.values, '-o', color=FS.ACC, lw=2.0, ms=5.0, zorder=3)
top = max(float(inj.injected_fwhm.max()), float(inj.width.max())) * 1.06
a.plot([0, top], [0, top], ls=(0, (4, 3)), color=FS.INK, lw=1.0, zorder=1)
a.text(top * 0.63, top * 0.78, 'one to one', fontsize=9.0, color=FS.INK,
       rotation=38, ha='center', va='center', bbox=FS.mask())
a.set_xlim(0, top); a.set_ylim(0, top)
a.set_xlabel('injected width (km FWHM)')
a.set_ylabel('measured corridor width (km)')

# (b) amplitude: stronger conduits give narrower corridors, not wider
am = sweep('RevealLO', 'amp')
if am:
    g = am[0].groupby('amp').width.median()
    b.plot(np.abs(g.index.to_numpy(float)), g.values, '-o', color=FS.BLU,
           lw=1.9, ms=5.0, zorder=3)
    b.plot([0], [float(am[1].width.median())], 'o', ms=5.5, mfc='white',
           mec=FS.BLU, mew=1.6, zorder=4)
    # The other three annotations in this figure carry FS.mask(); this one did not,
    # and it sits on the amplitude curve. Same treatment, for the same reason.
    b.text(0.04, float(am[1].width.median()), 'no conduit', fontsize=9.0,
           color=FS.BLU, va='top', bbox=FS.mask())
hotspot_line(b, where=0.55)
b.set_xlabel('conduit amplitude (per cent slow)')
b.set_ylabel('measured corridor width (km)')

# (c) tilt, over the range conduits are actually observed to occupy
tl = sweep('RevealLO', 'tilt')
sl = pd.read_csv(os.path.join(A.dir, 'plume_slant_RevealLO.csv'))
if tl:
    g = tl[0].groupby('tilt').width.median()
    off = g.index.to_numpy(float) * R_E * DEG
    c.plot(off, g.values, '-s', color=FS.ACC, lw=1.9, ms=4.8, zorder=3)
q1, q3 = float(sl.offset.quantile(.25)), float(sl.offset.quantile(.75))
c.axvspan(q1, q3, color=FS.GRY, alpha=0.22, linewidth=0, zorder=0)
hotspot_line(c, where=0.55)
# in the free space between the recovery curve and the hotspot line, rather
# than at the foot of the band where the label sat on the curve behind an
# opaque mask and hid a segment of it; placed after the hotspot line has set
# the axis range
_y0, _y1 = c.get_ylim()
c.text(0.5 * (q1 + q3), _y0 + 0.62 * (_y1 - _y0), 'observed\noffsets', ha='center',
       va='center', fontsize=9.0, color=FS.INK)
c.set_xlabel('lateral offset over the column (km)')
c.set_ylabel('measured corridor width (km)')

# (d) the continuity bound, each model against its own curve
for tag, col in MODELS:
    s = sweep(tag, 'duty')
    if not s:
        continue
    g = s[0].groupby('duty').width.median()
    d.plot(100 * g.index.to_numpy(float), g.values, '-o', color=col, lw=1.8,
           ms=4.2, label={'GLADM35': 'GLAD-M35', 'RevealLO_30km': 'RevealLO, 30 km'}.get(tag, tag), zorder=3)
    o = observed(tag)
    if np.isfinite(o):
        d.plot([0], [o], marker='<', color=col, ms=7.0, clip_on=False, zorder=4)
# The shaded band is READ from occupancy_bound.csv, which is the script that performs
# the test, rather than written here as a literal. It was 50 to 104 per cent, and the
# bound is now 75: a hard-coded band in a figure cannot follow the computation it claims
# to show, and this one had stopped following it.
_ob = os.path.join(A.dir, 'occupancy_bound.csv')
if not os.path.exists(_ob):
    raise SystemExit('out/occupancy_bound.csv not found; run occupancy_bound.py '
                     '--lateral 0.60 --ncell 2 before drawing this figure')
_o = pd.read_csv(_ob)
_runs = _o.groupby('occupancy').model.nunique()
_hit = _o.groupby('occupancy').excluded.sum()
_occ = sorted(float(x) for x in _runs.index)
_all = [x for x in _occ if _hit.loc[x] == _runs.loc[x]]
if _all:
    _lo = min(_all)
    _below = [x for x in _occ if x < _lo]
    _edge = 100.0 * (0.5 * (_lo + max(_below)) if _below else _lo)
    d.axvspan(_edge, 104, color=FS.GRY, alpha=0.20, linewidth=0, zorder=0)
    d.text(0.5 * (_edge + 104), d.get_ylim()[1], 'excluded', ha='center', va='top',
           fontsize=9.0, color=FS.INK)
d.set_xlim(-4, 104)
d.set_xlabel('column occupied (per cent)')
d.set_ylabel('measured corridor width (km)')
# The legend is opaque, so it must sit where there is nothing to hide. It used to be
# justified by the shaded band being in the lower left; the band is now read from
# occupancy_bound.csv and sits on the right, and the opaque box had begun covering the
# GLAD-M35 and SEMUCB-WM1 curves between 25 and 50 per cent occupancy. figstyle.check
# cannot see this: it tests text against drawn objects, and here it is the legend's own
# patch that does the hiding, with the text legitimately on top of it. So the room is
# made rather than borrowed - the axis is extended downwards until the legend fits below
# every curve.
_lo, _hi = d.get_ylim()
d.set_ylim(_lo - 0.42 * (_hi - _lo), _hi)
d.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=1.0,
         loc='lower left', fontsize=8.5, handlelength=1.4, borderaxespad=0.2)

for ax, ltr in zip((a, b, c, d), 'abcd'):
    # Above the axes rather than beside them: at x = -0.19 the letter lands on
    # the rotated y-axis label.
    ax.text(-0.02, 1.05, ltr, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='bottom', ha='right', bbox=FS.mask())
fig.tight_layout(w_pad=2.4, h_pad=2.2)
FS.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(f'{A.out}.{ext}', bbox_inches='tight')
print(f'wrote {A.out}.pdf and .png; text checks passed')
