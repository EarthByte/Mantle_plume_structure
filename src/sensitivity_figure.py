"""Calibration figure: what amplitude of conduit each model lets the search see.

Panel a is the recovery curve - the fraction of injected synthetic conduits
recovered as a function of their amplitude, one curve per model, over the
configurations that reach the detection floor. Panel b is the null distribution
of path costs against the observed hotspot costs for the paper's model, which is
the comparison every verdict rests on.
"""
from __future__ import annotations
import glob, json, os, sys, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

CM = 1 / 2.54
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F
F.apply(10.5)                   # drawn 18.2 cm wide, printed at 16
# 10.5 rather than 10: this figure is drawn at 20.5 cm and printed at 16.0, a factor
# of 0.78, so 10 pt reaches the page at 7.8 and falls under the 8 pt floor.
SMALL = 10.5                    # the secondary size used on this figure
INK, ACC, BLU, GRY = F.INK, F.ACC, F.BLU, F.GRY
DIR = os.environ.get('PLUME_DIR', 'out')
OUT = os.environ.get('PLUME_FIG', 'figures')
ORDER = ['RevealLO_05deg', 'RevealLO_1deg', 'REVEAL', 'GLADM35', 'SPiRaL',
         'SEMUCB-WM1']
NICE = {'RevealLO_30km': 'RevealLO (30 km)', 'RevealLO': 'RevealLO',
        'RevealLO_1deg': 'RevealLO (1°)', 'RevealLO_05deg': 'RevealLO (0.5°)',
        'REVEAL': 'REVEAL', 'GLADM35': 'GLAD-M35', 'SPiRaL': 'SPiRaL',
        'SEMUCB-WM1': 'SEMUCB-WM1'}
COL = {'RevealLO_05deg': ACC, 'RevealLO_1deg': '#d98f7a', 'REVEAL': BLU,
       'GLADM35': '#4E8C6A', 'SPiRaL': '#8A6BAE', 'SEMUCB-WM1': '#C08A2E'}
MARK = {'RevealLO_05deg': 's', 'RevealLO_1deg': 'o', 'REVEAL': '^',
        'GLADM35': 'D', 'SPiRaL': 'v', 'SEMUCB-WM1': 'P'}


def halo(t, lw=2.2):
    t.set_path_effects([pe.withStroke(linewidth=lw, foreground='white')])
    return t


# Unknown tags must not crash the figure: a run named in run_all.sh but absent
# from these tables is a labelling gap, not an error.
_SPARE_C = ['#7A7A7A', '#3E7C8C', '#A8553A', '#6B5B95', '#4E8C6A', '#C08A2E']
_SPARE_M = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


def col(tag, i):
    return COL.get(tag, _SPARE_C[i % len(_SPARE_C)])


def mark(tag, i):
    return MARK.get(tag, _SPARE_M[i % len(_SPARE_M)])


def nice(tag):
    return NICE.get(tag, str(tag).replace('_', ' '))


avail = sorted(os.path.basename(f)[len('sensitivity_'):-4]
               for f in glob.glob(f'{DIR}/sensitivity_*.csv'))
tags = [t for t in ORDER if t in avail] + [t for t in avail if t not in ORDER]
if not tags:
    raise SystemExit(f'no sensitivity_*.csv in {DIR}')
# constrained_layout reserves room for the axis labels it knows about, but not for the
# rotated hotspot names drawn above panel b in data coordinates, and it leaves the y label
# of panel b hard against the figure edge. Both were reported only once the text-on-object
# defect above them was fixed: the checker raises on the first failure it finds, so one
# defect can hide the next. The rectangle gives the label column and the name row their
# space explicitly.
fig, ax = plt.subplots(1, 2, figsize=(20.5 * CM, 9.0 * CM),
                       constrained_layout=True)
fig.get_layout_engine().set(rect=(0.010, 0.0, 0.986, 0.93))

a = ax[0]
a.axhspan(0.75, 1.04, color='#eef2ee', zorder=0)
for i_, t in enumerate(tags):
    s = pd.read_csv(f'{DIR}/sensitivity_{t}.csv')
    d = pd.read_csv(f'{DIR}/detection_{t}.csv')
    k = d[(d.n_cal >= 4) & (d.detect_cal >= 0.75)][
        ['s', 'z_target', 'channel', 'radius']]
    s = s.merge(k, on=['s', 'z_target', 'channel', 'radius'], how='inner')
    g = s.groupby('amp').detect.mean().sort_index(ascending=False)
    x = -g.index.values
    a.plot(x, g.values, '-', color=col(t, i_), lw=1.4, zorder=3)
    a.plot(x, g.values, mark(t, i_), color=col(t, i_), ms=4.2, mfc='white',
           mew=1.2, zorder=4, label=nice(t))
a.axhline(0.75, color=GRY, lw=0.8, ls='--')
halo(a.text(1.58, 0.762, 'detection floor', fontsize=SMALL, color=GRY, va='bottom',
            ha='right'))
a.set_xlabel('amplitude of the injected conduit (per cent, negative)')
a.set_ylabel('fraction recovered')
a.set_ylim(-0.03, 1.05)
a.set_xlim(0.4, 1.6)
# The key goes under the panel rather than in it. Six entries at reading size
# make a block tall enough to reach the curves in whichever corner it is put,
# and this panel is a rising diagonal, so no corner is free.
_h, _l = a.get_legend_handles_labels()
fig.legend(_h, _l, loc='outside lower left', ncol=3, frameon=False,
           handletextpad=0.4, columnspacing=1.6)
halo(a.text(-0.10, 1.02, 'a', transform=a.transAxes, fontsize=13,
            fontweight='bold', va='bottom', ha='right'))
# The half-recovery range is a number the running text also quotes, so it is read
# from what manuscript_stats.py wrote rather than worked out again here. This
# panel used to interpolate a mean where the statistic takes a median, and the
# two ended up describing the same quantity with different figures.
STATS = os.path.join(DIR, 'manuscript_stats.json')
if not os.path.exists(STATS):
    raise SystemExit(f'\n{STATS} is missing, so the half-recovery range this '
                     'panel\nannotates cannot be read. Run manuscript_stats.py '
                     'first; it is not\nrecomputed here, to keep the panel and '
                     'the text from drifting apart.\n')
lo_, hi_ = json.load(open(STATS))['calibration']['half_recovery_range_percent']
halo(a.text(0.97, 0.04, f'half recovery between {lo_:.2f} and {hi_:.2f} per cent',
            transform=a.transAxes, ha='right', fontsize=SMALL, color=INK,
            bbox=F.mask()))

b = ax[1]
# panel b uses whichever model a null snapshot was taken for
nulls = sorted(glob.glob(f'{DIR}/null_*.npz'))
if not nulls:
    raise SystemExit(f'no null_*.npz in {DIR}; run null_snapshot.py first')
tag = os.path.basename(nulls[0])[len('null_'):-4]
z = np.load(nulls[0], allow_pickle=True)
nul, cost, names, p5 = z['null'], z['cost'], z['hotspot'], float(z['p5'])
nul = nul[np.isfinite(nul)]
b.hist(nul, bins=44, color='#dfe4ea', edgecolor='white', linewidth=0.5,
       zorder=2, label=f'{len(nul)} random locations')
b.axvline(p5, color=BLU, lw=1.1, ls='--', zorder=4)
YT = b.get_ylim()[1]
# Unrotated, in axes coordinates, at the top left. Anchored to the percentile value it
# sat within a few points of an axis tick label whichever side of the line it went, and
# rotated text anchored by va/va in the unrotated frame is hard to reason about; the
# dashed line is what identifies the percentile, and the label only has to name it.
# Low and right of centre: the named hotspots are drawn across the top of the panel
# and their rotated boxes reach down into it, so the top strip is not available.
halo(b.text(0.62, 0.05, 'null, 5th percentile', transform=b.transAxes,
            ha='left', va='bottom', fontsize=SMALL, color=BLU, bbox=F.mask()))
ok = np.isfinite(cost) & (cost < p5)
y0 = YT * 0.045
b.scatter(cost[np.isfinite(cost) & ~ok], np.full((~ok & np.isfinite(cost)).sum(), y0),
          s=13, marker='v', facecolor='white', edgecolor=GRY, linewidth=0.7, zorder=5)
b.scatter(cost[ok], np.full(ok.sum(), y0), s=20, marker='o', facecolor=ACC,
          edgecolor=INK, linewidth=0.6, zorder=6)
# Labels are staggered and led by a thin line, because the rooted hotspots are
# tightly bunched below the null threshold and would otherwise overprint.
order = np.argsort(cost[ok])
idx = np.where(ok)[0][order]
# The rooted hotspots bunch into a narrow band of cost, so at reading size their
# names cannot stand over them and cannot be fitted between the threshold and the
# left edge either. Headroom is opened above the histogram and the names are
# spread across the whole width of the panel, each led back to its own marker.
b.set_ylim(0, YT * 1.62)
xs = np.linspace(*b.get_xlim(), len(idx) + 2)[1:-1]
for k, xl in zip(idx, xs):
    b.plot([cost[k], xl], [y0 + YT * 0.03, YT * 1.00], lw=0.5, color=ACC,
           alpha=0.55, zorder=4)
    halo(b.text(xl, YT * 1.04, str(names[k]).split('(')[0].split('/')[0].strip(),
                rotation=90, ha='center', va='bottom', fontsize=SMALL, color=ACC), 1.8)
b.set_xlabel('cheapest descending-path cost (km)')
b.set_ylabel('number of random locations')
b.legend(frameon=True, framealpha=0.82, edgecolor='none', loc='upper right', handletextpad=0.4,
         bbox_to_anchor=(1.0, 0.62))
for _t in b.get_legend().get_texts():
    _t.set_bbox(F.mask())
halo(b.text(0.98, 0.45, nice(tag), transform=b.transAxes, ha='right',
            fontsize=SMALL, color=GRY, va='top'))
halo(b.text(-0.10, 1.02, 'b', transform=b.transAxes, fontsize=13,
            fontweight='bold', va='bottom', ha='right'))
os.makedirs(OUT, exist_ok=True)
F.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/fig_sensitivity.{ext}', bbox_inches='tight',
                facecolor='white')
print('wrote fig_sensitivity  models:', ', '.join(tags))
