"""Figure: conduit lean in two depth regimes, and the direction it does not take.

Three rose diagrams of the angle between a conduit's lean and a predicted bearing,
so that concentration at zero means the prediction holds. The shallow panel is
against the reverse of absolute plate motion, the deep panel against the bearing
away from subduction in three models, and the third against the bearing to the
nearest ridge, which is flat. The flat panel is the point of the figure as much as
the other two: the same operator on the same conduits finds two directions and not
a third, so the nulls are informative.

Every misfit is recomputed here from the stored conduit paths rather than read from
a summary, so the figure cannot drift away from the numbers in the text. No titles.
"""
from __future__ import annotations
import argparse, json, math, os
import numpy as np, pandas as pd
from math import erfc, sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figstyle as FS

R_E, DEG = 6371.0, np.pi / 180.0
ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='out')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--ridges', default='data/ridges_presentday_zahirovic2022.csv')
ap.add_argument('--out', default='../figures/fig_lean')
A = ap.parse_args()
FS.apply(10.5)

hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180']).set_index('hotspot')


def gc(a, b, c, d):
    return R_E * np.arccos(np.clip(np.sin(a * DEG) * np.sin(c * DEG) +
                                   np.cos(a * DEG) * np.cos(c * DEG) *
                                   np.cos((d - b) * DEG), -1, 1))


def bearing(a, b, c, d):
    y = np.sin((d - b) * DEG) * np.cos(c * DEG)
    x = np.cos(a * DEG) * np.sin(c * DEG) - np.sin(a * DEG) * np.cos(c * DEG) * np.cos((d - b) * DEG)
    return np.degrees(np.arctan2(y, x)) % 360.0


def leans(tag, z0, z1):
    p = os.path.join(A.dir, f'conduit_paths_all_{tag}.json')
    if not os.path.exists(p):
        return {}
    out = {}
    for n, q in json.load(open(p)).items():
        if n not in hs.index:
            continue
        la = np.asarray(q['lat'], float); lo = np.asarray(q['lon'], float)
        zz = np.asarray(q['depth'], float)
        o = np.argsort(zz); la, lo, zz = la[o], lo[o], zz[o]
        m = (zz >= z0) & (zz <= z1)
        if m.sum() < 4:
            continue
        la2, lo2 = la[m], lo[m]
        e = R_E * DEG * (((lo2 - lo2[0] + 180) % 360) - 180) * np.cos(la2[0] * DEG)
        nn = R_E * DEG * (la2 - la2[0])
        if float(np.hypot(e[-1], nn[-1])) < 20:
            continue
        out[n] = math.degrees(math.atan2(e[-1], nn[-1])) % 360.0
    return out


def stats(mis):
    a = np.radians(np.asarray(mis, float))
    n = len(a); C, S = np.cos(a).mean(), np.sin(a).mean()
    return n, math.hypot(C, S), 0.5 * erfc((C * math.sqrt(2 * n)) / sqrt(2))


def wrap(x):
    return (np.asarray(x, float) + 180) % 360 - 180


# shallow: against the reverse of absolute plate motion
# The 50 Myr averaging window is the one the text reports. The choice does not
# carry the result: at this depth range every window from 10 to 80 Myr gives
# P between 0.0037 and 0.0137, with R between 0.271 and 0.369.
APM_WINDOW = 50.0
apm = pd.read_csv(os.path.join(A.dir, 'plume_slant_apm.csv'))
apm = apm[apm.window == APM_WINDOW].set_index('site')
sh = leans('RevealLO', 300.0, 660.0)
shallow = [wrap(v - ((float(apm.loc[k, 'apm_az']) + 180) % 360))
           for k, v in sh.items() if k in apm.index]

# deep: against the bearing away from subduction, three models
t = pd.read_csv(os.path.join(A.dir, 'hinge_migration.csv'))
tlon, tlat, al = t.lon.to_numpy(), t.lat.to_numpy(), t.arc_length.to_numpy()
deep = {}
for tag in ('RevealLO', 'REVEAL', 'GLADM35'):
    vals = []
    for k, v in leans(tag, 660.0, 1500.0).items():
        hla, hlo = float(hs.loc[k, 'lat']), float(hs.loc[k, 'lon_180'])
        d = gc(hla, hlo, tlat, tlon); sel = d <= 6000.0
        if sel.sum() < 20:
            continue
        w = al[sel] / np.maximum(d[sel], 200.0) ** 2
        br = np.radians(bearing(hla, hlo, tlat[sel], tlon[sel]))
        tb = math.degrees(math.atan2((np.sin(br) * w).sum(), (np.cos(br) * w).sum())) % 360.0
        vals.append(wrap(v - tb - 180.0))
    if vals:
        deep[tag] = vals

# ridge: against the bearing to the nearest ridge, within the interaction distance
rg = pd.read_csv(A.ridges)
RLA, RLO = rg.lat.to_numpy(float), rg.lon.to_numpy(float)
ridge = []
for k, v in sh.items():
    hla, hlo = float(hs.loc[k, 'lat']), float(hs.loc[k, 'lon_180'])
    d = gc(hla, hlo, RLA, RLO); j = int(np.argmin(d))
    if float(d[j]) <= 1400.0:
        ridge.append(wrap(v - bearing(hla, hlo, RLA[j], RLO[j])))

fig = plt.figure(figsize=(FS.PLACED_CM * FS.CM, 12.8 * FS.CM))
# Two grids rather than one. The rose row needs width for its captions and the
# cross-model row needs a deep left margin for its model names; a single shared margin
# that satisfies one pushes the other off the sheet, which is what the size check
# caught twice.
gtop = fig.add_gridspec(1, 3, left=0.055, right=0.945, top=0.96, bottom=0.58,
                        wspace=0.45)
gbot = fig.add_gridspec(1, 2, left=0.195, right=0.985, top=0.395, bottom=0.175,
                        wspace=0.10)
axes = [fig.add_subplot(gtop[0, k], projection='polar') for k in range(3)]
axd = [fig.add_subplot(gbot[0, 0]), fig.add_subplot(gbot[0, 1])]
BINS = np.arange(-180, 181, 30)


def rose(ax, vals, colour, label, pooled=False):
    n, R, P = stats(vals)
    h, _ = np.histogram(vals, bins=BINS)
    th = np.radians(BINS[:-1] + 15)
    ax.bar(th, h, width=np.radians(28), color=colour, alpha=0.75,
           edgecolor='white', linewidth=0.6, zorder=2)
    ax.plot([0, 0], [0, h.max() * 1.12], color=FS.INK, lw=1.6, zorder=3)
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(['0', '90', '180', '270'], fontsize=9)
    ax.set_yticklabels([])
    txt = (f'{label}\nn = {n}, R = {R:.3f}'
           + ('' if pooled else f', P = {P:.3g}'))
    ax.set_xlabel(txt, fontsize=9, labelpad=10)
    return R


rose(axes[0], shallow, FS.BLU, 'against plate motion\n300 to 660 km')
allthree = [v for vals in deep.values() for v in vals]
rose(axes[1], allthree, FS.ACC,
     'away from subduction\n660 to 1500 km, three models pooled', pooled=True)
rose(axes[2], ridge, FS.GRY, 'toward the nearest ridge\n300 to 660 km, within 1400 km')

# Panels d and e: the same test in every model, one panel per depth window. Both
# windows on one axis put two series on every row, where a count set beside one marker
# landed on the other and the untestable note sat across the other window's line. One
# series per panel removes that class of collision rather than tuning around it.
cm = pd.read_csv(os.path.join(A.dir, 'crossmodel_lean.csv'))
ORDER = ['RevealLO', 'RevealLO_30km', 'REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1']
SHOW = {'RevealLO': 'RevealLO', 'RevealLO_30km': 'RevealLO 30 km',
        'REVEAL': 'REVEAL', 'GLADM35': 'GLAD-M35', 'SPiRaL': 'SPiRaL',
        'SEMUCB-WM1': 'SEMUCB-WM1'}
PANELS = (('300-660', FS.BLU, 'against plate motion\n300 to 660 km'),
          ('660-1500', FS.ACC, 'away from subduction\n660 to 1500 km'))
for k, (win, colour, lab) in enumerate(PANELS):
    ax = axd[k]
    d = cm[cm.window == win].set_index('model')
    for i2, m in enumerate(ORDER):
        if m not in d.index:
            continue
        y = len(ORDER) - 1 - i2
        r, pv, n = d.at[m, 'R'], d.at[m, 'P'], int(d.at[m, 'n'])
        if not np.isfinite(r):
            ax.text(0.012, y, f'not testable, {n} paths', fontsize=8.5,
                    color=FS.GRY, va='center', ha='left')
            continue
        ax.plot([0, r], [y, y], color=colour, lw=1.4, alpha=0.5, zorder=1,
                solid_capstyle='butt')
        ax.plot([r], [y], marker='o', ms=7.5, zorder=3,
                color=colour if pv < 0.05 else 'white',
                markeredgecolor=colour, markeredgewidth=1.6)
        ax.text(r + 0.016, y, f'{n}', fontsize=8.5, color=FS.INK, va='center')
    ax.set_xlim(0, 0.44); ax.set_ylim(-0.7, len(ORDER) - 0.3)
    ax.set_xticks(np.arange(0, 0.401, 0.1))
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([SHOW[m] for m in ORDER][::-1] if k == 0 else [], fontsize=9)
    ax.tick_params(axis='both', labelsize=9)
    # The quantity is the axis label and sits close to the axis; the condition it was
    # measured under is a separate block below it, set tight. One three-line label
    # cannot do this, because line spacing there applies to every line at once and
    # either runs the two together or scatters all three.
    ax.set_xlabel('mean resultant length', fontsize=9, labelpad=5)
    ax.text(0.5, -0.36, lab, transform=ax.transAxes, ha='center', va='top',
            fontsize=9, linespacing=1.35)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(axis='x', color=FS.GRY, lw=0.5, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

for ax, ltr in zip(axes, 'abc'):
    ax.text(-0.10, 1.06, ltr, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='bottom')
for ax, ltr in zip(axd, 'de'):
    ax.text(-0.02 if ltr == 'e' else -0.30, 1.02, ltr, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom')

FS.check(fig)          # raises with the offending text if anything is too small,
                       # collides, or runs off the page; returns True otherwise
for ext in ('pdf', 'png'):
    fig.savefig(f'{A.out}.{ext}', bbox_inches='tight')
print(f'wrote {A.out}.pdf and .png; text checks passed')
