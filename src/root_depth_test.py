"""Does the root-depth walk recover a termination depth it is given?

The width negative in this study was retracted because the estimator was never
asked to measure a conduit of known width, and when it finally was it turned out
to invert: a 1400 km conduit came back narrower than a 400 km one. Root depth is
a second estimator of the same kind and must not be reported before the same
question is answered for it.

A Gaussian tube of known radius and amplitude is added to the real field from the
anchor down to a known termination depth and nothing below it, and the walk is run
at the axis. The ambient field is left in place, because a conduit terminating at
1000 km above a region that is independently slow at 2700 km is exactly the case
that decides whether a recovered deep root means anything, and removing the
ambient would define that difficulty away.

Sites are random by default. Injecting at hotspots measures recoverability where
it matters but confounds it with whatever is really there, so --at-hotspots is
available and is not the default.
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon, check_shells
from root_core import thresholds, walk as _walk
warnings.filterwarnings('ignore')

R_E, DEG = 6371.0, np.pi / 180.0

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=2)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--anchor', type=float, default=300.0)
ap.add_argument('--radius', type=float, default=400.0, help='cap radius searched, km')
ap.add_argument('--pct', type=float, default=20.0)
ap.add_argument('--tol', type=float, default=200.0)
ap.add_argument('--conduit-radius', type=float, default=400.0, dest='cr',
                help='Gaussian sigma of the injected tube, km')
ap.add_argument('--amps', default='0.5,1.0,2.0', help='injected amplitudes, per cent, comma separated')
ap.add_argument('--ends', default='660,1000,1500,2000,2500,2800',
                help='injected termination depths, km, comma separated')
ap.add_argument('--sites', type=int, default=25)
ap.add_argument('--at-hotspots', action='store_true', dest='at_hotspots')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--dir', default='out')
ap.add_argument('--suffix', default='')
A = ap.parse_args()

AMPS = [float(x) for x in A.amps.split(',')]
ENDS = [float(x) for x in A.ends.split(',')]
os.makedirs(A.dir, exist_ok=True)

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
check_shells(depth, arr, A.tag)
lat = np.asarray(lat, float)
lon = ((np.asarray(lon, float) + 180) % 360) - 180
o = np.argsort(lon); lon, arr = lon[o], arr[:, :, o]
thr = thresholds(arr, lat, lon, A.pct)
print(f'{A.tag}: {len(depth)} shells, {depth.min():.0f}-{depth.max():.0f} km', flush=True)

if A.at_hotspots:
    hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180'])
    SITES = [(str(r['hotspot']), float(r['lat']), float(r['lon_180'])) for _, r in hs.iterrows()]
else:
    rng = np.random.default_rng(202)
    SITES = [(f'site{i:03d}', float(np.degrees(np.arcsin(2 * rng.random() - 1))),
              float(360 * rng.random() - 180)) for i in range(A.sites)]
print(f'{len(SITES)} sites x {len(ENDS)} termination depths x {len(AMPS)} amplitudes', flush=True)

_mar = 2.5 * A.cr


def box(hlat, hlon):
    dlat = (A.radius + _mar) / (R_E * DEG) + abs(lat[1] - lat[0]) * 2
    jl = np.where(np.abs(lat - hlat) <= dlat)[0]
    dlon = ((A.radius + _mar) / (R_E * DEG * max(np.cos(hlat * DEG), 1e-3))
            + abs(lon[1] - lon[0]) * 2)
    il = np.where(np.abs(((lon - hlon + 180) % 360) - 180) <= min(dlon, 180.0))[0]
    if not len(jl) or not len(il):
        return None
    LA, LO = lat[jl][:, None], lon[il][None, :]
    d = R_E * np.arccos(np.clip(
        np.sin(hlat * DEG) * np.sin(LA * DEG) +
        np.cos(hlat * DEG) * np.cos(LA * DEG) * np.cos((LO - hlon) * DEG), -1, 1))
    return jl, il, d


def profile_of(sub, cap):
    out = np.full(sub.shape[0], np.nan)
    for i in range(sub.shape[0]):
        v = sub[i][cap]
        v = v[np.isfinite(v)]
        if len(v):
            out[i] = v.min()
    return out


rows = []
for si, (name, hlat, hlon) in enumerate(SITES):
    b = box(hlat, hlon)
    if b is None:
        continue
    jl, il, d = b
    cap = d <= A.radius
    if not cap.any():
        continue
    sub0 = np.asarray(arr[:, jl][:, :, il], float)
    horiz = np.exp(-0.5 * (d / A.cr) ** 2)[None, :, :]
    amb = profile_of(sub0, cap)
    r_amb, _ = _walk(depth, amb, amb < thr, A.anchor, A.tol)
    for amp in AMPS:
        for zend in ENDS:
            mask = (depth <= zend)[:, None, None].astype(float)
            sub = sub0 - amp * horiz * mask
            pr = profile_of(sub, cap)
            rec, gap = _walk(depth, pr, pr < thr, A.anchor, A.tol)
            rows.append(dict(site=name, lat=hlat, lon=hlon, amp=amp, injected=zend,
                             recovered=rec, gap_top=gap, ambient_root=r_amb,
                             basal=float(np.nanmean(amb[(depth >= 2400) & (depth <= 2850)]))))
    if (si + 1) % 10 == 0:
        print(f'  {si + 1}/{len(SITES)} sites', flush=True)

df = pd.DataFrame(rows)
p = os.path.join(A.dir, f'root_depth_test_{A.tag}{A.suffix}.csv')
df.to_csv(p, index=False)

print(f'\nrecovered termination depth against injected, conduit sigma {A.cr:.0f} km')
print(f'{"amp":>5s} {"injected":>9s} {"median rec":>11s} {"bias":>8s} '
      f'{"IQR":>9s} {"within 300 km":>14s} {"ran past by >500":>17s}')
for amp in AMPS:
    for zend in ENDS:
        s = df[(df.amp == amp) & (df.injected == zend)]
        r = s.recovered.fillna(A.anchor).to_numpy(float)
        if not len(r):
            continue
        q1, q3 = np.percentile(r, [25, 75])
        print(f'{amp:5.1f} {zend:9.0f} {np.median(r):11.0f} {np.median(r) - zend:+8.0f} '
              f'{q3 - q1:9.0f} {100 * np.mean(np.abs(r - zend) <= 300):13.0f}% '
              f'{100 * np.mean(r - zend > 500):16.0f}%')

print('\nmonotonic in injected depth? (the width estimator was not)')
for amp in AMPS:
    med = [float(np.median(df[(df.amp == amp) & (df.injected == z)]
                           .recovered.fillna(A.anchor))) for z in ENDS]
    mono = all(med[i + 1] >= med[i] - 1e-9 for i in range(len(med) - 1))
    print(f'  amp {amp:.1f}%: ' + ' -> '.join(f'{m:.0f}' for m in med)
          + ('   MONOTONIC' if mono else '   NOT MONOTONIC'))

print('\nhow often a conduit stopping at 660 km is read as reaching past 2400 km:')
for amp in AMPS:
    s = df[(df.amp == amp) & (df.injected == 660)]
    r = s.recovered.fillna(A.anchor).to_numpy(float)
    dp = s.basal.to_numpy(float)
    if len(r):
        print(f'  amp {amp:.1f}%: {100 * np.mean(r >= 2400):4.0f}% of sites; '
              f'among sites whose ambient base is slow (< -0.5%), '
              f'{100 * np.mean(r[dp < -0.5] >= 2400) if (dp < -0.5).any() else float("nan"):4.0f}%')
print(f'\nwrote {p}', flush=True)
