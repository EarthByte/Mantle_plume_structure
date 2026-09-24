"""Can the tracer recover a branch depth it was given?

This decides whether the depth questions can be attempted at all. Rudolph's
viscosity boundary near 1000 km and the post-spinel transition at 660 are 340 km
apart, so if injected branch depths come back with a scatter worse than about
150 km the two cannot be told apart and the programme stops at the catalogue.
That rule was set in PLUME_STRUCTURE_PLAN.md before this was run.

Y-shaped conduits are injected into the real RevealLO field at hotspot locations,
at the widths real plumes have rather than the widths a detector finds easy - the
mistake that cost this project three estimators. A single tube runs from the base
to the branch depth; above it two limbs diverge.

The scoring is PAIRED, and the first version of this file was not. It scored the
injected branch depth as the recovered branch nearest the injected one, which picks
from all branches found the one closest to the answer; with several ambient branches
available that metric cannot fail, and it duly reported a 60 km interquartile error
at 1000 km while an uninjected control reported 45 km at a depth that was never
injected. Here each site is traced WITHOUT the injection first, and a recovered
branch counts only if it is absent from that ambient set. The question then becomes
whether the injection ADDS a branch where it was put, which can be answered no.
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon, check_shells
from plume_core import Volume, Tracer, per_depth_threshold, R_E, DEG
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=2)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--pct', type=float, default=15.0, help='slow percentile per depth')
ap.add_argument('--min-area', type=float, default=2.0e5, dest='min_area')
ap.add_argument('--radii', default='300,500,700', help='injected tube sigma, km')
ap.add_argument('--amps', default='0.4,0.8,1.5', help='injected amplitude, per cent')
ap.add_argument('--branches', default='660,1000,1500,2000',
                help='injected branch depths, km')
ap.add_argument('--sep-ratios', default='1.5,2.5,4.0', dest='sep_ratios',
                help='limb separation at the top, as a multiple of the tube sigma. '
                     'Two Gaussians of width s placed closer than about 2s are one '
                     'blob, not two limbs, so a fixed separation in km is a different '
                     'experiment for every tube width and a Y that cannot be resolved '
                     'in principle gets scored as a failure to recover its depth. The '
                     'ratio at which branching becomes resolvable is itself a result: '
                     'real limbs are comparable in width to their separation, so this '
                     'may be the binding limit rather than depth precision.')
ap.add_argument('--box', type=float, default=28.0, help='half-width of the sub-box, deg')
ap.add_argument('--sites', type=int, default=12)
ap.add_argument('--new-tol', type=float, default=120.0, dest='new_tol',
                help='a recovered branch counts as new if no ambient branch lies '
                     'within this many km of it')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--dir', default='out')
ap.add_argument('--suffix', default='')
A = ap.parse_args()

RAD = [float(x) for x in A.radii.split(',')]
AMP = [float(x) for x in A.amps.split(',')]
SEP = [float(x) for x in A.sep_ratios.split(',')]
ZB = [float(x) for x in A.branches.split(',')]
os.makedirs(A.dir, exist_ok=True)

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
check_shells(depth, arr, A.tag)
lat = np.asarray(lat, float)
lon = ((np.asarray(lon, float) + 180) % 360) - 180
o = np.argsort(lon); lon, arr = lon[o], arr[:, :, o]
THR = per_depth_threshold(arr, lat, A.pct)
print(f'{A.tag}: {len(depth)} shells; global slow threshold runs '
      f'{np.nanmin(THR):+.2f} to {np.nanmax(THR):+.2f} per cent', flush=True)

hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180'])
SITES = [(str(r['hotspot']), float(r['lat']), float(r['lon_180']))
         for _, r in hs.iterrows()][:A.sites]
print(f'{len(SITES)} sites x {len(ZB)} branch depths x {len(RAD)} radii x '
      f'{len(AMP)} amplitudes', flush=True)

rows = []
for si, (name, hlat, hlon) in enumerate(SITES):
    jl = np.where(np.abs(lat - hlat) <= A.box)[0]
    dl = np.abs(((lon - hlon + 180) % 360) - 180)
    il = np.where(dl <= A.box / max(np.cos(hlat * DEG), 0.2))[0]
    if len(jl) < 20 or len(il) < 20:
        continue
    sub0 = np.asarray(arr[:, jl][:, :, il], float)
    blat, blon = lat[jl], lon[il]
    # The ambient branch set at this site, traced once with nothing injected.
    v0 = Volume(depth, blat, blon, sub0, min_area_km2=A.min_area, thresholds=THR)
    t0 = Tracer(v0).trace(hlat, hlon)
    amb = list(t0['branch_depths']) if (t0 and t0.get('found')) else []
    east = (R_E * DEG * (((blon - hlon + 180) % 360) - 180)
            * np.cos(hlat * DEG))[None, None, :]
    north = (R_E * DEG * (blat - hlat))[None, :, None]
    for zb in ZB:
        # Limbs separate linearly above the branch depth, up to --sep at the top.
        top = depth.min()
        frac = np.clip((zb - depth) / max(zb - top, 1.0), 0.0, 1.0)[:, None, None]
        for cr in RAD:
            for sr in SEP:
                off = 0.5 * (sr * cr) * frac
                d1 = (east - off) ** 2 + north ** 2
                d2 = (east + off) ** 2 + north ** 2
                shape = np.maximum(np.exp(-0.5 * d1 / cr ** 2),
                                   np.exp(-0.5 * d2 / cr ** 2))
                for amp in AMP:
                    fld = sub0 - amp * shape
                    vol = Volume(depth, blat, blon, fld, min_area_km2=A.min_area,
                                 thresholds=THR)
                    t = Tracer(vol).trace(hlat, hlon)
                    base = dict(site=name, zb=zb, cr=cr, amp=amp, sep_ratio=sr,
                                sep_km=sr * cr)
                    if not t or not t.get('found'):
                        rows.append(dict(**base, found=False))
                        continue
                    bd = list(t['branch_depths'])
                    new = [x for x in bd
                           if not amb or min(abs(x - a_) for a_ in amb) > A.new_tol]
                    near = min(new, key=lambda x: abs(x - zb)) if new else np.nan
                    rows.append(dict(**base, found=True,
                                     n_branch=t['n_branch'],
                                     n_ambient=len(amb), n_new=len(new),
                                     max_strands=t['max_strands'],
                                     recovered=near,
                                     err=(near - zb) if new else np.nan,
                                     root_km=t['root_km'], tilt=t['tilt_deg']))
    print(f'  {si + 1}/{len(SITES)} {name}', flush=True)

df = pd.DataFrame(rows)
p = os.path.join(A.dir, f'plume_trace_test_{A.tag}{A.suffix}.csv')
df.to_csv(p, index=False)

print(f'\n{"tube":>5s} {"sep/w":>6s} {"amp":>5s} {"z_branch":>9s} {"found":>6s} '
      f'{"2 strands":>10s} {"median err":>11s} {"IQR":>7s} {"|err|<150":>10s}')
for cr in RAD:
    for sr in SEP:
        for amp in AMP:
          for zb in ZB:
            s = df[(df.cr == cr) & (df.amp == amp) & (df.zb == zb)
                   & (df.sep_ratio == sr)]
            if not len(s):
                continue
            f = s[s.found == True]
            hit = f[np.isfinite(f.recovered)] if len(f) else f
            two = int((f.max_strands >= 2).sum()) if len(f) else 0
            if not len(hit):
                print(f'{cr:5.0f} {sr:6.1f} {amp:5.2f} {zb:9.0f} '
                      f'{len(f):5d}/{len(s):<5d} {two:10d} '
                      f'{"":>11s} {"":>7s} {"":>10s}')
                continue
            e = hit.err.to_numpy(float)
            q1, q3 = np.percentile(e, [25, 75])
            print(f'{cr:5.0f} {sr:6.1f} {amp:5.2f} {zb:9.0f} '
                  f'{len(f):5d}/{len(s):<5d} {two:10d} {np.median(e):+11.0f} '
                  f'{q3 - q1:7.0f} {100 * np.mean(np.abs(e) < 150):9.0f}%')

print(f'\nambient branches per site: median {df.n_ambient.median():.0f}; '
      f'new branches attributable to the injection: median {df.n_new.median():.0f}')
ok = df[(df.found == True) & np.isfinite(df.recovered)]
if len(ok):
    e = ok.err.to_numpy(float)
    q1, q3 = np.percentile(e, [25, 75])
    print(f'\noverall: {len(ok)} recovered of {len(df)} injected; '
          f'median error {np.median(e):+.0f} km, IQR {q3 - q1:.0f} km, '
          f'{100 * np.mean(np.abs(e) < 150):.0f}% within 150 km')
    print('The stopping rule set before this was run: an IQR worse than about')
    print('150 km means 660 and 1000 km cannot be separated, and the depth')
    print('questions should not be attempted.')
print(f'\nwrote {p}', flush=True)
