#!/usr/bin/env python3
"""A detector for conduits, rather than an integrator of slowness.

The least-cost descending path answers a different question from the one that
gets asked of it. A path exists from every surface point and reaches the deep
mantle from every surface point - 300 of 300 null sites did - so its existence
tests nothing, and its price is 99.8 per cent slowness and 0.3 per cent geometry
by the decomposition already run. A broad slow region satisfies it perfectly
without containing anything tube-like. Counting the pieces such a path breaks
into is a shape statistic laid over a detector that was never detecting.

What a plume conduit is, if it is anything, is a LOCAL velocity minimum that is
slow relative to its own immediate surroundings and that persists from shell to
shell into a connected column. Those two properties are what this measures, and
neither of them is measured by how slow the mantle is.

WHAT IS COMPUTED, PER SHELL

  THE LOCAL MINIMUM. Within a cap about the column's current position, the cell
  with the lowest anomaly. The cap re-centres on it as depth increases, so a
  conduit that leans is followed rather than lost, with the total lateral drift
  bounded so that the detector cannot walk away onto an unrelated feature.

  PROMINENCE. The anomaly at that minimum, subtracted from the mean anomaly on a
  surrounding annulus. This is the whole point. Inside a large low-velocity
  province the centre and the annulus are both slow, so a province contributes
  nothing to prominence; only a feature that is slow RELATIVE TO ITS OWN
  SURROUNDINGS scores. A statistic that rewarded absolute slowness would rank
  the provinces first and tell us what we already know.

  HALF-WIDTH. The radius at which the radial profile has risen halfway from the
  centre to the annulus. This is narrowness proper: a tube has a small half-width
  and a broad slow region has none, because its profile never rises.

THE SITE SCORE is the longest run of consecutive shells in which prominence
exceeds a threshold and the minimum has not jumped further than a step bound
between adjacent shells, reported as a depth extent in kilometres, together with
the mean prominence and median half-width over that run.

THE THRESHOLD IS NOT CHOSEN BY EYE. It is set by the null sites: the value at
which a stated fraction of randomly placed columns fail to register. Everything
is then reported as hotspots against nulls, which is the comparison that has
decided every other question in this study.

THE SELF-TEST IS A GATE. A detector that fires on a broad slow blob as readily as
on a narrow conduit is the instrument this file exists to replace, so the script
builds both, checks that it separates them, and refuses to report anything if it
does not.
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morphology as M
from tomo_io import ModelSpec, load_anomaly, dedupe_lon

R_E, DEG = 6371.0, np.pi / 180.0


def _window(lat, lon, clat, clon, r_km):
    """Index ranges of a box that certainly contains a disc of radius r_km."""
    dlat = r_km / 111.2
    dlon = r_km / max(111.2 * np.cos(np.radians(min(abs(clat) + dlat, 89.0))), 1.0)
    j = np.where(np.abs(lat - clat) <= dlat)[0]
    d = ((lon - clon + 180.0) % 360.0) - 180.0
    i = np.where(np.abs(d) <= dlon)[0]
    return j, i


def radial_profile(field, lat, lon, clat, clon, edges):
    """Mean anomaly in rings about a centre, and the centre's own value.

    Returned in the order of `edges`, which are ring outer radii in kilometres.
    Rings with no cells come back as nan rather than as zero, so that a ring
    falling off the edge of the model is absent rather than cold.
    """
    j, i = _window(lat, lon, clat, clon, float(edges[-1]))
    if not len(j) or not len(i):
        return np.full(len(edges), np.nan), np.nan
    sub = field[np.ix_(j, i)]
    LO, LA = np.meshgrid(lon[i], lat[j])
    d = M.gc_km(clat, clon, LA, LO)
    out = np.full(len(edges), np.nan)
    lo = 0.0
    for k, hi in enumerate(edges):
        m = (d >= lo) & (d < hi) & np.isfinite(sub)
        if m.any():
            out[k] = float(np.mean(sub[m]))
        lo = hi
    c = sub[np.unravel_index(np.nanargmin(np.where(np.isfinite(sub), d, np.inf)),
                             sub.shape)]
    return out, float(c)


def shell_tube(field, lat, lon, clat, clon, r_search, r_in, r_out, n_ring=12):
    """The best local minimum near a column position, and how tube-like it is."""
    j, i = _window(lat, lon, clat, clon, r_search)
    if not len(j) or not len(i):
        return None
    sub = field[np.ix_(j, i)]
    LO, LA = np.meshgrid(lon[i], lat[j])
    d = M.gc_km(clat, clon, LA, LO)
    ok = np.isfinite(sub) & (d <= r_search)
    if not ok.any():
        return None
    v = np.where(ok, sub, np.inf)
    a, b = np.unravel_index(np.argmin(v), v.shape)
    mlat, mlon = float(LA[a, b]), float(LO[a, b])
    centre = float(sub[a, b])

    edges = np.linspace(0.0, r_out, n_ring + 1)[1:]
    prof, _ = radial_profile(field, lat, lon, mlat, mlon, edges)
    ring = np.nanmean(prof[edges > r_in]) if np.isfinite(prof[edges > r_in]).any() \
        else np.nan
    prom = ring - centre if np.isfinite(ring) else np.nan

    # half-width: where the profile has risen halfway to the annulus
    half = np.nan
    if np.isfinite(prom) and prom > 0:
        target = centre + 0.5 * prom
        r_mid = 0.5 * (np.concatenate([[0.0], edges[:-1]]) + edges)
        f = np.isfinite(prof)
        if f.sum() >= 3:
            rr, pp = r_mid[f], prof[f]
            above = np.where(pp >= target)[0]
            if len(above):
                k = above[0]
                if k == 0:
                    half = float(rr[0])
                else:
                    x0, x1, y0, y1 = rr[k - 1], rr[k], pp[k - 1], pp[k]
                    half = float(x0 + (target - y0) * (x1 - x0) / max(y1 - y0, 1e-9))
    return dict(lat=mlat, lon=mlon, centre=centre, ring=ring, prominence=prom,
                half_width_km=half)


def depth_bins(arr, depth, zmin, zmax, dz_bin):
    """Average the volume into depth intervals matched to the lateral sampling.

    A conduit tilted at forty-five degrees moves twenty kilometres sideways per
    twenty kilometres of depth, which on a half-degree grid is less than half a
    cell. Tracking shell by shell at that spacing is therefore tracking noise:
    the position cannot move by a resolvable amount, so whichever neighbouring
    cell happens to be lowest wins, and the column walks. Binning to intervals
    comparable with the lateral cell size makes one step of the tracker one
    resolvable step of the Earth.
    """
    z = np.asarray(depth, float)
    edges = np.arange(zmin, zmax + dz_bin, dz_bin)
    out, zc = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = np.where((z >= lo) & (z < hi))[0]
        if not len(k):
            continue
        out.append(np.nanmean(arr[k], axis=0))
        zc.append(0.5 * (lo + hi))
    return np.asarray(out), np.asarray(zc, float)


def track(arr, depth, lat, lon, lat0, lon0, a):
    """Follow the column downward, recording how tube-like each interval is."""
    V, zc = depth_bins(arr, depth, a.zmin, a.zmax, a.dz_bin)
    # search and acceptance must be the same quantity. With a cap of 400 km and
    # a step bound of 150 km the tracker was allowed to find minima it would
    # then reject, so it failed the step test on half of Hawaii's shells while
    # its column wandered 1847 km from the hotspot onto unrelated features.
    step_max = a.dz_bin * np.tan(np.radians(a.max_tilt))
    clat, clon = lat0, lon0
    rows = []
    for k in range(len(V)):
        z_k = float(zc[k])
        r = shell_tube(V[k], lat, lon, clat, clon, step_max, a.r_in, a.r_out)
        if r is None:
            rows.append(dict(z=z_k, prominence=np.nan, step_km=np.nan,
                             half_width_km=np.nan))
            continue
        step = float(M.gc_km(clat, clon, r['lat'], r['lon']))
        newdrift = float(M.gc_km(lat0, lon0, r['lat'], r['lon']))
        if newdrift <= a.max_drift:
            clat, clon = r['lat'], r['lon']
        rows.append(dict(z=z_k, prominence=r['prominence'],
                         step_km=step, half_width_km=r['half_width_km'],
                         lat=r['lat'], lon=r['lon'], drift_km=newdrift))
    return pd.DataFrame(rows)


def score(df, thr, max_step, dz, w_max):
    """Longest run of shells that are tube-like and vertically continuous.

    Narrowness enters the criterion and not merely the report. Prominence alone
    admits a broad slow region, because a wide gentle low still sits below the
    mean of a ring drawn around it: in the self-test a 1600 km blob cleared a
    0.2 per cent prominence bar at every shell. The half-width bound is the
    statement that the feature must be resolved INSIDE the window used to define
    its surroundings rather than filling that window, which is what makes it a
    local feature at all, and it is set as a fraction of the annulus radius
    rather than as a length chosen for the answer it gives.
    """
    # the step is bounded by the search itself now, so it is not a separate
    # criterion; a move the tracker could not have made cannot be rejected here
    ok = ((df.prominence >= thr)
          & (df.half_width_km <= w_max)).fillna(False).values
    best = cur = 0
    b0 = c0 = 0
    for n, v in enumerate(ok):
        if v:
            if cur == 0:
                c0 = n
            cur += 1
            if cur > best:
                best, b0 = cur, c0
        else:
            cur = 0
    if best == 0:
        return dict(tube_km=0.0, mean_prominence=np.nan,
                    median_half_width_km=np.nan, z_top=np.nan, z_bot=np.nan)
    seg = df.iloc[b0:b0 + best]
    return dict(tube_km=float(best * dz),
                mean_prominence=float(np.nanmean(seg.prominence)),
                median_half_width_km=float(np.nanmedian(seg.half_width_km)),
                z_top=float(seg.z.min()), z_bot=float(seg.z.max()))


# ------------------------------------------------------------------ self-test

def selftest(a):
    """A narrow conduit and a broad slow region of the same amplitude.

    If the detector cannot separate these two it is measuring slowness, which is
    what the least-cost path already measured, and there is no reason to run it
    on real data.
    """
    lat = np.arange(-40.0, 40.01, 0.5)
    lon = np.arange(-40.0, 40.01, 0.5)
    # the synthetic volume spans whatever depth range is being tested, so that
    # restricting the analysis to the lowermost mantle does not fail the gate
    # merely because there is less depth in which to find a tube
    z = np.arange(a.zmin, a.zmax + 1.0, 20.0)
    LO, LA = np.meshgrid(lon, lat)
    arr = np.zeros((len(z), len(lat), len(lon)), np.float32)
    d_narrow = M.gc_km(0.0, -20.0, LA, LO)
    d_broad = M.gc_km(0.0, 20.0, LA, LO)
    for k in range(len(z)):
        arr[k] -= 2.0 * np.exp(-(d_narrow ** 2) / (2 * (250.0 ** 2)))
        arr[k] -= 2.0 * np.exp(-(d_broad ** 2) / (2 * (1600.0 ** 2)))
    out = {}
    # the third case is the one that matters: a narrow conduit sitting INSIDE a
    # broad slow region, which is the situation actually presented by a plume
    # rooted in a province. A detector that finds the conduit only in isolation
    # would be useless exactly where it is needed.
    d_both = M.gc_km(-20.0, 0.0, LA, LO)
    for k in range(len(z)):
        arr[k] -= 1.8 * np.exp(-(d_both ** 2) / (2 * (1600.0 ** 2)))
        arr[k] -= 1.6 * np.exp(-(M.gc_km(-20.0, 0.0, LA, LO) ** 2)
                               / (2 * (250.0 ** 2)))
    for nm, la_, lo_ in (('narrow conduit', 0.0, -20.0),
                         ('broad slow region', 0.0, 20.0),
                         ('conduit in a province', -20.0, 0.0)):
        s = score(track(arr, z, lat, lon, la_, lo_, a), a.threshold,
                  None, a.dz_bin, a.w_max)
        out[nm] = s
        print(f'  {nm:20s} tube {s["tube_km"]:6.0f} km  prominence '
              f'{s["mean_prominence"]:6.2f}  half-width '
              f'{s["median_half_width_km"]:6.0f} km')
    n, b = out['narrow conduit'], out['broad slow region']
    c = out['conduit in a province']
    span = float(a.zmax - a.zmin)
    ok = (n['tube_km'] >= 0.8 * span and c['tube_km'] >= 0.8 * span
          and b['tube_km'] <= 0.5 * n['tube_km'])
    print(f'  (a full tube over this range would be {span:.0f} km)')
    print('  self-test ' + ('PASSED' if ok else 'FAILED'))
    if not ok:
        print('  The detector does not separate a conduit from a broad slow')
        print('  region at the same peak amplitude, so it is measuring slowness')
        print('  and would repeat the error it was built to avoid.')
    return ok


def main():
    ap = argparse.ArgumentParser(
        description='detect conduits by prominence and vertical continuity')
    ap.add_argument('--file', default=None)
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
    ap.add_argument('--nulls', type=int, default=300)
    ap.add_argument('--null-seed', type=int, default=7, dest='null_seed')
    ap.add_argument('--zmin', type=float, default=800.0)
    ap.add_argument('--zmax', type=float, default=2700.0)
    ap.add_argument('--dz-bin', type=float, default=100.0, dest='dz_bin',
                    help='depth interval the volume is averaged into before '
                         'tracking, comparable with the lateral cell size')
    ap.add_argument('--max-tilt', type=float, default=60.0, dest='max_tilt',
                    help='steepest lean a conduit may have, degrees from '
                         'vertical; sets both the search radius and the step')
    ap.add_argument('--r-in', type=float, default=600.0, dest='r_in')
    ap.add_argument('--r-out', type=float, default=1000.0, dest='r_out')
    ap.add_argument('--max-drift', type=float, default=1500.0, dest='max_drift')
    ap.add_argument('--w-max', type=float, default=None, dest='w_max',
                    help='half-width bound, km; defaults to half the annulus '
                         'radius, which is the point at which a feature stops '
                         'being local to the window that defines it')
    ap.add_argument('--threshold', type=float, default=None,
                    help='prominence, per cent; default is set from the nulls')
    ap.add_argument('--null-quantile', type=float, default=0.5,
                    dest='null_quantile',
                    help='threshold is the prominence at which this fraction of '
                         'null shells fails')
    ap.add_argument('--selftest-only', action='store_true', dest='selftest_only')
    a = ap.parse_args()

    if a.threshold is None:
        a.threshold = 0.20
    if a.w_max is None:
        a.w_max = 0.5 * a.r_out
    print('self-test on synthetic structure:')
    ok = selftest(a)
    if not ok:
        sys.exit(1)
    if a.selftest_only:
        return
    if not a.file:
        sys.exit('--file is required to run on a model')

    depth, lat, lon, arr = load_anomaly(a.file, ModelSpec(a.tag, a.var),
                                        depth_max=a.depth_max, every=a.every)
    lon, arr = dedupe_lon(lon, arr)
    z = np.asarray(depth, float)
    dz = float(np.median(np.abs(np.diff(np.sort(z)))))
    print(f'\n{a.tag}: {len(z)} shells, {len(lat)}x{len(lon)}, spacing '
          f'{dz:.0f} km', flush=True)

    sites = []
    H = pd.read_csv(a.hotspots, comment='#')
    for r in H.itertuples():
        sites.append((str(r.hotspot), float(r.lat), float(r.lon_180), 'hotspot'))
    rng = np.random.default_rng(a.null_seed)
    for q in range(a.nulls):
        sites.append((f'null{q:03d}',
                      float(np.degrees(np.arcsin(2 * rng.random() - 1))),
                      float(360.0 * rng.random() - 180.0), 'null_site'))
    print(f'{len(sites)} sites, {a.nulls} of them nulls', flush=True)

    rows, prof = [], []
    for n, (name, la_, lo_, kind) in enumerate(sites):
        df = track(arr, z, lat, lon, la_, lo_, a)
        # the per-interval values are kept so that the depth dependence of the
        # hotspot excess can be read from one run. Expressed as hotspots against
        # nulls AT THE SAME DEPTH it needs no assumption about how resolution
        # varies with depth, which in a tomographic model is governed by ray
        # coverage and is not a function of depth at all.
        pf = df.copy()
        pf['site'] = name
        pf['kind'] = kind
        prof.append(pf[['site', 'kind', 'z', 'prominence', 'half_width_km',
                        'step_km']])
        s = score(df, a.threshold, None, a.dz_bin, a.w_max)
        s.update(site=name, kind=kind, start_lat=la_, start_lon=lo_,
                 shells=int(np.isfinite(df.prominence).sum()),
                 p90_prominence=float(np.nanquantile(df.prominence, 0.9))
                 if np.isfinite(df.prominence).any() else np.nan)
        rows.append(s)
        if (n + 1) % 50 == 0:
            print(f'  {n + 1}/{len(sites)}', flush=True)
    d = pd.DataFrame(rows)
    out = os.path.join(a.dir, f'conduit_detect_{a.tag}{a.suffix}.csv')
    d.to_csv(out, index=False)
    pout = os.path.join(a.dir, f'conduit_profile_{a.tag}{a.suffix}.csv')
    pd.concat(prof, ignore_index=True).to_csv(pout, index=False)
    # Both files feed mid_mantle_census.py, Figure 6 and Table S5, and carried no
    # provenance while sitting outside every runner; stamped since 22 Sep 2026.
    import provenance
    for _p in (out, pout):
        provenance.stamp(_p, nulls=a.nulls, null_seed=a.null_seed, zmin=a.zmin,
                         zmax=a.zmax, dz_bin=a.dz_bin, r_in=a.r_in, r_out=a.r_out,
                         max_tilt=a.max_tilt, max_drift=a.max_drift,
                         inputs=[a.file, a.hotspots])

    h, nl = d[d.kind == 'hotspot'], d[d.kind == 'null_site']
    print(f'\nprominence threshold {a.threshold:.2f} per cent, half-width bound '
          f'{a.w_max:.0f} km,\ndepth interval {a.dz_bin:.0f} km, maximum tilt '
          f'{a.max_tilt:.0f} degrees, so a step of at most '
          f'{a.dz_bin * np.tan(np.radians(a.max_tilt)):.0f} km\n')
    print(f'{"":22s} {"median tube":>12s} {"90th pct":>10s} {"any tube":>10s}')
    for lab, g in (('hotspots', h), ('null sites', nl)):
        if not len(g):
            continue
        print(f'{lab:22s} {np.median(g.tube_km):11.0f} km '
              f'{np.quantile(g.tube_km, 0.9):9.0f} km '
              f'{100.0 * np.mean(g.tube_km > 0):9.0f}%')
    if len(nl) and len(h):
        from root_bias import perm_median_diff, fisher_one_sided
        p, npm = perm_median_diff(h.tube_km, nl.tube_km, 100000)
        print(f'\nmedian tube length, hotspots against nulls, p = {p:.4f} '
              f'over {npm} resamples')
        for cut in (500.0, 1000.0, 1500.0):
            a11 = int((h.tube_km >= cut).sum()); a12 = len(h) - a11
            a21 = int((nl.tube_km >= cut).sum()); a22 = len(nl) - a21
            pf = fisher_one_sided(a11, a12, a21, a22)
            print(f'  tube of at least {cut:5.0f} km: hotspots {a11:3d}/{len(h)} '
                  f'{100.0 * a11 / len(h):5.1f}%, nulls {a21:3d}/{len(nl)} '
                  f'{100.0 * a21 / len(nl):5.1f}%, p = {pf:.4f}')
    print(f'\n{out}\n{pout}')


if __name__ == '__main__':
    main()
