#!/usr/bin/env python3
"""Where RevealLO differs from REVEAL, depth by depth.

RevealLO is REVEAL updated with 3.5-hour seismograms at 60-200 s, which add major-arc
and multi-orbit surface waves, then merged back onto REVEAL with the lowermost mantle
tapered to REVEAL from 2300 km down (Schiller et al., 2026; Supporting Information Text S10). That
description says what data were added; this measures what the update changed in the model.

At every depth of REVEAL's own sampling (99 shells, 0-2880 km), both models are reduced
to the anomaly the rest of this study uses - Voigt Vs, per cent departure from the
lateral mean of the shell, the convention of tomo_io.load_anomaly - and compared over
the sphere with cos(latitude) area weights. The convention is not re-implemented on
trust: REVEAL is read both here and through load_anomaly, and the run stops if the two
disagree.

One row per depth in out/model_update_<new>_<old>.csv:
  depth_km               shell depth
  mean_new_kms,          lateral mean Voigt Vs of each model (km/s), and the change in
  mean_old_kms, dmean_pct  that mean in per cent - the 1D part of the update, e.g. where
                         the update smoothed the 220, 400 and 670 km discontinuities
  rms_new, rms_old       area-weighted RMS anomaly (%) of each model
  rms_diff               area-weighted RMS of the anomaly difference new - old (%)
  ratio                  rms_diff / rms_old: the update as a share of the anomaly field
  corr                   area-weighted correlation of the two anomaly fields
  rms_diff_south,        rms_diff over the Southern and Northern hemispheres, which
  rms_diff_north         tests the report's claim that the added data improve the south
  area_valid             share of the sphere (area-weighted) where both models are finite;
                         RevealLO.nc has NaN cells in most of the 0 and 10 km shells and in
                         0.01-1% of the shells at 830-980 km, left out of the statistics

RevealLO.nc stores the whole Earth at 10 km in (lat, lon, depth) order, 5 GB, so it is
read in latitude blocks at REVEAL's depths only; nothing larger than a block is held.

  python3 model_update.py
  python3 model_update.py --new <file> --new-tag RevealLO --old <file> --old-tag REVEAL
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
import netCDF4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly
import provenance

ROOT = os.path.expanduser('~/mnt/Muller_mantle_tomography_subduction')
if not os.path.isdir(ROOT):
    ROOT = os.path.expanduser('~/Documents/Papers/in_prep/Muller_mantle_tomography_subduction')


def _var(ds, name):
    low = {k.lower(): k for k in ds.variables}
    return ds[low[name.lower()]]


def shells(path, depths_km, block=40):
    """Voigt Vs (km/s) at the requested depths, as (depth, lat, lon) in the file's own
    lat/lon order, plus the depth, lat and lon axes. Read in latitude blocks."""
    ds = netCDF4.Dataset(path)
    z = np.asarray(_var(ds, 'depth')[:], float)
    if np.nanmax(np.abs(z)) > 2e4:
        z = z / 1000.0
    idx = np.array([int(np.argmin(np.abs(z - d))) for d in depths_km])
    if not np.allclose(z[idx], depths_km, atol=0.5):
        raise SystemExit(f'{path}: requested depths are not all on its depth axis')
    lat = np.asarray(_var(ds, 'latitude')[:], float)
    lon = np.asarray(_var(ds, 'longitude')[:], float)
    vsv, vsh = _var(ds, 'vsv'), _var(ds, 'vsh')
    dims = [d.lower() for d in vsv.dimensions]
    out = np.empty((len(idx), len(lat), len(lon)), np.float64)
    if dims[-1] == 'depth':                       # (lat, lon, depth): block over lat
        for i0 in range(0, len(lat), block):
            i1 = min(len(lat), i0 + block)
            a = np.asarray(vsv[i0:i1, :, :])[:, :, idx]
            b = np.asarray(vsh[i0:i1, :, :])[:, :, idx]
            out[:, i0:i1, :] = np.moveaxis(np.sqrt((2 * a**2 + b**2) / 3.0), 2, 0)
    else:                                         # (depth, lat, lon)
        a = np.asarray(vsv[idx, :, :]); b = np.asarray(vsh[idx, :, :])
        out[:] = np.sqrt((2 * a**2 + b**2) / 3.0)
    ds.close()
    if np.nanmedian(out) > 100.0:                 # m/s, not km/s
        out = out / 1000.0
    return z[idx], lat, lon, out


def anomaly(v, lon):
    """tomo_io's convention: per cent from the unweighted lateral mean of the shell
    (taken over the file's own columns, a repeated wrap meridian included, as
    load_anomaly does), clipped at +-100; then rolled to -180..180 and the wrap
    meridian dropped."""
    m = np.nanmean(v, axis=(1, 2))               # xarray's mean skips NaN; so does this
    a = np.clip((v - m[:, None, None]) * 100.0 / m[:, None, None], -100, 100)
    lon = np.asarray(lon, float)
    if lon.max() > 180.0:
        roll = int(np.sum(lon > 180.0))
        lon = np.roll(np.where(lon > 180.0, lon - 360.0, lon), roll)
        a = np.roll(a, roll, axis=2)
    # Drop repeated meridians wherever they fall. A 0-360 file sampled at both ends
    # (RevealLO) repeats 0 deg in the middle after the roll, not at the ends, which
    # tomo_io.dedupe_lon does not catch; a -180..180 file (REVEAL) repeats 180 deg.
    lon = np.where(np.isclose(lon, 180.0), -180.0, lon)
    _, keep = np.unique(np.round(lon, 6), return_index=True)
    keep = keep[np.argsort(lon[keep])]           # ascending from -180
    return a[:, :, keep], lon[keep], m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', default=os.path.join(ROOT, 'REVEAL_mantle_tomography', 'RevealLO.nc'))
    ap.add_argument('--new-tag', default='RevealLO', dest='new_tag')
    ap.add_argument('--old', default=os.path.join(ROOT, 'REVEAL_vs_full.nc'))
    ap.add_argument('--old-tag', default='REVEAL', dest='old_tag')
    ap.add_argument('--dir', default='out')
    A = ap.parse_args()

    # REVEAL's own depth sampling is the common grid; RevealLO (10 km) contains it.
    with netCDF4.Dataset(A.old) as ds:
        zo = np.asarray(_var(ds, 'depth')[:], float)
    zo = zo / 1000.0 if np.nanmax(np.abs(zo)) > 2e4 else zo

    z, lat, lon_o, vo = shells(A.old, zo)
    _, lat_n, lon_n, vn = shells(A.new, zo)
    if not np.allclose(lat, lat_n):
        raise SystemExit('latitude axes differ')
    ao, lo, mo = anomaly(vo, lon_o)
    an, ln, mn = anomaly(vn, lon_n)
    if not np.allclose(lo, ln):
        raise SystemExit('longitude axes differ after rolling to -180..180')
    del vo, vn

    # The convention check: REVEAL through tomo_io must match REVEAL read here.
    zr, latr, lonr, ar = load_anomaly(A.old, ModelSpec(A.old_tag, 'voigt'))
    ar = ar[:, :, :-1] if np.isclose(lonr[-1] - lonr[0], 360.0) else ar   # wrap at 180
    if ar.shape != ao.shape or np.nanmax(np.abs(ar - ao)) > 1e-3:
        raise SystemExit('anomaly convention differs from tomo_io.load_anomaly')
    del ar

    # RevealLO.nc holds NaN cells: most of the 0 and 10 km shells, and 0.01-1% of the
    # shells at 830-980 km. A cell missing from either model is left out of every
    # statistic at its depth (zero weight), and the valid share is reported.
    ok = np.isfinite(an) & np.isfinite(ao)
    w = np.cos(np.radians(lat))[:, None, None].transpose(1, 0, 2) * ok
    an, ao = np.where(ok, an, 0.0), np.where(ok, ao, 0.0)
    south = (lat < 0)[None, :, None]
    wtot = np.cos(np.radians(lat)).sum() * len(lo)

    def wrms(x, mask=True):
        ww = w * mask
        return np.sqrt((x**2 * ww).sum(axis=(1, 2)) / ww.sum(axis=(1, 2)))

    def wcorr(x, y):
        sw = w.sum(axis=(1, 2))
        xm = (x * w).sum(axis=(1, 2)) / sw
        ym = (y * w).sum(axis=(1, 2)) / sw
        xc, yc = x - xm[:, None, None], y - ym[:, None, None]
        return (xc * yc * w).sum(axis=(1, 2)) / np.sqrt(
            (xc**2 * w).sum(axis=(1, 2)) * (yc**2 * w).sum(axis=(1, 2)))

    d = an - ao
    out = pd.DataFrame(dict(
        depth_km=z,
        mean_new_kms=mn, mean_old_kms=mo, dmean_pct=100.0 * (mn / mo - 1.0),
        rms_new=wrms(an), rms_old=wrms(ao), rms_diff=wrms(d),
        corr=wcorr(an, ao),
        rms_diff_south=wrms(d, south), rms_diff_north=wrms(d, ~south),
        area_valid=w.sum(axis=(1, 2)) / wtot))
    out['ratio'] = out.rms_diff / out.rms_old
    cols = ['depth_km', 'mean_new_kms', 'mean_old_kms', 'dmean_pct', 'rms_new', 'rms_old',
            'rms_diff', 'ratio', 'corr', 'rms_diff_south', 'rms_diff_north', 'area_valid']
    os.makedirs(A.dir, exist_ok=True)
    path = os.path.join(A.dir, f'model_update_{A.new_tag}_{A.old_tag}.csv')
    out[cols].to_csv(path + '.part', index=False, float_format='%.5g')
    os.replace(path + '.part', path)
    provenance.stamp(path, inputs=[A.new, A.old], new=A.new_tag, old=A.old_tag,
                     anomaly='voigt, % from unweighted shell mean (tomo_io)',
                     weights='cos(latitude)')
    print(f'wrote {path} ({len(out)} depths)')
    for zz in (100, 200, 400, 700, 1000, 1500, 2000, 2300, 2600, 2800):
        r = out.iloc[int(np.argmin(np.abs(out.depth_km - zz)))]
        print(f'  {r.depth_km:5.0f} km  diff {r.rms_diff:.3f}%  ratio {r.ratio:.2f}  '
              f'corr {r["corr"]:.3f}  S/N {r.rms_diff_south:.3f}/{r.rms_diff_north:.3f}  '
              f'dmean {r.dmean_pct:+.2f}%')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
