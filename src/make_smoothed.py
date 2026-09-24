#!/usr/bin/env python3
"""Laterally smoothed copies of a tomographic model, as ordinary model files.

The question this serves is what the best-resolved model shows that a coarser one
cannot. Answering it by comparing against other people's models answers a
different question, because those models differ in data, in theory and in
regularisation all at once, and their disagreement with RevealLO is a premise
rather than a result. The controlled version is to degrade RevealLO itself: take
the same inversion, the same data and the same analysis, and remove only spatial
detail.

A structure that survives smoothing to 800 km was never RevealLO's contribution.
A structure that disappears is exactly what the fine model adds, and whether it
is worth believing is then decided by injection recovery in the same model, not
by whether a degree-20 parameterisation happens to carry it.

The output is a normal model file with a single Voigt shear velocity `vs`, so
every script downstream treats a smoothed copy as just another tag and nothing
else has to change. Shells are read, smoothed and written one at a time, so the
memory cost is one shell rather than one model - the native file is over five
gigabytes and holding two copies of it is not possible on a workstation.

Smoothing is the same longitude-periodic, latitude-widened Gaussian the contrast
field uses, so the length scale is in kilometres everywhere rather than in
degrees, and a 400 km kernel is 400 km at the equator and at sixty degrees.
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contrast import _smooth_shell
from tomo_io import _coord, _var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--var', default='voigt',
                    help="'voigt' to build the Voigt average from vsv and vsh, "
                         "or the name of a velocity variable in the file")
    ap.add_argument('--sigma', nargs='+', type=float, required=True,
                    help='smoothing lengths in km, one output file each')
    ap.add_argument('--out-dir', default=None, dest='out_dir',
                    help='default is beside the input file')
    ap.add_argument('--tag', default=None,
                    help='base name of the outputs; default is the input stem')
    ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
    ap.add_argument('--every', type=int, default=1,
                    help='thin the depth axis, to trade detail for disk')
    a = ap.parse_args()

    ds = xr.open_dataset(a.file)
    latn, lonn = _coord(ds, ('latitude', 'lat')), _coord(ds, ('longitude', 'lon'))
    dnm = _coord(ds, ('depth', 'depth_km', 'radius_nondim'))
    z = np.asarray(ds[dnm].values, float)
    if np.nanmax(np.abs(z)) > 2e4:
        z = z / 1000.0
    keep = np.where(z <= a.depth_max)[0]
    keep = keep[np.argsort(z[keep])][::a.every]
    keep = np.sort(keep)
    lat = np.asarray(ds[latn].values, float)
    lon = np.asarray(ds[lonn].values, float)
    dlat, dlon = float(abs(lat[1] - lat[0])), float(abs(lon[1] - lon[0]))
    stem = a.tag or os.path.splitext(os.path.basename(a.file))[0]
    outdir = a.out_dir or os.path.dirname(os.path.abspath(a.file))
    os.makedirs(outdir, exist_ok=True)
    print(f'{a.file}: {len(keep)} of {len(z)} shells, {len(lat)}x{len(lon)}',
          flush=True)

    for sig in a.sigma:
        out = os.path.join(outdir, f'{stem}_s{int(round(sig))}.nc')
        if os.path.exists(out):
            print(f'  {os.path.basename(out)} exists, skipping')
            continue
        buf = np.empty((len(keep), len(lat), len(lon)), np.float32)
        for q, k in enumerate(keep):
            if a.var == 'voigt':
                vsv = _var(ds, 'vsv', a.file).isel({dnm: int(k)}).values
                vsh = _var(ds, 'vsh', a.file).isel({dnm: int(k)}).values
                v = np.sqrt((2.0 * np.asarray(vsv, float) ** 2
                             + np.asarray(vsh, float) ** 2) / 3.0)
            else:
                v = np.asarray(_var(ds, a.var, a.file).isel({dnm: int(k)}).values,
                               float)
            if v.shape != (len(lat), len(lon)):
                v = v.T
            # The shell mean is removed before smoothing and added back after, so
            # that the kernel acts on the anomaly and cannot bleed the radial
            # profile across depths. Smoothing the absolute velocity would give
            # the same result here only because the mean is laterally constant,
            # and would not if the file were ever cropped in latitude.
            m = float(np.nanmean(v))
            s = _smooth_shell(v - m, lat, dlat, dlon, float(sig))
            buf[q] = (np.where(np.isfinite(s), s, 0.0) + m).astype(np.float32)
            if q % 40 == 0:
                print(f'    sigma {sig:.0f} km: shell {q + 1}/{len(keep)} '
                      f'at {z[k]:.0f} km', flush=True)
        xr.Dataset(dict(vs=((dnm, latn, lonn), buf)),
                   coords={dnm: z[keep], latn: lat, lonn: lon},
                   attrs=dict(source=os.path.basename(a.file),
                              smoothing_km=float(sig),
                              note='laterally smoothed copy; Voigt shear velocity')
                   ).to_netcdf(out, encoding={'vs': dict(zlib=True, complevel=1)})
        del buf
        print(f'  wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB)', flush=True)
    ds.close()


if __name__ == '__main__':
    main()
