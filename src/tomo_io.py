"""Loading a tomographic model as a lateral-mean-removed per cent anomaly.

The same convention as generate_cross_sections.py in the GPlately-pyGMT
tutorials, so the numbers here and the cross-sections plotted from the same file
refer to the same quantity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

R_E = 6371.0


@dataclass(frozen=True)
class ModelSpec:
    name: str
    velocity_var: str = 'voigt'      # 'voigt' from vsv and vsh, or a variable name


def _var(ds, name, path):
    """Fetch a variable by name, ignoring case.

    Distributions of the same model differ in capitalisation - REVEAL ships vsv
    and vsh, the full RevealLO file ships VSV and VSH - and a case-sensitive
    lookup fails on one of them for no reason that matters.
    """
    if name in ds:
        return ds[name]
    lower = {str(k).lower(): k for k in ds.variables}
    if name.lower() in lower:
        return ds[lower[name.lower()]]
    raise SystemExit(f"variable '{name}' not in {path}; "
                     f"present: {list(ds.data_vars)}")


def _coord(ds, names):
    lower = {str(k).lower(): k for k in list(ds.coords) + list(ds.variables)}
    for n in names:
        if n in lower:
            return lower[n]
    raise SystemExit(f'no coordinate among {names} in the file; '
                     f'present: {list(ds.coords)}')


def load_anomaly(path, spec: ModelSpec, depth_max=None, every=1):
    """Return depth (km, increasing), latitude, longitude and the anomaly cube.

    depth_max and every crop and thin the depth axis before anything is read, so
    a file that samples the whole Earth radius at 10 km does not have to be held
    in memory in order to analyse the mantle. Both are applied to the depth
    coordinate, which is small, and the data variables are then read only for the
    shells that survive.
    """
    ds = xr.open_dataset(path)
    latn = _coord(ds, ('latitude', 'lat'))
    lonn = _coord(ds, ('longitude', 'lon'))

    dnm = _coord(ds, ('depth', 'depth_km', 'radius_nondim'))
    zc = np.asarray(ds[dnm].values, float)
    if np.nanmax(np.abs(zc)) > 2e4:            # metres, not kilometres
        zc = zc / 1000.0
    keep = np.arange(len(zc))
    if depth_max is not None:
        keep = keep[zc[keep] <= float(depth_max)]
    if every > 1:
        keep = keep[np.argsort(zc[keep])][::every]
    if len(keep) != len(zc):
        ds = ds.isel({dnm: np.sort(keep)})

    if spec.velocity_var == 'voigt':
        v = np.sqrt((2.0 * _var(ds, 'vsv', path) ** 2
                     + _var(ds, 'vsh', path) ** 2) / 3.0)
    else:
        v = _var(ds, spec.velocity_var, path)

    mean_layer = v.mean(dim=[latn, lonn])
    a = ((v - mean_layer) * 100.0 / mean_layer).clip(min=-100, max=100)

    dname = dnm if dnm in a.dims else [d for d in a.dims
                                       if d not in (latn, lonn)][0]
    depth = np.asarray(ds[dname].values, float)

    lat = np.asarray(ds[latn].values, float)
    lon = np.asarray(ds[lonn].values, float)
    arr = a.transpose(dname, latn, lonn).values.astype(np.float32)
    ds.close()

    # A depth axis in kilometres cannot exceed the Earth's radius, so a maximum
    # above about 2e4 means the file is in metres whatever the variable is named.
    if np.nanmax(np.abs(depth)) > 2e4:
        depth = depth / 1000.0
    if depth[0] > depth[-1]:
        order = np.argsort(depth)
        depth, arr = depth[order], arr[order]
    if lon.max() > 180.0:                      # 0-360 files
        roll = int(np.sum(lon > 180.0))
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        lon = np.roll(lon, roll)
        arr = np.roll(arr, roll, axis=2)
    return depth, lat, lon, arr


def dedupe_lon(lon, arr):
    """Drop a repeated wrap meridian.

    Several distributions sample longitude from -180 to +180 inclusive, so the
    first and last columns are the same meridian. The path search rolls the
    longitude axis to close the sphere, which would otherwise insert a
    zero-width step at the seam and let a path cross it for nothing.
    """
    lon = np.asarray(lon, float)
    if abs((float(lon[-1]) - float(lon[0])) - 360.0) < 1e-6:
        return lon[:-1], arr[:, :, :-1]
    return lon, arr


def check_shells(depth, arr, tag, z0=300.0, ratio=4.0):
    """Refuse a model whose depth shells are not internally consistent.

    A shell that has been assembled from more than one depth range has its
    lateral mean taken over velocities that do not belong together, and its
    anomalies are correspondingly enormous. The test is relative to the model
    itself rather than to an absolute per cent, because models differ several
    fold in the amplitude of their heterogeneity: each shell below z0 is
    summarised by the 99.9th percentile of the absolute anomaly, and a shell is
    flagged if that exceeds `ratio` times the median over all such shells.
    Healthy models sit between 1.6 and 2.3; a file we had mistakenly assembled
    from four depth ranges reached 13. The failure is silent and severe, since
    one spurious minimum makes a path through that cell almost free, so it is
    checked rather than assumed.
    """
    z = np.asarray(depth, float)
    k = np.where(z >= z0)[0]
    if len(k) < 5:
        return
    q = np.array([np.nanpercentile(np.abs(arr[i]), 99.9) for i in k])
    med = float(np.median(q))
    bad = k[q > ratio * med]
    if len(bad):
        print(f'{tag}: {len(bad)} depth shells are inconsistent with the rest '
              f'of the model (median shell spread {med:.2f} per cent):')
        for i in bad[:10]:
            print(f'    {z[i]:7.0f} km  spread {np.nanpercentile(np.abs(arr[i]), 99.9):6.2f} '
                  f'per cent  range {np.nanmin(arr[i]):+.1f} to {np.nanmax(arr[i]):+.1f}')
        raise SystemExit(f'{tag}: depth shells are not internally consistent; '
                         'check how the model file was assembled')
    print(f'  shell consistency check passed ({len(z)} shells, '
          f'worst shell {q.max() / med:.1f}x the median spread)', flush=True)
