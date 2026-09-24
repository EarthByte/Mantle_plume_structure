"""Which depth slice shows the conduits best, and the slice itself.

The map figure needs a tomographic background, and the depth to draw it at is a
choice that should be made from the recovered conduits rather than by eye. For
every depth level this samples the anomaly at the position each conduit occupies
at that depth, and expresses it as a percentile of the global distribution in the
same shell, so shells of different amplitude are comparable. The depth where that
percentile is lowest is where the conduits stand out most clearly against
everything else at their own depth.

    python3 plume_pipeline/conduit_depth_slice.py \
        --file ../REVEAL_mantle_tomography/RevealLO.nc --tag RevealLO

Writes out/conduit_depth_expression_<tag>.csv, prints the ranking, and saves the
chosen shell as out/slice_<tag>_<z>km.nc for the map figure. --depth fixes the
shell instead of choosing it, once the choice has been made once.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tomo_io

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True, help='the model netCDF')
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--var', default='voigt')
ap.add_argument('--dir', default='out', help='where the results live')
ap.add_argument('--zmin', type=float, default=100.0)
ap.add_argument('--zmax', type=float, default=800.0,
                help='the search is confined to the upper mantle and transition zone')
ap.add_argument('--depth', type=float, default=None,
                help='skip the search and extract this shell')
ap.add_argument('--rooted-only', action='store_true',
                help='use only conduits with root_fraction >= 0.5')
a = ap.parse_args()

paths = os.path.join(a.dir, f'conduit_paths_all_{a.tag}.json')
if not os.path.exists(paths):
    paths = os.path.join('results', f'conduit_paths_all_{a.tag}.json')
if not os.path.exists(paths):
    raise SystemExit(f'\nno conduit paths for {a.tag}; looked in {a.dir}/ and results/\n')
P = json.load(open(paths))
if a.rooted_only:
    P = {k: v for k, v in P.items() if float(v.get('root_fraction', 0)) >= 0.5}
    if not P:
        raise SystemExit('\nno conduit has root_fraction >= 0.5; drop --rooted-only\n')
print(f'{len(P)} conduits from {paths}')

spec = tomo_io.ModelSpec(name=a.tag, velocity_var=a.var)
dep, lat, lon, cube = tomo_io.load_anomaly(a.file, spec, depth_max=a.zmax + 50.0)
lon180 = ((np.asarray(lon, float) + 180.0) % 360.0) - 180.0
order = np.argsort(lon180)
lon180, cube = lon180[order], cube[:, :, order]
print(f'{a.tag}: {len(dep)} shells {dep.min():.0f}-{dep.max():.0f} km, '
      f'{len(lat)}x{len(lon180)} cells')


def at(iz, la, lo):
    """Nearest cell on a regular grid, which is rounding rather than a search."""
    j = np.abs(np.asarray(lat, float)[None, :] - np.asarray(la)[:, None]).argmin(1)
    i = np.abs(lon180[None, :] - np.asarray(lo)[:, None]).argmin(1)
    return cube[iz][j, i]


rows = []
keep = (dep >= a.zmin) & (dep <= a.zmax)
for iz in np.where(keep)[0]:
    z = float(dep[iz])
    la, lo = [], []
    for name, v in P.items():
        zz = np.asarray(v['depth'], float)
        if zz.min() > z or zz.max() < z:
            continue
        # the conduit's horizontal position where it crosses this shell
        k = np.argsort(zz)
        la.append(np.interp(z, zz[k], np.asarray(v['lat'], float)[k]))
        lo_ = np.interp(z, zz[k], ((np.asarray(v['lon'], float)[k] + 180) % 360) - 180)
        lo.append(lo_)
    if len(la) < 5:
        continue
    vals = at(iz, la, lo)
    shell = cube[iz][np.isfinite(cube[iz])]
    if shell.size == 0:
        continue
    pct = np.array([100.0 * (shell < v).mean() for v in vals if np.isfinite(v)])
    rows.append(dict(depth_km=z, n=len(pct), median_pct=float(np.median(pct)),
                     mean_anom=float(np.nanmean(vals)),
                     shell_sd=float(np.nanstd(shell))))

if not rows:
    raise SystemExit('\nno shell had five conduits crossing it in the search range\n')
t = pd.DataFrame(rows).sort_values('depth_km')
os.makedirs(a.dir, exist_ok=True)
f = os.path.join(a.dir, f'conduit_depth_expression_{a.tag}.csv')
t.to_csv(f, index=False)

best = float(t.loc[t.median_pct.idxmin(), 'depth_km']) if a.depth is None else a.depth
print(f'\n{"depth":>8} {"n":>4} {"median pct":>11} {"mean anom %":>12}')
for _, r in t.iterrows():
    mark = '  <-- lowest' if abs(r.depth_km - best) < 1e-6 else ''
    print(f'{r.depth_km:8.0f} {int(r.n):4d} {r.median_pct:11.1f} {r.mean_anom:12.3f}{mark}')
print(f'\nwrote {f}')
print(f'conduits are most clearly expressed at {best:.0f} km, where the median '
      f'conduit sits at the {t.median_pct.min():.0f}th percentile of that shell')

iz = int(np.abs(dep - best).argmin())
try:
    import xarray as xr
    # Several distributions carry both -180 and +180, which wrap to the same
    # meridian and leave a zero-width column; GMT then resamples the whole grid
    # onto a slightly wrong step and draws a white seam down the Pacific.
    _k = np.concatenate([[True], np.diff(lon180) > 0])
    _lon, _fld = lon180[_k], cube[iz][:, _k]
    _d = np.diff(_lon)
    if not np.allclose(_d, _d[0]):
        print('  warning: longitude step is still not constant after removing '
              'the duplicate meridian')
    da = xr.DataArray(_fld, coords=dict(lat=np.asarray(lat, float), lon=_lon),
                      dims=('lat', 'lon'), name='dvs',
                      attrs=dict(long_name='shear-velocity anomaly', units='per cent',
                                 depth_km=float(dep[iz]), model=a.tag))
    g = os.path.join(a.dir, f'slice_{a.tag}_{dep[iz]:.0f}km.nc')
    da.to_netcdf(g)
    print(f'wrote {g}')
except Exception as exc:
    print(f'could not write the slice: {exc.__class__.__name__}: {exc}')
