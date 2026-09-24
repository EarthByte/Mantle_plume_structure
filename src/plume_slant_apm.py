"""Absolute plate motion over each hotspot, and present-day subduction hinge migration.

This produces the plate-kinematic quantities the lean tests are measured AGAINST: the
time-averaged absolute plate velocity above each hotspot, as an azimuth and a speed over
several averaging windows, and the tessellated trench velocities that define the bearing
away from subduction.

It measures no lean of its own. A lean is a property of the traced conduits, and it is
measured from them over a stated depth window in apm_crossmodel.py (300-660 km, against
the reverse of plate motion) and in lean_confirm.py (660-1500 km, against the bearing
away from subduction). Both apply exclusion criteria fixed before any test statistic was
computed, which an alignment count cannot. An earlier version of this script carried a
lean azimuth, a misfit and an aligned-within-45-degrees count read from the slant table:
they duplicated the registered tests with a weaker statistic, and the lean_az column they
read is written by no producer, so this script could not run at all.

Keeping the file purely kinematic also closes a freshness hazard. Every column here now
depends on the plate model and the hotspot list alone, so the table cannot fall stale
against the traced paths, and no consumer can read a mixture of current and superseded
quantities out of one file.

Zahirovic et al. (2022) rather than Muller et al. (2025): the latter is a 1.8 Ga
deep-time model and is not the sharpest choice for Cenozoic kinematics, which is the
window that matters for a conduit's present shape.

The velocity is evaluated at the hotspot's FIXED position at each past time, not at
the reconstructed position of the plate that is above it today. The conduit sits
still in the mantle frame while plates pass over it, so the shear it feels is
whatever is overhead at each instant.

Hinge migration comes from the same reconstruction through tessellate_subduction_zones,
which returns a trench velocity and its azimuth per tessellated segment.
"""
from __future__ import annotations
import argparse, os, math, warnings
import numpy as np, pandas as pd
import provenance
warnings.filterwarnings('ignore')
import pygplates, gplately

R_E, DEG = 6371.0, np.pi / 180.0

ap = argparse.ArgumentParser()
ap.add_argument('--model', default=os.path.expanduser(
    '~/Documents/Papers/in_prep/Muller_mantle_tomography_subduction/Paper_hydration/'
    'Zahirovic2022_plate_model'))
ap.add_argument('--windows', default='10,30,50,80', help='averaging windows, Myr')
ap.add_argument('--step', type=float, default=5.0, help='sampling interval, Myr')
ap.add_argument('--hinge-time', type=float, default=0.0, dest='hinge_time')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--dir', default='out')
A = ap.parse_args()

rot = pygplates.RotationModel(os.path.join(A.model, 'CombinedRotations.rot'))
topo = [os.path.join(A.model, f) for f in
        ('Plate_Boundaries.gpml', 'Deforming_Networks_Active.gpml',
         'Feature_Geometries.gpml') if os.path.exists(os.path.join(A.model, f))]
stat = os.path.join(A.model, 'StaticGeometries', 'StaticPolygons',
                    'Global_EarthByte_GPlates_PresentDay_StaticPlatePolygons.shp')
recon = gplately.PlateReconstruction(rotation_model=rot, topology_features=topo,
                                     static_polygons=stat)
print(f'Zahirovic 2022 loaded: {len(topo)} topology files')

hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180']).set_index('hotspot')
sites = list(hs.index)
lons = np.array([float(hs.loc[s, 'lon_180']) for s in sites])
lats = np.array([float(hs.loc[s, 'lat']) for s in sites])

rows = []
for W in [float(x) for x in A.windows.split(',')]:
    times = np.arange(0.0, W + 1e-9, A.step)
    E = np.zeros(len(sites)); N = np.zeros(len(sites)); n_ok = np.zeros(len(sites))
    for t in times:
        try:
            e, n_ = recon.get_point_velocities(lons, lats, float(t), delta_time=A.step,
                                               return_east_north_arrays=True)
        except Exception:
            continue
        m = np.isfinite(e) & np.isfinite(n_)
        E[m] += e[m]; N[m] += n_[m]; n_ok[m] += 1
    ok = n_ok > 0
    E[ok] /= n_ok[ok]; N[ok] /= n_ok[ok]
    for i, site in enumerate(sites):
        if not ok[i]:
            continue
        az = math.degrees(math.atan2(E[i], N[i])) % 360.0
        spd = math.hypot(E[i], N[i])          # km/Myr, which is mm/yr
        rows.append(dict(site=site, window=W, apm_az=az, apm_mm_yr=spd))

d = pd.DataFrame(rows)
_sa = os.path.join(A.dir, 'plume_slant_apm.csv')
d.to_csv(_sa, index=False)
provenance.stamp(_sa, windows=A.windows, step=A.step, hinge_time=A.hinge_time,
                 plate_model=A.model, inputs=[A.hotspots])
print(f'\nabsolute plate motion above {d.site.nunique()} hotspots, '
      f'sampled every {A.step:.0f} Myr')
print(f'{"window":>7s} {"n":>4s} {"median speed":>14s} {"fastest":>24s}')
for w, g in d.groupby('window'):
    k = g.apm_mm_yr.idxmax()
    print(f'{w:6.0f}M {len(g):4d} {g.apm_mm_yr.median():9.0f} mm/yr '
          f'{g.loc[k, "site"]:>16s} {g.loc[k, "apm_mm_yr"]:4.0f}')

try:
    tess = recon.tessellate_subduction_zones(A.hinge_time, ignore_warnings=True)
    t = pd.DataFrame(tess, columns=['lon', 'lat', 'conv_rate', 'conv_angle',
                                    'trench_velocity', 'trench_velocity_angle',
                                    'arc_length', 'trench_azimuth_angle',
                                    'subducting_pid', 'trench_pid'])
    t.to_csv(os.path.join(A.dir, 'hinge_migration.csv'), index=False)
    def basin(lo, la):
        if -70 <= lo <= -20: return 'Atlantic'
        if 20 <= lo <= 140 and la < 30: return 'Indian'
        if lo >= 140 or lo <= -70: return 'Pacific'
        return 'other'
    t['basin'] = [basin(a, b) for a, b in zip(t.lon, t.lat)]
    print(f'\nhinge migration at {A.hinge_time:.0f} Ma, '
          f'{len(t)} tessellated trench segments (cm/yr, positive = advance):')
    for b, gg in t.groupby('basin'):
        print(f'  {b:>9s} n={len(gg):5d}  median trench velocity '
              f'{gg.trench_velocity.median():+6.2f}  '
              f'retreating {100 * (gg.trench_velocity < 0).mean():3.0f}%')
except Exception as ex:
    print(f'\nhinge migration unavailable: {ex}')
