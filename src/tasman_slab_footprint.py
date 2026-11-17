#!/usr/bin/env python3
"""Where the slab of Mather et al. (2026) sits in map view, rebuilt from the plate model.

Mather et al. (2026, Gondwana Research 160, 99-108, their Fig. 4) map the cumulative
slab area density beneath the SW Pacific: slab flux at every subduction zone of the
Mueller et al. (2019) plate model, 200 Ma to the present at 1 Myr steps, deposited at
the trench position in the model's mantle reference frame, with slab thickness from a
plate model of lithospheric cooling (Richards et al., 2018) and a density of
3000 kg/m3. Their slab "ribbon" is the material subducted at the d'Entrecasteaux
subduction zone, about 48-25 Ma, and their "windows" are cells that received no slab.

This rebuilds that map so it can be laid over the tomography. The map has no depth:
deposition beneath the trench is the vertical-sinking assumption of the method, and
how deep the material now sits is what the tomography is asked.

Per 1 Myr step: tessellate the subduction zones, take the trench-normal convergence,
sample the seafloor age of the subducting plate at the trench (the age grid of that
time, NaNs filled from the nearest valid cell), convert age to thickness with the
plate-model 1150 C isotherm (gplately.tools.plate_isotherm_depth), and deposit
  mass = 3000 kg/m3 * thickness * normal convergence * segment length * 1 Myr
in a 0.5 degree cell at the trench point.

Writes out/tasman_slab_footprint.npz: lon, lat, mass (kg, all ages), area_density
(kg/m2), last_time (Ma of the most recent deposit, NaN where none), and
mass_<a>_<b> for subduction-time bins.

  python3 tasman_slab_footprint.py            # 200-0 Ma, downloads the age grids once
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
if not hasattr(np, 'row_stack'):   # removed in NumPy 2.4, still called inside gplately 2.0
    np.row_stack = np.vstack
import gplately
from gplately import tools

ap = argparse.ArgumentParser()
ap.add_argument('--tmax', type=int, default=200)
ap.add_argument('--res', type=float, default=0.5)
ap.add_argument('--box', default='90,220,-75,15', help='lon0,lon1,lat0,lat1 (0-360)')
ap.add_argument('--out', default='out/tasman_slab_footprint.npz')
A = ap.parse_args()

RHO, MYR = 3000.0, 1.0e6 * 365.25 * 86400.0
R_E = 6371.0e3
BINS = [(0, 25), (25, 48), (48, 100), (100, 200)]

lo0, lo1, la0, la1 = (float(x) for x in A.box.split(','))
LON = np.arange(lo0, lo1 + 1e-9, A.res)
LAT = np.arange(la0, la1 + 1e-9, A.res)
nlo, nla = len(LON), len(LAT)
mass = np.zeros((nla, nlo))
last = np.full((nla, nlo), np.nan)
binned = {b: np.zeros((nla, nlo)) for b in BINS}

ds = gplately.download.DataServer('Muller2019')
rm, topo, _ = ds.get_plate_reconstruction_files()
model = gplately.PlateReconstruction(rm, topo)
t0 = time.time()
for t in range(A.tmax, -1, -1):
    d = model.tessellate_subduction_zones(t, ignore_warnings=True)
    lon, lat = d[:, 0] % 360.0, d[:, 1]
    vn = d[:, 2] * np.cos(np.radians(d[:, 3])) * 0.01           # m/yr, trench-normal
    seg = np.radians(d[:, 6]) * R_E                              # m
    m = (vn > 0) & (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)
    if not m.any():
        continue
    g = ds.get_age_grid(t)
    g.fill_NaNs(inplace=True)
    age = np.asarray(g.interpolate(((lon[m] + 180.0) % 360.0) - 180.0, lat[m]), float)
    age = np.clip(np.nan_to_num(age, nan=np.nanmedian(age)), 0.1, None)
    thick = np.asarray(tools.plate_isotherm_depth(age), float)    # m
    mk = RHO * thick * vn[m] * seg[m] * 1.0                       # kg per Myr step (vn in m/yr * 1e6 yr / 1e6)
    mk = mk * 1.0e6
    i = np.clip(np.round((lon[m] - lo0) / A.res).astype(int), 0, nlo - 1)
    j = np.clip(np.round((lat[m] - la0) / A.res).astype(int), 0, nla - 1)
    np.add.at(mass, (j, i), mk)
    last[j, i] = np.where(np.isnan(last[j, i]), t, np.minimum(last[j, i], t))
    for b in BINS:
        if b[0] <= t < b[1] or (t == b[1] == A.tmax):
            np.add.at(binned[b], (j, i), mk)
    if t % 10 == 0:
        print(f'  {t:4d} Ma  {m.sum():5d} trench points  ({time.time()-t0:.0f} s)', flush=True)

cell = (R_E ** 2) * np.radians(A.res) ** 2 * np.cos(np.radians(LAT))[:, None]
os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
np.savez_compressed(A.out, lon=LON, lat=LAT, mass=mass, area_density=mass / cell,
                    last_time=last,
                    **{f'mass_{a}_{b}': v for (a, b), v in binned.items()})
print(f'wrote {A.out}')
