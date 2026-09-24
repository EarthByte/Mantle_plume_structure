"""Present-day mid-ocean ridges from the Zahirovic 2022 topologies, as a point set.

MacLeod et al.'s active-ridge shapefile holds the 93 segments they selected for
their morphological comparison, not a global ridge map: it contains no Atlantic
segment north of 40 degrees, which puts the Azores 7344 km from the nearest
"active ridge" while it sits on the Mid-Atlantic Ridge. Any distance-to-ridge
measurement built on that file is meaningless. Their extinct catalogue is a
different matter and is complete, so it is used as it stands.
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd
import pygplates, gplately

R_E, DEG = 6371.0, np.pi / 180.0
ap = argparse.ArgumentParser()
ap.add_argument('--model', default=os.path.expanduser(
    '~/Documents/Papers/in_prep/Muller_mantle_tomography_subduction/Paper_hydration/'
    'Zahirovic2022_plate_model'))
ap.add_argument('--time', type=float, default=0.0)
ap.add_argument('--step-km', type=float, default=25.0, dest='step_km')
ap.add_argument('--out', default='data/ridges_presentday_zahirovic2022.csv')
A = ap.parse_args()

rot = pygplates.RotationModel(os.path.join(A.model, 'CombinedRotations.rot'))
topo = [os.path.join(A.model, f) for f in
        ('Plate_Boundaries.gpml', 'Deforming_Networks_Active.gpml',
         'Feature_Geometries.gpml') if os.path.exists(os.path.join(A.model, f))]
stat = os.path.join(A.model, 'StaticGeometries', 'StaticPolygons',
                    'Global_EarthByte_GPlates_PresentDay_StaticPlatePolygons.shp')
recon = gplately.PlateReconstruction(rotation_model=rot, topology_features=topo,
                                     static_polygons=stat)
gplot = gplately.PlotTopologies(recon, time=A.time)

la, lo = [], []
for name, gdf in (('ridges', gplot.get_ridges()),
                  ('ridges_and_transforms', gplot.get_ridges_and_transforms())):
    if gdf is None or not len(gdf):
        print(f'  {name}: none')
        continue
    print(f'  {name}: {len(gdf)} features')
    if name != 'ridges':
        continue
    for geom in gdf.geometry:
        if geom is None:
            continue
        parts = geom.geoms if geom.geom_type.startswith('Multi') else [geom]
        for p in parts:
            xy = np.asarray(p.coords, float)
            for i in range(len(xy) - 1):
                (x0, y0), (x1, y1) = xy[i], xy[i + 1]
                if abs(x1 - x0) > 180:
                    continue
                d = R_E * DEG * np.hypot((x1 - x0) * np.cos(0.5 * (y0 + y1) * DEG), y1 - y0)
                n = max(int(np.ceil(d / A.step_km)), 1)
                t = np.linspace(0, 1, n + 1)
                lo.append(x0 + t * (x1 - x0)); la.append(y0 + t * (y1 - y0))

la = np.concatenate(la); lo = ((np.concatenate(lo) + 180) % 360) - 180
os.makedirs(os.path.dirname(A.out), exist_ok=True)
pd.DataFrame(dict(lat=la, lon=lo)).to_csv(A.out, index=False)
print(f'\n{len(la)} ridge points written to {A.out}')
print(f'lat {la.min():+.1f} to {la.max():+.1f}, lon {lo.min():+.1f} to {lo.max():+.1f}')
n40 = int(((la > 40) & (lo > -60) & (lo < 20)).sum())
print(f'North Atlantic points above 40N: {n40}   '
      f'{"the Mid-Atlantic Ridge is present" if n40 else "STILL MISSING"}')
