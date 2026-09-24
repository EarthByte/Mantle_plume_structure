"""Global probability distribution of plume-path deflection depth, as Bao et al. define it.

They report a PDF of deflection depth, not a count of events, which is why our
"fourteen deflections per path" was never comparable with anything they published.
Every depth sample along every path at which the local direction change reaches 10
degrees contributes to the distribution, and each path contributes equally so that a
long path does not outvote a short one.

Their result, for eight hotspots in East Africa and the Indian Ocean: the highest
peak at 260 km, which is the top of their domain and which they attribute to
second-order asthenospheric convection rather than to plume dynamics, present for
every hotspot except Reunion; a second peak near 800 km with a range of 660 to 1000;
and a weaker plateau from 1000 to 1500 km.

This is the same measurement on all 49 hotspots of a global model, which has not
been done before.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd
import provenance

R_E, DEG = 6371.0, np.pi / 180.0

ap = argparse.ArgumentParser()
ap.add_argument('--paths', default='out/conduit_paths_all_RevealLO.json')
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--min-angle', type=float, default=10.0, dest='min_angle')
ap.add_argument('--window-km', type=float, default=300.0, dest='window')
ap.add_argument('--z-top', type=float, default=260.0, dest='z_top',
                help='top of the domain; Bao et al. use 260 km and their highest peak '
                     'sits on it, so the same boundary is used here to keep the '
                     'comparison honest rather than removing their artefact')
ap.add_argument('--bin-km', type=float, default=50.0, dest='bin_km')
ap.add_argument('--region', default=None)
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--dir', default='out')
A = ap.parse_args()

paths = json.load(open(A.paths))
if A.region == 'bao':
    hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180']).set_index('hotspot')
    paths = {k: v for k, v in paths.items() if k in hs.index
             and 15.0 <= float(hs.loc[k, 'lon_180']) <= 110.0
             and -55.0 <= float(hs.loc[k, 'lat']) <= 30.0}
print(f'{len(paths)} paths, deflection >= {A.min_angle:.0f} deg over a '
      f'{A.window:.0f} km window, domain from {A.z_top:.0f} km')

edges = np.arange(A.z_top, 2900.0 + A.bin_km, A.bin_km)
centres = 0.5 * (edges[:-1] + edges[1:])
pdf = np.zeros(len(centres))
per_site, rows = {}, []
for name, p in paths.items():
    la = np.asarray(p['lat'], float)
    lo = np.asarray(p['lon'], float)
    z = np.asarray(p['depth'], float)
    if len(z) < 12:
        continue
    o = np.argsort(z)
    la, lo, z = la[o], lo[o], z[o]
    e = R_E * DEG * (((lo - lo[0] + 180) % 360) - 180) * np.cos(la[0] * DEG)
    n_ = R_E * DEG * (la - la[0])
    dz = float(np.median(np.diff(z)))
    w = max(2, int(round(A.window / max(dz, 1.0))))
    hit = []
    for k in range(w, len(z) - w):
        if z[k] < A.z_top:
            continue
        v1 = np.array([e[k] - e[k - w], n_[k] - n_[k - w]])
        v2 = np.array([e[k + w] - e[k], n_[k + w] - n_[k]])
        if np.linalg.norm(v1) < 30 or np.linalg.norm(v2) < 30:
            continue
        c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
        if ang >= A.min_angle:
            hit.append((float(z[k]), ang))
    if not hit:
        continue
    h = np.histogram([d for d, _ in hit], bins=edges)[0].astype(float)
    if h.sum() > 0:
        pdf += h / h.sum()          # each path contributes equally
    per_site[name] = [d for d, _ in hit]
    for d_, a_ in hit:
        rows.append(dict(site=name, depth=d_, angle=a_))

pdf /= max(pdf.sum(), 1e-9)
_out = os.path.join(A.dir, f'bao_pdf_{A.tag}.csv')
pd.DataFrame(rows).to_csv(_out, index=False)
provenance.stamp(_out, min_angle=A.min_angle, window_km=A.window, z_top=A.z_top,
                 bin_km=A.bin_km, region=A.region, inputs=[A.paths, A.hotspots])

print(f'\n{len(per_site)} of {len(paths)} paths contribute\n')
print(f'{"depth":>9s} {"density":>8s}')
for c, v in zip(centres, pdf):
    if v <= 0:
        continue
    print(f'{c:6.0f} km {v:8.3f}  {"#" * int(round(200 * v))}')

k = np.argsort(-pdf)[:6]
print('\nhighest bins: ' + ', '.join(f'{centres[i]:.0f} km ({pdf[i]:.3f})' for i in sorted(k, key=lambda t: -pdf[t])))
for lo_, hi_ in ((A.z_top, 400), (660, 1000), (1000, 1500), (1500, 2200), (2200, 2900)):
    m = (centres >= lo_) & (centres < hi_)
    print(f'  {lo_:5.0f}-{hi_:5.0f} km carries {100 * pdf[m].sum():5.1f}% of the distribution')
shallow = sum(1 for v in per_site.values() if any(d < 400 for d in v))
print(f'\n{shallow} of {len(per_site)} hotspots show a deflection above 400 km '
      f'(Bao: all but Reunion)')
