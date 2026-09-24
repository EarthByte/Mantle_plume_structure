"""Repair a corridor mask file written before the depth-axis slice was fixed.

corridor_all.py wrote each mask as m[::coarsen] over the whole depth range while
writing the depth coordinate as z[k_seed:k0+1:coarsen], so the mask carried more
shells than its own axis and every depth attributed to a corridor cell was wrong.
The script that produced the file is fixed; this repairs files already on disk.

The repair is a trim, and it is exact only when the seed shell falls on a
coarsening step, so that the shells the axis wants are a subset of the shells the
mask holds. That is checked rather than assumed, and the alignment is confirmed
independently by requiring the trimmed-off shallow planes to be empty: no corridor
can occupy a shell above the depth its path starts at. If either check fails the
file is left alone and must be regenerated.
"""
from __future__ import annotations
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--masks', default='out/corridor_masks_RevealLO.npz')
ap.add_argument('--out', default=None, help='default: alongside, with _fixed')
A = ap.parse_args()

z = np.load(A.masks)
depth, lat, lon = z['depth'], z['lat'], z['lon']
coarsen, tau = int(z['coarsen']), float(z['tau'])
sites = [k for k in z.files if k not in ('depth', 'lat', 'lon', 'coarsen', 'tau')]
if not sites:
    raise SystemExit('no masks in file')

n_axis, n_mask = len(depth), z[sites[0]].shape[0]
print(f'{os.path.basename(A.masks)}: {len(sites)} masks, axis {n_axis} shells, '
      f'mask {n_mask} shells, coarsen {coarsen}, tau {tau}')
if n_axis == n_mask:
    raise SystemExit('axis and masks already agree; nothing to repair')

step = float(depth[1] - depth[0])
dz = step / coarsen
j0f = float(depth[0]) / step
j0 = int(round(j0f))
print(f'  native shell spacing {dz:g} km, seed depth {depth[0]:g} km, '
      f'implied mask offset {j0f:.4f}')
if abs(j0f - j0) > 1e-6:
    raise SystemExit(f'seed shell does not fall on a coarsening step ({j0f:.4f}); '
                     'the shells the axis needs were never written. Regenerate.')
if j0 + n_axis > n_mask:
    raise SystemExit(f'mask holds {n_mask} shells, trim needs {j0 + n_axis}. Regenerate.')

bad = [s for s in sites if z[s][:j0].any()]
if bad:
    raise SystemExit(f'{len(bad)} masks occupy shells above the seed depth '
                     f'({bad[:3]}); the alignment is not what it appears. Regenerate.')
print(f'  trimming [{j0}:{j0 + n_axis}]; all {len(sites)} masks empty above the seed '
      f'depth, which confirms the offset')

out = A.out or A.masks.replace('.npz', '_fixed.npz')
fixed = {s: z[s][j0:j0 + n_axis] for s in sites}
np.savez_compressed(out, coarsen=coarsen, tau=tau, depth=depth, lat=lat, lon=lon, **fixed)
chk = np.load(out)
assert chk[sites[0]].shape[0] == len(chk['depth'])
occ = {s: int(fixed[s].sum()) for s in sites}
print(f'  wrote {os.path.basename(out)}: masks now {fixed[sites[0]].shape[0]} shells, '
      f'matching the axis')
print(f'  occupied cells per corridor: median {int(np.median(list(occ.values())))}, '
      f'range {min(occ.values())}-{max(occ.values())}')
