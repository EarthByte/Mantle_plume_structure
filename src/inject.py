"""Synthetic conduits, used to calibrate the search on each model.

The conduit is a Gaussian in horizontal distance from an axis that migrates with
depth. It is tapered over the deepest 200 km so that it does not end in a step
at the base.

Two deflection styles are offered, and both matter. In 'transition' the axis
holds its position, steps across the transition zone where the viscosity
contrast concentrates horizontal flow, then rises vertically again - so the
conduit is close to a sequence of vertical and horizontal runs. In 'uniform' the
same total offset is spread evenly over the whole mantle, giving a conduit that
leans steadily from base to surface.

The distinction is not cosmetic. A search that prices a step by its length finds
the first shape easily, because both of its segments lie along the grid; the
second asks the search to pay for lateral travel at every depth, and is the case
a steadily leaning plume actually presents. Calibrating on the first alone would
report a detection rate the method does not have for the second.
"""
from __future__ import annotations

import numpy as np

R_E = 6371.0


def gc_km(lat0, lon0, LAT, LON):
    p0, p1 = np.radians(lat0), np.radians(LAT)
    dl = np.radians(LON - lon0)
    return R_E * np.arccos(np.clip(np.sin(p0) * np.sin(p1) +
                                   np.cos(p0) * np.cos(p1) * np.cos(dl), -1, 1))


def inject_tilted(arr, depth, lat, lon, hlat, hlon, radius_km, amp, tilt_deg,
                  z0=100.0, z1=2800.0, azimuth=45.0, profile='transition'):
    """Add a conduit of peak amplitude `amp` per cent to the anomaly cube.

    `profile` is 'transition' for deflection concentrated across the transition
    zone, or 'uniform' for a conduit that leans at a constant angle. With
    tilt_deg of zero the two are the same vertical conduit.
    """
    out = arr.copy()
    LO, LA = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
    LO = ((LO + 180) % 360) - 180
    for k, z in enumerate(np.asarray(depth, float)):
        if z < z0 or z > z1:
            continue
        if profile == 'uniform':
            f = (z - z0) / max(z1 - z0, 1)
        elif z <= 410:
            f = 0.15 * (z - z0) / max(410 - z0, 1)
        elif z <= 660:
            f = 0.15 + 0.5 * (z - 410) / 250.0
        else:
            f = 0.65 + 0.35 * (z - 660) / max(z1 - 660, 1)
        off = tilt_deg * f
        clat = hlat + off * np.cos(np.radians(azimuth))
        clon = hlon + off * np.sin(np.radians(azimuth)) / max(
            np.cos(np.radians(clat)), 0.2)
        d = gc_km(clat, clon, LA, LO)
        w = np.exp(-(d ** 2) / (2.0 * (radius_km / 2.0) ** 2))
        taper = 1.0 if z < z1 - 200 else max(0.0, (z1 - z) / 200.0)
        out[k] += (amp * w * taper).astype(out.dtype)
    return out
