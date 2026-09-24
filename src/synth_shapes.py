"""Synthetic conduit morphologies, beyond the straight and singly deflected pair.

The calibration behind the classification injects three shapes: a vertical
conduit, one whose offset is taken up across the transition zone, and one that
leans at a constant angle. All three are single, continuous, tube-like
structures, so the recovery rate they establish is the recovery rate for a pipe.
The question here is what happens to structures that are coherent but not pipes,
and answering it needs shapes the calibration has never contained.

Each generator returns an axis: latitude and longitude as functions of depth,
together with a radius and an amplitude scale that may themselves vary with
depth, and a weight for structures made of more than one limb. The injector then
lays a Gaussian of that radius about the axis at every shell, so a branch, a
sheet and a gap are built from the same primitive as the original conduits and
differ only in the axis and the profiles - which is what makes the recovery rates
comparable with the ones already published.

WHAT THESE TESTS CAN AND CANNOT SHOW

Injecting a structure into an already inverted model and recovering it tests the
path search, not the seismic inversion. It shows what the analysis would find if
a structure of that amplitude and scale were present in the model. It says
nothing about whether the seismic data would have put it there. Every statement
built on these numbers has to carry that distinction.
"""
from __future__ import annotations

import numpy as np

R_E = 6371.0
DEG = np.pi / 180.0


def _offset(hlat, hlon, d_deg, az_deg):
    """A point d_deg away from the hotspot along the bearing az_deg."""
    la = hlat + d_deg * np.cos(az_deg * DEG)
    lo = hlon + d_deg * np.sin(az_deg * DEG) / np.clip(np.cos(la * DEG), 0.2, None)
    return la, lo


def _f(z, z0, z1):
    return np.clip((np.asarray(z, float) - z0) / max(z1 - z0, 1.0), 0.0, 1.0)


def limbs(name, z, hlat, hlon, z0=100.0, z1=2800.0, offset_deg=10.0,
          radius_km=300.0, azimuth=45.0):
    """Axes for one morphology, as a list of (lat, lon, radius, scale, weight).

    Each entry is an array over the depths in `z`; scale multiplies the amplitude
    and weight sets the share of the amplitude a limb carries where limbs
    overlap. Returns [] for an unknown name so the caller can fail loudly.
    """
    z = np.asarray(z, float)
    f = _f(z, z0, z1)
    one = np.ones_like(z)
    R = radius_km * one

    if name == 'vertical':
        la, lo = np.full_like(z, hlat), np.full_like(z, hlon)
        return [(la, lo, R, one, one)]

    if name == 'uniform':
        la, lo = _offset(hlat, hlon, offset_deg * f, azimuth)
        return [(la, lo, R, one, one)]

    if name == 'transition':
        g = np.where(z <= 410, 0.15 * _f(z, z0, 410.0),
                     np.where(z <= 660, 0.15 + 0.5 * _f(z, 410.0, 660.0),
                              0.65 + 0.35 * _f(z, 660.0, z1)))
        la, lo = _offset(hlat, hlon, offset_deg * g, azimuth)
        return [(la, lo, R, one, one)]

    if name == 's_shaped':
        # one bend in the upper half of the mantle and an opposite one below, so
        # the depth-averaged lean is small and the curvature is not
        g = np.sin(2.0 * np.pi * f) * 0.5 + f * 0.25
        la, lo = _offset(hlat, hlon, offset_deg * g, azimuth)
        return [(la, lo, R, one, one)]

    if name == 'reversing':
        # the azimuth turns through ninety degrees at mid-mantle while the
        # displacement keeps growing: a structure no single tilt can describe
        d = offset_deg * f
        az = azimuth + 90.0 * f
        la = hlat + d * np.cos(az * DEG)
        lo = hlon + d * np.sin(az * DEG) / np.clip(np.cos(la * DEG), 0.2, None)
        return [(la, lo, R, one, one)]

    if name == 'bifurcating':
        # two limbs above the mid mantle that merge into one trunk below it
        split = np.clip((1500.0 - z) / 900.0, 0.0, 1.0)
        out = []
        for sgn in (+1.0, -1.0):
            la, lo = _offset(hlat, hlon, offset_deg * f + 0.0, azimuth)
            d = 5.0 * split * sgn
            la2 = la + d * np.cos((azimuth + 90.0) * DEG)
            lo2 = lo + d * np.sin((azimuth + 90.0) * DEG) / np.clip(
                np.cos(la2 * DEG), 0.2, None)
            out.append((la2, lo2, R * 0.8, one, 0.5 + 0.5 * (1 - split)))
        return out

    if name == 'sheet':
        # a wall rather than a tube: parallel limbs across strike, spaced at half
        # the conduit radius so that the envelope is flat rather than corrugated.
        # Spacing them at the full radius leaves a trough between limbs that
        # costs as much to cross as leaving the structure, and the corridor then
        # correctly reports a tube inside one limb - a sheet made of separate
        # conduits is not a sheet, and the test would be of the injector.
        out = []
        span = radius_km / 2.0 / 111.19
        for q in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
            la, lo = _offset(hlat, hlon, offset_deg * f, azimuth)
            d = span * q
            la2 = la + d * np.cos((azimuth + 90.0) * DEG)
            lo2 = lo + d * np.sin((azimuth + 90.0) * DEG) / np.clip(
                np.cos(la2 * DEG), 0.2, None)
            out.append((la2, lo2, R, one, one))
        return out

    if name == 'tapered':
        # amplitude falls by a factor of three from base to surface
        la, lo = _offset(hlat, hlon, offset_deg * f, azimuth)
        return [(la, lo, R, 0.33 + 0.67 * f, one)]

    if name == 'gapped':
        # a 400 km interval of the lower mantle carries nothing
        la, lo = _offset(hlat, hlon, offset_deg * f, azimuth)
        sc = np.where((z > 1400.0) & (z < 1800.0), 0.0, 1.0)
        return [(la, lo, R, sc, one)]

    if name == 'narrow_in_broad':
        # a conduit of the standard radius inside a province five times wider
        # and a quarter as strong, the case in which a route through the
        # conduit is barely cheaper than a route through its surroundings
        la, lo = _offset(hlat, hlon, offset_deg * f, azimuth)
        return [(la, lo, R, one, one),
                (la, lo, R * 5.0, 0.25 * one, one)]

    if name == 'twin_branch':
        # two conduits 8 degrees apart sharing one basal source
        out = []
        for sgn in (+1.0, -1.0):
            merge = np.clip((z - 2000.0) / 600.0, 0.0, 1.0)
            d = 4.0 * sgn * (1.0 - merge)
            la = hlat + d * np.cos((azimuth + 90.0) * DEG) + offset_deg * f * np.cos(azimuth * DEG)
            lo = hlon + (d * np.sin((azimuth + 90.0) * DEG)
                         + offset_deg * f * np.sin(azimuth * DEG)) / np.clip(
                np.cos(la * DEG), 0.2, None)
            out.append((la, lo, R * 0.8, one, one))
        return out

    if name == 'broken_beside_whole':
        # THE CASE THE CALIBRATION HAS NEVER CONTAINED, and the only one in which a
        # continuity term can show a benefit. Two structures of equal amplitude are
        # injected together:
        #
        #   limb A, beneath the site, vertical and BROKEN - three 200 km intervals
        #           carry nothing, so it is a chain of blobs rather than a tube
        #   limb B, leaning away to offset_deg, CONTINUOUS from top to base
        #
        # A is the shorter route and its blobs are as slow as B, so a search that
        # integrates slowness along a path should take A; B is the structure a
        # reader would call the conduit. Ground truth is B. Every other synthetic in
        # this file and in inject.py is continuous, so none of them can separate a
        # search that follows structure from one that merely finds cheap material.
        la_a, lo_a = np.full_like(z, hlat), np.full_like(z, hlon)
        gaps = ((900.0, 1100.0), (1500.0, 1700.0), (2100.0, 2300.0))
        sc_a = np.ones_like(z)
        for g0, g1 in gaps:
            sc_a = np.where((z > g0) & (z < g1), 0.0, sc_a)
        la_b, lo_b = _offset(hlat, hlon, offset_deg * f, azimuth)
        return [(la_a, lo_a, R, sc_a, one), (la_b, lo_b, R, one, one)]

    return []


NAMES = ('vertical', 'uniform', 'transition', 's_shaped', 'reversing',
         'bifurcating', 'sheet', 'tapered', 'gapped', 'narrow_in_broad',
         'twin_branch', 'broken_beside_whole')


def inject(arr, depth, lat, lon, hlat, hlon, name, amp, radius_km=300.0,
           offset_deg=10.0, azimuth=45.0, z0=100.0, z1=2800.0, taper_km=200.0):
    """Add a synthetic morphology of peak amplitude `amp` per cent to the cube."""
    z = np.asarray(depth, float)
    ls = limbs(name, z, hlat, hlon, z0=z0, z1=z1, offset_deg=offset_deg,
               radius_km=radius_km, azimuth=azimuth)
    if not ls:
        raise SystemExit(f'unknown synthetic morphology {name!r}')
    out = arr.copy()
    LO, LA = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
    LO = ((LO + 180.0) % 360.0) - 180.0
    p1, l1 = LA * DEG, LO * DEG
    for k, zz in enumerate(z):
        if zz < z0 or zz > z1:
            continue
        taper = 1.0 if zz < z1 - taper_km else max(0.0, (z1 - zz) / taper_km)
        if taper <= 0:
            continue
        add = np.zeros(LA.shape, np.float32)
        for la_, lo_, R_, sc_, w_ in ls:
            if float(sc_[k]) == 0.0:
                continue
            p0 = float(la_[k]) * DEG
            dl = l1 - float(lo_[k]) * DEG
            d = R_E * np.arccos(np.clip(np.sin(p0) * np.sin(p1) +
                                        np.cos(p0) * np.cos(p1) * np.cos(dl), -1, 1))
            g = np.exp(-(d ** 2) / (2.0 * (float(R_[k]) / 2.0) ** 2))
            add = np.minimum(add, (amp * float(sc_[k]) * float(w_[k]) * g
                                   ).astype(np.float32)) if amp < 0 else \
                  np.maximum(add, (amp * float(sc_[k]) * float(w_[k]) * g
                                   ).astype(np.float32))
        out[k] += (add * taper).astype(out.dtype)
    return out
