"""Axis coherence: is the slow material beneath a site organised into an axis?

Every statistic that has failed in this project took the minimum anomaly within a
disc at each depth, which asks whether anything slow is nearby rather than whether
a column is present. Over a large low-velocity province the answer is yes almost
everywhere, which is why hotspots — selected for sitting over exactly those
provinces — could not be separated from their surroundings.

This asks a different question. A conduit has an axis: the local minimum sits at
nearly the same place at every depth, possibly on a tilt. Ambient slow material in
a province has no axis, because the local minimum jumps between unrelated patches
from depth to depth. The discriminating quantity is therefore the scatter of the
track about a fitted line, which is a statement about organisation rather than
about amplitude.

Two design points decide whether this works.

The track is built from INDEPENDENT minima at each depth. A tracker that constrains
each step to lie near the previous one manufactures coherence from any field
whatever, and the statistic would then measure the constraint rather than the
Earth. Nothing here carries information between depths.

Only genuine interior local minima count. A smooth regional gradient — the flank of
a province — pushes the minimum of a disc onto the disc's rim at every depth, which
produces a beautifully coherent track that is an artefact of the gradient and not a
conduit. Requiring the minimum to be lower than all its neighbours, and to sit away
from the rim, removes that. The fraction of depths supplying such a minimum is
itself reported, because a conduit should supply one at most depths and a gradient
supplies none.
"""
from __future__ import annotations
import numpy as np

R_E, DEG = 6371.0, np.pi / 180.0


def local_frame(lat0, lon0, lat, lon):
    """Local east/north in km relative to (lat0, lon0)."""
    dlon = ((np.asarray(lon, float) - lon0 + 180) % 360) - 180
    east = R_E * DEG * dlon * np.cos(lat0 * DEG)
    north = R_E * DEG * (np.asarray(lat, float) - lat0)
    return east, north


def theilsen(x, y):
    """Median of pairwise slopes; robust to the occasional wild track point."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    sl = []
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        ok = np.abs(dx) > 1e-9
        if ok.any():
            sl.append((y[i + 1:][ok] - y[i]) / dx[ok])
    if not sl:
        return np.nan, np.nan
    s = float(np.median(np.concatenate(sl)))
    return s, float(np.median(y - s * x))


class AxisTracker:
    """Interior local minima within a disc, one per depth, found independently."""

    def __init__(self, depth, lat, lon, arr, radius, rim_frac=0.85):
        self.depth = np.asarray(depth, float)
        self.lat = np.asarray(lat, float)
        self.lon = np.asarray(lon, float)
        self.arr = arr
        self.radius = float(radius)
        self.rim_frac = float(rim_frac)

    def _box(self, hlat, hlon):
        dlat = self.radius / (R_E * DEG) + abs(self.lat[1] - self.lat[0]) * 2
        jl = np.where(np.abs(self.lat - hlat) <= dlat)[0]
        dlon = (self.radius / (R_E * DEG * max(np.cos(hlat * DEG), 1e-3))
                + abs(self.lon[1] - self.lon[0]) * 2)
        il = np.where(np.abs(((self.lon - hlon + 180) % 360) - 180) <= min(dlon, 180.0))[0]
        if len(jl) < 3 or len(il) < 3:
            return None
        LA, LO = self.lat[jl][:, None], self.lon[il][None, :]
        d = R_E * np.arccos(np.clip(
            np.sin(hlat * DEG) * np.sin(LA * DEG) +
            np.cos(hlat * DEG) * np.cos(LA * DEG) * np.cos((LO - hlon) * DEG), -1, 1))
        return jl, il, d

    def track(self, hlat, hlon, kz, sub=None):
        """Return (depths, east_km, north_km, amp) for depths supplying a local minimum.

        sub, when given, is the field ALREADY restricted to this site's box, which is
        how the calibration injects a conduit without rebuilding the whole volume.
        Its shape must match the box this site would produce.
        """
        b = self._box(hlat, hlon)
        if b is None:
            return None
        jl, il, d = b
        if (d <= self.radius).sum() < 4:
            return None
        if sub is None:
            sub = self.arr[:, jl][:, :, il]
        elif sub.shape[1:] != (len(jl), len(il)):
            raise ValueError(f'injected box {sub.shape[1:]} does not match this '
                             f'site\'s box {(len(jl), len(il))}')
        zz, ee, nn, aa, rim = [], [], [], [], 0
        in_disc = d <= self.radius
        for i in kz:
            g = np.asarray(sub[i], float)
            if not np.isfinite(g[in_disc]).any():
                continue
            fl = np.where(in_disc & np.isfinite(g), g, np.inf)
            k = int(np.argmin(fl))
            r, c = divmod(k, g.shape[1])
            # The gradient artefact this guards against is a minimum pinned to the
            # rim by a smooth regional slope, which yields a perfectly coherent
            # track that is not a conduit. Requiring the minimum to lie inside the
            # disc rather than on its edge removes that, and unlike a strict
            # eight-neighbour test it does not depend on the model's grid spacing —
            # SEMUCB-WM1's two-degree cells make a 400 km disc three cells across,
            # where a strict local minimum almost never exists.
            if d[r, c] > self.radius * self.rim_frac:
                rim += 1
                continue
            e, n_ = local_frame(hlat, hlon, self.lat[jl][r], self.lon[il][c])
            zz.append(self.depth[i]); ee.append(float(e)); nn.append(float(n_))
            aa.append(float(g[r, c]))
        self.last_rim_frac = rim / max(len(kz), 1)
        if len(zz) < 5:
            return None
        return np.array(zz), np.array(ee), np.array(nn), np.array(aa)


def coherence(tr, n_depths, axial_tol_km=300.0):
    """Scatter of a track about a robust straight line, and how complete it is.

    scatter_km is the median distance of the track points from the fitted line, in
    the horizontal plane. tilt_km_per_1000 is how far the fitted axis moves per
    1000 km of depth, which is a measurement in its own right rather than a
    nuisance: a conduit may lean, and a leaning conduit is still an axis.
    """
    if tr is None:
        return dict(scatter_km=np.nan, fill=0.0, axial_frac=0.0, tilt=np.nan,
                    offset_km=np.nan, n_track=0)
    z, e, n_, _ = tr
    se, ie = theilsen(z, e)
    sn, in_ = theilsen(z, n_)
    if not np.isfinite(se) or not np.isfinite(sn):
        return dict(scatter_km=np.nan, fill=len(z) / max(n_depths, 1),
                    axial_frac=0.0, tilt=np.nan, offset_km=np.nan, n_track=len(z))
    re_ = e - (ie + se * z)
    rn = n_ - (in_ + sn * z)
    # Scatter alone is not comparable between sites, because a track with few
    # points fits a line better than a full one: an ambient site supplying minima
    # at a seventh of its depths scores a LOWER scatter than an injected conduit
    # supplying them everywhere, which inverts the statistic. axial_frac folds both
    # failure modes into one bounded number - the fraction of the whole depth window
    # that both yields an interior minimum and places it on a single line. A conduit
    # approaches 1; a regional gradient supplies no interior minima and scores near
    # 0; unorganised slow patches supply them but off the line and also score low.
    return dict(scatter_km=float(np.median(np.hypot(re_, rn))),
                fill=len(z) / max(n_depths, 1),
                axial_frac=float(np.sum(np.hypot(re_, rn) <= axial_tol_km)
                                 / max(n_depths, 1)),
                tilt=float(np.hypot(se, sn) * 1000.0),
                offset_km=float(np.hypot(np.median(e), np.median(n_))),
                n_track=len(z))


def disc_benchmark(radius, n_depths, rim_frac=0.85, n=4000, seed=3):
    """Scatter expected when the minima carry no axis at all.

    Points uniform on the disc the tracker is allowed to use, fitted the same way.
    This is the value a site with no organised structure should return, and it is
    what the observed scatter has to beat for the statistic to mean anything.
    """
    rg = np.random.default_rng(seed)
    R = radius * rim_frac
    out = []
    for _ in range(n):
        rr = R * np.sqrt(rg.random(n_depths))
        th = 2 * np.pi * rg.random(n_depths)
        z = np.linspace(0.0, 1.0, n_depths)
        e, n_ = rr * np.cos(th), rr * np.sin(th)
        se, ie = theilsen(z, e); sn, in_ = theilsen(z, n_)
        if not np.isfinite(se):
            continue
        out.append(float(np.median(np.hypot(e - (ie + se * z), n_ - (in_ + sn * z)))))
    return float(np.median(out)) if out else np.nan
