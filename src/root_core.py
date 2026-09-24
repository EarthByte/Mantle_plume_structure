"""The root-depth measurement itself, shared by the measurement and its calibration.

root_depth.py measures real sites and root_depth_test.py measures injected
conduits whose termination depth is known. If those two ran different code the
calibration would not calibrate anything, so the walk, the threshold and the cap
search live here and both import them.
"""
from __future__ import annotations
import numpy as np

R_E, DEG = 6371.0, np.pi / 180.0


def wquantile(v, w, q):
    """Percentile of a shell weighted by cell area, so the poles do not dominate."""
    m = np.isfinite(v)
    v, w = v[m], w[m]
    if not len(v):
        return np.nan
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q / 100.0 * w.sum(), c, v))


def thresholds(arr, lat, lon, pct):
    """The per-depth slow threshold: a percentile of each shell's own distribution.

    A fixed per cent means different things in a model that renders anomalies
    strongly and one that renders them weakly, and different things at 800 km and
    at 2800 km where the amplitudes differ by a factor of several. This does not.
    """
    aw = np.repeat(np.cos(np.asarray(lat, float) * DEG)[:, None], len(lon), axis=1).ravel()
    return np.array([wquantile(arr[i].ravel(), aw, pct) for i in range(arr.shape[0])])


class Profiler:
    """The lowest anomaly within a cap of the given radius, at every depth."""

    def __init__(self, depth, lat, lon, arr, radius):
        self.depth = np.asarray(depth, float)
        self.lat = np.asarray(lat, float)
        self.lon = np.asarray(lon, float)
        self.arr = arr
        self.radius = float(radius)
        self._dlat = self.radius / (R_E * DEG) + abs(self.lat[1] - self.lat[0]) * 2

    def profile(self, hlat, hlon):
        jl = np.where(np.abs(self.lat - hlat) <= self._dlat)[0]
        if not len(jl):
            return None
        dlon = (self.radius / (R_E * DEG * max(np.cos(hlat * DEG), 1e-3))
                + abs(self.lon[1] - self.lon[0]) * 2)
        dl = np.abs(((self.lon - hlon + 180) % 360) - 180)
        il = np.where(dl <= min(dlon, 180.0))[0]
        if not len(il):
            return None
        LA = self.lat[jl][:, None]
        LO = self.lon[il][None, :]
        d = R_E * np.arccos(np.clip(
            np.sin(hlat * DEG) * np.sin(LA * DEG) +
            np.cos(hlat * DEG) * np.cos(LA * DEG) * np.cos((LO - hlon) * DEG), -1, 1))
        m = d <= self.radius
        if not m.any():
            return None
        sub = self.arr[:, jl][:, :, il]
        out = np.full(len(self.depth), np.nan)
        for i in range(len(self.depth)):
            s = sub[i][m]
            s = s[np.isfinite(s)]
            if len(s):
                out[i] = s.min()
        return out


def walk(depth, prof, slow, anchor, tol):
    """Deepest depth reachable walking down from the anchor, bridging gaps up to tol.

    Returns the root depth and the depth at which the walk stopped, which is the
    top of the gap that ended it rather than the bottom of the last slow patch.
    """
    depth = np.asarray(depth, float)
    k = np.where(depth >= anchor)[0]
    if not len(k):
        return np.nan, np.nan
    ok = slow[k] & np.isfinite(prof[k])
    if not ok.any():
        return np.nan, np.nan
    start = int(np.argmax(ok))
    root, gap_top, i, j = depth[k[start]], np.nan, start, start
    while j + 1 < len(k):
        j += 1
        if ok[j]:
            root, i = depth[k[j]], j
        elif depth[k[j]] - depth[k[i]] > tol:
            gap_top = depth[k[i]]
            break
    return float(root), float(gap_top)
