"""Plume structures as objects: segment the volume, then measure the objects.

Every failed measurement in this project probed the volume at hotspot points and
asked whether the numbers there differed from numbers elsewhere. That question is
answerable only by null testing, it re-proves an existence nobody disputes, and it
was defeated three times by the fact that hotspots sit over the slowest parts of
the mantle by construction. This does the thing a person does by eye instead: find
the connected slow bodies, then measure their shape.

The topology comes from a lobe graph rather than a three-dimensional skeleton. At
each depth the slow mask is split into lobes; lobes in adjacent depth slices are
linked when they overlap in map view. Branch points are then vertices of that
graph, read directly, without the spurious spurs skeletonisation produces on fat
noisy objects. Every lobe carries a centroid, an area and an amplitude, so the
centre-line, the slant, the cross-sectional profile and a temperature proxy all
come out of one pass.

Traced downward from a hotspot, a plume that splits upward appears as two tracked
lobes merging into one, so a branch is recorded where the merge happens.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage

R_E, DEG = 6371.0, np.pi / 180.0


def per_depth_threshold(arr, lat, pct):
    """Slow threshold per depth shell: a percentile of that shell's own distribution.

    A fixed per cent means different things at 800 km and at 2800 km, where the
    amplitudes differ several-fold. Weighted by cell area so the poles do not
    dominate the percentile.
    """
    w = np.repeat(np.cos(np.asarray(lat, float) * DEG)[:, None], arr.shape[2], axis=1).ravel()
    out = np.empty(arr.shape[0])
    for i in range(arr.shape[0]):
        v = arr[i].ravel()
        m = np.isfinite(v)
        if not m.any():
            out[i] = np.nan
            continue
        o = np.argsort(v[m])
        vv, ww = v[m][o], w[m][o]
        c = np.cumsum(ww) - 0.5 * ww
        out[i] = float(np.interp(pct / 100.0 * ww.sum(), c, vv))
    return out


def label_wrapped(mask):
    """Connected components on one depth slice, joined across the longitude seam.

    scipy has no periodic option, so the seam is stitched afterwards: any pair of
    labels touching across the last and first columns is merged. Without this a
    conduit sitting on the antimeridian is split into two structures.
    """
    lab, n = ndimage.label(mask)
    if n < 2:
        return lab, n
    left, right = lab[:, 0], lab[:, -1]
    pairs = {(int(a), int(b)) for a, b in zip(left, right) if a and b}
    if not pairs:
        return lab, n
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    remap = np.zeros(n + 1, int)
    nxt = 0
    for i in range(1, n + 1):
        r = find(i)
        if remap[r] == 0:
            nxt += 1
            remap[r] = nxt
        remap[i] = remap[r]
    return remap[lab], nxt


class Volume:
    """The slow mask, its lobes per depth, and the links between adjacent depths."""

    def __init__(self, depth, lat, lon, arr, pct=15.0, min_area_km2=2.0e5,
                 thresholds=None):
        """thresholds, when given, are used instead of being computed from arr.

        The calibration builds a Volume on a regional sub-box so that injecting a
        conduit does not mean relabelling the globe. The slow threshold must still
        come from the whole model, or it would be a percentile of whatever happens
        to be inside the box and the segmentation would change with the box.
        """
        self.depth = np.asarray(depth, float)
        self.lat = np.asarray(lat, float)
        self.lon = np.asarray(lon, float)
        self.arr = arr
        self.thr = (per_depth_threshold(arr, self.lat, pct)
                    if thresholds is None else np.asarray(thresholds, float))
        dlat = abs(self.lat[1] - self.lat[0]) * DEG * R_E
        dlon = abs(self.lon[1] - self.lon[0]) * DEG * R_E
        self.cell = (dlat * dlon * np.cos(self.lat * DEG))[:, None]
        self.min_area = float(min_area_km2)
        self.labels, self.lobes = [], []
        for i in range(len(self.depth)):
            m = np.isfinite(arr[i]) & (arr[i] < self.thr[i])
            lab, n = label_wrapped(m)
            rec = {}
            for k in range(1, n + 1):
                sel = lab == k
                a = float((self.cell * sel).sum())
                if a < self.min_area:
                    lab[sel] = 0
                    continue
                jj, ii = np.nonzero(sel)
                wgt = np.repeat(self.cell, len(self.lon), axis=1)[jj, ii]
                # Longitude is circular, so its centroid is the angle of the mean
                # unit vector rather than the mean of the values, which would put a
                # body straddling the antimeridian somewhere near Greenwich.
                a_ = self.lon[ii] * DEG
                clon = float(np.degrees(np.arctan2((np.sin(a_) * wgt).sum(),
                                                   (np.cos(a_) * wgt).sum())))
                clat = float((self.lat[jj] * wgt).sum() / wgt.sum())
                v = arr[i][sel]
                rec[k] = dict(area=a, lat=clat, lon=clon,
                              amp_min=float(np.nanmin(v)), amp_mean=float(np.nanmean(v)),
                              n=int(sel.sum()))
            self.labels.append(lab)
            self.lobes.append(rec)

    def links(self, i, j):
        """Which lobes at depth i overlap which at depth j, in map view."""
        a, b = self.labels[i], self.labels[j]
        both = (a > 0) & (b > 0)
        if not both.any():
            return {}
        pa, pb = a[both], b[both]
        out = {}
        for k in np.unique(pa):
            out[int(k)] = sorted({int(x) for x in np.unique(pb[pa == k])})
        return out

    def lobe_at(self, i, hlat, hlon):
        """The lobe at depth i containing, or nearest to, a point."""
        rec = self.lobes[i]
        if not rec:
            return None
        j = int(np.argmin(np.abs(self.lat - hlat)))
        k = int(np.argmin(np.abs(((self.lon - hlon + 180) % 360) - 180)))
        v = int(self.labels[i][j, k])
        if v:
            return v
        best, bd = None, np.inf
        for key, r in rec.items():
            d = R_E * np.arccos(np.clip(
                np.sin(hlat * DEG) * np.sin(r['lat'] * DEG) +
                np.cos(hlat * DEG) * np.cos(r['lat'] * DEG) *
                np.cos((r['lon'] - hlon) * DEG), -1, 1))
            if d < bd:
                best, bd = key, d
        return best if bd <= 800.0 else None


def _circmean(lons, w):
    a = np.asarray(lons, float) * DEG
    w = np.asarray(w, float)
    return float(np.degrees(np.arctan2((np.sin(a) * w).sum(), (np.cos(a) * w).sum())))


def _gc(alat, alon, blat, blon):
    return R_E * np.arccos(np.clip(
        np.sin(alat * DEG) * np.sin(blat * DEG) +
        np.cos(alat * DEG) * np.cos(blat * DEG) * np.cos((blon - alon) * DEG), -1, 1))


class Tracer:
    """Walk the lobe graph downward from a hotspot and record the structure.

    Downward is the direction of travel, so a plume that splits upward appears as
    two tracked lobes arriving at one deeper lobe. That merge is the branch, and it
    is recorded at the depth where it happens. The reverse case, one lobe linking to
    several deeper ones, is a structure with more than one root and is recorded
    separately, because the two mean different things: strands joining as we descend
    is a conduit that branches on its way up, while strands separating as we descend
    is a conduit fed from several places.
    """

    def __init__(self, vol: Volume):
        self.v = vol

    def trace(self, hlat, hlon, z_top=300.0, z_bot=2850.0, seed_search=5):
        v = self.v
        idx = np.where((v.depth >= z_top) & (v.depth <= z_bot))[0]
        if len(idx) < 5:
            return None
        seed, i0 = None, None
        for k in idx[:seed_search]:
            seed = v.lobe_at(int(k), hlat, hlon)
            if seed is not None:
                i0 = int(k)
                break
        if seed is None:
            return dict(site=None, found=False)

        front = {seed}
        prof, branches, splits = [], [], []
        last = i0
        for i in range(i0, int(idx[-1])):
            rec = v.lobes[i]
            act = [rec[l] for l in front if l in rec]
            if act:
                w = np.array([a['area'] for a in act], float)
                prof.append(dict(
                    depth=float(v.depth[i]), n_strand=len(act),
                    area=float(w.sum()),
                    lat=float((np.array([a['lat'] for a in act]) * w).sum() / w.sum()),
                    lon=_circmean([a['lon'] for a in act], w),
                    amp_min=float(min(a['amp_min'] for a in act)),
                    amp_mean=float((np.array([a['amp_mean'] for a in act]) * w).sum() / w.sum())))
                last = i
            link = v.links(i, i + 1)
            nxt = {}
            for lab in front:
                for ch in link.get(lab, []):
                    nxt.setdefault(ch, []).append(lab)
            for ch, par in nxt.items():
                if len(par) > 1:
                    branches.append(dict(depth=float(v.depth[i + 1]), n=len(par)))
            for lab in front:
                ch = link.get(lab, [])
                if len(ch) > 1:
                    splits.append(dict(depth=float(v.depth[i + 1]), n=len(ch)))
            front = set(nxt)
            if not front:
                break

        if len(prof) < 3:
            return dict(site=None, found=False)
        root = float(v.depth[last])
        deeper = self._slow_below(hlat, hlon, last, int(idx[-1]))
        return dict(found=True, root_km=root, n_depths=len(prof),
                    branch_depths=[b['depth'] for b in branches],
                    n_branch=len(branches),
                    max_strands=int(max(p['n_strand'] for p in prof)),
                    split_depths=[s['depth'] for s in splits],
                    slow_below=deeper, profile=prof,
                    **self._geometry(prof))

    def _slow_below(self, hlat, hlon, i_last, i_end, radius=800.0):
        """Is there slow material beneath where the traced column stopped?

        This is the discriminant that separates a plume impeded at a barrier, which
        has slow material below the break, from one that genuinely originates at
        that depth, which does not.
        """
        v = self.v
        n = 0
        for i in range(i_last + 1, i_end + 1):
            for r in v.lobes[i].values():
                if _gc(hlat, hlon, r['lat'], r['lon']) <= radius:
                    n += 1
                    break
        span = max(i_end - i_last, 1)
        return float(n / span)

    def _geometry(self, prof):
        """Slant of the centre-line, and where its direction changes."""
        z = np.array([p['depth'] for p in prof])
        la = np.array([p['lat'] for p in prof])
        lo = np.array([p['lon'] for p in prof])
        e = R_E * DEG * (((lo - lo[0] + 180) % 360) - 180) * np.cos(la[0] * DEG)
        n_ = R_E * DEG * (la - la[0])
        if len(z) < 5:
            return dict(tilt_deg=np.nan, offset_km=np.nan, defl_depths=[])
        # Overall slant: horizontal travel per unit depth, as an angle from vertical.
        dz = z[-1] - z[0]
        off = float(np.hypot(e[-1], n_[-1]))
        tilt = float(np.degrees(np.arctan2(off, abs(dz)))) if dz else np.nan
        # Direction changes, measured over a window wide enough not to be noise.
        w = max(3, len(z) // 10)
        defl = []
        for k in range(w, len(z) - w):
            v1 = np.array([e[k] - e[k - w], n_[k] - n_[k - w]])
            v2 = np.array([e[k + w] - e[k], n_[k + w] - n_[k]])
            if np.linalg.norm(v1) < 50 or np.linalg.norm(v2) < 50:
                continue
            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            ang = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
            if ang >= 10.0:
                defl.append((float(z[k]), ang))
        # Keep the strongest deflection in each contiguous run, so one bend is one
        # event rather than a dozen adjacent depths.
        keep, run = [], []
        for d_, a_ in defl:
            if run and abs(d_ - run[-1][0]) <= (z[1] - z[0]) * 1.5:
                run.append((d_, a_))
            else:
                if run:
                    keep.append(max(run, key=lambda t: t[1]))
                run = [(d_, a_)]
        if run:
            keep.append(max(run, key=lambda t: t[1]))
        return dict(tilt_deg=tilt, offset_km=off,
                    defl_depths=[d_ for d_, _ in keep],
                    defl_angles=[a_ for _, a_ in keep])
