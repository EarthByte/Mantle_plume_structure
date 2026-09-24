"""Shape measurements on a three-dimensional weight field beneath a hotspot.

The input is one weight per cell per depth: either the fraction of retained
configurations whose optimal path passes through the cell (the ensemble), or the
fraction for which the cell lies on a route costing no more than a fixed excess
over the optimum (the corridor). Both are sensitivity maps over methodological
choices, not posterior probabilities, and nothing below treats them as
probabilities.

The measurements are chosen to separate the hypotheses the paper sets out, and
deliberately kept few. Width separates a tube from a fan; anisotropy separates a
tube from a sheet; the number of components separates one structure from several;
azimuth against depth separates a straight lean from a curve; and the weight
present at each depth separates a continuous structure from a segmented one. A
single composite score would blur exactly the distinctions the analysis exists to
make, so none is formed.

Everything is computed in an azimuthal equidistant projection about the weighted
centroid of each shell, so a cap several thousand kilometres across is measured
without the distortion a flat latitude-longitude approximation introduces at high
latitude or at large offset.
"""
from __future__ import annotations

import numpy as np

R_E = 6371.0
DEG = np.pi / 180.0


def gc_km(lat0, lon0, LAT, LON):
    p0, p1 = lat0 * DEG, np.asarray(LAT, float) * DEG
    dl = (np.asarray(LON, float) - lon0) * DEG
    return R_E * np.arccos(np.clip(np.sin(p0) * np.sin(p1) +
                                   np.cos(p0) * np.cos(p1) * np.cos(dl), -1, 1))


def bearing_deg(lat0, lon0, LAT, LON):
    """Initial great-circle bearing from (lat0, lon0), degrees clockwise from north."""
    p0, p1 = lat0 * DEG, np.asarray(LAT, float) * DEG
    dl = (np.asarray(LON, float) - lon0) * DEG
    y = np.sin(dl) * np.cos(p1)
    x = np.cos(p0) * np.sin(p1) - np.sin(p0) * np.cos(p1) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360.0


def weighted_centroid(w, LAT, LON):
    """Centroid of a weight field on the sphere, by averaging unit vectors."""
    tw = float(np.sum(w))
    if tw <= 0:
        return np.nan, np.nan, 0.0
    p, l = LAT * DEG, LON * DEG
    x = np.sum(w * np.cos(p) * np.cos(l)); y = np.sum(w * np.cos(p) * np.sin(l))
    z = np.sum(w * np.sin(p))
    n = np.hypot(np.hypot(x, y), z)
    if n <= 0:
        return np.nan, np.nan, tw
    return (float(np.degrees(np.arcsin(z / n))),
            float(np.degrees(np.arctan2(y, x))), tw)


def _tangent(w, LAT, LON, clat, clon):
    """East and north offsets from the centroid, azimuthal equidistant, in km."""
    d = gc_km(clat, clon, LAT, LON)
    th = bearing_deg(clat, clon, LAT, LON) * DEG
    return d * np.sin(th), d * np.cos(th), d


def shell_metrics(w, LAT, LON, hlat, hlon, comp_min=0.10, comp_floor=0.10,
                  cell_km=0.0):
    """Width, shape and location of one shell of the weight field.

    comp_floor is the weight a cell must carry to belong to a component at all.
    Components are counted on a thresholded mask rather than on the support of
    the field, because the support of a smooth field is a single connected set
    however many structures it contains: two well separated conduits joined by a
    skin of cells one configuration in fifty visits are counted as one, and the
    branching the analysis exists to detect is never seen. Weights here are
    fractions of the retained configurations, so the threshold reads directly as
    the agreement a cell must command.

    comp_min is then the share of the shell weight a surviving component must
    carry to be counted as a separate structure rather than as a fringe.
    """
    out = dict(weight=0.0, offset_km=np.nan, azimuth_deg=np.nan,
               r50_km=np.nan, r90_km=np.nan, anisotropy=np.nan,
               n_components=0, main_share=np.nan, clat=np.nan, clon=np.nan)
    w = np.asarray(w, float)
    tw = float(np.sum(w))
    if not np.isfinite(tw) or tw <= 0:
        return out
    clat, clon, _ = weighted_centroid(w, LAT, LON)
    out.update(weight=tw, clat=clat, clon=clon,
               offset_km=float(gc_km(hlat, hlon, clat, clon)),
               azimuth_deg=float(bearing_deg(hlat, hlon, clat, clon)))

    e, n, d = _tangent(w, LAT, LON, clat, clon)
    m = w > 0
    if m.sum() == 1:
        # one cell: the structure is narrower than the grid, not of zero width,
        # and reporting nothing here would drop every well-focused shell out of
        # the median. Width is set to zero and shape to circular, both of which
        # are what "unresolved at this sampling" means.
        out.update(r50_km=0.0, r90_km=0.0, anisotropy=1.0)
    elif m.sum() >= 2:
        ww = w[m]
        order = np.argsort(d[m])
        c = np.cumsum(ww[order]) / ww.sum()
        dd = d[m][order]
        out['r50_km'] = float(np.interp(0.5, c, dd))
        out['r90_km'] = float(np.interp(0.9, c, dd))
        ee, nn = e[m], n[m]
        cov = np.array([[np.sum(ww * ee * ee), np.sum(ww * ee * nn)],
                        [np.sum(ww * ee * nn), np.sum(ww * nn * nn)]]) / ww.sum()
        ev = np.linalg.eigvalsh(cov)
        # A structure one cell wide has no measurable short axis, and the ratio
        # of its eigenvalues is then a statement about floating point rather
        # than about shape - it reached 1e8 on a vertical path. Both eigenvalues
        # are floored at the variance a single cell would have, so the widest
        # anisotropy a one-cell-wide feature can report is its true extent
        # divided by the grid.
        floor = (cell_km / 2.0) ** 2 if cell_km else 1e-12
        ev = np.clip(ev, floor, None)
        out['anisotropy'] = float(np.sqrt(ev[1] / ev[0]))

    lab, nlab = _label_periodic(w >= comp_floor)
    if nlab:
        share = np.array([float(np.sum(w[lab == q])) / tw for q in range(1, nlab + 1)])
        out['n_components'] = int(np.sum(share >= comp_min))
        out['main_share'] = float(share.max())
    return out


def _label_periodic(mask):
    """Connected components with the longitude axis joined, four-connected.

    A structure straddling the date line is one structure. Labelling on a plain
    array splits it in two and reports a branching that is an artefact of where
    the grid was cut, which is the sort of thing that would be believed.
    """
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n <= 1:
        return lab, n
    left, right = lab[:, 0], lab[:, -1]
    pairs = {(min(a, b), max(a, b)) for a, b in zip(left, right) if a and b and a != b}
    if pairs:
        parent = {q: q for q in range(1, n + 1)}

        def find(q):
            while parent[q] != q:
                parent[q] = parent[parent[q]]
                q = parent[q]
            return q

        for a, b in pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        remap = {q: find(q) for q in range(1, n + 1)}
        roots = sorted(set(remap.values()))
        idx = {r: i + 1 for i, r in enumerate(roots)}
        out = np.zeros_like(lab)
        for q, r in remap.items():
            out[lab == q] = idx[r]
        return out, len(roots)
    return lab, n


def profile(W, depth, lat, lon, hlat, hlon, weight_floor=0.0, comp_min=0.10,
            comp_floor=0.10):
    """Shell metrics for every depth, as a list of dictionaries."""
    LO, LA = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
    LO = ((LO + 180.0) % 360.0) - 180.0
    dlat_km = abs(float(lat[1] - lat[0])) * DEG * R_E if len(lat) > 1 else 0.0
    dlon_km = (abs(float(lon[1] - lon[0])) * DEG * R_E *
               max(np.cos(hlat * DEG), 0.05)) if len(lon) > 1 else 0.0
    cell_km = float(np.sqrt(max(dlat_km * dlon_km, 1e-9)))
    rows = []
    for k, z in enumerate(np.asarray(depth, float)):
        w = np.where(W[k] > weight_floor, W[k], 0.0)
        r = shell_metrics(w, LA, LO, hlat, hlon, comp_min=comp_min,
                          comp_floor=comp_floor, cell_km=cell_km)
        r['depth_km'] = float(z)
        rows.append(r)
    return rows


def azimuth_turning(rows, zmin=400.0, zmax=2700.0, wmin=0.0, off_min=200.0):
    """Total and net change of corridor azimuth with depth, in degrees.

    Total turning is the sum of absolute shell-to-shell changes and measures
    curvature; net turning is the change from the shallowest to the deepest
    populated shell and measures whether the structure leans one way throughout.
    A structure that reverses has large total and small net turning, which is the
    signature the straight-tilt picture cannot produce.
    """
    # A shell whose centroid sits almost above the hotspot has no meaningful
    # azimuth: the bearing to a point 30 km away swings through a hundred
    # degrees on a shift of one cell, and summing those swings reports a
    # curvature that belongs to the grid. Shells displaced by less than off_min
    # are therefore left out of the turning, not counted as zero.
    az = [(r['depth_km'], r['azimuth_deg'], r['weight']) for r in rows
          if np.isfinite(r['azimuth_deg']) and r['weight'] > wmin
          and np.isfinite(r['offset_km']) and r['offset_km'] >= off_min
          and zmin <= r['depth_km'] <= zmax]
    if len(az) < 3:
        return np.nan, np.nan
    a = np.array([q[1] for q in az], float)
    d = (np.diff(a) + 180.0) % 360.0 - 180.0
    net = (a[-1] - a[0] + 180.0) % 360.0 - 180.0
    return float(np.abs(d).sum()), float(net)


def gap_fraction(rows, zmin=400.0, zmax=2700.0, floor=1e-9):
    """Share of the depth interval carrying no weight at all."""
    sel = [r for r in rows if zmin <= r['depth_km'] <= zmax]
    if not sel:
        return np.nan
    return float(np.mean([r['weight'] <= floor for r in sel]))


def path_metrics(paths, hlat, hlon):
    """Tortuosity and endpoint dispersion over an ensemble of traced paths.

    `paths` is a list of (depth, lat, lon) arrays, one per configuration.
    """
    lens, ends, deepest = [], [], []
    for z, la, lo in paths:
        if len(z) < 2:
            continue
        seg = np.sqrt(np.diff(z) ** 2 +
                      gc_km(la[:-1], lo[:-1], la[1:], lo[1:]) ** 2)
        lens.append(float(seg.sum()))
        straight = float(np.hypot(z[-1] - z[0], gc_km(la[0], lo[0], la[-1], lo[-1])))
        ends.append((float(la[-1]), float(lo[-1])))
        deepest.append(float(z[-1]))
        lens[-1] = lens[-1] / max(straight, 1e-6)
    if not lens:
        return dict(tortuosity=np.nan, endpoint_spread_km=np.nan,
                    endpoint_lat=np.nan, endpoint_lon=np.nan, n_paths=0,
                    deepest_km=np.nan)
    ela = np.array([e[0] for e in ends]); elo = np.array([e[1] for e in ends])
    clat, clon, _ = weighted_centroid(np.ones_like(ela), ela, elo)
    spread = float(np.median(gc_km(clat, clon, ela, elo)))
    return dict(tortuosity=float(np.median(lens)), endpoint_spread_km=spread,
                endpoint_lat=clat, endpoint_lon=clon, n_paths=len(lens),
                deepest_km=float(np.median(deepest)))


def jaccard_stability(masks):
    """Median pairwise Jaccard overlap of a list of boolean volumes."""
    v = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            a, b = masks[i], masks[j]
            u = np.logical_or(a, b).sum()
            if u:
                v.append(np.logical_and(a, b).sum() / u)
    return float(np.median(v)) if v else np.nan
