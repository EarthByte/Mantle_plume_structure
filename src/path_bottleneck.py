#!/usr/bin/env python3
"""The bottleneck path: the descending route whose WORST material is as slow as possible.

NAMED path_bottleneck, not bottleneck, and it has to stay that way. `bottleneck` is a
real package that pandas imports as an optional accelerator, and because scripts here
run from this directory a module of that name shadows it: pandas asks the shadow for
its __version__, does not find one, and every import of pandas in this workflow dies.

WHY NOT CONNECTED COMPONENTS. The first attempt bisected on a level and took the
connected body containing the hotspot. It failed, and the failure is instructive:
at every level where a body reaches 2700 km the slow cells already percolate
globally, so the body is the whole slow mantle and its centroid sits 8,000 to
16,000 km from the hotspot. Connectivity at a fixed level is a property of the
planet, not of a plume. The bottleneck has to be a property of a PATH.

THE FIELD. B[k, x] is the best achievable worst-anomaly over all descending routes
from cell x to the target depth: the min over routes of the max over the route.
It obeys the same dynamic program the cost field does with (sum, plus) replaced by
(max, min), so it is built by the same shell-by-shell sweep and the same
within-shell relaxation, and it needs no threshold anywhere.

THE RULE IS LEXICOGRAPHIC. The bottleneck alone is degenerate - many routes share
one worst point - so among the routes that achieve it we take the cheapest under
the existing cost. That returns a single path and leaves the calibrated cost
function doing the fine-grained work it is good at, while continuity decides the
question it was never able to answer.
"""
from __future__ import annotations
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_cost import (SLANT_KM, _OFF, _cap_indices, _n_horiz, _offsets, _shift2,
                       _spans, cell_widths, channel_field)


def _relax_max(cur, a_k):
    """One min-max relaxation sweep within a shell.

    Stepping sideways into a neighbour costs nothing but forces the path through
    the material it lands on, so the value carried is the worse of the two.
    """
    out = cur
    for sh in (1, -1):
        m = np.maximum(np.roll(cur, sh, axis=0), a_k)
        if sh == 1:
            m[0] = np.inf
        else:
            m[-1] = np.inf
        out = np.minimum(out, m)
    for sh in (1, -1):
        out = np.minimum(out, np.maximum(np.roll(cur, sh, axis=1), a_k))
    return out


def bottleneck_field(anom, contrast, depth, lat, lon, z_target=2700.0, n_relax=6,
                     channel='min', a_clip=6.0, slant_km=SLANT_KM, h_max_km=None,
                     ncell=1):
    """B[k, x]: the slowest-possible worst point on a descending route to the target.

    Lower is better, in the same units as the anomaly: a value of -0.6 means every
    cell on the best route is at least 0.6 per cent slow. There is no `lateral`
    weight and no length: a bottleneck does not care how far the route runs, only
    what it must pass through. That is the whole point of the criterion.
    """
    z = np.asarray(depth, float)
    a = np.clip(channel_field(anom, contrast, channel), -a_clip, a_clip)
    dlat, _dlon = cell_widths(lat, lon)
    n_relax = _n_horiz(h_max_km, dlat, n_relax)
    dk_set = _spans(z, len(z) - 1, slant_km)
    k0 = int(np.where(z >= z_target)[0][0]) if (z >= z_target).any() else len(z) - 1

    B = np.full(a.shape, np.inf)
    B[k0] = a[k0]
    for _ in range(n_relax):
        B[k0] = _relax_max(B[k0], a[k0])

    for k in range(k0 - 1, -1, -1):
        cur = np.maximum(a[k], np.maximum(a[k + 1], B[k + 1]))   # straight down
        for dk in dk_set:
            kk = k + dk
            if kk > k0:
                continue
            # a slanted move passes through every shell it spans, so the worst of
            # them is what the route has to accept
            span = a[k:kk + 1].max(axis=0)
            for dj, di in _offsets(ncell):
                nc = max(abs(dj), abs(di))
                w = span
                if nc > 1:
                    for t in range(1, nc):
                        w = np.maximum(w, _shift2(span, dj * t // nc, di * t // nc))
                cand = np.maximum(np.maximum(a[k], w),
                                  np.maximum(_shift2(span, dj, di),
                                             _shift2(B[kk], dj, di)))
                np.minimum(cur, cand, out=cur)
        for _ in range(n_relax):
            cur = _relax_max(cur, a[k])
        B[k] = cur
    return B


def site_bottleneck(B, depth, lat, lon, hlat, hlon, radius_deg=3.0,
                    seed_depth=200.0):
    """The best bottleneck available anywhere in the seed cap."""
    jj, ii = _cap_indices(depth, lat, lon, hlat, hlon, radius_deg, seed_depth)
    k = int(np.argmin(np.abs(np.asarray(depth, float) - seed_depth)))
    blk = B[k][np.ix_(jj, ii)]
    return float(np.nanmin(blk)) if np.isfinite(blk).any() else float('nan')
