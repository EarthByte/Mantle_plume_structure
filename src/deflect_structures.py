#!/usr/bin/env python3
"""How far does an imaged structure lean, and can a lean be told from a wander?

Deflection is the last of the axes a plume taxonomy uses and the only one with
prior reason to be recoverable: corridor displacement was partly reproducible
between independent inversion families, at rank correlations of +0.34 to +0.47,
where corridor SHAPE was not reproducible at all. That makes it worth measuring
and worth calibrating, in that order.

THE OBJECT is a connected component of the prominence field, as for branching: an
imaged structure that is slow relative to its own surroundings, found without
reference to any hotspot, and blind to the provinces by construction.

THE MEASUREMENT is the horizontal displacement of that structure's centre between
its deepest and shallowest shells, the azimuth of that displacement, and - the
part that matters - whether the drift is systematic or a wander. A structure that
contains ambient material will have a centre that moves shell to shell whether or
not anything leans, and the total displacement alone cannot tell those apart. The
monotonicity is the fraction of shell-to-shell steps whose component along the
overall drift is positive: a true lean approaches one, a random walk sits near a
half, and the calibration says which is which for this field.

THE FLOOR injects conduits of known lean into the real model, from vertical to
1600 km of offset over the lower mantle, and asks what comes back. The vertical
case is the control that decides everything: whatever displacement and
monotonicity a conduit with NO lean returns is the level below which a measured
lean means nothing. This is the same control that exposed the branch-depth
estimator, which returned the same depth for a branch, a different branch and no
branch at all.
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morphology as M
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from contrast import contrast_field
from prominence_map import label_3d_periodic
from branch_structures import prominence_of, pieces_in_shell

R_E, DEG = 6371.0, np.pi / 180.0


def bearing(lat0, lon0, lat1, lon1):
    p0, p1 = np.radians(lat0), np.radians(lat1)
    dl = np.radians(lon1 - lon0)
    y = np.sin(dl) * np.cos(p1)
    x = np.cos(p0) * np.sin(p1) - np.sin(p0) * np.cos(p1) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def chains_from_field(P, LA, LO, thr, link_km, min_cells=4):
    """Shell-wise components linked between adjacent shells into chains.

    Requiring three-dimensional connectivity breaks a leaning conduit: the
    calibration showed the tracked structure shortening from 1460 km when
    vertical to 540 km at 1100 km of lean, because the piece at one depth stops
    touching the piece above it. Linking by centroid distance instead, as
    component_continuity does for corridors, keeps a leaning conduit as one
    object. The link distance is the one parameter this introduces, and the
    vertical control says whether it has been set too loosely: chaining
    unrelated components together would give a vertical conduit a drift it does
    not have.
    """
    shells = []
    for k in range(P.shape[0]):
        shells.append(pieces_in_shell(P[k] >= thr, LA, LO, 0.0, min_cells))
    nodes, index = [], []
    for k, cs in enumerate(shells):
        index.append([])
        for c in cs:
            index[k].append(len(nodes))
            nodes.append((k, c['lat'], c['lon'], c['cells']))
    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k in range(len(shells) - 1):
        a_, b_ = shells[k], shells[k + 1]
        if not a_ or not b_:
            continue
        la1 = np.array([c['lat'] for c in a_]); lo1 = np.array([c['lon'] for c in a_])
        la2 = np.array([c['lat'] for c in b_]); lo2 = np.array([c['lon'] for c in b_])
        for i2 in range(len(a_)):
            d = M.gc_km(la1[i2], lo1[i2], la2, lo2)
            for j2 in np.where(d <= link_km)[0]:
                ra, rb = find(index[k][i2]), find(index[k + 1][int(j2)])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for n_, (k, la_, lo_, cells) in enumerate(nodes):
        groups.setdefault(find(n_), []).append((k, la_, lo_, cells))
    return list(groups.values())


def chain_centres(chain, n_shell):
    """One centre per shell for a chain: the largest component in that shell."""
    out = np.full((n_shell, 2), np.nan)
    best = {}
    for k, la_, lo_, cells in chain:
        if k not in best or cells > best[k][2]:
            best[k] = (la_, lo_, cells)
    for k, (la_, lo_, _) in best.items():
        out[k] = (la_, lo_)
    return out


def track_centre(mask3d, zz, LA, LO, min_cells=4):
    """Centre of the largest piece of a structure in each shell."""
    out = []
    for k in range(mask3d.shape[0]):
        ps = pieces_in_shell(mask3d[k], LA, LO, 0.0, min_cells)
        if not ps:
            out.append((np.nan, np.nan))
            continue
        b = max(ps, key=lambda p: p['cells'])
        out.append((b['lat'], b['lon']))
    return np.asarray(out, float)


def drift_stats(cen, zz):
    """Total displacement base to top, its azimuth, and how systematic it is."""
    ok = np.isfinite(cen[:, 0])
    if ok.sum() < 4:
        return dict(drift_km=np.nan, azimuth_deg=np.nan, monotonicity=np.nan,
                    span_km=np.nan)
    idx = np.where(ok)[0]
    # zz ascends with depth index in the caller, so the deepest is the last
    deep, shal = idx[-1], idx[0]
    d = float(M.gc_km(cen[deep, 0], cen[deep, 1], cen[shal, 0], cen[shal, 1]))
    az = float(bearing(cen[deep, 0], cen[deep, 1], cen[shal, 0], cen[shal, 1]))
    steps = 0, 0
    fwd = 0
    tot = 0
    for a_, b_ in zip(idx[:-1], idx[1:]):
        if not (np.isfinite(cen[a_, 0]) and np.isfinite(cen[b_, 0])):
            continue
        s = float(M.gc_km(cen[b_, 0], cen[b_, 1], cen[a_, 0], cen[a_, 1]))
        if s <= 0:
            continue
        ab = bearing(cen[b_, 0], cen[b_, 1], cen[a_, 0], cen[a_, 1])
        # component of this step along the overall drift direction
        if abs(((ab - az + 180.0) % 360.0) - 180.0) <= 90.0:
            fwd += 1
        tot += 1
    return dict(drift_km=d, azimuth_deg=az,
                monotonicity=(fwd / tot) if tot else np.nan,
                span_km=float(abs(zz[deep] - zz[shal])))


def main():
    ap = argparse.ArgumentParser(
        description='deflection of imaged structures, with a vertical control')
    ap.add_argument('--file', required=True)
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
    ap.add_argument('--zmin', type=float, default=800.0)
    ap.add_argument('--zmax', type=float, default=2700.0)
    ap.add_argument('--sigma', type=float, default=800.0)
    ap.add_argument('--threshold', type=float, default=0.65)
    ap.add_argument('--min-extent', type=float, default=1000.0, dest='min_extent')
    ap.add_argument('--offsets', nargs='+', type=float,
                    default=[0., 200., 400., 700., 1100., 1600.])
    ap.add_argument('--inject-radius', type=float, default=250.0,
                    dest='inject_radius')
    ap.add_argument('--inject-amp', type=float, default=1.2, dest='inject_amp')
    ap.add_argument('--floor-sites', type=int, default=8, dest='floor_sites')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--catchment', type=float, default=500.0)
    ap.add_argument('--link-km', type=float, default=150.0, dest='link_km',
                    help='components in adjacent shells closer than this are one '
                         'structure; 0 restores three-dimensional connectivity')
    ap.add_argument('--mode', choices=['floor', 'measure', 'both'], default='both')
    a = ap.parse_args()

    depth, lat, lon, arr = load_anomaly(a.file, ModelSpec(a.tag, a.var),
                                        depth_max=a.depth_max, every=a.every)
    lon, arr = dedupe_lon(lon, arr)
    z = np.asarray(depth, float)
    k = np.where((z >= a.zmin) & (z <= a.zmax))[0]
    k = k[np.argsort(z[k])]
    zz = z[k]
    dz = float(np.median(np.abs(np.diff(zz))))
    print(f'{a.tag}: {len(zz)} shells {zz.min():.0f}-{zz.max():.0f} km',
          flush=True)
    P0 = -contrast_field(arr[k], lat, lon, a.sigma)
    LO, LA = np.meshgrid(lon, lat)
    LO = ((LO + 180.0) % 360.0) - 180.0

    if a.mode in ('floor', 'both'):
        rng = np.random.default_rng(a.seed)
        sites = []
        while len(sites) < a.floor_sites:
            la_ = float(np.degrees(np.arcsin(2 * rng.random() - 1)))
            if abs(la_) < 50.0:
                sites.append((la_, float(360.0 * rng.random() - 180.0)))
        rows = []
        for si, (clat, clon) in enumerate(sites):
            cw = max(np.cos(np.radians(clat)), 0.2)
            for off in a.offsets:
                cache = {}

                def prom_for(o_km):
                    key = int(round(o_km / 100.0))
                    if key not in cache:
                        dlon_ = (key * 100.0) / 111.2 / cw
                        g = np.exp(-(M.gc_km(clat, clon + dlon_, LA, LO) ** 2)
                                   / (2.0 * (a.inject_radius / 2.0) ** 2))
                        cache[key] = prominence_of(-a.inject_amp * g, lat, lon,
                                                   a.sigma)
                    return cache[key]

                rad = off + 800.0
                jw = np.where(np.abs(lat - clat) <= rad / 111.2)[0]
                dl = ((lon - clon + 180.0) % 360.0) - 180.0
                iw = np.where(np.abs(dl) <= rad / 111.2 / cw)[0]
                if len(jw) < 6 or len(iw) < 6:
                    continue
                sub = np.ix_(jw, iw)
                vol = []
                for ki in range(len(zz)):
                    frac = (zz.max() - zz[ki]) / max(zz.max() - zz.min(), 1)
                    vol.append(((P0[ki] + prom_for(off * frac))[sub]
                                >= a.threshold))
                vol = np.stack(vol)
                l3, n3 = ndi.label(vol)
                jc = int(np.argmin(np.abs(lat[jw] - clat)))
                ic = int(np.argmin(np.abs(((lon[iw] - clon + 180) % 360) - 180)))
                own = [int(l3[ki, jc, ic]) for ki in range(len(zz))
                       if zz[ki] > zz.max() - 400]
                own = [o for o in own if o > 0]
                q = max(set(own), key=own.count) if own else 0
                if not q:
                    continue
                if a.link_km > 0:
                    Pw = np.stack([(P0[ki] + prom_for(
                        off * (zz.max() - zz[ki])
                        / max(zz.max() - zz.min(), 1)))[sub]
                        for ki in range(len(zz))])
                    ch = chains_from_field(Pw, LA[sub], LO[sub], a.threshold,
                                           a.link_km)
                    if not ch:
                        continue
                    # the chain whose deepest member is nearest the injection
                    def score(c):
                        deep = max(c, key=lambda t: t[0])
                        return M.gc_km(clat, clon, deep[1], deep[2])
                    best = min(ch, key=score)
                    cen = chain_centres(best, len(zz))
                else:
                    cen = track_centre(l3 == q, zz, LA[sub], LO[sub])
                st = drift_stats(cen, zz)
                st.update(site=si, injected_km=off)
                rows.append(st)
            print(f'  floor site {si + 1}/{len(sites)}', flush=True)
        fd = pd.DataFrame(rows)
        fo = os.path.join(a.dir, f'deflect_floor_{a.tag}{a.suffix}.csv')
        fd.to_csv(fo, index=False)
        t = fd.groupby('injected_km').agg(
            recovered_km=('drift_km', 'median'),
            monotonicity=('monotonicity', 'median'),
            span_km=('span_km', 'median')).round(2)
        print('\nFLOOR: a conduit injected with a known lean over the window\n')
        print(t.to_string())
        v = fd[fd.injected_km == 0]
        print(f'\na VERTICAL conduit returns {v.drift_km.median():.0f} km of '
              f'apparent drift at monotonicity {v.monotonicity.median():.2f}')
        r = fd.dropna(subset=['injected_km', 'drift_km'])
        if len(r) > 4:
            rho = float(np.corrcoef(r.injected_km, r.drift_km)[0, 1])
            print(f'recovered against injected: r = {rho:+.2f}')
        print(f'{fo}')
    if a.mode == 'floor':
        return

    print('\nbuilding linked structures over the whole field', flush=True)
    chains = chains_from_field(P0, LA, LO, a.threshold, a.link_km)
    rows, foots = [], []
    for q, ch in enumerate(chains, 1):
        ks = sorted({t[0] for t in ch})
        span = (max(ks) - min(ks) + 1) * dz
        if span < a.min_extent:
            continue
        cen = chain_centres(ch, len(zz))
        st = drift_stats(cen, zz)
        st.update(structure=q, extent_km=float(span))
        rows.append(st)
        foots.append((q, np.array([[t[1], t[2]] for t in ch])))
    S = pd.DataFrame(rows)
    print(f'{len(S)} linked structures spanning at least {a.min_extent:g} km',
          flush=True)
    H = pd.read_csv(a.hotspots, comment='#')
    hs = set()
    for q, pts in foots:
        for r in H.itertuples():
            if float(M.gc_km(r.lat, r.lon_180, pts[:, 0], pts[:, 1]).min()) \
                    <= a.catchment:
                hs.add(q)
                break
    S['has_hotspot'] = S.structure.isin(hs)
    out = os.path.join(a.dir, f'deflect_structures_{a.tag}{a.suffix}.csv')
    S.to_csv(out, index=False)
    w, wo = S[S.has_hotspot], S[~S.has_hotspot]
    from root_bias import perm_median_diff
    print(f'\n{len(w)} structures carry a hotspot within {a.catchment:g} km, '
          f'{len(wo)} do not\n')
    print(f'{"":22s} {"with":>8s} {"without":>9s} {"p":>9s}')
    for lab_, col in (('drift km', 'drift_km'),
                      ('monotonicity', 'monotonicity'),
                      ('depth span km', 'span_km')):
        x, y = w[col].dropna(), wo[col].dropna()
        if len(x) < 3 or len(y) < 3:
            continue
        p, _ = perm_median_diff(x, y, 50000)
        print(f'{lab_:22s} {np.median(x):8.2f} {np.median(y):9.2f} {p:9.4f}')
    print(f'\n{out}')


if __name__ == '__main__':
    main()
