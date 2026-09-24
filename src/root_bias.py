#!/usr/bin/env python3
"""Is a corridor's deep end where the plume is, or where the province is?

The search minimises a cost that falls with slowness, and the slowest material
in the lowermost mantle is the interior of a large low-shear-velocity province.
A descending path that is free to choose where it bottoms out will therefore
tend to bottom out in a province interior WHETHER OR NOT a plume is there. Every
statement of the form "these hotspots root inside the provinces and those root at
the margins" is worthless until that tendency is measured, because the interior
is the cost function's preferred destination by construction.

This is not a hypothetical worry. It is the direct consequence of the same
property that makes the search work at all, and it has the same standing as the
resolution floor: a number the analysis has to carry, not an objection to be
argued away.

TWO MEASUREMENTS, IN INCREASING COST

  THE SINK TEST reuses paths that have already been traced. The run carries
  null sites - surface points drawn uniformly over the sphere, with no hotspot
  and no reason to have a plume beneath them. Their corridors are priced and
  traced by exactly the machinery that priced and traced the hotspots. Where
  their deep ends fall, relative to the provinces, is where the search sends a
  path in the absence of a plume. Three rates are then comparable:

      the province area fraction   - what landing at random would give
      the null endpoint rate       - what the search gives with no plume
      the hotspot endpoint rate    - what is observed

  The first-to-second step is the search's attraction to provinces. The
  second-to-third step is the only part that can be attributed to plumes. A
  hotspot rate compared against the area fraction, skipping the middle term, is
  the mistake this test exists to prevent.

  THE PULL TEST costs a cost field per site. Conduits are injected with their
  roots at KNOWN signed margin distances, from deep inside a province to well
  outside it, and recovered by the same search. Recovered margin distance
  against true margin distance gives the bias directly, in kilometres: a slope
  below one is inward pull, and an intercept below zero is a systematic offset
  toward the interior. The conduit is injected vertically and the search is
  started directly above its root, so any lateral movement of the recovered end
  is bias and not tilt.

  Each injected site is paired with a trace from the same surface point through
  the UNMODIFIED model. That control says where the background alone would have
  sent the path, so the injected result can be read as the conduit winning
  against the province rather than merely as agreeing with it.

WHAT THE OUTCOME LICENSES

  If the null endpoints are no more likely to land inside a province than the
  area fraction, and the pull test returns a slope near one, then margin distance
  is a measurement and the interior-against-margin contrast can be made - subject
  still to the 375 km disagreement between province models, which is a separate
  and additive uncertainty.

  If the null endpoints pile into the provinces, the correct statement is that
  the deep end of a corridor locates the province and not the plume root, and the
  interior-against-margin contrast cannot be made from corridor endpoints at all.
  That is a publishable negative result about a widely used class of method, and
  it is considerably better than the same contrast published without the test.
"""
from __future__ import annotations

import argparse, os, sys
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morphology as M
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from llsvp_geometry import province_mask, boundary_cells
from contrast import contrast_field
from path_cost import cost_field, channel_field
from corridor import trace_consistent
from inject import inject_tilted

R_E, DEG = 6371.0, np.pi / 180.0
SIGMA, N_RELAX, SEED_DEPTH = 800.0, 6, 200.0
CANON = (('Pacific', -15.0, -170.0), ('African', -5.0, 25.0))


# ---------------------------------------------------------------- provinces

def load_province(a):
    """The province geometry, from a vote map or from a second model.

    Returns the label array, the kept province ids, their names, the coordinate
    meshes and the cos-weighted area fraction of all provinces together. The
    area fraction is the chance-level rate the null rate has to be judged
    against, so it is computed here rather than left to the caller.
    """
    if a.province_vote:
        vz = np.load(a.province_vote)
        vote, lat, lon = vz['vote'], vz['lat'], vz['lon']
        nfam = int(vz['n_families'][0])
        src = f'vote>={a.min_votes}of{nfam}'
        field = -vote.astype(np.float32)
        lab, n = M._label_periodic(field <= -float(a.min_votes))
        order = [q for _, q in sorted(((int((lab == q).sum()), q)
                                       for q in range(1, n + 1)), reverse=True)]
    else:
        depth, lat, lon, arr = load_anomaly(
            a.province_file, ModelSpec(a.province_tag or 'prov', a.province_var),
            depth_max=a.depth_max, every=a.every)
        lon, arr = dedupe_lon(lon, arr)
        field, thr, lab, order = province_mask(arr, np.asarray(depth, float),
                                               a.z0, a.z1, a.pct)
        src = a.province_tag or os.path.splitext(
            os.path.basename(a.province_file))[0]
    LO, LA = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
    LO = ((LO + 180.0) % 360.0) - 180.0
    keep = [q for q in order if (lab == q).sum() >= a.min_cells]
    w = np.cos(LA * DEG)
    frac = float(np.sum(np.isin(lab, keep) * w) / np.sum(w))
    names, taken = {}, set()
    for q in keep[:2]:
        cl, co = M.weighted_centroid((lab == q) * w, LA, LO)[:2]
        best = sorted(((float(M.gc_km(cl, co, la_, lo_)), nm)
                       for nm, la_, lo_ in CANON))
        names[q] = next((nm for _, nm in best if nm not in taken), f'region{q}')
        taken.add(names[q])
    for q in keep[2:]:
        cl, co = M.weighted_centroid((lab == q) * w, LA, LO)[:2]
        names[q] = f'other({cl:+.0f},{co:+.0f})'
    edges = {q: boundary_cells(lab, q) for q in keep}
    return dict(lab=lab, keep=keep, names=names, edges=edges, LA=LA, LO=LO,
                lat=np.asarray(lat, float), lon=np.asarray(lon, float),
                area_frac=frac, source=src)


def signed_margin(lat0, lon0, P):
    """Signed great-circle distance to the nearest province boundary.

    Negative inside, positive outside, computed cell by cell for this one point
    rather than by a Euclidean transform of the whole grid, which on a half
    degree mesh is wrong by the cosine of the latitude.
    """
    d = M.gc_km(lat0, lon0, P['LA'], P['LO'])
    j, i = np.unravel_index(np.argmin(d), d.shape)
    q = int(P['lab'][j, i])
    inside = q in P['keep']
    best = None
    for qq in P['keep']:
        e = P['edges'][qq]
        if not e.any():
            continue
        dd = float(np.min(d[e]))
        if best is None or dd < best[0]:
            best = (dd, qq)
    if best is None:
        return np.nan, 'none', inside
    return ((-1.0 if inside else 1.0) * best[0],
            P['names'].get(best[1], 'none'), inside)


def signed_field(P, max_edge=1200):
    """Signed margin distance over the whole grid, negative inside a province.

    Built by taking the exact great-circle distance to a subsample of boundary
    cells rather than by a Euclidean distance transform, which on a latitude
    -longitude grid is wrong by the cosine of the latitude. The subsample makes
    the field accurate to roughly the spacing between the boundary cells kept,
    which is far below anything this analysis claims, and it is used only for
    distributions over many points; the handful of real hotspot positions are
    still measured exactly by signed_margin.
    """
    if 'signed' in P:
        return P['signed']
    LA, LO = P['LA'], P['LO']
    dmin = np.full(LA.shape, np.inf)
    for q in P['keep']:
        ej, ei = np.where(P['edges'][q])
        if not len(ej):
            continue
        step = max(1, len(ej) // max_edge)
        for j, i in zip(ej[::step], ei[::step]):
            np.minimum(dmin, M.gc_km(LA[j, i], LO[j, i], LA, LO), out=dmin)
    P['signed'] = np.where(np.isin(P['lab'], P['keep']), -dmin, dmin)
    return P['signed']


def province_geometry(P, resolution, n=40000, seed=3):
    """How far from a margin a province actually is, over its own area.

    Davies, Goes and Sambridge (2015) showed that the apparent concentration of
    hotspots near the African margin follows from that province being elongated:
    more than three quarters of its interior lies within ten degrees of its own
    edge. A margin association is therefore not evidence for margin generation
    unless it beats a sample drawn uniformly over the province area, and that is
    the null used here rather than a uniform sample over the sphere.

    Reported alongside is the fraction of each province lying further from its
    margin than the disagreement between province models, which is the part of
    the province where an interior classification survives that uncertainty.
    """
    sf = signed_field(P)
    LA = P['LA']
    w = np.cos(LA * DEG)
    rows = []
    rng = np.random.default_rng(seed)
    for q in P['keep']:
        m = P['lab'] == q
        d = sf[m]
        ww = w[m]
        pr = ww / ww.sum()
        idx = rng.choice(len(d), size=min(n, 8 * len(d)), p=pr, replace=True)
        smp = d[idx]
        rows.append(dict(province=P['names'].get(q, str(q)),
                         cells=int(m.sum()),
                         area_pct=100.0 * float(ww.sum() / w.sum()),
                         median_margin_km=float(np.median(smp)),
                         within_1100km_pct=100.0 * float(np.mean(smp > -1100.0)),
                         beyond_resolution_pct=100.0 * float(
                             np.mean(smp < -resolution)),
                         sample=smp))
    return rows


# ---------------------------------------------------------------- statistics

def fisher_one_sided(a11, a12, a21, a22):
    """P(X >= a11) for the 2x2 table, exactly, by the hypergeometric sum.

    Written out rather than taken from scipy so that the test runs wherever the
    rest of the workflow runs, and so that the direction being tested is visible
    in the code instead of hidden in a keyword.
    """
    n, r1, c1 = a11 + a12 + a21 + a22, a11 + a12, a11 + a21
    hi = min(r1, c1)
    tot = comb(n, c1)
    return float(sum(comb(r1, x) * comb(n - r1, c1 - x)
                     for x in range(a11, hi + 1)) / tot)


def perm_median_diff(x, y, n=200000, seed=11):
    """Two-sided permutation p for a difference in medians.

    The sample sizes here are a dozen hotspots against a few dozen nulls, so an
    exact enumeration is out of reach and the resampled p is quoted with the
    number of resamples that produced it.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, 0
    obs = abs(float(np.median(x) - np.median(y)))
    pool = np.concatenate([x, y])
    k = len(x)
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(float(np.median(pool[:k]) - np.median(pool[k:]))) >= obs - 1e-12:
            hit += 1
    return (hit + 1) / (n + 1), n


# ---------------------------------------------------------------- sink test

def sink(a, P):
    """Where the already-traced corridors end, hotspots against nulls.

    The full run traces its null sites but deliberately does not store their
    paths, so this reads whatever is in the archive and defers to `endpoints`
    when the nulls are missing.
    """
    f = os.path.join(a.dir, f'morph_paths_{a.tag}{a.suffix}.npz')
    if not os.path.exists(f):
        sys.exit(f'no traced paths at {f}')
    Z = np.load(f)
    ends = {}
    for k in Z.files:
        pz, pla, plo = Z[k].astype(float)
        ends.setdefault(k.split('|')[0], []).append((pla[-1], plo[-1], pz[-1]))
    rows = []
    for s, u in ends.items():
        la = float(np.median([v[0] for v in u]))
        lo = float(np.median([v[1] for v in u]))
        dz = float(np.median([v[2] for v in u]))
        m, prov, ins = signed_margin(la, lo, P)
        rows.append(dict(site=s,
                         kind='null_site' if s.startswith('null') else 'hotspot',
                         end_lat=la, end_lon=lo, end_depth_km=dz,
                         margin_km=m, province=prov, inside=bool(ins),
                         n_configs=len(u)))
    d = pd.DataFrame(rows).sort_values(['kind', 'margin_km'])
    out = os.path.join(a.dir, f'root_bias_sink_{a.tag}{a.suffix}.csv')
    d.to_csv(out, index=False)
    return sink_stats(d, P, a, out)


def endpoints(a, P):
    """Trace hotspots and nulls together, in one configuration, and compare.

    The stored run priced sixty nulls and threw their paths away, and re-running
    it to recover them would cost hours for a quantity that needs one cost field.
    Tracing both populations here also removes a mismatch that would otherwise
    have to be argued away: the hotspot endpoints and the null endpoints come
    from the same field, the same move set and the same grid spacing, so the
    comparison between them is not carrying a resolution difference.

    A null is only a null if it is treated exactly as a hotspot, which includes
    the requirement that its corridor reach the deep mantle at all. The fraction
    of nulls that do is reported, because a search that drives every surface
    point to the base is making a different claim from one that does not.
    """
    depth, lat, lon, arr = load_anomaly(a.file, ModelSpec(a.tag, a.var),
                                        depth_max=a.depth_max, every=a.every)
    lon, arr = dedupe_lon(lon, arr)
    print(f'{a.tag}: {len(depth)} shells, {len(lat)}x{len(lon)}', flush=True)
    print(f'provinces on a {P["lab"].shape[0]}x{P["lab"].shape[1]} grid',
          flush=True)
    # The two grids do not have to agree. A corridor end is a latitude and a
    # longitude, and signed_margin measures from those to the nearest province
    # boundary cell by great-circle distance, so a model node-registered from
    # -90 to 90 and a vote map cell-centred at the half degree describe the same
    # sphere. An earlier version required the shapes to match and refused to run
    # on exactly the pair of grids this study uses.
    con = contrast_field(arr, lat, lon, SIGMA)
    C = cost_field(arr, con, depth, lat, lon, s=a.s, z_target=a.z_target,
                   n_relax=N_RELAX, channel=a.channel)
    print('cost field done', flush=True)

    sites = []
    H = pd.read_csv(a.hotspots, comment='#')
    for r in H.itertuples():
        sites.append((str(r.hotspot), float(r.lat), float(r.lon_180), 'hotspot'))
    rng = np.random.default_rng(a.null_seed)
    for q in range(a.nulls):
        # the label is null_site rather than null because pandas reads the bare
        # string "null" back from a csv as a missing value, which silently
        # emptied the null population on the first run of this test
        sites.append((f'null{q:03d}',
                      float(np.degrees(np.arcsin(2 * rng.random() - 1))),
                      float(360.0 * rng.random() - 180.0), 'null_site'))
    print(f'{len(sites)} sites, {a.nulls} of them nulls', flush=True)

    # A traced path stops at the cost field's target surface, not at the core
    # mantle boundary, so "reached the deep mantle" means reaching that surface.
    # Judging it against a depth below the target marked every path a failure,
    # hotspots included, and emptied the comparison.
    basal = a.basal if a.basal is not None else a.z_target - 50.0
    print(f'a corridor counts as reaching the deep mantle at {basal:g} km',
          flush=True)
    rows = []
    for k, (name, la_, lo_, kind) in enumerate(sites):
        tr = trace_consistent(C, arr, con, depth, lat, lon, la_, lo_, s=a.s,
                              radius_deg=2.0, n_relax=N_RELAX,
                              channel=a.channel)
        rec = dict(site=name, kind=kind, start_lat=la_, start_lon=lo_)
        if tr is None:
            rec.update(reached=False)
        else:
            pz, pla, plo = tr
            dz = float(pz[-1])
            rec.update(end_lat=float(pla[-1]), end_lon=float(plo[-1]),
                       end_depth_km=dz, reached=bool(dz >= basal))
            if rec['reached']:
                m, prov, ins = signed_margin(float(pla[-1]), float(plo[-1]), P)
                rec.update(margin_km=m, province=prov, inside=bool(ins))
                rec['moved_km'] = float(M.gc_km(la_, lo_, pla[-1], plo[-1]))
        rows.append(rec)
        if (k + 1) % 20 == 0:
            print(f'  {k + 1}/{len(sites)}', flush=True)
    d = pd.DataFrame(rows)
    out = os.path.join(a.dir, f'root_bias_endpoints_{a.tag}{a.suffix}.csv')
    d.to_csv(out, index=False)
    for kind in ('hotspot', 'null_site'):
        g = d[d.kind == kind]
        if len(g):
            print(f'{kind:10s} {int(g.reached.sum()):3d} of {len(g):3d} '
                  f'corridors reach {basal:g} km')
    return sink_stats(d[d.reached.fillna(False)], P, a, out)


def sink_stats(d, P, a, out):
    geo = province_geometry(P, a.resolution)
    print(f'\n{"province":22s} {"area":>6s} {"median":>8s} {"within":>8s} '
          f'{"beyond":>8s}')
    print(f'{"":22s} {"":>6s} {"margin":>8s} {"1100 km":>8s} '
          f'{a.resolution:6.0f} km')
    for g in geo:
        print(f'{g["province"]:22s} {g["area_pct"]:5.1f}% '
              f'{g["median_margin_km"]:8.0f} {g["within_1100km_pct"]:7.0f}% '
              f'{g["beyond_resolution_pct"]:7.0f}%')
    print('an elongated province is mostly near its own edge, so a hotspot')
    print('close to a margin is only evidence for margin generation if it is')
    print('closer than a point drawn at random from inside the province')

    h = d[d.kind == 'hotspot']
    nl = d[d.kind.isin(['null', 'null_site'])]
    print(f'\nprovinces from {P["source"]}, covering '
          f'{100 * P["area_frac"]:.1f} per cent of the sphere')
    if not len(nl):
        print('\nNO NULL SITES SURVIVED. The sink test cannot run, and without')
        print('it the endpoint locations carry no interpretation. In sink mode')
        print('the stored paths hold no nulls, because the full run discards')
        print('them; use --mode endpoints, which traces its own.')
        return d
    print(f'{len(h)} hotspots, {len(nl)} nulls\n')
    ih, inl = int(h.inside.sum()), int(nl.inside.sum())
    print(f'{"":22s} {"inside":>8s} {"of":>5s} {"rate":>7s}')
    print(f'{"expected by area":22s} {"":>8s} {"":>5s} '
          f'{100 * P["area_frac"]:6.1f}%')
    print(f'{"null corridors":22s} {inl:8d} {len(nl):5d} '
          f'{100 * inl / len(nl):6.1f}%')
    print(f'{"hotspot corridors":22s} {ih:8d} {len(h):5d} '
          f'{100 * ih / len(h):6.1f}%')
    p_attract = fisher_one_sided(inl, len(nl) - inl,
                                 int(round(P['area_frac'] * len(nl))),
                                 len(nl) - int(round(P['area_frac'] * len(nl))))
    p_plume = fisher_one_sided(ih, len(h) - ih, inl, len(nl) - inl)
    print(f'\nsearch attraction, nulls against area   p = {p_attract:.4f}')
    print(f'plume signal, hotspots against nulls    p = {p_plume:.4f}')
    pm, nperm = perm_median_diff(h.margin_km, nl.margin_km, a.perm)
    print(f'median margin  hotspots {np.nanmedian(h.margin_km):+8.0f} km, '
          f'nulls {np.nanmedian(nl.margin_km):+8.0f} km, '
          f'p = {pm:.4f} over {nperm} resamples')

    # the published test: hotspots that land inside a province, against points
    # drawn uniformly over that province's area
    hin = h[h.inside.fillna(False)]
    if len(hin) >= 3 and geo:
        pool = np.concatenate([g['sample'] for g in geo])
        obs = float(np.median(hin.margin_km))
        rng = np.random.default_rng(23)
        k = len(hin)
        draws = np.median(rng.choice(pool, size=(min(a.perm, 50000), k)), axis=1)
        pu = float((np.sum(np.abs(draws - np.median(pool))
                           >= abs(obs - np.median(pool))) + 1)
                   / (len(draws) + 1))
        print(f'\n{len(hin)} hotspots inside a province, median margin '
              f'{obs:+.0f} km')
        print(f'uniform over province area would give {np.median(pool):+.0f} km, '
              f'p = {pu:.4f}')
        if pu > 0.05:
            print('so their distribution within the provinces is not '
                  'distinguishable from\nuniform, which is the result Davies, '
                  'Goes and Sambridge reached in 2015')
    print(f'\n{out}')
    if p_attract < 0.05 and p_plume > 0.05:
        print('\nThe nulls land inside the provinces more often than area alone')
        print('would give, and the hotspots are not distinguishable from the')
        print('nulls. On this evidence the deep end of a corridor locates the')
        print('province and not the plume root, and no interior-against-margin')
        print('contrast can be drawn from these endpoints.')
    return d


# ---------------------------------------------------------------- pull test

_W = {}


def _sites_at(P, targets, tol, reps, seed, sep_km):
    """Grid cells whose signed margin distance is near each target.

    Replicates are forced apart by sep_km so that three sites at one target
    distance are three independent settings rather than three neighbours of one
    cell, which would give a spuriously tight spread.
    """
    LA, LO = P['LA'], P['LO']
    signed = signed_field(P)
    rng = np.random.default_rng(seed)
    out = []
    for t in targets:
        cand = np.argwhere(np.abs(signed - t) <= tol)
        if not len(cand):
            print(f'  no cells within {tol:g} km of a margin distance of {t:+g}')
            continue
        rng.shuffle(cand)
        chosen = []
        for j, i in cand:
            la_, lo_ = float(LA[j, i]), float(LO[j, i])
            if all(M.gc_km(la_, lo_, c[1], c[2]) > sep_km for c in chosen):
                chosen.append((f'm{t:+05.0f}_{len(chosen)}', la_, lo_,
                               float(signed[j, i])))
            if len(chosen) >= reps:
                break
        out.extend(chosen)
    return out


def _one_site(q):
    """One injected conduit: cost field on the modified model, then a trace."""
    name, hlat, hlon, true_m = _W['sites'][q]
    arr, depth, lat, lon = _W['arr'], _W['depth'], _W['lat'], _W['lon']
    inj = inject_tilted(arr, depth, lat, lon, hlat, hlon,
                        _W['radius'], _W['amp'], 0.0, z0=100.0, z1=2800.0)
    con = contrast_field(inj, lat, lon, SIGMA)
    C = cost_field(inj, con, depth, lat, lon, s=_W['s'], z_target=_W['zt'],
                   n_relax=N_RELAX, channel=_W['ch'])
    tr = trace_consistent(C, inj, con, depth, lat, lon, hlat, hlon, s=_W['s'],
                          radius_deg=2.0, n_relax=N_RELAX, channel=_W['ch'])
    if tr is None:
        return name, None
    pz, pla, plo = tr
    return name, (float(pla[-1]), float(plo[-1]), float(pz[-1]))


def pull(a, P):
    depth, lat, lon, arr = load_anomaly(a.file, ModelSpec(a.tag, a.var),
                                        depth_max=a.depth_max, every=a.every)
    lon, arr = dedupe_lon(lon, arr)
    print(f'{a.tag}: {len(depth)} shells, {len(lat)}x{len(lon)}', flush=True)
    # injection sites are chosen on the province grid and injected by latitude
    # and longitude, so the two grids are independent here as well
    targets = [float(t) for t in a.targets]
    sites = _sites_at(P, targets, a.tol, a.reps, a.seed, a.separation)
    if not sites:
        sys.exit('no injection sites found')
    print(f'{len(sites)} injection sites over {len(targets)} target distances',
          flush=True)

    con0 = contrast_field(arr, lat, lon, SIGMA)
    C0 = cost_field(arr, con0, depth, lat, lon, s=a.s, z_target=a.z_target,
                    n_relax=N_RELAX, channel=a.channel)
    print('background cost field done; tracing controls', flush=True)
    base = {}
    for name, hlat, hlon, _ in sites:
        tr = trace_consistent(C0, arr, con0, depth, lat, lon, hlat, hlon,
                              s=a.s, radius_deg=2.0, n_relax=N_RELAX,
                              channel=a.channel)
        base[name] = (None if tr is None
                      else (float(tr[1][-1]), float(tr[2][-1])))
    del C0

    _W.update(arr=arr, depth=depth, lat=lat, lon=lon, sites=sites,
              radius=a.radius, amp=a.amp, s=a.s, zt=a.z_target, ch=a.channel)
    res = {}
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(a.jobs) as pool:
            for k, (nm, v) in enumerate(pool.imap_unordered(
                    _one_site, range(len(sites))), 1):
                res[nm] = v
                print(f'  {k}/{len(sites)} {nm}', flush=True)
    else:
        for k in range(len(sites)):
            nm, v = _one_site(k)
            res[nm] = v
            print(f'  {k + 1}/{len(sites)} {nm}', flush=True)

    rows = []
    for name, hlat, hlon, true_m in sites:
        r = res.get(name)
        rec = dict(site=name, root_lat=hlat, root_lon=hlon,
                   true_margin_km=true_m)
        if r is None:
            rec.update(rec_margin_km=np.nan, pull_km=np.nan,
                       moved_km=np.nan, base_margin_km=np.nan)
        else:
            rm, prov, ins = signed_margin(r[0], r[1], P)
            rec.update(rec_lat=r[0], rec_lon=r[1], rec_depth_km=r[2],
                       rec_margin_km=rm, province=prov, inside=bool(ins),
                       pull_km=rm - true_m,
                       moved_km=float(M.gc_km(hlat, hlon, r[0], r[1])))
            b = base.get(name)
            rec['base_margin_km'] = (np.nan if b is None
                                     else signed_margin(b[0], b[1], P)[0])
        rows.append(rec)
    d = pd.DataFrame(rows)
    out = os.path.join(a.dir, f'root_bias_pull_{a.tag}{a.suffix}.csv')
    d.to_csv(out, index=False)

    print(f'\n{"target km":>10s} {"n":>3s} {"recovered km":>13s} '
          f'{"pull km":>9s} {"background km":>14s}')
    for t in targets:
        g = d[np.abs(d.true_margin_km - t) <= a.tol]
        if not len(g):
            continue
        print(f'{t:10.0f} {len(g):3d} {np.nanmean(g.rec_margin_km):13.0f} '
              f'{np.nanmean(g.pull_km):9.0f} '
              f'{np.nanmean(g.base_margin_km):14.0f}')
    ok = d.dropna(subset=['true_margin_km', 'rec_margin_km'])
    if len(ok) >= 3:
        b, c = np.polyfit(ok.true_margin_km, ok.rec_margin_km, 1)
        r = float(np.corrcoef(ok.true_margin_km, ok.rec_margin_km)[0, 1])
        print(f'\nrecovered = {b:.2f} x true {c:+.0f} km, r = {r:+.2f}')
        print(f'mean pull {np.nanmean(ok.pull_km):+.0f} km, '
              f'median {np.nanmedian(ok.pull_km):+.0f} km')
        if b < 0.5 or np.nanmean(ok.pull_km) < -a.tol:
            print('\nThe recovered root is displaced toward the province')
            print('interior by more than the tolerance the sites were chosen')
            print('to, so margin distance measured from a corridor end is')
            print('biased and the interior-against-margin contrast needs a')
            print('correction of this size before it can be read.')
    print(f'\n{out}')
    return d


def main():
    ap = argparse.ArgumentParser(
        description='measure whether corridor ends locate plumes or provinces')
    ap.add_argument('--mode', choices=['geometry', 'sink', 'endpoints', 'pull'],
                    default='endpoints',
                    help='endpoints traces hotspots and nulls together in one '
                         'configuration; sink reuses stored paths, which carry '
                         'no nulls; pull injects roots at known margin distances')
    ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
    ap.add_argument('--nulls', type=int, default=300)
    ap.add_argument('--null-seed', type=int, default=7, dest='null_seed')
    ap.add_argument('--basal', type=float, default=None,
                    help='depth at which a corridor counts as reaching the deep '
                         'mantle, hotspots and nulls judged alike; defaults to '
                         'just above the cost target, which is where a traced '
                         'path ends')
    ap.add_argument('--file', default=None,
                    help='model the corridors are traced in; needed for pull')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--province-vote', default=None, dest='province_vote')
    ap.add_argument('--min-votes', type=int, default=3, dest='min_votes')
    ap.add_argument('--province-file', default=None, dest='province_file')
    ap.add_argument('--province-tag', default=None, dest='province_tag')
    ap.add_argument('--province-var', default='voigt', dest='province_var')
    ap.add_argument('--z0', type=float, default=2600.0)
    ap.add_argument('--z1', type=float, default=2880.0)
    ap.add_argument('--pct', type=float, default=20.0)
    ap.add_argument('--min-cells', type=int, default=200, dest='min_cells')
    ap.add_argument('--perm', type=int, default=200000)
    ap.add_argument('--resolution', type=float, default=375.0,
                    help='median disagreement between province models, in km')
    ap.add_argument('--targets', nargs='+', type=float,
                    default=[-1500, -1000, -600, -300, 0, 300, 600])
    ap.add_argument('--tol', type=float, default=100.0,
                    help='how near a cell must be to a target margin distance')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--separation', type=float, default=2000.0)
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--radius', type=float, default=300.0)
    ap.add_argument('--amp', type=float, default=-1.5)
    ap.add_argument('--s', type=float, default=0.5)
    ap.add_argument('--z-target', type=float, default=2400.0, dest='z_target')
    ap.add_argument('--channel', default='min')
    ap.add_argument('--jobs', type=int, default=1)
    a = ap.parse_args()
    if not a.province_vote and not a.province_file:
        sys.exit('give --province-vote or --province-file: a province taken '
                 'from the same model the corridors were traced in cannot test '
                 'anything about where those corridors end')
    P = load_province(a)
    if a.mode == 'geometry':
        for g in province_geometry(P, a.resolution):
            print(f'{g["province"]:22s} area {g["area_pct"]:5.1f}%  median '
                  f'margin {g["median_margin_km"]:6.0f} km  within 1100 km '
                  f'{g["within_1100km_pct"]:3.0f}%  beyond '
                  f'{a.resolution:.0f} km {g["beyond_resolution_pct"]:3.0f}%')
    elif a.mode == 'sink':
        sink(a, P)
    elif a.mode == 'endpoints':
        if not a.file:
            sys.exit('--file is required to trace endpoints')
        endpoints(a, P)
    else:
        if not a.file:
            sys.exit('--file is required for the pull test')
        pull(a, P)


if __name__ == '__main__':
    main()
