#!/usr/bin/env python3
"""Where each corridor meets the deep mantle, relative to the province it meets.

The plume-generation-zone hypothesis places plumes at the MARGINS of the large
low-shear-velocity provinces; other pictures place them over the interior, and
Bao et al. (2026) find several distinct domains within the African province
alone. Testing any of that needs the province as a geometrical object rather than
as a shaded region on a map: a boundary, a signed distance from it, and some
measure of where within the interior a point sits.

WHAT IS BUILT

  THE PROVINCE. Cells in a depth window above the core-mantle boundary whose
  anomaly falls below a threshold, grouped into connected regions with longitude
  joined so a province straddling the date line is one province. The two largest
  are the Pacific and African provinces; smaller ones are kept and reported
  because discarding them would decide in advance that there are only two.

  THE MARGIN DISTANCE. Signed great-circle distance from a point to the nearest
  boundary cell, negative inside the province and positive outside. This is
  computed exactly, cell by cell, for the handful of points that need it rather
  than approximated over the whole grid by a Euclidean transform, which on a
  0.5 degree grid is wrong by the cosine of the latitude.

  THE PILE THICKNESS. The height above the core-mantle boundary over which the
  anomaly stays below the threshold, at each cell. A thermochemical pile has
  topography, and "interior ridge" means a crest of that topography rather than
  simply a point far from the edge. A point is placed on the ridge or the flank
  by its thickness percentile within its own province, so the two provinces are
  compared on their own terms and not against each other's amplitude.

THE CIRCULARITY, AND HOW IT IS BROKEN

Margin distance, pile thickness and the anomaly at a corridor's end are not three
measurements. On the first run they correlated at 0.89, -0.80 and -0.78, because
all three are functions of one thresholded field: the province is defined by the
anomaly, the pile thickness is the vertical run of sub-threshold cells, and the
margin is where the anomaly crosses the threshold. Worse, the search whose
corridors are being placed minimises that same field, so a path priced on
slowness ends where the mantle is slowest, which is the thickest part of a pile.
"Corridors terminate on pile crests" is then a restatement of the cost function
rather than an observation about plumes.

--province-file breaks it. The province is defined in a DIFFERENT tomographic
model from the one the corridors were traced in, so the geometry a corridor is
placed within no longer comes from the field that placed it. Agreement between
the two then means something; agreement within one model does not. The cost of
this is that the two models must be describing the same provinces, which they do
at the scale of a province and do not at the scale of a conduit - so this is
legitimate here and would not be for anything smaller.

THE THRESHOLD

There is no natural one, so it is a percentile of the window and the analysis is
repeated over several. A province boundary defined at the slowest 15 per cent and
one defined at the slowest 25 per cent are different objects, and a margin
distance quoted without the threshold it came from is not a measurement. Every
output carries the threshold that produced it.
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morphology as M
from tomo_io import ModelSpec, load_anomaly, dedupe_lon

R_E, DEG = 6371.0, np.pi / 180.0


def province_mask(V, z, z0, z1, pct):
    """Connected slow regions in a depth window, labelled, longitude joined."""
    k = np.where((z >= z0) & (z <= z1))[0]
    field = np.nanmean(V[k], axis=0)
    thr = float(np.nanpercentile(field, pct))
    lab, n = M._label_periodic(field <= thr)
    sizes = [(int((lab == q).sum()), q) for q in range(1, n + 1)]
    sizes.sort(reverse=True)
    return field, thr, lab, [q for _, q in sizes]


def boundary_cells(lab, q):
    """Cells of province q with at least one neighbour outside it.

    The inner boundary, not the outer: a cell that belongs to the province and
    touches something that does not. Longitude wraps, latitude does not, so a
    province reaching the edge of the grid in latitude has that edge as a
    boundary, which is correct - the model stops there.
    """
    m = lab == q
    n = np.roll(m, 1, axis=1) & np.roll(m, -1, axis=1)
    n &= np.pad(m[:-1], ((1, 0), (0, 0)), constant_values=False)
    n &= np.pad(m[1:], ((0, 1), (0, 0)), constant_values=False)
    return m & ~n


def pile_thickness(V, z, thr, zbase):
    """Height above the deepest shell over which the anomaly stays below thr."""
    k = np.where(z <= zbase)[0]
    k = k[np.argsort(z[k])][::-1]                 # deepest first
    slow = V[k] <= thr
    # the run of consecutive slow shells starting at the base
    run = np.zeros(slow.shape[1:], int)
    alive = np.ones(slow.shape[1:], bool)
    for q in range(len(k)):
        alive &= slow[q]
        run += alive
    dz = float(np.median(np.abs(np.diff(np.sort(z[k])))))
    return run * dz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
    ap.add_argument('--z0', type=float, default=2600.0,
                    help='top of the window the province is defined in')
    ap.add_argument('--z1', type=float, default=2880.0)
    ap.add_argument('--pct', nargs='+', type=float, default=[15.0, 20.0, 25.0],
                    help='thresholds, as percentiles of the window')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--suffix', default='',
                    help='selects the path archive to read, '
                         'morph_paths_<tag><suffix>.npz. It does NOT name the '
                         'output: the output carries the province model instead, '
                         'so two runs differing only in where the provinces came '
                         'from cannot overwrite each other')
    ap.add_argument('--out-suffix', default=None, dest='out_suffix',
                    help='override the automatic output name')
    ap.add_argument('--province-vote', default=None, dest='province_vote',
                    help='province_vote.npz from province_agreement.py. The '
                         'province is then the set of cells at least --min-votes '
                         'independent inversion families agree on, which is the '
                         'defensible object: single models place the boundary a '
                         'median of 375 km apart, so a margin distance smaller '
                         'than that is not resolved by any one of them')
    ap.add_argument('--min-votes', type=int, default=3, dest='min_votes')
    ap.add_argument('--province-file', default=None, dest='province_file',
                    help='define the provinces in THIS model rather than in the '
                         'one the corridors came from, breaking the circularity '
                         'between the field the search minimises and the geometry '
                         'it is placed in')
    ap.add_argument('--province-tag', default=None, dest='province_tag')
    ap.add_argument('--province-var', default='voigt', dest='province_var')
    ap.add_argument('--min-cells', type=int, default=200, dest='min_cells',
                    help='a labelled region smaller than this is not a province')
    a = ap.parse_args()

    if a.province_vote:
        vz = np.load(a.province_vote)
        vote, lat, lon = vz['vote'], vz['lat'], vz['lon']
        nfam = int(vz['n_families'][0])
        prov_src = f'vote>={a.min_votes}of{nfam}'
        print(f'provinces from the agreement of at least {a.min_votes} of '
              f'{nfam} independent families', flush=True)
        # the vote count stands in for the anomaly: a cell is province where the
        # families agree, and "thr" is the vote level rather than a velocity
        arr = -vote[None, :, :].astype(np.float32)
        depth = np.array([0.5 * (a.z0 + a.z1)])
        # the vote level stands in for the percentile, so the summary and the
        # output column mean "agreed by this many families" rather than nothing
        a.pct = [float(a.min_votes)]
    elif a.province_file:
        ptag = a.province_tag or os.path.splitext(
            os.path.basename(a.province_file))[0]
        depth, lat, lon, arr = load_anomaly(a.province_file,
                                            ModelSpec(ptag, a.province_var),
                                            depth_max=a.depth_max, every=a.every)
        print(f'provinces defined in {ptag}, corridors traced in {a.tag}: the '
              f'geometry and the search no longer share a field', flush=True)
        prov_src = ptag
    else:
        depth, lat, lon, arr = load_anomaly(a.file, ModelSpec(a.tag, a.var),
                                            depth_max=a.depth_max, every=a.every)
        print(f'!! provinces and corridors both from {a.tag}. Margin distance, '
              f'pile thickness and\n!! endpoint anomaly are then functions of the '
              f'same field the search minimises,\n!! and none of them can '
              f'corroborate the others. Use --province-file.', flush=True)
        prov_src = a.tag
    if not a.province_vote:
        lon, arr = dedupe_lon(lon, arr)
    z = np.asarray(depth, float)
    LO, LA = np.meshgrid(lon, lat)
    LO = ((LO + 180.0) % 360.0) - 180.0

    P = np.load(os.path.join(a.dir, f'morph_paths_{a.tag}{a.suffix}.npz'))
    ends = {}
    for k in P.files:
        pz, pla, plo = P[k].astype(float)
        ends.setdefault(k.split('|')[0], []).append((pla[-1], plo[-1], pz[-1]))
    sites = {s: (float(np.median([v[0] for v in u])),
                 float(np.median([v[1] for v in u])),
                 float(np.median([v[2] for v in u]))) for s, u in ends.items()}
    print(f'{len(sites)} hotspots with a basal corridor end', flush=True)

    rows = []
    for pct in a.pct:
        if a.province_vote:
            field = arr[0]
            thr = -float(a.min_votes)
            lab, n = M._label_periodic(field <= thr)
            sizes = sorted(((int((lab == q).sum()), q)
                            for q in range(1, n + 1)), reverse=True)
            order = [q for _, q in sizes]
        else:
            field, thr, lab, order = province_mask(arr, z, a.z0, a.z1, pct)
        keep = [q for q in order if (lab == q).sum() >= a.min_cells]
        area = {q: float(np.sum((lab == q) * np.cos(LA * DEG)))
                / float(np.sum(np.cos(LA * DEG))) for q in keep}
        # name the two largest by where their area-weighted centre falls
        # Named by which canonical province centre the area-weighted centroid is
        # nearer to, and each name used once. A rule on longitude alone gave both
        # of the two largest the same name whenever their centroids happened to
        # fall the same side of the cut.
        CANON = (('Pacific', -15.0, -170.0), ('African', -5.0, 25.0))
        names, taken = {}, set()
        cents = {}
        for q in keep:
            w = (lab == q) * np.cos(LA * DEG)
            cents[q] = M.weighted_centroid(w, LA, LO)[:2]
        for q in keep[:2]:
            clat, clon = cents[q]
            best = sorted(((float(M.gc_km(clat, clon, la_, lo_)), nm)
                           for nm, la_, lo_ in CANON))
            names[q] = next((nm for _, nm in best if nm not in taken), f'region{q}')
            taken.add(names[q])
        for q in keep[2:]:
            clat, clon = cents[q]
            names[q] = f'other({clat:+.0f},{clon:+.0f})'
        # A vote map has no depth axis, so there is no pile to measure. Saying
        # so is better than returning a thickness of zero that would be read as
        # a thin pile rather than as an absent measurement.
        thick = (np.full(field.shape, np.nan) if a.province_vote
                 else pile_thickness(arr, z, thr, a.z1))
        print(f'\nthreshold {pct:g} per cent = {thr:+.3f}: {len(keep)} regions '
              f'of at least {a.min_cells} cells, areas '
              + ', '.join(f'{names[q]} {100 * area[q]:.1f}%' for q in keep))
        for q in keep[:2]:
            print(f'    {names[q]:>8s} centroid {cents[q][0]:+.0f}, '
                  f'{cents[q][1]:+.0f}')

        edges = {q: boundary_cells(lab, q) for q in keep}
        for s, (slat, slon, sdep) in sites.items():
            d = M.gc_km(slat, slon, LA, LO)
            j, i = np.unravel_index(np.argmin(d), d.shape)
            q = int(lab[j, i])
            inside = q in keep
            # nearest boundary of the province it is in, or of the nearest one
            best = None
            for qq in keep:
                e = edges[qq]
                if not e.any():
                    continue
                dd = float(np.min(d[e]))
                if best is None or dd < best[0]:
                    best = (dd, qq)
            margin = best[0] if best else np.nan
            host = q if inside else (best[1] if best else -1)
            sign = -1.0 if inside else 1.0
            th = float(thick[j, i])
            inprov = (thick[lab == host] if host in keep else np.array([]))
            inprov = inprov[np.isfinite(inprov)] if inprov.size else inprov
            th_pct = (float(100.0 * np.mean(inprov <= th))
                      if inprov.size and np.isfinite(th) else np.nan)
            rows.append(dict(site=s, province_model=prov_src,
                             threshold_pct=pct, threshold_dvs=thr,
                             end_lat=slat, end_lon=slon, end_depth_km=sdep,
                             inside=bool(inside),
                             province=names.get(host, 'none'),
                             margin_km=sign * margin,
                             # on the vote path this column holds the number of
                             # families agreeing at the endpoint, not a velocity;
                             # it was printed under a dVs heading and read as
                             # "-4.00 per cent" when it meant "all four agree"
                             dvs_at_end=(np.nan if a.province_vote
                                         else float(field[j, i])),
                             votes_at_end=(float(-field[j, i])
                                           if a.province_vote else np.nan),
                             # the region a point falls in may exist and still be
                             # too small to be a province; without this, a
                             # hotspot inside its own small slow patch reads as
                             # simply outside, which is not the same thing
                             host_cells=int((lab == q).sum()) if q else 0,
                             host_kept=bool(q in keep),
                             thickness_km=th, thickness_pct=th_pct))
    d = pd.DataFrame(rows)
    osfx = a.out_suffix if a.out_suffix is not None else (
        f'{a.suffix}_prov-{prov_src}' if a.province_file else a.suffix)
    outfile = os.path.join(a.dir, f'llsvp_geometry_{a.tag}{osfx}.csv')
    d.to_csv(outfile, index=False)

    mid = a.pct[len(a.pct) // 2]
    g = d[d.threshold_pct == mid].sort_values('margin_km')
    where = (f'agreed by at least {mid:g} of the independent families'
             if a.province_vote else f'at the {mid:g} per cent threshold')
    print(f'\n{where}, ordered from deepest inside the province to furthest '
          f'outside\n')
    vote_mode = bool(a.province_vote)
    c4 = 'votes' if vote_mode else 'dVs'
    print(f'{"hotspot":26s} {"province":>9s} {"margin km":>10s} {c4:>7s} '
          + ('' if vote_mode else f'{"pile km":>8s} {"pile pct":>9s}')
          + f'{"own patch":>11s}')
    for _, r in g.iterrows():
        v = r.votes_at_end if vote_mode else r.dvs_at_end
        patch = ('-' if r.host_kept or not r.host_cells
                 else f'{int(r.host_cells)} cells')
        print(f'{r.site:26s} {r.province:>9s} {r.margin_km:10.0f} {v:7.2f} '
              + ('' if vote_mode
                 else f'{r.thickness_km:8.0f} {r.thickness_pct:9.0f}')
              + f'{patch:>11s}')
    print(f'\nprovinces from {prov_src}, corridors from {a.tag}.')
    print('margin distance is negative inside the province and positive outside;')
    print('pile pct is the thickness percentile within the host province, so a')
    print('high value is a crest of the pile and a low one its thin flank.')
    print(f'\nwrote {os.path.basename(outfile)}, all {len(a.pct)} thresholds')


if __name__ == '__main__':
    main()
