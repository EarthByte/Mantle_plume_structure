#!/usr/bin/env python3
"""Where does the tomography offer the search more than one answer?

THE QUESTION A REVIEWER ASKS AT AFAR. The traced route there descends through a set
of interrupted anomalies east of the volcano while a continuous limb lies to the west.
Decomposed, the western detour is 2.7 per cent longer in the length the objective
charges and 0.03 per cent slower in the anomaly it integrates, where it would need
0.09 per cent to pay for itself. The two descents are separated by six hundredths of
one per cent in mean shear-velocity anomaly, which is far below what the model
resolves. Afar is not a bad fit; it is a tie, and a least-cost criterion has to break
ties.

That answer is only worth anything if the class it puts Afar in can be counted. So
this measures, for every hotspot, what the SECOND deep structure costs.

HOW. Trace the route the search takes. Then forbid everything within a stated radius
of where that route roots, below a stated depth, and solve the identical problem
again. The result is the cheapest descent that does NOT use the chosen structure, and
the ratio of the two costs is how strongly the field prefers its answer. A site whose
second-best structure costs 30 per cent more has one candidate root; a site whose
second-best costs 2 per cent more has two, and which one is reported is decided
by differences the tomography cannot resolve.

The exclusion is a disc around the root rather than a hemisphere: a side constraint
asks "is there something to the west", which is a question about the figure, while
this asks "is there another structure at all", which is a question about the field.

  python3 competing_routes.py --file <model.nc> --tag RevealLO
"""
from __future__ import annotations
import argparse, gc, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import path_config
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from contrast import contrast_field
from path_cost import cost_field, trace_path, channel_field

R_E, DEG, SIGMA, SEED_DEPTH = 6371.0, np.pi / 180.0, 800.0, 200.0


def gc_km(la1, lo1, la2, lo2):
    p1, p2 = la1 * DEG, np.asarray(la2, float) * DEG
    dl = (np.asarray(lo2, float) - lo1) * DEG
    h = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2)
    return 2 * R_E * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _nearest(v, x):
    return int(np.argmin(np.abs(np.asarray(v, float) - float(x))))


def _decompose(z, la, lo, a, s, lateral):
    """Total cost, the length the objective charges, and the anomaly it integrates.

    Reconstructed from the path nodes, so it agrees with the cost field to about half
    a per cent rather than exactly; the ratios it is read for are unaffected.
    """
    o = np.argsort(z)
    z, la, lo, a = z[o], la[o], lo[o], a[o]
    h = gc_km(la[:-1], lo[:-1], la[1:], lo[1:])
    L = np.hypot(np.diff(z), lateral * h)
    am = 0.5 * (a[:-1] + a[1:])
    return (float((L * np.exp(s * am)).sum()), float(L.sum()), float(h.sum()),
            float(np.average(am, weights=L)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
    ap.add_argument('--lateral', type=float, default=0.60)
    ap.add_argument('--ncell', type=int, default=2)
    ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
    ap.add_argument('--sites', default='', help='default: every hotspot in the table')
    ap.add_argument('--exclude-km', type=float, default=500.0, dest='exclude_km',
                    help='radius of the disc forbidden around the chosen root')
    ap.add_argument('--exclude-below', type=float, default=1500.0,
                    dest='exclude_below',
                    help='the disc applies at and below this depth, so the exclusion '
                         'is of a ROOT and not of a whole column')
    ap.add_argument('--suffix', default='',
                    help='appended to the output name, so a run on null sites cannot '
                         'overwrite the hotspot run it is compared against')
    ap.add_argument('--save-paths', default=None, dest='save_paths',
                    help='also write both routes per site as JSON, so the figures can '
                         'draw the second track where the two are indistinguishable. '
                         'Without this only the endpoints survive and the alternative '
                         'cannot be plotted.')
    ap.add_argument('--max-sites', type=int, default=0, dest='max_sites',
                    help='stop after this many NEW sites and exit cleanly. Only for a '
                         'machine with little memory to spare - the 3 GB Cowork sandbox, '
                         'where loading RevealLO alone reaches the ceiling. On a desktop '
                         'leave it at 0, no limit, and run once: the output is written '
                         'after every site through a temporary file, so if the process '
                         'is ever killed the next run resumes where it stopped.')
    ap.add_argument('--halfwidth', type=float, default=40.0,
                    help='half-width of the regional window each site is solved in')
    A = ap.parse_args()

    c = path_config.load(A.tag, A.dir)
    hm = getattr(c, 'h_max', None)
    hm = None if hm is None or (isinstance(hm, float) and hm != hm) else float(hm)
    ztar = float(c.z_target)

    depth, LATV, LONV, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                          depth_max=A.depth_max, every=A.every)
    LONV, arr = dedupe_lon(LONV, arr)
    CON = contrast_field(arr, LATV, LONV, SIGMA)
    z = np.asarray(depth, float)
    print(f'model loaded: {arr.shape}', flush=True)

    hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180'])
    want = [s.strip() for s in A.sites.split(',') if s.strip()] or \
        [str(r.hotspot) for _, r in hs.iterrows()]
    kw = dict(s=float(c.s), z_target=ztar, n_relax=6, channel=str(c.channel),
              h_max_km=hm, lateral=float(A.lateral), ncell=int(A.ncell))
    tkw = dict(s=float(c.s), radius_deg=float(c.radius), seed_depth=SEED_DEPTH,
               n_relax=6, channel=str(c.channel), h_max_km=hm,
               lateral=float(A.lateral), ncell=int(A.ncell))
    CFG = dict(cfg_s=float(c.s), cfg_z_target=ztar, cfg_channel=str(c.channel),
               cfg_h_max=(hm if hm is not None else float('nan')),
               cfg_radius=float(c.radius), cfg_lateral=float(A.lateral),
               cfg_ncell=int(A.ncell), cfg_exclude_km=float(A.exclude_km),
               cfg_exclude_below=float(A.exclude_below))

    OUT = os.path.join(A.dir, f'competing_routes_{A.tag}{A.suffix}.csv')
    PJ = os.path.join(A.dir, A.save_paths) if A.save_paths else None
    tracks = json.load(open(PJ)) if (PJ and os.path.exists(PJ)) else {}

    # Resume. Forty-nine sites at two cost fields each is long enough that something
    # will interrupt it, and a run that starts again from the beginning each time is a
    # run nobody finishes. A file computed under a different configuration is refused
    # rather than extended, for the reason frontier.py now carries the same guard: a
    # resumed file that mixes two cost fields is worse than no resume at all.
    rows, have = [], set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        bad = next((k for k, v in CFG.items()
                    if k not in prev.columns
                    or (isinstance(v, str) and len(prev[k].dropna())
                        and str(prev[k].dropna().iloc[0]) != v)
                    or (not isinstance(v, str) and v == v
                        and len(prev[k].dropna())
                        and abs(float(prev[k].dropna().iloc[0]) - float(v)) > 1e-9)),
                   None)
        if bad:
            raise SystemExit(
                f'{OUT} was computed under a different configuration ({bad} differs).\n'
                f'Move it aside and run again:\n'
                f'  mkdir -p to_delete && mv {OUT} to_delete/')
        rows = prev.to_dict('records')
        have = set(prev.site.astype(str))
        if PJ:
            # A site is only done when its TRACKS are on disk too. The first sites
            # measured here were run before --save-paths existed, so the table knew
            # them and the figure could not draw them; resuming on the table alone
            # would have left that gap permanently invisible.
            short = {k for k in have
                     if k not in tracks or 'second' not in tracks.get(k, {})}
            if short:
                print(f'  {len(short)} site(s) have a row but no saved track and will '
                      f'be recomputed', flush=True)
                have -= short
                rows = [r for r in rows if str(r.get('site')) not in short]
        print(f'resuming: {len(have)} sites already measured', flush=True)

    n_new = 0
    for name in want:
        if str(name) in have:
            continue
        if A.max_sites and n_new >= A.max_sites:
            print(f'reached --max-sites {A.max_sites}; run again to continue',
                  flush=True)
            break
        n_new += 1
        r = hs[hs.hotspot.astype(str) == name]
        if not len(r):
            continue
        la0, lo0 = float(r.iloc[0].lat), float(r.iloc[0].lon_180)
        jm = np.abs(LATV - la0) <= A.halfwidth
        im = np.abs(((LONV - lo0 + 180.0) % 360.0) - 180.0) <= A.halfwidth
        lat_b, lon_b = LATV[jm], LONV[im]
        anom = np.ascontiguousarray(arr[:, jm][:, :, im])
        con = np.ascontiguousarray(CON[:, jm][:, :, im])
        ach = np.clip(channel_field(anom, con, str(c.channel)), -6.0, 6.0)

        C = cost_field(anom, con, depth, lat_b, lon_b, **kw)
        tr = trace_path(C, anom, con, depth, lat_b, lon_b, la0, lo0, **tkw)
        if tr is None:
            print(f'  {name:26s} no route', flush=True)
            del C; gc.collect(); continue
        z1, la1, lo1 = (np.asarray(v, float) for v in tr)
        k0, j0, i0 = (_nearest(depth, SEED_DEPTH), _nearest(lat_b, la1[0]),
                      _nearest(lon_b, lo1[0]))
        c_free = float(C[k0, j0, i0])
        v1 = np.array([ach[_nearest(depth, zz), _nearest(lat_b, aa),
                           _nearest(lon_b, oo)]
                       for zz, aa, oo in zip(z1, la1, lo1)], float)
        o = np.argsort(z1)
        end_la, end_lo = float(la1[o][-1]), float(lo1[o][-1])
        w1, L1, H1, a1 = _decompose(z1, la1, lo1, v1, float(c.s), float(A.lateral))
        del C; gc.collect()

        # forbid the chosen root and solve the identical problem again
        d2 = gc_km(end_la, end_lo, lat_b[:, None] * np.ones((1, len(lon_b))),
                   np.ones((len(lat_b), 1)) * lon_b[None, :])
        ban = np.zeros(anom.shape, bool)
        ban[z >= A.exclude_below] = (d2 <= A.exclude_km)[None, :, :]
        C2 = cost_field(anom, con, depth, lat_b, lon_b, forbid=ban, **kw)
        tr2 = trace_path(C2, anom, con, depth, lat_b, lon_b, la0, lo0, **tkw)
        row = dict(**CFG, site=name, cost=c_free, end_lat=end_la, end_lon=end_lo,
                   worst=float(v1.max()), charged_km=L1, horiz_km=H1, mean_a=a1)
        if tr2 is None:
            row.update(alt_reached=False)
            print(f'  {name:26s} no second structure within the window', flush=True)
        else:
            z2, la2, lo2 = (np.asarray(v, float) for v in tr2)
            c_alt = float(C2[k0, _nearest(lat_b, la2[0]), _nearest(lon_b, lo2[0])])
            v2 = np.array([ach[_nearest(depth, zz), _nearest(lat_b, aa),
                               _nearest(lon_b, oo)]
                           for zz, aa, oo in zip(z2, la2, lo2)], float)
            o2 = np.argsort(z2)
            e2la, e2lo = float(la2[o2][-1]), float(lo2[o2][-1])
            w2, L2, H2, a2 = _decompose(z2, la2, lo2, v2, float(c.s), float(A.lateral))
            # what the second structure would have to be slower by, to win
            need = np.log(L2 / L1) / float(c.s) if L1 > 0 else np.nan
            # WHERE the two routes part company. If they share the upper mantle and
            # differ only below the transition zone, then the shallow geometry the
            # paper reports - tilt, offset at 660 - is determinate even at sites whose
            # root is not, and the two roots are two limbs of one imaged object rather
            # than two answers. This is a statement about which reported quantities
            # survive the indeterminacy. It is NOT evidence that a conduit branches:
            # two nearly equally cheap routes sharing a segment is a property of the
            # cost surface, which is the error branch_structures.py exists to replace,
            # and the injected-split floor says a real branch is recovered in 4 to 10
            # per cent of cases.
            zs = np.arange(SEED_DEPTH, ztar + 1.0, 50.0)
            o1 = np.argsort(z1)
            la_a = np.interp(zs, z1[o1], la1[o1]); lo_a = np.interp(zs, z1[o1], lo1[o1])
            la_b = np.interp(zs, z2[o2], la2[o2]); lo_b = np.interp(zs, z2[o2], lo2[o2])
            sep = gc_km(0.0, 0.0, 0.0, 0.0) * 0  # placeholder to keep dtype float
            sep = np.array([float(gc_km(a1_, o1_, b1_, b2_))
                            for a1_, o1_, b1_, b2_ in zip(la_a, lo_a, la_b, lo_b)])
            over = np.where(sep > 300.0)[0]
            row.update(**{f'sep_{int(zq)}km': float(np.interp(zq, zs, sep))
                          for zq in (400, 660, 1000, 1500, 2000)},
                       split_depth_km=(float(zs[over[0]]) if len(over) else np.nan),
                       sep_max_km=float(sep.max()))
            row.update(alt_reached=True, alt_cost=c_alt,
                       margin_pct=100.0 * (c_alt / c_free - 1.0),
                       alt_end_lat=e2la, alt_end_lon=e2lo,
                       alt_worst=float(v2.max()), alt_charged_km=L2,
                       alt_horiz_km=H2, alt_mean_a=a2,
                       needs_slower_pct=float(need),
                       is_slower_pct=float(a1 - a2),
                       shortfall_pct=float(need - (a1 - a2)),
                       separation_km=float(gc_km(end_la, end_lo, e2la, e2lo)))
            print(f'  {name:26s} second structure +{row["margin_pct"]:5.1f} %  '
                  f'{row["separation_km"]:5.0f} km away  shortfall '
                  f'{row["shortfall_pct"]:+.2f} %', flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT + '.part', index=False)
        os.replace(OUT + '.part', OUT)
        if PJ:
            e = dict(first=dict(depth=z1.tolist(), lat=la1.tolist(),
                                lon=lo1.tolist()))
            if tr2 is not None:
                e['second'] = dict(depth=z2.tolist(), lat=la2.tolist(),
                                   lon=lo2.tolist())
            e['margin_pct'] = row.get('margin_pct', float('nan'))
            tracks[str(name)] = e
            json.dump(tracks, open(PJ + '.part', 'w'))
            os.replace(PJ + '.part', PJ)
        del C2, anom, con, ach; gc.collect()

    d = pd.DataFrame(rows)
    ok = d[d.alt_reached == True] if 'alt_reached' in d else d
    print(f'\nwrote {OUT}')
    if len(ok):
        print(f'\n{len(ok)} of {len(d)} sites have a second deep structure in reach.')
        for t in (2.0, 5.0, 10.0):
            print(f'  within {t:4.0f} per cent of the chosen route: '
                  f'{int((ok.margin_pct <= t).sum()):2d}')
        print(f'  median margin {ok.margin_pct.median():.1f} per cent')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
