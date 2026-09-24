#!/usr/bin/env python3
"""Are the separate corridors at native sampling structures, or speckle?

The scale ladder shows multi-component corridors beneath six of the twelve
targets at RevealLO's native sampling and none at all once the model is smoothed
by 200 km. That is either the fine model resolving something coarser models
cannot, or it is noise at the native scale averaging out - and the two produce
the same signature in a per-shell component count, so counting components cannot
distinguish them.

Depth can. A real structure is continuous: a component at one depth has a
neighbour at the next, and the chain runs for hundreds of kilometres. Speckle is
not: components appear in isolated shells, and the chains are one or two shells
long. This links components between adjacent shells by proximity of their
centroids and reports how far each chain runs.

The linking distance is set by the grid, not by a plausible lean. RevealLO's
shells are ten kilometres apart, so a conduit leaning at forty-five degrees moves
only ten kilometres between them - far below the fifty-five kilometre cell on
which the components are resolved. A centroid can shift by a whole cell between
adjacent shells from discretisation alone, so anything tighter than about two
cells would break real chains, and the default is set there. Because that
threshold is generous enough to join things that are not the same object, every
result is reported at the chosen distance AND at half of it: a chain length that
survives halving is a chain, and one that collapses was an artefact of the
linking.

WHAT A POSITIVE RESULT WOULD AND WOULD NOT MEAN

Chains that run for hundreds of kilometres would show the components are coherent
in depth rather than shell-by-shell noise. They would not show that the seismic
data resolve them - that question belongs to the inversion, not to this analysis,
and is answered by the injection floor and by the model's own resolution
analysis, not here.
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morphology as M

R_E, DEG = 6371.0, np.pi / 180.0


def shell_components(w, LAT, LON, floor):
    """Centroid, weight and cell count of every component in one shell."""
    lab, n = M._label_periodic(w >= floor)
    out = []
    for q in range(1, n + 1):
        m = lab == q
        ww = np.where(m, w, 0.0)
        tw = float(ww.sum())
        if tw <= 0:
            continue
        clat, clon, _ = M.weighted_centroid(ww, LAT, LON)
        out.append(dict(lat=clat, lon=clon, weight=tw, cells=int(m.sum())))
    return out


def branching(shells):
    """Splits and merges counted UPWARD, from the deep mantle toward the surface.

    Direction is the whole point and was missing from the first attempt. A tree
    of plumes divides as it rises: read upward it splits and rarely rejoins. A
    network does both at similar rates. Read downward the same structure would
    show the opposite, so a count without a stated direction says nothing.

    The measure is the change in the number of components between adjacent
    shells, summed separately over increases and decreases. It deliberately does
    not use the linking distance: the chain analysis already depends on that
    choice, and a second measure that shares the dependence would not be a check
    on it. An earlier version counted from the reach graph instead and returned
    identical split and merge counts wherever every component lay within reach of
    every other, which is a statement about the linking distance rather than
    about the structure.
    """
    n = [len(c) for c in shells][::-1]            # deep to shallow
    if len(n) < 2:
        return 0, 0
    d = np.diff(n)
    return int(np.sum(d[d > 0])), int(-np.sum(d[d < 0]))


def link(shells, depth, link_km):
    """Chain components between adjacent shells by centroid proximity.

    Greedy and weight-ordered: the heaviest component of a shell claims its
    nearest unclaimed predecessor. A component that finds none starts a chain.

    """
    chains, active = [], []
    for k, comps in enumerate(shells):
        used = set()
        nxt = []
        for c in sorted(comps, key=lambda x: -x['weight']):
            best, bi = link_km + 1.0, None
            for i, a in enumerate(active):
                if i in used:
                    continue
                d = float(M.gc_km(a['lat'], a['lon'], c['lat'], c['lon']))
                if d < best:
                    best, bi = d, i
            if bi is None:
                ch = dict(k0=k, k1=k, w=[c['weight']], lat=[c['lat']],
                          lon=[c['lon']], cells=[c['cells']])
                chains.append(ch)
                nxt.append(dict(lat=c['lat'], lon=c['lon'], ch=ch))
            else:
                used.add(bi)
                ch = active[bi]['ch']
                ch['k1'] = k
                ch['w'].append(c['weight'])
                ch['lat'].append(c['lat'])
                ch['lon'].append(c['lon'])
                ch['cells'].append(c['cells'])
                nxt.append(dict(lat=c['lat'], lon=c['lon'], ch=ch))
        active = nxt
    z = np.asarray(depth, float)
    for ch in chains:
        ch['z0'], ch['z1'] = float(z[ch['k0']]), float(z[ch['k1']])
        ch['extent_km'] = ch['z1'] - ch['z0']
        ch['n_shells'] = len(ch['w'])
        ch['mean_weight'] = float(np.mean(ch['w']))
    return chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='RevealLO')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--tau', type=float, default=0.02)
    ap.add_argument('--floor', type=float, default=0.10,
                    help='agreement a cell must command to belong to a component')
    ap.add_argument('--link-km', type=float, default=110.0, dest='link_km',
                    help='how far a centroid may move between adjacent shells '
                         'before it is treated as a different object. About two '
                         'grid cells; everything is also reported at half this')
    ap.add_argument('--zmin', type=float, default=800.0)
    ap.add_argument('--zmax', type=float, default=2700.0)
    ap.add_argument('--min-extent', type=float, default=400.0, dest='min_extent',
                    help='depth a chain must span to count as coherent, km')
    ap.add_argument('--include-nulls', action='store_true', dest='include_nulls',
                    help='keep the matched null sites, so that a conduit count '
                         'can be judged against how often one arises with no '
                         'hotspot present')
    a = ap.parse_args()

    f = os.path.join(a.dir, f'morph_volumes_{a.tag}{a.suffix}.npz')
    store = np.load(f)
    depth, lat, lon = store['depth'], store['lat'], store['lon']
    keys = sorted({k.split('|', 1)[1] for k in store.files
                   if k.startswith('occ|')})
    if not a.include_nulls:
        keys = [k for k in keys if not k.startswith('null')]
    else:
        # the census asks how many hotspots carry more than one conduit, and
        # that number means nothing without knowing how often a site with no
        # hotspot does. The nulls are already in the store; excluding them was
        # right while this script only fed the morphology metrics.
        nn = sum(1 for k in keys if k.startswith('null'))
        print(f'{len(keys) - nn} hotspots and {nn} null sites', flush=True)
    sel = np.where((depth >= a.zmin) & (depth <= a.zmax))[0]

    rows, summary = [], []
    for key in keys:
        vk = f'cor{a.tau:g}|{key}'
        if vk not in store.files:
            continue
        W = store[vk]
        jb, ib = store[f'jb|{key}'], store[f'ib|{key}']
        hlat, hlon = store[f'site|{key}']
        LO, LA = np.meshgrid(lon[ib], lat[jb])
        LO = ((LO + 180.0) % 360.0) - 180.0
        shells = [shell_components(W[k], LA, LO, a.floor) for k in sel]
        chains = link(shells, depth[sel], a.link_km)
        tight = link(shells, depth[sel], a.link_km / 2.0)
        spl, mrg = branching(shells)
        keep = [c for c in chains if c['extent_km'] >= a.min_extent]
        keep.sort(key=lambda c: -c['extent_km'])
        keep_t = [c for c in tight if c['extent_km'] >= a.min_extent]
        for c in keep:
            rows.append(dict(site=key, z0=c['z0'], z1=c['z1'],
                             extent_km=c['extent_km'], n_shells=c['n_shells'],
                             mean_weight=c['mean_weight'],
                             offset_km=float(M.gc_km(hlat, hlon,
                                                     np.mean(c['lat']),
                                                     np.mean(c['lon']))),
                             azimuth_deg=float(M.bearing_deg(hlat, hlon,
                                                             np.mean(c['lat']),
                                                             np.mean(c['lon'])))))
        summary.append(dict(site=key, n_chains=len(chains),
                            n_coherent=len(keep),
                            n_coherent_tight=len(keep_t),
                            longest_km=max([c['extent_km'] for c in chains],
                                           default=0.0),
                            longest_tight_km=max([c['extent_km'] for c in tight],
                                                 default=0.0),
                            second_km=(sorted([c['extent_km'] for c in chains],
                                              reverse=True) + [0.0, 0.0])[1],
                            second_tight_km=(sorted([c['extent_km'] for c in tight],
                                                    reverse=True) + [0.0, 0.0])[1],
                            splits=spl, merges=mrg, n_shells=len(sel),
                            # splits and merges per hundred shells, so a window
                            # of a different depth or a model of a different
                            # sampling can be compared with this one
                            split_rate=100.0 * spl / max(len(sel), 1),
                            merge_rate=100.0 * mrg / max(len(sel), 1)))
    S = pd.DataFrame(summary).sort_values('n_coherent', ascending=False)
    D = pd.DataFrame(rows)
    S.to_csv(os.path.join(a.dir, f'component_chains_{a.tag}{a.suffix}.csv'),
             index=False)
    D.to_csv(os.path.join(a.dir, f'component_chains_detail_{a.tag}{a.suffix}.csv'),
             index=False)

    print(f'corridor at tau = {a.tau:g}, {a.zmin:.0f} to {a.zmax:.0f} km. '
          f'Components are linked between adjacent shells within\n'
          f'{a.link_km:.0f} km, and again within {a.link_km / 2:.0f} km; the '
          f'second set is the one to believe.\n')
    print(f'{"":26s} {"":>17s} {"longest chain":>15s} {"second chain":>15s}')
    _w, _t = f'{a.link_km:.0f} km', f'{a.link_km / 2:.0f} km'
    print(f'{"hotspot":26s} {"coherent chains":>17s} {_w:>7s} {_t:>7s} '
          f'{_w:>7s} {_t:>7s} {"splits":>7s} {"merges":>7s}')
    for _, r in S.iterrows():
        print(f'{r.site:26s} {r.n_coherent:8d} {r.n_coherent_tight:8d} '
              f'{r.longest_km:7.0f} {r.longest_tight_km:7.0f} '
              f'{r.second_km:7.0f} {r.second_tight_km:7.0f} '
              f'{r.split_rate:7.1f} {r.merge_rate:7.1f}')
    print(f'\ncoherent means a chain spanning at least {a.min_extent:.0f} km of depth.')
    print('A site with two or more coherent chains AT BOTH linking distances has')
    print('genuinely separate corridors; one whose second chain collapses when')
    print('the linking is halved had shell-by-shell speckle, however many')
    print('components any single shell appeared to contain. A count that RISES')
    print('when the linking is halved is also informative: the loose distance')
    print('was merging chains that the tight one keeps apart.')
    print()
    print('Splits and merges are counted UPWARD, from the deep mantle toward the')
    print('surface, per hundred shells. A TREE of plumes divides as it rises, so')
    print('it splits and rarely rejoins; a NETWORK does both at similar rates.')
    print('Those are two of the conceptual models in the literature, and the ratio')
    print('is what separates them here rather than an impression from a rendering.')
    if len(D):
        print(f'\nthe coherent chains, longest first\n')
        print(f'{"hotspot":26s} {"depth range":>16s} {"extent":>7s} '
              f'{"offset":>7s} {"azimuth":>8s} {"weight":>7s}')
        for _, r in D.sort_values(['site', 'extent_km'],
                                  ascending=[True, False]).iterrows():
            print(f'{r.site:26s} {r.z0:7.0f} to {r.z1:5.0f} {r.extent_km:7.0f} '
                  f'{r.offset_km:7.0f} {r.azimuth_deg:8.0f} {r.mean_weight:7.2f}')
    print(f'\nwrote component_chains_{a.tag}{a.suffix}.csv and its detail table')


if __name__ == '__main__':
    main()
