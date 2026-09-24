#!/usr/bin/env python3
"""Path ensembles and near-optimal corridors beneath selected hotspots.

The classification reduces each hotspot, under each retained configuration, to
one cost and one path. That is enough to ask whether a hotspot is exceptional and
not enough to ask what is beneath it, because a single optimum looks the same
whether it is a lone route through expensive mantle or one draw from a wide
family of equivalent routes. This produces the two fields that separate them.

  ENSEMBLE. The optimal path is traced under every configuration that survived
  calibration, and the paths are accumulated into an occupancy volume: the
  fraction of configurations whose route passes through each cell. It is a
  sensitivity map over methodological choice, not a posterior, and is named
  occupancy throughout so that it cannot be read as one.

  CORRIDOR. For each configuration the forward arrival field is built as well, so
  that every cell carries the excess cost of the cheapest complete route through
  it. Cells whose excess is below a fixed fraction of the optimum form the
  corridor: the routes the tomography cannot distinguish from the best one. The
  fraction is fixed on the command line and reported at three values, so that no
  single tolerance can be chosen after seeing the shapes it produces.

Both are accumulated in a box about each hotspot and written as volumes, together
with the traced paths and a per-configuration table of costs.

WHY THE TRACED COST IS CHECKED AGAINST THE FIELD

The cost field bounds the lateral run at one depth by a number of relaxation
sweeps, which is h_max divided by the meridional cell width; the tracer bounds it
by h_max in kilometres. Off the equator a zonal cell is narrower, so the tracer
may take more zonal steps at one depth than the field ever priced, and would then
return a route cheaper than the field's own optimum. Every traced path is
therefore priced under the field's rules and compared with the field's optimum,
and the excess is reported. It is a property of the existing search, not of this
extension, and it is measured rather than assumed to be small.
"""
from __future__ import annotations

import argparse, json, os, sys, warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon, check_shells
from contrast import contrast_field
from path_cost import (SLANT_KM, _OFF, _cap_indices, _spans, cell_widths,
                       channel_field, cost_field, site_cost, trace_path)
from corridor import (forward_field, excess_cost, self_check,
                      trace_consistent)

warnings.filterwarnings('ignore')

SIGMA = 800.0
N_RELAX = 6
MIN_DETECT = 0.75
N_CAL_MIN = 4
SEED_DEPTH = 200.0
TAUS = (0.01, 0.02, 0.05)
BOX_DEG = 45.0

_W = {}                                   # read-only state shared with workers


# ---------------------------------------------------------------- configurations
def retained(dirname, tag):
    """The configurations that survived calibration, in a fixed order.

    Read from the same detection table and with the same rule the classification
    uses, so the ensemble is the ensemble of the published method rather than of
    a family chosen here.
    """
    det = pd.read_csv(os.path.join(dirname, f'detection_{tag}.csv'))
    import path_config as _pc
    keep = _pc.retained(det, _pc.ensemble_channel(tag, dirname))
    return keep.sort_values(['s', 'z_target', 'channel', 'h_max', 'radius']
                            ).reset_index(drop=True)


def stratify(keep, n):
    """A subset spanning the retained grid rather than its first n rows.

    Configurations are ordered by the cost trade-off first, so taking a prefix
    would sample one corner of the grid and call the result a sensitivity test.
    Rows are taken at even spacing through the sorted table instead, which keeps
    both ends of every swept axis.
    """
    if n is None or n >= len(keep):
        return keep
    idx = np.unique(np.linspace(0, len(keep) - 1, int(n)).round().astype(int))
    return keep.iloc[idx].reset_index(drop=True)


def field_key(c):
    return (float(c.s), float(c.z_target), str(c.channel), _hmax(c))


def _hmax(c):
    v = getattr(c, 'h_max', None)
    return None if v is None or (isinstance(v, float) and v != v) else float(v)


# ------------------------------------------------------------------------- boxes
def box_indices(lat, lon, hlat, hlon, box_deg=BOX_DEG):
    """Latitude and longitude indices of a box about a hotspot.

    Storage only: every field is computed globally, so the box never becomes a
    wall the search can lean against. Its half-width in longitude grows as
    1/cos(latitude) so that the box is about as wide in kilometres everywhere,
    and saturates to the whole sphere near the poles.
    """
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    j = np.where(np.abs(lat - hlat) <= box_deg)[0]
    cl = max(np.cos(np.radians(min(abs(hlat) + box_deg, 89.0))), 0.05)
    half = min(180.0, box_deg / cl)
    if half >= 179.0:
        i = np.arange(len(lon))
    else:
        d = ((lon - hlon + 180.0) % 360.0) - 180.0
        i = np.where(np.abs(d) <= half)[0]
        if len(i) and (i[-1] - i[0] + 1) != len(i):      # wrapped: reorder
            i = np.concatenate([i[i > len(lon) // 2], i[i <= len(lon) // 2]])
    return j, i


# ------------------------------------------------------------------- path pricing
def price_path(z, la, lo, a, depth, lat, lon, s, slant_km, h_max_km, n_relax):
    """Cost of a traced path under the rules the cost field was built from.

    Returns the total, and the number of lateral steps taken at any one depth in
    excess of what the field admits. A path that exceeds the field's reach is
    cheaper than the field's optimum for a reason that is a property of the
    search, not of the mantle.
    """
    z = np.asarray(z, float); la = np.asarray(la, float); lo = np.asarray(lo, float)
    dlat, dlon = cell_widths(lat, lon)
    jj = np.abs(np.asarray(lat, float)[None, :] - la[:, None]).argmin(axis=1)
    dl = ((np.asarray(lon, float)[None, :] - lo[:, None] + 180.0) % 360.0) - 180.0
    ii = np.abs(dl).argmin(axis=1)
    kk = np.abs(np.asarray(depth, float)[None, :] - z[:, None]).argmin(axis=1)
    R = max(1, int(round(float(h_max_km) / dlat))) if h_max_km else n_relax
    total, run, over = 0.0, 0, 0
    for q in range(len(z) - 1):
        k1, j1, i1 = kk[q], jj[q], ii[q]
        k2, j2, i2 = kk[q + 1], jj[q + 1], ii[q + 1]
        if k2 == k1:                                   # lateral
            w = dlat if j2 != j1 else float(dlon[j1, i1])
            total += w * float(np.exp(s * a[k1, j1, i1]))
            run += 1
            over = max(over, run - R)
        else:
            m1 = float(a[k1:k2 + 1, j1, i1].mean())
            m2 = float(a[k1:k2 + 1, j2, i2].mean())
            lateral = 0.0
            if j2 != j1:
                lateral = abs(j2 - j1) * dlat
            elif i2 != i1:
                lateral = float(dlon[j1, i1])
            L = float(np.hypot(float(depth[k2] - depth[k1]), lateral))
            total += L * float(np.exp(s * 0.5 * (m1 + m2)))
            run = 0
    return total, int(over)


# ------------------------------------------------------------------------ worker
def _one_config(q):
    """One configuration: cost field, traced paths, and optionally corridors."""
    arr, con, depth, lat, lon = (_W['arr'], _W['con'], _W['depth'],
                                 _W['lat'], _W['lon'])
    cfg = _W['cfgs'][q]
    boxes, want_corr = _W['boxes'], _W['corridor']
    sites = _W['sites']
    s, zt, ch, hm = cfg['s'], cfg['z_target'], cfg['channel'], cfg['h_max']
    rd = cfg['radius']
    C = cost_field(arr, con, depth, lat, lon, s=s, z_target=zt, n_relax=N_RELAX,
                   channel=ch, h_max_km=hm)
    a = np.clip(channel_field(arr, con, ch), -6.0, 6.0)
    z = np.asarray(depth, float)
    k0 = int(np.where(z >= zt)[0][0]) if (z >= zt).any() else len(z) - 1
    k_seed = int(np.argmin(np.abs(z - SEED_DEPTH)))

    out = {}
    for name, hlat, hlon in sites:
        rec = dict(config=q, site=name)
        rec['cost'] = float(site_cost(C, depth, lat, lon, hlat, hlon, rd, SEED_DEPTH))
        tr = trace_consistent(C, arr, con, depth, lat, lon, hlat, hlon, s=s,
                              radius_deg=rd, n_relax=N_RELAX, channel=ch,
                              h_max_km=hm)
        if tr is None:
            rec['path'] = None
        else:
            pz, pla, plo = tr
            priced, over = price_path(pz, pla, plo, a, depth, lat, lon, s,
                                      SLANT_KM, hm, N_RELAX)
            rec['path'] = (pz.astype(np.float32), pla.astype(np.float32),
                           plo.astype(np.float32))
            rec['traced_cost'] = priced
            rec['reach_excess_steps'] = over
        if want_corr:
            D = forward_field(arr, con, depth, lat, lon, hlat, hlon, s=s,
                              z_target=zt, radius_deg=rd, seed_depth=SEED_DEPTH,
                              n_relax=N_RELAX, channel=ch, h_max_km=hm)
            dC, cbest = excess_cost(D, C, k_seed, k0)
            ok, _mins = self_check(D, C, k_seed, k0, c_best=rec['cost'])
            rec['corridor_ok'] = bool(ok)
            rec['c_best'] = float(cbest)
            jb, ib = boxes[name]
            sub = dC[:, jb][:, :, ib]
            rec['corridor'] = {}
            for t in (_W['taus'] if not name.startswith('null') else _W['taus'][1:2]):
                m = np.isfinite(sub) & (sub <= t * cbest)
                rec['corridor'][t] = np.flatnonzero(m.ravel()).astype(np.int32)
                rec.setdefault('corridor_cells', {})[t] = int(m.sum())
            del D, dC, sub
        out[name] = rec
    del C, a
    print(f'  config {q}: s={s} zt={zt:.0f} {ch} h{hm if hm else 0:.0f} r{rd}',
          flush=True)
    return q, out


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
    ap.add_argument('--targets', default=None,
                    help='file with one hotspot name per line; default is the '
                         'primary sample written by track_classes.py')
    ap.add_argument('--corridor', action='store_true',
                    help='also build the forward field and the near-optimal '
                         'corridor. Roughly doubles the work per hotspot')
    ap.add_argument('--configs', type=int, default=None,
                    help='use this many retained configurations, spread evenly '
                         'through the sorted grid; default is all of them')
    ap.add_argument('--nulls', type=int, default=0,
                    help='matched random locations to measure alongside, so '
                         'every morphology metric carries its own null')
    ap.add_argument('--null-seed', type=int, default=7, dest='null_seed')
    ap.add_argument('--box-deg', type=float, default=BOX_DEG, dest='box_deg')
    ap.add_argument('--null-box-deg', type=float, default=30.0,
                    dest='null_box_deg',
                    help='smaller box for the matched null locations, which need\n                         metrics rather than figures')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--force-channel', default=None, dest='force_channel',
                    choices=('min', 'anom', 'contrast'),
                    help='override the cost channel of every retained '
                         'configuration, keeping the rest of each one. This is '
                         'the control for what the search is measuring: the '
                         'anomaly is relative to the depth shell and so is '
                         'negative throughout a large low-velocity province '
                         'whether or not a conduit is present, while the '
                         'contrast is relative to the local background and so '
                         'responds to a conduit and not to the province. Note '
                         'that a forced channel has NOT been calibrated against '
                         'the synthetic conduits for that channel, so the '
                         'configurations are the ones calibration retained for '
                         'the original channel; a properly calibrated control '
                         'needs classify.py --channels')
    ap.add_argument('--suffix', default='')
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)

    keep = stratify(retained(a.dir, a.tag), a.configs)
    cfgs = [dict(s=float(r.s), z_target=float(r.z_target),
                 channel=(a.force_channel or str(r.channel)),
                 h_max=_hmax(r), radius=float(r.radius))
            for _, r in keep.iterrows()]
    if a.force_channel:
        # the same parameter set can appear twice once the channel is forced,
        # because 'min' and 'anom' rows differing only in channel collapse
        seen, uniq = set(), []
        for c in cfgs:
            k = (c['s'], c['z_target'], c['channel'], c['h_max'], c['radius'])
            if k not in seen:
                seen.add(k)
                uniq.append(c)
        print(f'channel forced to {a.force_channel}: {len(cfgs)} retained '
              f'configurations collapse to {len(uniq)} distinct ones', flush=True)
        cfgs = uniq
    print(f'{len(cfgs)} configurations', flush=True)

    hs = pd.read_csv(a.hotspots).dropna(subset=['lat', 'lon_180'])
    if a.targets and os.path.exists(a.targets):
        names = [l.strip() for l in open(a.targets) if l.strip()
                 and not l.startswith('#')]
    else:
        names = list(hs.hotspot.astype(str))
    sites = []
    for n in names:
        r = hs[hs.hotspot.astype(str) == n]
        if not len(r):
            print(f'  {n}: not in the hotspot table, skipped')
            continue
        sites.append((n, float(r.iloc[0].lat), float(r.iloc[0].lon_180)))
    if a.nulls:
        rng = np.random.default_rng(a.null_seed)
        for q in range(a.nulls):
            sites.append((f'null{q:03d}',
                          float(np.degrees(np.arcsin(2 * rng.random() - 1))),
                          float(360.0 * rng.random() - 180.0)))
    print(f'{len(sites)} sites ({len(sites) - a.nulls} named)', flush=True)

    depth, lat, lon, arr = load_anomaly(a.file, ModelSpec(a.tag, a.var),
                                        depth_max=a.depth_max, every=a.every)
    lon, arr = dedupe_lon(lon, arr)
    check_shells(depth, arr, a.tag)
    print(f'{a.tag}: {len(depth)} shells, {len(lat)}x{len(lon)}', flush=True)
    con = contrast_field(arr, lat, lon, SIGMA)

    # The lowermost-mantle setting of every site, hotspot and null alike, so
    # that the null distribution a metric is judged against can be restricted to
    # locations sitting over material as slow as the hotspot does. A corridor
    # inside a large low-velocity province is being compared with corridors
    # inside such provinces, not with corridors under the whole Earth.
    kdeep = np.where((depth >= 2600.0) & (depth <= 2880.0))[0]
    deepmap = np.nanmean(arr[kdeep], axis=0) if len(kdeep) else None

    boxes = {n: box_indices(lat, lon, la, lo,
                            a.null_box_deg if n.startswith('null') else a.box_deg)
             for n, la, lo in sites}
    _W.update(arr=arr, con=con, depth=depth, lat=lat, lon=lon, cfgs=cfgs,
              sites=sites, boxes=boxes, taus=TAUS, corridor=bool(a.corridor))

    # Accumulators are counts, not fractions: a count fits in one byte for any
    # number of configurations this study can afford, and at 40 null locations
    # the difference between one byte and four is the difference between a run
    # that fits in memory and one that does not.
    acc, meta = {}, {}
    for name, la_, lo_ in sites:
        jb, ib = boxes[name]
        shape = (len(depth), len(jb), len(ib))
        tset = TAUS if not name.startswith('null') else TAUS[1:2]
        acc[name] = dict(occ=np.zeros(shape, np.uint8),
                         **{f'{t:g}': np.zeros(shape, np.uint8) for t in tset})
        meta[name] = (la_, lo_, jb, ib, tset)
    rows, paths_out = [], {}

    def consume(q, res):
        for name, rec in res.items():
            jb, ib = meta[name][2], meta[name][3]
            row = dict(config=q, site=name, cost=rec['cost'],
                       **{k: cfgs[q][k] for k in ('s', 'z_target', 'channel',
                                                  'h_max', 'radius')})
            if rec.get('path') is not None:
                pz, pla, plo = rec['path']
                if not name.startswith('null'):
                    paths_out[f'{name}|{q}'] = np.stack([pz, pla, plo])
                row['traced_cost'] = rec['traced_cost']
                row['reach_excess_steps'] = rec['reach_excess_steps']
                _rasterise(acc[name]['occ'], pz, pla, plo, depth, lat, lon, jb, ib)
            if 'corridor' in rec:
                row['corridor_ok'] = rec['corridor_ok']
                row['c_best'] = rec['c_best']
                for t, idx in rec['corridor'].items():
                    acc[name][f'{t:g}'].reshape(-1)[idx] += 1
                    row[f'cells_{t:g}'] = rec['corridor_cells'][t]
            rows.append(row)

    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.jobs) as pool:
            for q, res in pool.imap_unordered(_one_config, range(len(cfgs)),
                                              chunksize=1):
                consume(q, res)
    else:
        for q in range(len(cfgs)):
            consume(*_one_config(q))

    sfx, n_cfg = a.suffix, len(cfgs)
    np.savez_compressed(os.path.join(a.dir, f'morph_paths_{a.tag}{sfx}.npz'),
                        **paths_out)
    store = dict(depth=depth, lat=lat, lon=lon, taus=np.array(TAUS),
                 n_config=np.array([n_cfg]))
    for name, (la_, lo_, jb, ib, tset) in meta.items():
        key = _key(name)
        store[f'occ|{key}'] = (acc[name]['occ'].astype(np.float32) / n_cfg)
        store[f'jb|{key}'] = jb
        store[f'ib|{key}'] = ib
        store[f'site|{key}'] = np.array([la_, lo_], float)
        store[f'deep|{key}'] = np.array([_cap_mean(deepmap, lat, lon, la_, lo_)])
        if a.corridor:
            for t in tset:
                store[f'cor{t:g}|{key}'] = (acc[name][f'{t:g}'].astype(np.float32)
                                            / n_cfg)
    np.savez_compressed(os.path.join(a.dir, f'morph_volumes_{a.tag}{sfx}.npz'),
                        **store)
    df = pd.DataFrame(rows).sort_values(['site', 'config'])
    df.to_csv(os.path.join(a.dir, f'morph_configs_{a.tag}{sfx}.csv'), index=False)
    print(f'\nwrote morph_volumes_{a.tag}{sfx}.npz, morph_paths_{a.tag}{sfx}.npz '
          f'and morph_configs_{a.tag}{sfx}.csv')
    if 'reach_excess_steps' in df:
        bad = df[df.reach_excess_steps > 0]
        print(f'traced paths exceeding the field reach: {len(bad)} of {len(df)}')
        if len(bad):
            d = (bad.traced_cost - bad.cost) / bad.cost
            print(f'  their traced cost against the field optimum: median '
                  f'{d.median():+.4%}, most negative {d.min():+.4%}')
    if a.corridor and 'corridor_ok' in df:
        print(f'corridor self-check passed for {int(df.corridor_ok.sum())} '
              f'of {int(df.corridor_ok.notna().sum())} site-configurations')


def _cap_mean(deepmap, lat, lon, hlat, hlon, radius_km=800.0):
    """Mean lowermost-mantle anomaly in a cap, the province measure of province.py."""
    if deepmap is None:
        return np.nan
    LO, LA = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
    LO = ((LO + 180.0) % 360.0) - 180.0
    d = 6371.0 * np.arccos(np.clip(
        np.sin(np.radians(hlat)) * np.sin(np.radians(LA)) +
        np.cos(np.radians(hlat)) * np.cos(np.radians(LA)) *
        np.cos(np.radians(LO - hlon)), -1, 1))
    m = d <= radius_km
    return float(np.nanmean(deepmap[m])) if m.any() else np.nan


def _key(name):
    return name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')


def _rasterise(vol, pz, pla, plo, depth, lat, lon, jb, ib):
    """Add one to every box cell this path visits, counting each cell once.

    A path may pass through the same cell twice, at two lateral steps of one
    depth. Adding on every point would then weight that cell twice for a single
    configuration, and the occupancy would no longer be the fraction of
    configurations that use the cell, which is the only thing it is allowed to
    mean.
    """
    z = np.asarray(depth, float); la_ = np.asarray(lat, float)
    lo_ = np.asarray(lon, float)
    k = np.abs(z[None, :] - np.asarray(pz, float)[:, None]).argmin(axis=1)
    j = np.abs(la_[None, :] - np.asarray(pla, float)[:, None]).argmin(axis=1)
    dl = ((lo_[None, :] - np.asarray(plo, float)[:, None] + 180.0) % 360.0) - 180.0
    i = np.abs(dl).argmin(axis=1)
    jmap = -np.ones(len(la_), int); jmap[jb] = np.arange(len(jb))
    imap = -np.ones(len(lo_), int); imap[ib] = np.arange(len(ib))
    jj, ii = jmap[j], imap[i]
    ok = (jj >= 0) & (ii >= 0)
    if not ok.any():
        return
    flat = np.unique(np.ravel_multi_index((k[ok], jj[ok], ii[ok]), vol.shape))
    vol.reshape(-1)[flat] += 1


if __name__ == '__main__':
    main()
