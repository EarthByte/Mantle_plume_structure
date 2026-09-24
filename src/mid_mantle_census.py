#!/usr/bin/env python3
"""Where, in depth, do hotspot conduits differ from ambient slow structure?

Every quantity the paper already measures is read here interval by interval, so that
the same five depth bands carry the tilt, the width, the prominence, the deflection
share and the root depth, and the hotspot value stands beside the ambient value where
an ambient set exists. The bands are chosen around the mid-mantle viscosity increase
of Rudolph et al. (2015): above the 660 km discontinuity, the low-viscosity channel
between 660 and 1000 km, the interval just beneath the increase, the mid lower mantle,
and the basal 500 km.

Nothing here loads a model. It reads:

  conduit_paths_all_<tag>.json    the traced hotspot paths (paths.py)
  ambient_paths_<tag>.json        the traced ambient paths, where they exist
                                  (tilt_recovery.py --amp 0 --save-paths)
  corridor_profiles_<tag>.csv     corridor width against depth (corridor_all.py)
  conduit_profile_<tag>[_xm].csv  surface-anchored prominence and half width against
                                  depth, hotspots and null sites (conduit_detect.py)
  root_depth_<tag>.csv            depth reached by the connected slow column, hotspots
                                  and null sites (root_depth.py)

and computes, per path and per band, the apparent tilt (lateral length over vertical
length, in degrees) and the path-weighted share of deflections, with the deflection
definition of bao_pdf.py (direction change of at least 10 degrees over a 300 km
window, domain from 260 km). Where hotspot and ambient paths exist for the same
model, the difference of medians is tested by permutation.

Writes

  out/band_tilt_<tag>.csv           one row per path: population, tilt in each band
  out/mid_mantle_summary.csv        long form: tag, quantity, band, hotspot, ambient,
                                    n_hotspot, n_ambient, difference, P

    python3 mid_mantle_census.py                      (all tags with paths)
    python3 mid_mantle_census.py --tags RevealLO,GLADM35
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd
import provenance

R_E, DEG = 6371.0, np.pi / 180.0
BANDS = (('400-660', 400.0, 660.0), ('660-1000', 660.0, 1000.0),
         ('1000-1500', 1000.0, 1500.0), ('1500-2200', 1500.0, 2200.0),
         ('2200-2700', 2200.0, 2700.0))
ALL_TAGS = ('RevealLO', 'RevealLO_30km', 'REVEAL', 'GLADM35', 'SPiRaL', 'SEMUCB-WM1')
MIN_ANGLE, WINDOW_KM, Z_TOP = 10.0, 300.0, 260.0     # bao_pdf.py's defaults
N_PERM, SEED = 20000, 0


def _xyz(lat, lon, z):
    r = R_E - z
    la, lo = lat * DEG, lon * DEG
    return r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)


def band_tilt(p):
    """Apparent tilt per band: the lateral length of the path within the band over
    its vertical length, as an angle from the vertical."""
    lat = np.asarray(p['lat'], float); lon = np.asarray(p['lon'], float)
    z = np.asarray(p['depth'], float)
    o = np.argsort(z); lat, lon, z = lat[o], lon[o], z[o]
    rec = {}
    for nm, a, b in BANDS:
        m = (z >= a) & (z <= b)
        if m.sum() < 3:
            rec[nm] = np.nan; continue
        x, y, w = _xyz(lat[m], lon[m], z[m])
        seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(w) ** 2)
        dz = np.abs(np.diff(z[m]))
        lateral = np.sqrt(np.maximum(seg ** 2 - dz ** 2, 0.0)).sum()
        rec[nm] = float(np.degrees(np.arctan2(lateral, dz.sum())))
    return rec


def deflections(p):
    """Deflection depths of one path, as bao_pdf.py defines them."""
    la = np.asarray(p['lat'], float); lo = np.asarray(p['lon'], float)
    z = np.asarray(p['depth'], float)
    if len(z) < 12:
        return []
    o = np.argsort(z); la, lo, z = la[o], lo[o], z[o]
    e = R_E * DEG * (((lo - lo[0] + 180) % 360) - 180) * np.cos(la[0] * DEG)
    n_ = R_E * DEG * (la - la[0])
    dz = float(np.median(np.diff(z)))
    w = max(2, int(round(WINDOW_KM / max(dz, 1.0))))
    hit = []
    for k in range(w, len(z) - w):
        if z[k] < Z_TOP:
            continue
        v1 = np.array([e[k] - e[k - w], n_[k] - n_[k - w]])
        v2 = np.array([e[k + w] - e[k], n_[k + w] - n_[k]])
        if np.linalg.norm(v1) < 30 or np.linalg.norm(v2) < 30:
            continue
        c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if np.degrees(np.arccos(np.clip(c, -1, 1))) >= MIN_ANGLE:
            hit.append(float(z[k]))
    return hit


def deflection_share(paths):
    """Path-weighted share of deflections in each band, and the number of paths
    that deflect at all."""
    tot = np.zeros(len(BANDS)); n = 0
    for p in paths.values():
        h = deflections(p)
        if not h:
            continue
        h = np.asarray(h); n += 1
        tot += np.array([((h >= a) & (h < b)).sum() for _, a, b in BANDS]) / len(h)
    return (tot / max(n, 1)), n


def horizon_ratio(paths, lo=800.0, hi=1200.0, z0=660.0, z1=2200.0):
    """Share of the deflections between z0 and z1 that fall in lo-hi, path-weighted,
    divided by the share expected from its thickness if deflections were spread
    uniformly over z0-z1. A horizon at the viscosity increase would give a value well
    above one; ambient advection gives about one."""
    num = den = 0.0
    for p in paths.values():
        h = np.asarray(deflections(p))
        if not len(h):
            continue
        m = (h >= z0) & (h < z1)
        if not m.any():
            continue
        num += ((h >= lo) & (h < hi)).sum() / len(h)
        den += m.sum() / len(h)
    return (num / den) / ((hi - lo) / (z1 - z0)) if den > 0 else np.nan


def perm_p(h, a, rng):
    h = h[~np.isnan(h)]; a = a[~np.isnan(a)]
    if len(h) < 3 or len(a) < 3:
        return np.nan
    obs = np.median(h) - np.median(a); x = np.r_[h, a]; n = len(h)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        p = rng.permutation(len(x)); null[i] = np.median(x[p[:n]]) - np.median(x[p[n:]])
    return float(np.mean(np.abs(null) >= abs(obs)))


def band_median(df, zcol, vcol, kind=None):
    d = df if kind is None else df[df['kind'] == kind]
    out = {}
    for nm, a, b in BANDS:
        m = (d[zcol] >= a) & (d[zcol] <= b)
        out[nm] = (float(d.loc[m, vcol].median()) if m.any() else np.nan, int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='out')
    ap.add_argument('--tags', default=','.join(ALL_TAGS))
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    rows, inputs = [], []

    def add(tag, quantity, band, h, am, nh, na, P=np.nan):
        rows.append(dict(tag=tag, quantity=quantity, band=band, hotspot=h, ambient=am,
                         n_hotspot=nh, n_ambient=na,
                         difference=(h - am) if np.isfinite(am) else np.nan, P=P))

    for tag in a.tags.split(','):
        pth = os.path.join(a.dir, f'conduit_paths_all_{tag}.json')
        if not os.path.exists(pth):
            print(f'{tag}: no paths, skipped'); continue
        H = json.load(open(pth)); inputs.append(pth)
        amb = os.path.join(a.dir, f'ambient_paths_{tag}.json')
        A_ = json.load(open(amb)) if os.path.exists(amb) else {}
        if A_:
            inputs.append(amb)
        print(f'{tag}: {len(H)} hotspot paths, {len(A_)} ambient paths')

        # ---- tilt per band
        T = pd.DataFrame({k: band_tilt(v) for k, v in H.items()}).T
        T.insert(0, 'population', 'hotspot')
        if A_:
            TA = pd.DataFrame({k: band_tilt(v) for k, v in A_.items()}).T
            TA.insert(0, 'population', 'ambient')
            T = pd.concat([T, TA])
        T.index.name = 'site'
        tout = os.path.join(a.dir, f'band_tilt_{tag}.csv')
        T.to_csv(tout)
        provenance.stamp(tout, inputs=[pth] + ([amb] if A_ else []))
        for nm, _, _ in BANDS:
            h = T.loc[T.population == 'hotspot', nm].to_numpy(float)
            am = T.loc[T.population == 'ambient', nm].to_numpy(float) if A_ else np.array([])
            add(tag, 'tilt_deg', nm, float(np.nanmedian(h)),
                float(np.nanmedian(am)) if A_ else np.nan,
                int(np.isfinite(h).sum()), int(np.isfinite(am).sum()),
                perm_p(h, am, rng) if A_ else np.nan)

        # ---- deflection share per band, and the per-path median deflection depth
        sh, nh = deflection_share(H)
        sa, na = deflection_share(A_) if A_ else (np.full(len(BANDS), np.nan), 0)
        for (nm, _, _), x, y in zip(BANDS, sh, sa):
            add(tag, 'deflection_share', nm, float(x), float(y), nh, na)
        add(tag, 'horizon_ratio_800_1200', 'column', float(horizon_ratio(H)),
            float(horizon_ratio(A_)) if A_ else np.nan, nh, na)
        dh = np.array([np.median(d) for d in (deflections(p) for p in H.values()) if d])
        da = np.array([np.median(d) for d in (deflections(p) for p in A_.values()) if d])
        add(tag, 'deflection_depth_median_km', 'column', float(np.median(dh)),
            float(np.median(da)) if len(da) else np.nan, len(dh), len(da),
            perm_p(dh, da, rng) if len(da) else np.nan)

        # ---- corridor width per band (hotspot corridors only)
        cw = os.path.join(a.dir, f'corridor_profiles_{tag}.csv')
        if os.path.exists(cw):
            W = pd.read_csv(cw); inputs.append(cw)
            for nm, (v, n) in band_median(W, 'depth', 'width_km').items():
                add(tag, 'corridor_width_km', nm, v, np.nan, n, 0)

        # ---- surface-anchored prominence and half width, hotspots against null sites
        for suf in ('', '_xm'):
            cp = os.path.join(a.dir, f'conduit_profile_{tag}{suf}.csv')
            if os.path.exists(cp):
                C = pd.read_csv(cp); inputs.append(cp)
                for q in ('prominence', 'half_width_km'):
                    hb = band_median(C, 'z', q, 'hotspot')
                    nb = band_median(C, 'z', q, 'null_site')
                    for nm, _, _ in BANDS:
                        add(tag, q, nm, hb[nm][0], nb[nm][0], hb[nm][1], nb[nm][1])
                break

        # ---- shell percentile per band, when band_percentile.py has been run
        bp = os.path.join(a.dir, f'band_percentile_{tag}.csv')
        if os.path.exists(bp):
            Pc = pd.read_csv(bp); inputs.append(bp)
            for nm, _, _ in BANDS:
                h = Pc.loc[Pc.population == 'hotspot', nm].to_numpy(float)
                am = Pc.loc[Pc.population == 'ambient', nm].to_numpy(float)
                add(tag, 'shell_percentile', nm, float(np.nanmedian(h)),
                    float(np.nanmedian(am)) if len(am) else np.nan,
                    int(np.isfinite(h).sum()), int(np.isfinite(am).sum()),
                    perm_p(h, am, rng) if len(am) else np.nan)

        # ---- root depth, hotspots against null sites
        rd = os.path.join(a.dir, f'root_depth_{tag}.csv')
        if os.path.exists(rd):
            Rt = pd.read_csv(rd); inputs.append(rd)
            h = Rt[Rt.kind == 'hotspot']; n = Rt[Rt.kind != 'hotspot']
            add(tag, 'root_ge_2600_share', 'column', float((h.root_km >= 2600).mean()),
                float((n.root_km >= 2600).mean()), len(h), len(n))
            add(tag, 'root_km_median', 'column', float(h.root_km.median()),
                float(n.root_km.median()), len(h), len(n))
            add(tag, 'deep_frac_median', 'column', float(h.deep_frac.median()),
                float(n.deep_frac.median()), len(h), len(n))

    S = pd.DataFrame(rows)
    out = os.path.join(a.dir, 'mid_mantle_summary.csv')
    S.to_csv(out, index=False)
    provenance.stamp(out, min_angle=MIN_ANGLE, window_km=WINDOW_KM, z_top=Z_TOP,
                     n_perm=N_PERM, seed=SEED, inputs=inputs)
    pd.set_option('display.width', 200)
    for q in S.quantity.unique():
        print(f'\n== {q}')
        print(S[S.quantity == q].pivot_table(index='band', columns='tag',
              values=['hotspot', 'ambient'], sort=False).round(3).to_string())
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
