"""Does corridor width measure the structure beneath a site, or the site itself?

Corridor width is quoted for all 49 hotspots, 263 to 613 km. Before any of that
becomes a plume property it has to be shown that the number responds to the width of
what is actually there. The alternative is that it responds to where you are: a
least-cost corridor may be broad wherever the model is bland and narrow wherever it
is structured, in which case width is a property of the tomography's local character
and says nothing about a conduit.

Conduits of known width are injected into ambient locations - uniform points on the
sphere at least 1000 km from any Courtillot hotspot, so nothing real is underneath -
and the corridor is measured exactly as it is for a real hotspot. Every site is also
measured with no injection at all, so each injected width has its own ambient trace
to be scored against. Scoring an injection against anything other than the same site
untouched is what went wrong in the earlier morphology tests.

PRE-REGISTRATION. Fixed here before the script was first run, and not to be
revised after seeing the output.

  Direction.  Measured corridor width increases with injected conduit width.

  Primary statistic.  Spearman rho between injected FWHM and measured corridor
  width, over all injections.

  The discriminating comparison.  rho_injected against rho_ambient, the same
  correlation computed against the width measured at that site with no conduit
  present. If width tracks the site more strongly than the conduit, it is a
  property of the model and not of the plume, however well it correlates.

  Decision rule.  Corridor width is usable as a plume property if and only if
  rho_injected >= 0.5 at P < 0.01 AND rho_injected > rho_ambient AND the median
  measured width does not decrease across the five injected levels.

  If rho_injected >= 0.5 but the medians are not monotonic, width is reported as
  an ordinal indicator only, narrow against broad, never as a number in km.

  If rho_injected < 0.5, or if rho_ambient is the larger, corridor width is
  withdrawn as a plume property. It remains the error bar on the traced path,
  which is what it is by construction and needs no calibration to be that.

The injection is applied in place over a band of latitude rows and undone
afterwards, because inject_tilted copies the whole cube and a second copy does not
fit beside the contrast, cost and forward fields. It is verified against
inject_tilted on the real grid at startup and the run aborts if they differ.
"""
from __future__ import annotations
import argparse, os, sys, time, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from contrast import contrast_field
from path_cost import cost_field, site_cost
import path_config
from corridor import forward_field, excess_cost, self_check
from inject import inject_tilted, gc_km
warnings.filterwarnings('ignore')

R_E, DEG = 6371.0, np.pi / 180.0
SEED_DEPTH, N_RELAX, SIGMA = 200.0, 6, 800.0
TAU = 0.02
AMP = -1.0
# Width varies systematically with depth - 325 km in the transition zone rising to
# 627 km at 2700 - and the per-hotspot median across depth hides real structure, so
# the quantity to be reported is per band. The calibration therefore has to
# calibrate per band as well, or it validates a number the paper will not use.
BANDS = ((200.0, 660.0, 'upper_tz'), (660.0, 1500.0, 'upper_lm'),
         (1500.0, 2700.0, 'deep_lm'))
RADII = (200.0, 300.0, 400.0, 500.0, 600.0)
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0)) / 2.0
Z0_INJ, Z1_INJ = 100.0, 2800.0
MIN_HOTSPOT_KM = 1000.0

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--var', default='voigt')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max')
ap.add_argument('--sites', type=int, default=12)
ap.add_argument('--seed', type=int, default=51, help='the null seed used elsewhere')
ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
ap.add_argument('--pilot', action='store_true',
                help='3 sites and 3 widths, to size the full run before committing')
ap.add_argument('--skip-verify', action='store_true', dest='skip_verify',
                help='skip the in-place injector check against inject_tilted')
ap.add_argument('--verify-only', action='store_true', dest='verify_only',
                help='run the injector verification and stop, writing nothing. Use '
                     'this rather than --sites 0, which used to reach the writer '
                     'with an empty frame and truncate the calibration it should '
                     'have been protecting.')
ap.add_argument('--amps', default=None,
                help='comma-separated amplitudes; sweeps amplitude at one fixed '
                     'radius instead of sweeping radius. Separate question from the '
                     'pre-registered calibration and reported separately.')
ap.add_argument('--sweep-radius', type=float, default=400.0, dest='sweep_radius')
ap.add_argument('--tilts', default=None,
                help='comma-separated conduit tilts in degrees, swept at one radius '
                     'and one amplitude. Separates tilt from incoherence as the '
                     'reason real corridors are wider than injected ones.')
ap.add_argument('--sweep-azimuth', type=float, default=45.0, dest='sweep_azimuth')
ap.add_argument('--duty', default=None,
                help='comma-separated fractions of the depth column the conduit '
                     'actually occupies, swept at one radius, amplitude and tilt. '
                     'Tests discontinuity directly instead of inferring it as the '
                     'residual after tilt and amplitude are excluded.')
ap.add_argument('--segment-km', type=float, default=400.0, dest='segment_km',
                help='length of one present-plus-absent cycle, km')
ap.add_argument('--segments', default=None,
                help='comma-separated cycle lengths in km, swept at one occupancy. '
                     'The 400 km default was chosen arbitrarily; this says whether '
                     'the inferred occupancy depends on it.')
ap.add_argument('--sweep-duty', type=float, default=0.25, dest='sweep_duty',
                help='occupancy held fixed while sweeping segment length; 0.25 sits '
                     'on the steep part of the continuity curve')
# Defaulted to the CALIBRATED move set. These were 1 and 1.0, the superseded set,
# so every caller had to remember to override them and a runner that did not - mine -
# produced an hour of sweeps at a move set no result uses. A default that is never the
# right answer is a defect, not a neutral starting point.
ap.add_argument('--ncell', type=int, default=2,
                help='lateral cells a slanted move may cross; calibrated value 2. The '
                     'injected and observed corridors must be built under the SAME move '
                     'set as the paths they are compared with')
ap.add_argument('--lateral', type=float, default=0.60,
                help='weight on the lateral component of a move; calibrated 0.60')
ap.add_argument('--dir', default='out')
A = ap.parse_args()

SEGS = [float(x) for x in A.segments.split(',')] if A.segments else None
DUTY = [float(x) for x in A.duty.split(',')] if A.duty else None
TILTS = [float(x) for x in A.tilts.split(',')] if A.tilts else None
AMPS = [float(x) for x in A.amps.split(',')] if A.amps else None
if AMPS and all(a > 0 for a in AMPS):
    # argparse reads a leading '-' as an option, so --amps -0.5,... fails unless
    # written --amps=-0.5,...  Plumes are slow, so bare magnitudes are accepted
    # and negated here rather than leaving a command that only works one way.
    print(f'amplitudes given as magnitudes; using {[-a for a in AMPS]}', flush=True)
    AMPS = [-a for a in AMPS]
radii = RADII[::2] if A.pilot else RADII
if AMPS or TILTS or DUTY or SEGS:
    radii = [A.sweep_radius]
n_sites = 3 if A.pilot else A.sites


def _hmax(c):
    v = getattr(c, 'h_max', None)
    return None if v is None or (isinstance(v, float) and v != v) else float(v)


# The configuration is READ from out/path_config_<tag>.json, never re-derived.
# Re-deriving it as the median of the retained set makes it depend on which
# configurations happened to clear a 75 per cent floor on eighteen injections,
# and after the move-set retrace that median moved from s=0.4 z=2700 r=3.0 to
# s=0.3 z=2800 r=2.0. A null traced under one configuration and hotspots under
# another are two different searches and must not be compared.
c = path_config.load(A.tag, A.dir)
hm = _hmax(c)
print(f'calibrated configuration, same as corridor_all.py: s={c.s} '
      f'z_target={c.z_target:.0f} channel={c.channel} radius={c.radius} h_max={hm}',
      flush=True)

depth, lat, lon, arr = load_anomaly(A.file, ModelSpec(A.tag, A.var),
                                    depth_max=A.depth_max, every=A.every)
lon, arr = dedupe_lon(lon, arr)
arr = np.asarray(arr, dtype=np.float64)
z = np.asarray(depth, float)
LATV, LONV = np.asarray(lat, float), np.asarray(lon, float)
k0 = int(np.where(z >= float(c.z_target))[0][0]) if (z >= float(c.z_target)).any() else len(z) - 1
k_seed = int(np.argmin(np.abs(z - SEED_DEPTH)))
dlat = abs(LATV[1] - LATV[0]) * DEG * R_E
dlon = abs(LONV[1] - LONV[0]) * DEG * R_E
cell_km2 = dlat * dlon * np.cos(LATV * DEG)
print(f'model loaded; seed shell {z[k_seed]:.0f} km, target shell {z[k0]:.0f} km',
      flush=True)


def patch_rows(hlat, radius_km):
    """Latitude rows the conduit can reach, at eight Gaussian widths.

    Five sigma truncates the Gaussian at exp(-12.5) = 3.7e-06, which is physically
    negligible but is 3.7e-06 more than zero, and the verification against
    inject_tilted is written to a tolerance of 1e-12. Eight sigma leaves
    exp(-32) = 1.3e-14, under the tolerance, so the check passes on its merits
    instead of being argued around.
    """
    reach = 8.0 * (radius_km / 2.0) / (R_E * DEG)
    return np.where(np.abs(LATV - hlat) <= reach)[0]


def _present(zk, duty, seg_km):
    """Is the conduit present at this depth?

    A duty of 1 is a continuous conduit and reproduces inject_tilted exactly, which
    is what the startup verification checks. Below 1 the conduit occupies that
    fraction of each cycle and is simply absent over the rest, which is the
    discontinuity the width gap is attributed to.
    """
    if duty >= 1.0:
        return True
    return ((zk - Z0_INJ) % seg_km) < duty * seg_km


def apply_patch(a, hlat, hlon, radius_km, amp, tilt_deg=0.0, azimuth=45.0,
                duty=1.0, seg_km=400.0):
    """Add a conduit in place, tilted if asked. Returns what is needed to undo it.

    With tilt the centre migrates with depth exactly as inject_tilted does under
    profile='uniform', so the row band must also cover the whole excursion.
    """
    reach_deg = 8.0 * (radius_km / 2.0) / (R_E * DEG) + abs(tilt_deg)
    rows = np.where(np.abs(LATV - hlat) <= reach_deg)[0]
    if not len(rows):
        return None
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    saved = a[:, r0:r1, :].copy()
    LO, LA = np.meshgrid(((LONV + 180) % 360) - 180, LATV[r0:r1])
    hlo180 = ((hlon + 180) % 360) - 180
    if tilt_deg == 0.0:
        d = gc_km(hlat, hlo180, LA, LO)
        w0 = np.exp(-(d ** 2) / (2.0 * (radius_km / 2.0) ** 2))
    for k, zk in enumerate(z):
        if zk < Z0_INJ or zk > Z1_INJ:
            continue
        if not _present(zk, duty, seg_km):
            continue
        taper = 1.0 if zk < Z1_INJ - 200 else max(0.0, (Z1_INJ - zk) / 200.0)
        if tilt_deg == 0.0:
            w = w0
        else:
            f = (zk - Z0_INJ) / max(Z1_INJ - Z0_INJ, 1)
            off = tilt_deg * f
            clat = hlat + off * np.cos(np.radians(azimuth))
            clon = hlo180 + off * np.sin(np.radians(azimuth)) / max(
                np.cos(np.radians(clat)), 0.2)
            d = gc_km(clat, clon, LA, LO)
            w = np.exp(-(d ** 2) / (2.0 * (radius_km / 2.0) ** 2))
        a[k, r0:r1, :] += amp * w * taper
    return (r0, r1, saved)


def undo_patch(a, u):
    if u is not None:
        r0, r1, saved = u
        a[:, r0:r1, :] = saved


if not A.skip_verify:
    t = time.time()
    hla, hlo, rad = float(LATV[len(LATV) // 3]), 12.0, RADII[-1]
    # Both the vertical and the tilted path are checked, because the tilt sweep uses
    # a different branch and an unverified injector is what this guard exists to stop.
    for _tilt in (0.0, 10.0):
        ref = inject_tilted(arr, z, LATV, LONV, hla, hlo, rad, AMP, _tilt,
                            z0=Z0_INJ, z1=Z1_INJ, azimuth=A.sweep_azimuth,
                            profile='uniform')
        u = apply_patch(arr, hla, hlo, rad, AMP, _tilt, A.sweep_azimuth)
        fa, fr = np.isfinite(arr), np.isfinite(ref)
        same_mask = bool(np.array_equal(fa, fr))
        both = fa & fr
        dmax = float(np.max(np.abs(arr[both] - ref[both]))) if both.any() else float('nan')
        n_fin = int(both.sum())
        undo_patch(arr, u)
        del ref, fa, fr, both
        print(f'  injector at tilt {_tilt:.0f} deg agrees with inject_tilted to '
              f'{dmax:.3e} over {n_fin} finite cells, NaN pattern preserved: '
              f'{same_mask}', flush=True)
        if not np.isfinite(dmax) or not same_mask or n_fin == 0 or dmax > 1e-12:
            raise SystemExit(f'the in-place injector does not reproduce inject_tilted '
                             f'at tilt {_tilt} (max difference {dmax}, masks equal '
                             f'{same_mask}, {n_fin} finite cells)')
    print(f'injector verified ({time.time() - t:.0f} s)', flush=True)
    ref = inject_tilted(arr, z, LATV, LONV, hla, hlo, rad, AMP, 0.0,
                        z0=Z0_INJ, z1=Z1_INJ, profile='uniform')
    u = apply_patch(arr, hla, hlo, rad, AMP)
    # The model carries NaNs, so a plain max is nan and `nan > tol` is False - the
    # guard would pass by failing to compute. Compare where both are finite, require
    # the finite masks to match, and require the result itself to be a real number.
    fa, fr = np.isfinite(arr), np.isfinite(ref)
    same_mask = bool(np.array_equal(fa, fr))
    both = fa & fr
    dmax = float(np.max(np.abs(arr[both] - ref[both]))) if both.any() else float('nan')
    n_fin = int(both.sum())
    undo_patch(arr, u)
    del ref, fa, fr, both
    print(f'in-place injector agrees with inject_tilted to {dmax:.3e} over '
          f'{n_fin} finite cells, NaN pattern preserved: {same_mask} '
          f'({time.time() - t:.0f} s)', flush=True)
    if not np.isfinite(dmax) or not same_mask or n_fin == 0 or dmax > 1e-12:
        raise SystemExit('the in-place injector does not reproduce inject_tilted '
                         f'(max difference {dmax}, masks equal {same_mask}, '
                         f'{n_fin} finite cells); the calibration would not be '
                         'measuring the same conduit')

if A.verify_only:
    raise SystemExit('injector verified; nothing written (--verify-only)')

hs = pd.read_csv(A.hotspots).dropna(subset=['lat', 'lon_180'])
HLA, HLO = hs.lat.to_numpy(float), hs.lon_180.to_numpy(float)
rng = np.random.default_rng(A.seed)
sites, tries = [], 0
while len(sites) < n_sites and tries < 20000:
    tries += 1
    slo = float(360 * rng.random() - 180)
    sla = float(np.degrees(np.arcsin(2 * rng.random() - 1)))
    if float(gc_km(sla, slo, HLA, HLO).min()) >= MIN_HOTSPOT_KM:
        sites.append((f'amb{len(sites):02d}', sla, slo))
print(f'{len(sites)} ambient sites, all at least {MIN_HOTSPOT_KM:g} km from any '
      f'hotspot, from {tries} draws', flush=True)


def corridor_width(a, hla, hlo):
    """Median equivalent diameter of the corridor at TAU, as corridor_all.py measures it."""
    con = np.asarray(contrast_field(a, LATV, LONV, SIGMA), dtype=np.float64)
    C = np.asarray(cost_field(a, con, z, LATV, LONV, s=float(c.s),
                              z_target=float(c.z_target), n_relax=N_RELAX,
                              channel=str(c.channel), h_max_km=hm,
                              lateral=float(A.lateral), ncell=int(A.ncell)),
                   dtype=np.float64)
    cost = float(site_cost(C, z, LATV, LONV, hla, hlo, float(c.radius), SEED_DEPTH))
    if not np.isfinite(cost):
        del con, C
        return None
    D = forward_field(a, con, z, LATV, LONV, hla, hlo, s=float(c.s),
                      z_target=float(c.z_target), radius_deg=float(c.radius),
                      seed_depth=SEED_DEPTH, n_relax=N_RELAX,
                      channel=str(c.channel), h_max_km=hm,
                      lateral=float(A.lateral), ncell=int(A.ncell))
    dC, cbest = excess_cost(D, C, k_seed, k0)
    if dC is None:
        del con, C, D
        return None
    ok, _ = self_check(D, C, k_seed, k0, c_best=cost)
    del D
    m = np.asarray(dC, dtype=np.float64) <= TAU * cbest
    del dC, con, C
    w, zs = [], []
    for k in range(k_seed, k0 + 1):
        a_ = float((m[k] * cell_km2[:, None]).sum())
        if a_ > 0:
            w.append(2.0 * np.sqrt(a_ / np.pi)); zs.append(float(z[k]))
    del m
    if not w:
        return None
    zk = np.asarray(zs, float)
    wa = np.asarray(w, float)
    rec = dict(ok=bool(ok), width=float(np.median(wa)), c_best=float(cbest))
    for lo, hi, lab in BANDS:
        m = (zk >= lo) & (zk < hi)
        rec[f'w_{lab}'] = float(np.median(wa[m])) if m.sum() >= 3 else np.nan
    return rec


rows, t0 = [], time.time()
_per = (len(SEGS) if SEGS else len(DUTY) if DUTY else len(TILTS) if TILTS
        else len(AMPS) if AMPS else len(radii))
todo = len(sites) * (1 + _per)
done = 0
for nm, sla, slo in sites:
    base = corridor_width(arr, sla, slo)
    done += 1
    _r = dict(site=nm, lat=sla, lon=slo, injected_radius=0.0, injected_fwhm=0.0,
              ok=None if base is None else base['ok'],
              width=None if base is None else base['width'])
    for _, _, lab in BANDS:
        _r[f'w_{lab}'] = None if base is None else base.get(f'w_{lab}')
    rows.append(_r)
    shown = 'failed' if base is None else f'{base["width"]:.0f} km'
    print(f'  [{done}/{todo}] {nm} ambient {shown}'
          f'   {time.time() - t0:.0f} s elapsed', flush=True)
    if SEGS:
        plan = [(A.sweep_radius, AMP, 0.0, A.sweep_duty, g_) for g_ in SEGS]
    elif DUTY:
        plan = [(A.sweep_radius, AMP, 0.0, d_, A.segment_km) for d_ in DUTY]
    elif TILTS:
        plan = [(A.sweep_radius, AMP, t, 1.0, A.segment_km) for t in TILTS]
    elif AMPS:
        plan = [(A.sweep_radius, a, 0.0, 1.0, A.segment_km) for a in AMPS]
    else:
        plan = [(r_, AMP, 0.0, 1.0, A.segment_km) for r_ in radii]
    for rad, amp, tlt, dty, seg in plan:
        u = apply_patch(arr, sla, slo, rad, amp, tlt, A.sweep_azimuth, dty, seg)
        r = corridor_width(arr, sla, slo)
        undo_patch(arr, u)
        done += 1
        _r = dict(site=nm, lat=sla, lon=slo, injected_radius=rad, amp=amp, tilt=tlt,
                  duty=dty, segment_km=seg, injected_fwhm=rad * FWHM, ok=None if r is None else r['ok'],
                  width=None if r is None else r['width'],
                  ambient_width=None if base is None else base['width'])
        for _, _, lab in BANDS:
            _r[f'w_{lab}'] = None if r is None else r.get(f'w_{lab}')
            _r[f'amb_{lab}'] = None if base is None else base.get(f'w_{lab}')
        rows.append(_r)
        shown = 'failed' if r is None else f'{r["width"]:.0f} km'
        print(f'  [{done}/{todo}] {nm} r={rad:.0f} {shown}'
              f'   {time.time() - t0:.0f} s elapsed', flush=True)

def _tag_vals(v):
    """Name a sweep by what it swept, so two sweeps never share an output file."""
    return '-'.join(('%g' % abs(x)) for x in v)


suffix = ('_segsweep_' + _tag_vals(SEGS) if SEGS
          else '_dutysweep_' + _tag_vals(DUTY) if DUTY
          else '_tiltsweep_' + _tag_vals(TILTS) if TILTS
          else '_ampsweep_' + _tag_vals(AMPS) if AMPS
          else '_pilot' if A.pilot else '')
d = pd.DataFrame(rows)
# Provenance. Three of the four occupancy sweeps in the paper were run at the old
# move set and there was no way to tell from the files, only from their dates. A
# sweep that cannot say which move set produced it cannot be compared with an
# observed corridor, because the observed side moved too.
d['lateral'] = float(A.lateral)
d['ncell'] = int(A.ncell)
for _k in ('s', 'z_target', 'channel', 'radius', 'h_max'):
    if hasattr(c, _k):
        d[f'cfg_{_k}'] = getattr(c, _k)
out = os.path.join(A.dir, f'corridor_width_calibration_{A.tag}{suffix}.csv')
# Never let a run with nothing in it overwrite a real calibration. An empty frame
# reaching this line once truncated a 60-injection run that took an hour.
if not len(d) or 'width' not in d or not d.width.notna().any():
    raise SystemExit(f'no corridors measured; refusing to write {out} and destroy '
                     'whatever is already there')
d.to_csv(out, index=False)
print(f'\nwrote {out}')
print(f'{time.time() - t0:.0f} s for {todo} corridors, '
      f'{(time.time() - t0) / max(todo, 1):.0f} s each')
if A.pilot:
    full = len(RADII) + 1
    print(f'a full run of {A.sites} sites is {A.sites * full} corridors, about '
          f'{A.sites * full * (time.time() - t0) / max(todo, 1) / 60:.0f} minutes')

inj = d[(d.injected_radius > 0) & d.width.notna() & d.ambient_width.notna()]
print(f'\n{len(inj)} injected corridors measured, self-check passed on '
      f'{int(inj.ok.sum())}')
if len(inj) < 6:
    raise SystemExit('too few for the pre-registered test')

from scipy.stats import spearmanr
r_inj, p_inj = spearmanr(inj.injected_fwhm, inj.width)
r_amb, p_amb = spearmanr(inj.ambient_width, inj.width)
print(f'\nmeasured width against INJECTED width : rho = {r_inj:+.3f}, P = {p_inj:.3g}')
print(f'measured width against AMBIENT width  : rho = {r_amb:+.3f}, P = {p_amb:.3g}')
print('\nmedian measured width by injected level:')
g = inj.groupby('injected_fwhm').width.agg(['median', 'count'])
for f_, r_ in g.iterrows():
    print(f'  injected {f_:5.0f} km FWHM   measured {r_["median"]:6.0f} km   '
          f'n = {int(r_["count"])}')
med = g['median'].to_numpy(float)
mono = bool(np.all(np.diff(med) >= 0))
print(f'\nmedians non-decreasing across levels: {mono}')
bias = float(np.median(inj.width - inj.injected_fwhm))
print(f'median measured minus injected: {bias:+.0f} km')

if AMPS or TILTS or DUTY or SEGS:
    print('\n[Sweep mode: injected width is constant here, so the correlation '
          'above is\n undefined and the pre-registered decision does NOT apply. '
          'That verdict comes\n only from a radius sweep. Skipping it.]')
else:
    print('\nPRE-REGISTERED DECISION')
    if r_inj >= 0.5 and p_inj < 0.01 and r_inj > r_amb and mono:
        print('  corridor width IS usable as a plume property')
    elif r_inj >= 0.5 and p_inj < 0.01 and r_inj > r_amb:
        print('  ordinal only: report narrow against broad, never a number in km')
    else:
        why = ('it tracks the site more strongly than the conduit'
               if r_amb >= r_inj else 'it does not track the injected width')
        print(f'  corridor width is WITHDRAWN as a plume property, because {why}.')
        print('  It remains the error bar on the traced path, which needs no calibration.')

if not (AMPS or TILTS or DUTY or SEGS):
    print('\nPER BAND, declared before this script was first run. The primary rule')
    print('above is unchanged; this calibrates the quantity the paper will actually')
    print('report, and a band passes on the same terms: rho against injected beats')
    print("rho against that band's own ambient width, at P < 0.01.")
    print(f'{"band":<12s}{"n":>5s}{"rho injected":>14s}{"P":>10s}'
          f'{"rho ambient":>13s}  verdict')
    for _, _, lab in BANDS:
        g = inj.dropna(subset=[f'w_{lab}', f'amb_{lab}'])
        if len(g) < 6:
            print(f'{lab:<12s}{len(g):5d}   too few')
            continue
        ri, pi = spearmanr(g.injected_fwhm, g[f'w_{lab}'])
        ra, _ = spearmanr(g[f'amb_{lab}'], g[f'w_{lab}'])
        ok = (ri >= 0.5) and (pi < 0.01) and (ri > ra)
        print(f'{lab:<12s}{len(g):5d}{ri:+14.3f}{pi:10.4f}{ra:+13.3f}  '
              f'{"usable" if ok else "not usable"}')

if TILTS:
    print('\nTILT SWEEP: does tilting a conduit of fixed strength widen its corridor')
    print('toward the width real hotspots show? If it does, the amplitude gap is')
    print('partly geometry rather than incoherence.')
    try:
        _h = pd.read_csv(os.path.join(A.dir, f'corridor_summary_{A.tag}.csv'))
        _obs = float(_h[_h.ok == True]['width_med_0.02'].median())
    except Exception:
        _obs = float('nan')
    print(f'  real hotspots, median corridor width: {_obs:.0f} km')
    print(f'  injected at {AMP:+.2f} per cent, radius {A.sweep_radius:.0f} km')
    print(f'{"tilt":>7s}{"n":>4s}{"median width":>14s}')
    for t_, g in inj.groupby('tilt'):
        print(f'{t_:6.0f}d{len(g):4d}{g.width.median():13.0f}k')
    print(f'{"ambient":>7s}{len(d[d.injected_radius == 0]):4d}'
          f'{d[d.injected_radius == 0].width.median():13.0f}k')

if AMPS:
    print('\nAMPLITUDE SWEEP, a separate question from the calibration above: at what')
    print('conduit strength does the corridor reach the width real hotspots show?')
    try:
        _h = pd.read_csv(os.path.join(A.dir, f'corridor_summary_{A.tag}.csv'))
        _obs = float(_h[_h.ok == True]['width_med_0.02'].median())
    except Exception:
        _obs = float('nan')
    print(f'  real hotspots, median corridor width: {_obs:.0f} km')
    print(f'{"amplitude":>11s}{"n":>4s}{"median width":>14s}')
    for a_, g in inj.groupby('amp'):
        print(f'{a_:+11.2f}{len(g):4d}{g.width.median():13.0f}k')
    print(f'{"ambient":>11s}{len(d[d.injected_radius == 0]):4d}'
          f'{d[d.injected_radius == 0].width.median():13.0f}k')

if DUTY:
    print('\nCONTINUITY SWEEP: how much of the depth column must a conduit actually')
    print('occupy to give the corridor width real hotspots show? This tests')
    print('discontinuity directly rather than inferring it as a residual.')
    try:
        _h = pd.read_csv(os.path.join(A.dir, f'corridor_summary_{A.tag}.csv'))
        _obs = float(_h[_h.ok == True]['width_med_0.02'].median())
    except Exception:
        _obs = float('nan')
    print(f'  real hotspots, median corridor width: {_obs:.0f} km')
    print(f'  injected at {AMP:+.2f} per cent, radius {A.sweep_radius:.0f} km, '
          f'vertical, in {A.segment_km:.0f} km cycles')
    print(f'{"occupied":>10s}{"n":>4s}{"median width":>14s}')
    _g = inj.groupby('duty').width.median()
    for d_, w_ in _g.items():
        print(f'{100 * d_:9.0f}%{int((inj.duty == d_).sum()):4d}{w_:13.0f}k')
    print(f'{"ambient":>10s}{len(d[d.injected_radius == 0]):4d}'
          f'{d[d.injected_radius == 0].width.median():13.0f}k')
    if len(_g) > 1 and np.isfinite(_obs):
        _x = _g.index.to_numpy(float)[::-1]; _y = _g.to_numpy(float)[::-1]
        if _y[0] <= _obs <= _y[-1] or _y[-1] <= _obs <= _y[0]:
            print(f'\n  the observed {_obs:.0f} km corresponds to a conduit occupying '
                  f'about {100 * np.interp(_obs, _y, _x):.0f} per cent of the column')

if SEGS:
    print(f'\nSEGMENT-LENGTH SWEEP at {100 * A.sweep_duty:.0f} per cent occupancy.')
    print('The 400 km cycle was chosen arbitrarily. If corridor width barely moves')
    print('across cycle length, the inferred occupancy does not depend on that choice.')
    print(f'{"cycle km":>10s}{"n":>4s}{"median width":>14s}')
    _g = inj.groupby('segment_km').width.median()
    for s_, w_ in _g.items():
        print(f'{s_:10.0f}{int((inj.segment_km == s_).sum()):4d}{w_:13.0f}k')
    print(f'{"ambient":>10s}{len(d[d.injected_radius == 0]):4d}'
          f'{d[d.injected_radius == 0].width.median():13.0f}k')
    print(f'\n  spread across cycle lengths: {_g.max() - _g.min():.0f} km')
