"""Recover the conduit geometry the search selects, for the cross-section figure.

For each hotspot the optimal descending path is traced under every configuration
that survived calibration, and the median configuration's path is written out
together with the along-track and cross-track position of each point relative to
the great circle of the corresponding cross-section, so that the panel can show
which parts of the path lie in the plane of the section and which do not.
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import ModelSpec, load_anomaly, dedupe_lon
from contrast import contrast_field
from path_cost import cost_field, trace_path
import path_config
import provenance
warnings.filterwarnings('ignore')

def _hmax(c):
    """The horizontal reach of the chosen configuration, if the table records one."""
    v = getattr(c, 'h_max', None)
    return None if v is None or (isinstance(v, float) and v != v) else float(v)


R_E = 6371.0

def along_cross(lat0, lon0, az, la, lo):
    """Along-track angle (deg, signed) and cross-track distance (km) of points."""
    la0, lo0, a = np.radians(lat0), np.radians(lon0), np.radians(az)
    la_, lo_ = np.radians(la), np.radians(lo)
    d13 = np.arccos(np.clip(np.sin(la0) * np.sin(la_) +
                            np.cos(la0) * np.cos(la_) * np.cos(lo_ - lo0), -1, 1))
    th13 = np.arctan2(np.sin(lo_ - lo0) * np.cos(la_),
                      np.cos(la0) * np.sin(la_) -
                      np.sin(la0) * np.cos(la_) * np.cos(lo_ - lo0))
    dxt = np.arcsin(np.sin(d13) * np.sin(th13 - a))
    dat = np.arccos(np.clip(np.cos(d13) / np.maximum(np.cos(dxt), 1e-9), -1, 1))
    sign = np.where(np.cos(th13 - a) < 0, -1.0, 1.0)
    return np.degrees(dat) * sign, np.degrees(dxt) * np.pi / 180.0 * R_E


def best_azimuth(lat0, lon0, la, lo, depth, zmax=2700.0):
    """Azimuth of the great circle through the hotspot that best contains the path.

    Chosen by least absolute cross-track distance, so the plane of each section
    contains the conduit the search actually selected. Deeper points are weighted
    by depth, because it is the deep part of the path whose continuity is at
    issue and the shallow part is anchored at the hotspot in any case.
    """
    w = np.clip(np.asarray(depth, float), 0.0, zmax)
    best, arg = np.inf, 0.0
    for a in np.arange(0.0, 180.0, 1.0):
        _, dxt = along_cross(lat0, lon0, a, la, lo)
        v = float(np.sum(w * np.abs(dxt)))
        if v < best:
            best, arg = v, a
    return arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--depth-max', type=float, default=2880.0,
                    dest='depth_max',
                    help='ignore shells deeper than this, in km. The default '
                         'sits just above the core-mantle boundary at 2891 km, '
                         'because a file that samples the whole Earth radius '
                         'carries shells in which the shear velocity collapses')
    ap.add_argument('--ncell', type=int, default=1,
                    help='lateral cells a single slanted move may span. 1 was the whole move set and caps the steepest expressible tilt near 64 degrees in this model; 2 raises it to 76. Chosen with --lateral by slant_calibrate.py on hotspot paths against a matched ambient set.')
    ap.add_argument('--lateral', type=float, default=1.0,
                    help='weight on horizontal travel in the path cost. 1.0 is the Euclidean step length used before the tilt calibration; 0.40 is the value chosen by tilt_recovery.py under its pre-registered rule. Left at 1.0 the search abandons strongly tilted conduits, recovering a median 72 per cent of an injected lateral offset against 93 per cent at 0.40.')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--refreeze', action='store_true',
                    help='re-derive the configuration from the current detection '
                         'table and overwrite the frozen one. Every path, corridor, '
                         'offset, tilt and width in the paper is computed under it, '
                         'so this is never incidental to another change')
    ap.add_argument('--sections', default='sections.json',
                    help='hotspot: azimuth, the great circles the panels are cut on')
    ap.add_argument('--sections-all', default='sections_all.json',
                    dest='sections_all',
                    help='the same, for every hotspot. Written by the paper model '
                         'under --all --fix-azimuths and read by every other '
                         'model, so all of them are cut on the same planes')
    ap.add_argument('--fix-azimuths', action='store_true',
                    help='derive the section azimuths from this model and write them')
    ap.add_argument('--all', action='store_true', dest='all_hotspots',
                    help='trace every hotspot in the table, not only the nine the '
                         'paper cuts sections on, and derive an azimuth for each. '
                         'Writes conduit_paths_all_<tag>.json for the atlas figure; '
                         'costs one extra trace per hotspot on a cost field that '
                         'has already been built, so it is nearly free')
    ap.add_argument('--s', type=float, default=None,
                    help='trace under this s instead of the frozen one, for this '
                         'run only. The frozen file is never written. The output '
                         'file name carries the override, so a production file is '
                         'never overwritten by an exploratory trace.')
    ap.add_argument('--channel', default=None,
                    help='likewise for the cost channel (anom, contrast, min)')
    A = ap.parse_args()
    MODEL, VAR, EVERY = A.file, A.var, A.every
    AZOUT = A.sections
    OUT = os.path.join(A.dir, ('conduit_paths_all_' if A.all_hotspots
                                else 'conduit_paths_') + f'{A.tag}.json')
    # The configuration is read from out/path_config_<tag>.json, not re-derived.
    # See path_config.py: the median-of-the-retained-set selector it replaces moves
    # on single-trial differences in an eighteen-injection binomial, so a change in
    # the move set could silently retrace the whole paper under a different one.
    c = path_config.load(A.tag, A.dir, refreeze=A.refreeze)

    # An override changes the cost for this run and nothing else. path_config.py
    # explains why the configuration is frozen rather than re-derived: the selector
    # it replaced moved on single-trial differences in an eighteen-injection
    # binomial. That argument is about re-deriving silently, not about never
    # examining the choice, so an override is allowed, is announced, and is written
    # to its own file. Adopting one means re-freezing deliberately, not renaming a
    # file.
    _ov = []
    if A.s is not None or A.channel is not None:
        if A.fix_azimuths:
            raise SystemExit('--fix-azimuths with an override would recut the '
                             'section planes under a configuration that is not the '
                             'frozen one, and every model is meant to be cut on the '
                             'same planes. Run the override without it.')
        c = c.copy()
        if A.s is not None:
            _ov.append(f's{float(A.s):g}')
            c['s'] = float(A.s)
        if A.channel is not None:
            _ov.append(str(A.channel))
            c['channel'] = str(A.channel)
        OUT = OUT[:-len('.json')] + '_' + '_'.join(_ov) + '.json'
        print('OVERRIDE, this run only; out/path_config_%s.json is unchanged: %s'
              % (A.tag, ' '.join(_ov)))
        print(f'  writing {OUT}')

    depth, lat, lon, arr = load_anomaly(MODEL, ModelSpec(A.tag, VAR),
                                        depth_max=A.depth_max, every=EVERY)
    lon, arr = dedupe_lon(lon, arr)
    con = contrast_field(arr, lat, lon, 800.0)
    C = cost_field(arr, con, depth, lat, lon, s=float(c.s),
                   z_target=float(c.z_target), n_relax=6, channel=str(c.channel),
                   h_max_km=_hmax(c), lateral=float(A.lateral), ncell=int(A.ncell))
    _dcache = {}

    # Under --all the planes come from the all-hotspot file: derived once from
    # the paper model and reused by the rest, so a hotspot is cut on the same
    # great circle in every model and the panels can be compared. Deriving them
    # per model would give each its own plane and quietly break that.
    if A.all_hotspots and not A.fix_azimuths and os.path.exists(A.sections_all):
        az = json.load(open(A.sections_all))
    else:
        az = json.load(open(AZOUT))
    cls = pd.read_csv(os.path.join(A.dir, f'classification_{A.tag}.csv'))
    hs = pd.read_csv('hotspots_courtillot2003.csv').dropna(
        subset=['lat', 'lon_180'])

    # Either the nine great circles the paper's sections are cut on, or every
    # hotspot in the table. In the second case there is no stored azimuth to
    # start from, so each is derived from the path the search recovers there.
    if A.all_hotspots:
        todo = [(str(r.hotspot), az.get(str(r.hotspot), 0.0)) for _, r in hs.iterrows()]
    else:
        todo = list(az.items())

    out, az_path = {}, {}
    for name, a_ in todo:
        r = hs[hs.hotspot.astype(str) == name]
        if not len(r):
            r = hs[hs.hotspot.astype(str).str.startswith(name.split('(')[0].strip())]
        if not len(r):
            print(f'  {name}: not in hotspot table'); continue
        r = r.iloc[0]
        tr = trace_path(C, arr, con, depth, lat, lon, float(r.lat), float(r.lon_180),
                        s=float(c.s), radius_deg=float(c.radius), n_relax=6,
                        channel=str(c.channel), h_max_km=_hmax(c), lateral=float(A.lateral), ncell=int(A.ncell), cache=_dcache)
        if tr is None:
            print(f'  {name}: no path'); continue
        z, la, lo = tr
        # The section great circles are fixed once, by the model the paper
        # figure uses, so that every model is shown on the same planes and the
        # panels can be compared directly.
        azp = (best_azimuth(float(r.lat), float(r.lon_180), la, lo, z)
               if (A.fix_azimuths or (A.all_hotspots and str(r.hotspot) not in az))
               else float(az.get(str(r.hotspot), a_)))
        dat, dxt = along_cross(float(r.lat), float(r.lon_180), azp, la, lo)
        az_path[name] = azp
        f = cls[cls.hotspot.astype(str) == str(r.hotspot)]
        out[str(r.hotspot)] = dict(
            azimuth=azp, azimuth_old=float(a_ if np.isscalar(a_) else a_[0]),
            lat=la.tolist(), lon=lo.tolist(), depth=z.tolist(),
            along_deg=dat.tolist(), cross_km=dxt.tolist(),
            root_fraction=float(f.root_fraction.iloc[0]) if len(f) else np.nan)
        off = R_E * np.arccos(np.clip(
            np.sin(np.radians(la[0])) * np.sin(np.radians(la)) +
            np.cos(np.radians(la[0])) * np.cos(np.radians(la)) *
            np.cos(np.radians(lo - lo[0])), -1, 1))
        print(f'  {name:26s} az {float(a_):5.0f} -> {azp:5.0f} deg   zmax='
              f'{z.max():5.0f}  offset@2000={off[np.argmin(abs(z - 2000))]:5.0f} km'
              f'  max|cross|={np.abs(dxt).max():5.0f} km')
    json.dump(out, open(OUT, 'w'))
    # The override, if any, is already in OUT's name; recording it here as well means a
    # reader does not have to parse a filename to know what the paths were traced under.
    provenance.stamp(OUT, config=c, lateral=float(A.lateral), ncell=int(A.ncell),
                     s_override=A.s, channel_override=A.channel,
                     all_hotspots=bool(A.all_hotspots),
                     fix_azimuths=bool(A.fix_azimuths), n_paths=len(out),
                     inputs=[MODEL, A.sections_all if A.all_hotspots else AZOUT,
                             os.path.join(A.dir, f'classification_{A.tag}.csv')])
    if A.fix_azimuths:
        json.dump(az_path, open(A.sections_all if A.all_hotspots else AZOUT, 'w'),
                  indent=1)
        print(f'wrote {A.sections_all if A.all_hotspots else AZOUT}')
    print(f'wrote {OUT} with {len(out)} paths')


if __name__ == '__main__':
    main()
