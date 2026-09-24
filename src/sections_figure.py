"""Annular-wedge cross-sections in pyGMT, adapted from T27.

The wedge geometry (polar projection P<w>c+a, region [theta_min, theta_max,
r_min, r_max], CMB inner arc, phase-transition arcs) follows the T27 notebook.
Differences: RevealLO rather than REVEAL, a Voigt shear average rather than vsv
alone, and the colour scale is FIXED across every panel and saturates at the
model's own strong-anomaly amplitude rather than at a value chosen to make
structures look continuous — which is the specific practice Foulger et al.
(2013, their Fig. 17) identify as manufacturing apparent whole-mantle plumes.
"""
import argparse, math, os, re, sys, warnings
import matplotlib
import numpy as np, pandas as pd, xarray as xr, pygmt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tomo_io import _var, _coord
warnings.filterwarnings('ignore')

R_E, MINZ, MAXZ = 6371.0, 50.0, 2890.0
KM_DEG = R_E * np.pi / 180.0   # exact spherical km per degree, for distance ticks
ROOT_HI, ROOT_LO = 0.80, 0.20  # same thresholds classify.py uses for the ensemble_class wording
LIM = 1.8                      # per cent, fixed for every panel
import model_names
from figstyle import PLACED_CM   # one number, in one place: a point-size floor
                                 # checked against the wrong width is not a check
PHASE = [410, 660, 1000]
NP = 221
HALF = 30.0

_ap = argparse.ArgumentParser()
_ap.add_argument('--file', required=True, help='the model netCDF to plot')
_ap.add_argument('--tag', required=True,
                 help='names the outputs and picks corridor_summary_<tag>.csv')
_ap.add_argument('--var', default='voigt')
_ap.add_argument('--dir', default='out')
_ap.add_argument('--sections', default='sections.json')
_ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
_ap.add_argument('--depth-max', type=float, default=2880.0, dest='depth_max',
                 help='ignore shells deeper than this, in km; the default sits '
                      'above the core-mantle boundary at 2891 km')
_ap.add_argument('--out', default=None)
_ap.add_argument('--paths', default=None,
                 help='read the traced paths from this file instead of '
                      'conduit_paths_all_<tag>.json, so an alternative '
                      'configuration can be drawn without overwriting the '
                      'production trace')
_ap.add_argument('--all', action='store_true', dest='all_hotspots',
                 help='every hotspot in the table rather than the nine the paper '
                      'cuts sections on, paginated. Reads conduit_paths_all_<tag>'
                      '.json, which paths.py --all writes')
_ap.add_argument('--sites', default=None,
                 help='comma-separated hotspot names, drawn in the order given. '
                      'The paper figure is a chosen set in the order the argument '
                      'is made, which is neither alphabetical nor a width ranking.')
_ap.add_argument('--top', type=int, default=0,
                 help='the N highest-ranked hotspots on one page rather than all '
                      'of them, for the per-model panels in the supplement')
_ap.add_argument('--second-tracks', default=None, dest='second_tracks',
                 help='competing_tracks_<tag>.json, written by competing_routes.py '
                      '--save-paths. Where the second descent is within the '
                      'indifference margin the panel draws BOTH, because the cost '
                      'difference between them is below what the model resolves and a '
                      'single track would assert a choice the data do not make. '
                      'Defaults to out/competing_tracks_<tag>.json if it exists.')
_ap.add_argument('--second-margin', type=float, default=3.0, dest='second_margin',
                 help='draw the second descent when it costs no more than this per '
                      'cent above the first. 3.0 is exp(s*0.10)-1 at s=0.3: a tenth '
                      'of a per cent in mean anomaly, which no model here resolves. '
                      'Fixed in CONTINUITY_PREREGISTRATION.md.')
_ap.add_argument('--second-crosstrack', type=float, default=500.0,
                 dest='second_xt',
                 help='the second descent is drawn dashed while it lies within this '
                      'many km of the section plane and dotted beyond it. The plane '
                      'was fixed to contain the FIRST descent, so the second is in '
                      'general out of it, and a single style would assert that a '
                      'structure 1900 km away is in this section.')
_ap.add_argument('--per-page', type=int, default=12, dest='per_page',
                 help='panels per figure under --all: 4 rows of --ncols, so the '
                      'default with the default --ncols is a 4x3 grid')
_ap.add_argument('--ncols', type=int, default=3)
_A = _ap.parse_args()
SRC, VAR, TAG = _A.file, _A.var, _A.tag
if _A.top > 0:                      # --top is --all, truncated to one page
    _A.all_hotspots = True
    _A.per_page = _A.top
if _A.sites:                        # --sites is --all, restricted and reordered
    _A.all_hotspots = True
    _A.sites = [x.strip() for x in _A.sites.split(',') if x.strip()]
    _A.per_page = len(_A.sites)

def pt_num(p):
    """The drawn-canvas point size pt() quotes as text, as a number.

    Kept separate from pt() so text_width_cm() below can measure a string at
    the exact size fig.text() will set it at, rather than re-deriving the same
    ceiling-and-scale arithmetic a second time somewhere it could drift out of
    step with this one.
    """
    return math.ceil(p * DX * NC / PLACED_CM * 10) / 10   # never round a size down


def pt(p):
    """A point size given at the size the figure is printed, not drawn.

    The panels are drawn on a canvas half again as wide as the column they are
    placed in, which keeps the tomography sharp and shrinks every label by the
    same factor. Sizes are quoted here as they will appear on the page, so
    changing the panel pitch cannot quietly make the type unreadable.
    """
    return f'{pt_num(p):.1f}p'


_AFM_PATH = os.path.join(os.path.dirname(matplotlib.__file__),
                          'mpl-data', 'fonts', 'afm', 'phvr8a.afm')


def _hv_width_table():
    """Helvetica glyph widths (1000-unit em), from the Adobe AFM matplotlib ships.

    figstyle.py already imports matplotlib, so this is not a new dependency.
    The legend box below used to be sized by a fixed width chosen once and
    never revisited - the same failure mode that let the colour bar sit on top
    of the legend text above until a third row grew past where it was placed.
    A hand-guessed box width would go stale the same way the moment a legend
    line changes; measuring the actual glyphs cannot.

    For every character this legend or the colour-bar label ever sets -
    letters, digits, space, hyphen, comma, period, percent, parentheses - the
    AFM file's C field (the character code) already equals the ASCII code,
    because Helvetica's base encoding matches ASCII over that range. No
    separate glyph-name table is needed to look a character up by it.
    """
    table = {}
    try:
        with open(_AFM_PATH) as fh:
            for line in fh:
                if not line.startswith('C '):
                    continue
                m = re.match(r'C\s+(-?\d+)\s*;\s*WX\s+(\d+)', line)
                if m and int(m.group(1)) >= 0:
                    table[int(m.group(1))] = int(m.group(2))
    except OSError:
        pass   # AFM path changed under us - fall back to _HV_FALLBACK for every glyph
    return table


_HV_WIDTHS = _hv_width_table()
_HV_FALLBACK = 600   # wider than any letter this legend or bar label actually sets,
                      # so a lookup miss over-, not under-, estimates the width


def text_width_cm(s, pt_size):
    """The width Helvetica renders s at, in drawn-canvas centimetres.

    pt_size is the DRAWN size - pt_num(9), not 9 - so the result sits in the
    same units as WL, DX and every other length in this file: all of them are
    drawn-canvas centimetres, shrunk together to PLACED_CM only when the
    finished PDF is placed in the manuscript.
    """
    units = sum(_HV_WIDTHS.get(ord(c), _HV_FALLBACK) for c in s)
    return units / 1000 * pt_size * 0.0352778   # 1 pt = 1/72 in = 2.54 cm/72


def colour_key(fig, bar, label_at, size, w=6.5, h=0.30):
    """The colour bar, with its numbers placed by hand.

    GMT sets the type on a colour bar from the square root of the bar's length,
    so on a bar this short no value of FONT_ANNOT_PRIMARY makes the numbers
    readable: at 5.4 cm they come out under half the size asked for. The bar is
    drawn with ticks alone and the numbers are placed here, at the size the rest
    of the figure is set in. label_at turns a position in centimetres relative to
    the centre of the bar into a piece of text, and differs between the bar in
    the legend box and the one that hangs below a panel.
    """
    # The numbers below are placed at v/LIM * w/2 either side of the anchor, so this
    # function only works if the anchor IS the centre of the bar. GMT does not agree
    # by default: for -Dg, -Dj, -Dx and -Dn the bar is anchored by its BOTTOM LEFT
    # corner, and only -DJ centres it by taking the mirror of the reference code. The
    # legend-box bar is placed with g<x>/<y> and its numbers therefore sat half a bar
    # width - 3.25 cm - to the left of the colours they name. Rather than leave each
    # caller to know that, the centring is imposed here.
    # -DJ is no exception. There the default justification is the MIRROR of the
    # reference code, so JBC anchors the bar by its TOP centre; the numbers, computed
    # for a centre anchor, then sat 0.02 cm inside the bar rather than 0.13 cm below
    # it. Centring both callers is the only version where the arithmetic here is the
    # arithmetic GMT uses. This moves the panel bar up by h/2, 1.5 mm.
    anchor = bar if '+j' in bar else bar + '+jMC'
    fig.colorbar(cmap=True, position=f'{anchor}+w{w}c/{h}c+h', frame=['xf0.5'])
    for v in (-1, 0, 1):
        label_at(v / LIM * w / 2, -(h / 2 + 0.13), f'{v:g}', size)
    label_at(0.0, -(h / 2 + 0.62), 'shear-velocity anomaly (%)', size)


def great_circle(lat0, lon0, az, half, n=NP):
    d = np.radians(np.linspace(-half, half, n))
    la0, lo0, a = np.radians(lat0), np.radians(lon0), np.radians(az)
    la = np.arcsin(np.sin(la0)*np.cos(d) + np.cos(la0)*np.sin(d)*np.cos(a))
    lo = lo0 + np.arctan2(np.sin(a)*np.sin(d)*np.cos(la0),
                          np.cos(d) - np.sin(la0)*np.sin(la))
    return np.degrees(d), np.degrees(la), (np.degrees(lo)+180) % 360 - 180

print('loading', TAG)
# Variable and coordinate names, depth units and the core shells are handled by
# the same helpers the analysis uses, rather than being assumed here. This file
# read its own netCDF directly and so missed every normalisation tomo_io does.
ds = xr.open_dataset(SRC)
_LAT = _coord(ds, ('latitude', 'lat'))
_LON = _coord(ds, ('longitude', 'lon'))
_DEP = _coord(ds, ('depth', 'depth_km', 'radius_nondim'))

_z = np.asarray(ds[_DEP].values, float)
if np.nanmax(np.abs(_z)) > 2e4:            # metres, not kilometres
    _z = _z / 1000.0
_keep = np.where(_z <= _A.depth_max)[0]
if len(_keep) != len(_z):
    ds = ds.isel({_DEP: _keep})
    _z = _z[_keep]

vs = (_var(ds, VAR, SRC) if VAR != 'voigt'
      else np.sqrt((2*_var(ds, 'vsv', SRC)**2
                    + _var(ds, 'vsh', SRC)**2)/3.0)).rename('vs')
mean = vs.mean(dim=[_LAT, _LON])
anom = ((vs - mean)/mean*100.0)
anom = anom.rename({_LAT: 'latitude', _LON: 'longitude'}) if (
    _LAT != 'latitude' or _LON != 'longitude') else anom
if _z[0] > _z[-1]:
    anom = anom.isel({_DEP: slice(None, None, -1)})
    _z = _z[::-1]
# Some files are stored (latitude, longitude, depth) rather than depth-first.
# Fix the order here so that nothing downstream has to know or care.
anom = anom.transpose(_DEP, 'latitude', 'longitude')
depth = _z

# Close the sphere explicitly. A great circle that crosses the antimeridian
# otherwise asks for longitudes outside the file's range, which interpolate to
# NaN and leave a white stripe through the panel. One periodic column is added
# at each end, and a repeated wrap meridian is dropped first so the added column
# is not a duplicate.
_lonv = anom['longitude'].values.astype(float)
# Drop a repeated wrap meridian BEFORE any shift. A file sampling 0 to 360
# inclusive, or -180 to 180 inclusive, carries the same meridian twice; shifting
# first leaves two entries at the same longitude and interpolation then fails on
# a non-unique index.
if abs((_lonv[-1] - _lonv[0]) - 360.0) < 1e-6:
    anom = anom.isel(longitude=slice(0, -1))
    _lonv = _lonv[:-1]
if _lonv.max() > 180.0:                    # 0-360 file, e.g. RevealLO
    anom = anom.assign_coords(
        longitude=np.where(_lonv > 180.0, _lonv - 360.0, _lonv)).sortby('longitude')
    _lonv = anom['longitude'].values.astype(float)
if len(np.unique(_lonv)) != len(_lonv):
    _u, _i = np.unique(_lonv, return_index=True)
    anom = anom.isel(longitude=np.sort(_i))
    _lonv = anom['longitude'].values.astype(float)
_left = anom.isel(longitude=[-1]).assign_coords(longitude=[_lonv[-1] - 360.0])
_right = anom.isel(longitude=[0]).assign_coords(longitude=[_lonv[0] + 360.0])
anom = xr.concat([_left, anom, _right], dim='longitude')
LON0 = float(anom['longitude'].values.min())
LON1 = float(anom['longitude'].values.max())


def _check_text(pdf):
    """Check the rendered page for unreadable or colliding text, and say so.

    This figure is drawn with PyGMT, so figstyle.check - which walks a live matplotlib
    figure - cannot see it, and the atlas was the one part of the figure set with no
    text check at all: 49 panels over six pages, every one of them labelled. figcheck.py
    reads the PDF instead, which is the right instrument here and the only one that works
    on a figure this script did not build in memory.

    It needs pdfplumber, which wants a Pillow newer than several unrelated packages
    accept and so lives in .venv-figcheck rather than in front of whatever else is
    installed. A missing environment is reported rather than passed over: a check that
    silently does nothing is worse than no check, because the run still prints success.
    """
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    venv = os.path.join(here, '.venv-figcheck', 'bin', 'python3')
    exe = venv if os.path.exists(venv) else sys.executable
    r = subprocess.run([exe, os.path.join(here, 'figcheck.py'), pdf],
                       capture_output=True, text=True, cwd=here)
    out = (r.stdout or '') + (r.stderr or '')
    if 'needs pdfplumber' in out:
        print('  TEXT NOT CHECKED: figcheck needs pdfplumber. Create the environment '
              'once with\n    python3 -m venv .venv-figcheck && '
              '.venv-figcheck/bin/pip install -q pdfplumber', flush=True)
        return
    for line in out.splitlines():
        if line.strip():
            print('  ' + line, flush=True)
    if r.returncode != 0:
        raise SystemExit(f'{os.path.basename(pdf)} failed its text check')


SECOND_XT = _A.second_xt


def wedge(fig, name, lat, lon, az, label, w=7.2, cbar=False, label_depths=False,
          path=None, axis=None, second=None):
    d, la, lo = great_circle(lat, lon, az, HALF)
    # A path point landing just outside the file's longitude range - -180.0
    # when the grid starts at -179.5 - interpolates to NaN and leaves a white
    # stripe through the panel. Wrap such points to the other end instead.
    lo_s = np.where(lo < LON0, lo + 360.0, lo)
    lo_s = np.where(lo_s > LON1, lo_s - 360.0, lo_s)
    s = anom.interp(latitude=('path', la), longitude=('path', lo_s),
                    kwargs={'fill_value': np.nan}
                    ).transpose(_DEP, 'path').values
    k = (depth >= MINZ) & (depth <= MAXZ)
    dm, data = depth[k], s[k]
    r = R_E - dm
    if r[0] > r[-1]:
        r, data = r[::-1], data[::-1]
    ru = np.linspace(r.min(), r.max(), len(r))
    du = np.empty_like(data)
    for j in range(data.shape[1]):
        du[:, j] = np.interp(ru, r, data[:, j])
    # Coordinates stay at full float64 precision, even though the data payload
    # is float32 - casting the y (radius) axis to float32 quantises adjacent
    # values enough that GMT reads the spacing as irregular (it isn't; this is
    # exactly the linspace construction above, just rounded), which produced a
    # spurious 'NX must be an integer' warning on every panel.
    da = xr.DataArray(du.astype(np.float32),
                      coords={'y': ru,
                              'x': np.linspace(-HALF, HALF, data.shape[1])},
                      dims=('y','x'))
    reg = [-HALF, HALF, float(ru[0]), float(ru[-1])]
    proj = f'P{w}c+a'
    pygmt.makecpt(cmap='roma', series=[-LIM, LIM, 0.02], continuous=True)
    fig.basemap(region=reg, projection=proj, frame='+n')
    fig.grdimage(grid=da, cmap=True, region=reg, projection=proj)
    t = np.linspace(-HALF, HALF, 240)
    fig.plot(x=t, y=np.full_like(t, ru[-1]), pen='1.0p,black')
    fig.plot(x=t, y=np.full_like(t, ru[0]), pen='1.3p,black')
    for e in (-HALF, HALF):
        fig.plot(x=[e, e], y=[ru[0], ru[-1]], pen='0.7p,black')
    fig.plot(x=[0,0], y=[ru[0], ru[-1]], pen='0.5p,black,dashed')
    # Distance ticks, in degrees of great-circle arc - the native unit of the
    # polar projection this wedge is drawn on, so GMT places them correctly
    # along the curved bottom edge instead of by hand (a first attempt at
    # hand-placed km ticks fought the projection and produced overlapping
    # text). The km equivalent is given once, in the figure-level caption.
    # Annotated on the south (CMB) side, not the north: the north edge is
    # already carrying the hotspot's name, triangle and root-fraction label at
    # x=0, and the two collided there.
    fig.basemap(region=reg, projection=proj,
               frame=['xa10f5', 'wenS'] if HALF >= 15 else ['xa5f1', 'wenS'])
    # which end is which: a hollow circle at -HALF, a filled one at +HALF,
    # repeated on the locator globe below so the two are never a guess.
    fig.plot(x=[-HALF], y=[ru[-1]], style='c0.14c', fill='white', pen='0.8p,black', no_clip=True)
    fig.plot(x=[HALF], y=[ru[-1]], style='c0.14c', fill='black', pen='0.8p,black', no_clip=True)
    # 410 and 660 lie 250 km apart, which on this radial scale is a third of the
    # height of the type, so the depths cannot be stacked down the edge of the
    # wedge. Each sits on its own arc instead, at its own angle along it, over a
    # white box; they are drawn on the first panel and the arcs are the same on
    # all of them.
    # the shallowest goes right, clear of the locator globe in the top-left
    # corner, and each next one further left, staying off the centre line where
    # the conduit is drawn. The depths carry no unit: three of "410 km" will not
    # fit across a wedge this size, and the caption gives them in kilometres.
    _AT = (21.0, 7.0, -7.0)
    for _j, pz in enumerate(PHASE):
        rd = R_E - pz
        if ru[0] < rd < ru[-1]:
            fig.plot(x=t, y=np.full_like(t, rd), pen='0.5p,gray30,dashed')
            if label_depths:
                fig.text(x=_AT[_j % len(_AT)], y=rd, text=f'{pz}',
                         justify='BC', offset='0c/0.05c', fill='white@25',
                         clearance='0.05c/0.02c',
                         font=f'{pt(8.5)},Helvetica,gray20', no_clip=True)
    fig.text(x=HALF, y=ru[0], text='CMB', justify='ML', offset='0.15c/0c',
             font=f'{pt(8.5)},Helvetica-Bold,gray20', no_clip=True)
    if path is not None:
        pa = np.asarray(path['along_deg'], float)
        pdep = np.asarray(path['depth'], float)
        pz = R_E - pdep
        k2 = (pa >= -HALF) & (pa <= HALF) & (pz >= ru[0]) & (pz <= ru[-1])
        # The search seeds anywhere within its cap around the hotspot and can
        # take several lateral steps at the seed depth before it ever commits
        # to descending - real behaviour of a greedy trace over a cost field,
        # not noise, but drawing it the same as the descent below makes the
        # entry point look like part of the conduit. It is shown thin and
        # dotted instead, and a matching dotted line bridges the gap from the
        # hotspot's own marker down to wherever the trace actually starts -
        # both the seed depth (typically 200 km, not the surface) and the cap
        # radius (up to a few hundred km) make that gap real, not an artefact,
        # and hiding it is what produced the unexplained jog.
        if k2.sum() > 1:
            i0 = int(np.argmax(k2))
            fig.plot(x=[0.0, pa[i0]], y=[ru[-1], pz[i0]], pen='0.6p,gray50,dotted')
            hunt_end = i0
            while hunt_end + 1 < len(pdep) and pdep[hunt_end + 1] == pdep[i0]:
                hunt_end += 1
            hk = np.arange(i0, hunt_end + 1)
            hk = hk[k2[hk]]
            if len(hk) > 1:
                fig.plot(x=pa[hk], y=pz[hk], pen='0.6p,gray50,dotted')
            dk = np.arange(max(hunt_end, i0), len(pdep))
            dk = dk[k2[dk]]
            if len(dk) > 1:
                # One style for every descent. The three-way split this replaces
                # encoded how often a hotspot beat a random-start null, which is
                # a withdrawn result; drawing it would assert it.
                fig.plot(x=pa[dk], y=pz[dk], pen='2.4p,black')
                fig.plot(x=pa[dk], y=pz[dk], pen='1.1p,white')
    if second is not None:
        # The second descent, drawn only where the two are indistinguishable. It is
        # projected onto this panel's own great circle rather than carrying a stored
        # along_deg, because the section azimuth was fixed for the first route and the
        # second has to be shown on the same plane to be comparable. Dashed, same
        # weight: it is the same kind of object, not a lesser one.
        sz = np.asarray(second['depth'], float)
        sa, sx = along_cross(lat, lon, az, np.asarray(second['lat'], float),
                             np.asarray(second['lon'], float))
        sr = R_E - sz
        o = np.argsort(sz)
        sa, sr, sx = sa[o], sr[o], np.abs(sx[o])
        ks = (sa >= -HALF) & (sa <= HALF) & (sr >= ru[0]) & (sr <= ru[-1])
        # The section plane was fixed to contain the FIRST descent, so the second
        # is in general out of it. Drawing all of it the same way would put a
        # structure that is 1900 km off this plane on the same footing as one that
        # is in it. The part within SECOND_XT of the plane is drawn dashed; beyond
        # that it is dotted and faint, so the reader can see where the second
        # descent is really in this section and where it is only projected into it.
        # The second descent seeds within the same cap as the first, so it too
        # starts some distance from the hotspot marker; bridge that gap the same
        # way, or its start looks like an unexplained jog.
        if ks.any():
            i0 = int(np.argmax(ks))
            fig.plot(x=[0.0, sa[i0]], y=[ru[-1], sr[i0]], pen='0.6p,gray50,dotted')
        near = ks & (sx <= SECOND_XT)
        far = ks & (sx > SECOND_XT)
        # Dashed in both, because dashed is what 'second descent' means in the key;
        # the part out of the plane is faint rather than dotted, since dotted
        # already means 'within the search cap' and the two would be read as one.
        for msk, cas, pen in ((far, '2.2p,white@60', '0.9p,black@60,4_2'),
                              (near, '2.6p,white@20', '1.1p,black,4_2')):
            # contiguous runs only: a single polyline through a gap would draw a
            # segment the route never takes
            idx = np.where(msk)[0]
            if len(idx) < 2:
                continue
            for grp in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
                if len(grp) < 2:
                    continue
                fig.plot(x=sa[grp], y=sr[grp], pen=cas)
                fig.plot(x=sa[grp], y=sr[grp], pen=pen)
    if axis is not None:
        az_, at_ = axis
        zz = np.asarray(az_['depth'], float)
        aa, _ = along_cross(lat, lon, az, np.asarray(az_['lat'], float),
                            np.asarray(az_['lon'], float))
        rr = R_E - zz
        ka = (aa >= -HALF) & (aa <= HALF) & (rr >= ru[0]) & (rr <= ru[-1])
        if ka.sum() > 2:
            fig.plot(x=aa[ka], y=rr[ka], pen='3.0p,white@40')
            fig.plot(x=aa[ka], y=rr[ka], pen='1.4p,#B4442E')
            # the axis is drawn only as deep as slow material continues, so the
            # depth at which it stops is itself the measurement
            fig.plot(x=[aa[ka][-1]], y=[rr[ka][-1]], style='c0.11c',
                     fill='#B4442E', pen='0.4p,white')
    fig.plot(x=[0], y=[ru[-1]], style='t0.34c', fill='white', pen='0.9p,black', no_clip=True)
    fig.text(x=0, y=ru[-1], text=model_names.hotspot(name), justify='BC',
             offset='0c/1.02c',
             font=f'{pt(10)},Helvetica-Bold,black', no_clip=True)
    lab = label
    if axis is not None and np.isfinite(axis[1]):
        lab = f'{label}   @~t@~ {axis[1]:.2f}' if label else f'@~t@~ {axis[1]:.2f}'
    # clear of the apex marker, which the larger type used to sit on top of
    fig.text(x=0, y=ru[-1], text=lab, justify='BC', offset='0c/0.46c',
             font=f'{pt(8.5)},Helvetica,gray30', no_clip=True)
    # locator globe: the section line and its midpoint, orthographic
    try:
        with fig.inset(position='jTL+w1.75c+o-0.15c/-0.05c', box=None):
            fig.coast(region='g', projection=f'G{lon:.1f}/{lat:.1f}/1.75c',
                      land='gray82', water='white', shorelines='0.2p,gray55',
                      frame='g')
            fig.plot(x=lo, y=la, pen='1.2p,black')
            fig.plot(x=[lo[0]], y=[la[0]], style='c0.10c', fill='white', pen='0.5p,black')
            fig.plot(x=[lo[-1]], y=[la[-1]], style='c0.10c', fill='black', pen='0.5p,black')
            fig.plot(x=[lon], y=[lat], style='a0.26c', fill='#B4442E',
                     pen='0.4p,black')
    except Exception as e:
        print('  inset skipped for', name, e)
    if cbar:
        colour_key(fig, 'JBC+o0c/0.8c',
                   lambda dx, dy, t, sz: fig.text(
                       text=t, position='BC', offset=f'{dx}c/{dy - 0.8}c',
                       justify='TC', font=f'{sz},Helvetica,black', no_clip=True),
                   pt(8.5))

# Section azimuths are not chosen. Each is the azimuth of the great circle
# through the hotspot that best contains the least-cost path recovered by the
# search, so the plane of each section contains the conduit the classifier
# actually used and the white line drawn on each panel is that conduit. An
# arbitrary azimuth would cut a leaning conduit obliquely and make it look
# discontinuous, which is the artefact these sections are most often accused of.
import json as _json
# The nine fixed section planes are only needed for the nine-panel figure, which
# the paper no longer carries: the all-hotspot atlas took its place when the
# study moved to a journal with room for all 49. Under --all/--top the planes
# come from sections_all.json instead, so sections.json need not exist.
_AZ = ({} if _A.all_hotspots or not os.path.exists(_A.sections)
       else _json.load(open(_A.sections)))


def along_cross(lat0, lon0, az, la, lo):
    """Along-track angle (deg, signed) and cross-track distance (km) of points.

    The same construction paths.py uses, repeated here rather than imported so
    that drawing a figure does not pull in the search itself.
    """
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
# The anomaly axis, if conduit_axis has been run for this model. It answers a
# different question from the route: the route is where a descending path is
# cheapest, the axis is where the slow material actually is. Showing only the
# route invites the reader to read its shape as the shape of the plume, which it
# is not - a broad anomaly is chorded by the cheapest route through it.
_axf = os.path.join(_A.dir, f'axis_paths_{TAG}.json')
_AXIS = _json.load(open(_axf)) if os.path.exists(_axf) else {}
_axg = os.path.join(_A.dir, f'conduit_axis_{TAG}.csv')
_AXG = pd.read_csv(_axg).set_index('hotspot') if os.path.exists(_axg) else None
_pf = (_A.paths if _A.paths else
       os.path.join(_A.dir, ('conduit_paths_all_' if _A.all_hotspots
                             else 'conduit_paths_') + f'{TAG}.json'))
if not os.path.exists(_pf):
    raise SystemExit(f'{_pf} not found'
                     + (' - run paths.py --all first' if _A.all_hotspots else ''))
_PATHS = _json.load(open(_pf))


def _path_for(key):
    for k, v in _PATHS.items():
        if str(k).lower().startswith(key.lower()[:6]):
            return v
    return None


_ST = None
_stp = _A.second_tracks or os.path.join(_A.dir, f'competing_tracks_{_A.tag}.json')
if os.path.exists(_stp):
    _ST = _json.load(open(_stp))
    _nin = sum(1 for v in _ST.values()
               if 'second' in v and float(v.get('margin_pct', 1e9)) <= _A.second_margin)
    print(f'  second tracks: {_nin} of {len(_ST)} sites are within '
          f'{_A.second_margin:g} per cent and will be drawn twice')
else:
    print(f'  NOTE: {_stp} absent, so every panel asserts a single descent. '
          f'Run competing_routes.py --save-paths to know whether that is true.')


def _second_for(key):
    """The second descent, when the two are indistinguishable. Otherwise None."""
    if _ST is None:
        return None
    for k, v in _ST.items():
        if str(k).lower().startswith(key.lower()[:6]):
            if 'second' not in v:
                return None
            m = float(v.get('margin_pct', float('nan')))
            return v['second'] if (m == m and m <= _A.second_margin) else None
    return None


def _axis_for(key):
    for k, v in _AXIS.items():
        if str(k).lower().startswith(key.lower()[:6]):
            tort = np.nan
            if _AXG is not None and k in _AXG.index:
                tort = float(_AXG.loc[k, 'tortuosity'])
            return v, tort
    return None, np.nan
_HS = pd.read_csv(_A.hotspots)


def _pos(key):
    m = _HS[_HS.hotspot.astype(str).str.lower().str.startswith(key.lower()[:6])]
    if not len(m):
        raise SystemExit(f'{key} not in {_A.hotspots}')
    return float(m.lat.iloc[0]), float(m.lon_180.iloc[0])


if _A.all_hotspots:
    # Every hotspot the search returned a path for, ranked as the study ranks
    # them: by how many independent models call it rooted, then by the mean root
    # fraction across them. Ranking on this model alone would put the panels in
    # an order the paper never uses. The single-model root fraction is the
    # fallback for a run where the comparison table has not been built yet.
    # Panel order. The atlas is a reference, so it is alphabetical: a reader
    # looking for one hotspot should be able to find it. The previous ordering
    # ranked by how many models called a hotspot deep-rooted, which sorts the
    # atlas by a withdrawn result and tells the reader a story the paper does not
    # make. Where a subset is asked for, it is taken by corridor width, narrowest
    # first, which is calibrated and means best constrained.
    _names = sorted(_PATHS)

    # Grouped by track support, so the atlas separates the cases the method follows
    # from the cases it does not, instead of interleaving them alphabetically and
    # leaving a reader to discover the difference panel by panel. Within a group the
    # order is alphabetical, so the atlas is still usable as a reference.
    #
    # The groups are DESCRIPTIVE. They come from the fraction of 300 ambient paths
    # that found slower material than the hotspot in each depth band, with a cut at
    # 0.10 that is a presentational choice and not a test; the tested result is the
    # rank comparison of the whole distributions in track_support_population_<tag>.csv.
    _ts = os.path.join(_A.dir, f'track_support_{_A.tag}.csv')
    _GROUPS = {}
    if os.path.exists(_ts):
        _t = pd.read_csv(_ts, index_col=0)
        if 'group' in _t.columns:
            _GROUPS = _t['group'].to_dict()
            _names = sorted(_names, key=lambda n: (_GROUPS.get(n, '9 unclassified'), n))
            print('atlas grouped by track support:')
            for _g in sorted(set(_GROUPS.get(n, '9 unclassified') for n in _names)):
                _m = [n for n in _names if _GROUPS.get(n, '9 unclassified') == _g]
                print(f'  {_g:22s} {len(_m):2d}  {", ".join(_m[:4])}'
                      + (' ...' if len(_m) > 4 else ''))
    else:
        print(f'no {_ts}; atlas stays alphabetical')

    if _A.sites:
        # Fail on a name that is not there rather than quietly drawing fewer
        # panels than were asked for: a figure short one panel is not obviously
        # wrong when you look at it.
        _bad = [n for n in _A.sites if n not in _PATHS]
        if _bad:
            raise SystemExit('no traced path for: ' + ', '.join(_bad)
                             + '\navailable: ' + ', '.join(sorted(_PATHS)))
        _names = list(_A.sites)
    elif _A.top > 0:
        _cs = os.path.join(_A.dir, f'corridor_summary_{TAG}.csv')
        if os.path.exists(_cs):
            _c = pd.read_csv(_cs)
            _c = _c[_c.ok == True].set_index('site')['width_med_0.02'].to_dict()
            _names = sorted(_names, key=lambda n: (_c.get(n, float('inf')), n))
        else:
            print('no corridor summary; --top falls back to alphabetical order')
        _names = _names[:_A.top]
    PANELS = [(n, float(_HS[_HS.hotspot.astype(str) == n].lat.iloc[0]),
               float(_HS[_HS.hotspot.astype(str) == n].lon_180.iloc[0]),
               float(_PATHS[n]['azimuth'])) for n in _names]
else:
    PANELS = [(k, *_pos(k), _AZ[k]) for k in
              ('Macdonald', 'Samoa', 'Tahiti', 'Iceland', 'Reunion', 'Baja',
               'Hawaii', 'Kerguelen', 'Yellowstone')]

# Panel sub-labels come from the corridor, which is calibrated, rather than from
# the classification, which is withdrawn. Each panel carries the corridor width in
# the three depth bands and whether the descent ends inside a province, so the
# reader can see how well constrained the path drawn beside it actually is.
LABELS = {}
_pf = os.path.join(_A.dir, f'corridor_profiles_{TAG}.csv')
_rt = os.path.join(_A.dir, f'plume_roots_{TAG}.csv')
if os.path.exists(_pf):
    _p = pd.read_csv(_pf)
    _bands = ((200, 660), (660, 1500), (1500, 2700))
    _r = pd.read_csv(_rt).set_index('site') if os.path.exists(_rt) else None
    for _k, _, _, _ in PANELS:
        _g = _p[_p.site == _k]
        if not len(_g):
            continue
        _w = []
        for _lo, _hi in _bands:
            _m = _g[(_g.depth >= _lo) & (_g.depth < _hi)]
            _w.append(f'{_m.width_km.median():.0f}' if len(_m) else '-')
        _lab = '/'.join(_w) + ' km'
        if _r is not None and _k in _r.index:
            _lab += ', province' if bool(_r.loc[_k, 'root_in']) else ', outside'
        # The support group leads the sub-label, so a reader knows which group a panel
        # belongs to without counting back to the start of the atlas. The word
        # 'corridor' and the phrase 'in a' come out to pay for it: the caption defines
        # both, and a longer sub-label is a collision waiting to happen at this pitch.
        _tag = {'1 deep signature': 'deep', '2 upper or mid only': 'upper/mid',
                '3 weak throughout': 'weak', '4 disjointed': 'disjointed'}
        _g = globals().get('_GROUPS', {}).get(_k)
        if _g in _tag:
            _lab = f'{_tag[_g]} | ' + _lab
        LABELS[_k] = _lab

NC = max(1, int(_A.ncols))
PER = max(1, int(_A.per_page)) if _A.all_hotspots else len(PANELS)
DX, DY = 8.1, 6.4                       # panel pitch, centimetres
# --sites and --top already trim PANELS to exactly one page (PER == len(PANELS)),
# so only true --all pagination (the cross-section atlas) reaches the merge below.
_ATLAS = _A.all_hotspots and not _A.sites and not _A.top
if _ATLAS and len(PANELS) > PER:
    _nfull, _rem = divmod(len(PANELS), PER)
    # A trailing page holding only the remainder is a nearly blank figure. Merged
    # into the last full page instead, its one extra row carries the legend beside
    # the lone panel rather than under a page that is otherwise empty.
    _sizes = [PER] * _nfull if _rem == 0 else [PER] * (_nfull - 1) + [PER + _rem]
    PAGES, _i = [], 0
    for _sz in _sizes:
        PAGES.append(PANELS[_i:_i + _sz]); _i += _sz
else:
    PAGES = [PANELS[i:i + PER] for i in range(0, len(PANELS), PER)]
out = _A.out or os.path.join(
    # Everything else in this study writes to Paper_plumes/figures. Writing to
    # plume_pipeline/figures instead left two directories holding same-named
    # files, and the stale copy was picked up once.
    os.path.join('..', 'figures'), f'fig_wedges_{TAG}'
    + ('_sel' if _A.sites else
       (f'_top{_A.top}' if _A.top else ('_all' if _A.all_hotspots else ''))))
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

for pno, page in enumerate(PAGES, 1):
    fig = pygmt.Figure()
    pygmt.config(FONT_ANNOT_PRIMARY=pt(8.5), FONT_LABEL=pt(9), MAP_FRAME_PEN='0.8p')
    nrow = 0
    for i, (n, la, lo, az) in enumerate(page):
        if i:
            col = i % NC
            fig.shift_origin(xshift=f'{DX}c' if col else f'{-DX * (NC - 1)}c',
                             yshift='0c' if col else f'{-DY}c')
            if not col:
                nrow += 1
        # the colour bar hangs off the last panel of the page
        wedge(fig, n, la, lo, az, LABELS.get(n, ''),
              cbar=(not _A.all_hotspots and i == 7), label_depths=(i == 0),
              path=_path_for(n),
              second=_second_for(n),
              axis=(lambda t: (t[0], t[1]) if t[0] else None)(_axis_for(n)))

    # legend, below the last row of this page
    back = (len(page) - 1) % NC
    # A key, not an explanation. Every sentence that used to sit here at 7 point
    # is in the caption file written alongside the figure, which is where a
    # reader looks for it and where it can be set at reading size.
    _legend = [
        ([('2.4p,black',), ('1.1p,white',)], 'least-cost descent'),
        ([('0.6p,gray50,dotted',)], 'within the search cap'),
    ]
    # The axis is drawn only when conduit_axis has been run for this model, so
    # the key claims it only then. Every atlas page and Figure 5 carried this
    # entry with no axis on any panel (no axis_paths file exists for any model).
    if _AXIS:
        _legend.append(([('3.0p,white@40',), ('1.4p,#B4442E',)], 'axis of the slow structure'))
    # Only claim the second descent in the key when a panel on this page carries
    # one. A key entry for something absent from the figure is a worse defect than
    # no key at all.
    if _ST is not None and any(_second_for(n) is not None for n, _, _, _ in page):
        _legend.append(([('2.6p,white@20',), ('1.1p,black,4_2',)],
                        f'second descent, within {_A.second_margin:g} per cent of the first'))
    # THE BOX GROWS DOWNWARD, NEVER UP. GMT anchors a frame at its lower-left corner,
    # which is the current origin, so making the box taller raises its top - and the
    # top is what has to clear the rotated -30 and 30 degree tick labels of the last
    # row. Sizing the box to the key on 21 September moved it up by one row and put
    # 'least-cost descent' under a tick label; figcheck caught it. The original box
    # holds four rows, so it is kept for up to four, and any extra height is taken
    # by moving the origin down by the same amount so the top stays where it was.
    _KEY_DY = 0.62
    _rows = max(4, len(_legend))
    # The box used to reserve two or three lines for a caption drawn on the
    # figure. The caption is in the manuscript now, so the box is sized to what
    # it actually holds. Where the colour bar sits inside it, the box has to
    # reach as far down as the bar's own ink does - len(_legend) rows, a 0.15 cm
    # gap, then the bar itself and its unit label 0.77 cm below that (see
    # colour_key) - not a fixed pad that was set when the key held two rows and
    # never revisited when a third and fourth entry were added.
    # The unit label is anchored at its own TOP (justify='TC'), so its ink hangs
    # roughly one more font-height below that anchor point, not merely sitting at
    # it. The box was sized only to the anchor's y before - correct for the
    # numeric tick labels immediately above it, which are short and sans
    # descenders, but not for a label with descenders in 'velocity' and
    # 'anomaly'. Dietmar caught the box cutting through it on the first real
    # render; measured directly against that page, the label's ink ran 13.0 pt
    # top to bottom at its own 13.0 pt (pt_num(8.5)) size, so one full
    # font-height is exactly the missing term, not a guess.
    _UNIT_H = pt_num(8.5) * 0.0352778
    if _A.all_hotspots:
        _bot = -(len(_legend) * _KEY_DY + 0.15 + 0.77 + _UNIT_H + 0.10)
    else:
        _bot = -(_rows * _KEY_DY + 0.25)
    # GMT anchors this frame at its own lower-left corner (the plot origin), so a
    # taller box does not grow downward in absolute terms - it grows upward, into
    # whatever sits above it. The -4.3 below was calibrated against a 4-row,
    # no-colour-bar box (4*_KEY_DY + 0.25 tall); _grow is how much taller the
    # ACTUAL box is than that baseline, and the origin is pushed down by exactly
    # that much so the top stays where -4.3 put it, regardless of what drove the
    # extra height. The colour-bar branch never fed its own height into this
    # compensation before - _grow was still computed from len(_legend) alone,
    # blind to the 0.15+0.77+_UNIT_H+0.10 the colour-bar branch adds on top -
    # which is why a taller unit label pushed the box up into the last row's
    # rotated tick labels rather than only growing down: the defect Dietmar found
    # on the very next render after the label-clearance fix, and the same
    # mechanism the original 'least-cost descent under a tick label' defect
    # (see above) was.
    _BASELINE_BOT = -(4 * _KEY_DY + 0.25)
    _grow = _BASELINE_BOT - _bot
    # A lone panel left in an otherwise-empty last row (the merged remainder page,
    # e.g. 13 panels on a grid of 12) gets the legend beside it, in that row's
    # spare columns, instead of under a grid that is mostly blank there. That only
    # works when the box is no wider than NC-1 columns, so in atlas mode it is
    # narrowed to exactly that width and reused unchanged - centred below a full
    # grid or beside a lone panel - rather than sized separately for each.
    _nfull_rows, _extra = divmod(len(page), NC)
    _beside = _ATLAS and _nfull_rows > 0 and _extra == 1
    # The box used to be as wide as the full panel grid regardless of what it
    # held - correct as an upper bound (it must never run into the neighbouring
    # panel column, or in the beside case past the spare columns it is placed
    # in) but far wider than a two- or three-line key needs, which is what this
    # narrows. WL is now the actual content's width - the longest legend line,
    # or the colour bar and its unit label when one is drawn here - plus a flat
    # clearance, capped at that same structural upper bound so it only ever
    # shrinks the box, never grows it past where the beside case expects the
    # spare columns to start.
    _WL_MARGIN = 0.3   # clearance past the widest line or the bar, on the same
                       # scale as the 0.10-0.15 cm margins the box's height uses below
    _LEFT_PAD = 0.15   # the line-sample swatches start flush at the box's own left
                       # border with no inset of their own (unlike the text, already
                       # offset 1.05 cm right of them) - measured at zero clearance
                       # on the same rendered page. This shifts swatch and text alike,
                       # on the same 0.10-0.15 cm scale as the box's other margins,
                       # and is folded into the width below so the box still clears
                       # the longest line by _WL_MARGIN past this new start.
    _leg_w = max((_LEFT_PAD + 1.05 + text_width_cm(_txt, pt_num(9))
                 for _, _txt in _legend), default=0.0)
    _bar_w = (max(6.5, text_width_cm('shear-velocity anomaly (%)', pt_num(8.5)))
             if _A.all_hotspots else 0.0)
    _WL_MAX = (NC - 1) * DX if _ATLAS else DX * NC
    WL = min(_WL_MAX, max(_leg_w, _bar_w) + _WL_MARGIN)
    if _beside:
        # The extra row's one panel is in column 0, and no column shift has
        # happened since it was drawn, so the current origin is already that
        # row's left edge. The box goes one panel-width to its right, in the
        # row's own spare columns, not below it. Vertically, a panel's own ink
        # runs about 2.1 cm above its origin to 4.3 cm below (the same two
        # numbers the centred placement clears by; their difference is DY,
        # since rows are packed with no gap between them), so the box is
        # centred on that same span rather than on the panel's bare origin.
        _span_mid = (2.1 - 4.3) / 2
        fig.shift_origin(xshift=f'{DX}c',
                         yshift=f'{_span_mid - (0.28 + _bot) / 2}c')
    else:
        fig.shift_origin(xshift=f'{-DX * back + (DX * NC - WL) / 2}c',
                         yshift=f'{-4.3 - _grow}c')
    fig.basemap(region=[0, WL, _bot, 0.28], projection=f'X{WL}c/{0.28 - _bot}c',
                frame=0)
    _ly = 0.0
    for _pens, _txt in _legend:
        for (_pen,) in _pens:
            fig.plot(x=[_LEFT_PAD, _LEFT_PAD + 0.9], y=[_ly, _ly], pen=_pen,
                     no_clip=True)
        fig.text(x=_LEFT_PAD + 1.05, y=_ly, text=_txt, justify='ML',
                 font=f'{pt(9)},Helvetica,black', no_clip=True)
        _ly -= _KEY_DY
    if _A.all_hotspots:                 # colour bar below the key rows, not among them
        # It used to sit at a fixed y chosen when the key held two rows, and never
        # moved when a third (the second-descent entry) was added, so it landed
        # squarely on that row's own text - the exact defect a text-vs-drawn-object
        # check (figcheck._object_boxes) now catches on its own rather than by eye.
        # One clear row-pitch below the last row used - not _rows, which pads to a
        # minimum of four and would leave a gap even for a two-row key - keeps the
        # bar off the text with the same margin the rows keep off each other.
        _kx = WL / 2
        _ky = -len(_legend) * _KEY_DY - 0.15
        colour_key(fig, f'g{_kx}/{_ky}',
                   lambda dx, dy, t, sz: fig.text(
                       x=_kx + dx, y=_ky + dy, text=t, justify='TC',
                       font=f'{sz},Helvetica,black', no_clip=True),
                   pt(8.5))
    _cap = (
        'Annular cross-sections through the shear-velocity anomaly beneath each '
        'hotspot, along the great circle whose azimuth the axis scan itself '
        'selected. The x-axis is degrees of great-circle arc from the hotspot, '
        f'so 10 degrees is {10 * KM_DEG:.0f} km. Dashed arcs mark 410, 660 and '
        '1000 km depth, labelled in kilometres on the first panel and drawn on '
        'all of them, and the solid inner arc marks the core-mantle boundary. The colour '
        'scale is fixed across every panel and saturates at the amplitude of the '
        "model's own strong anomalies rather than at a value chosen to make "
        'structures appear continuous. The white-on-black line is the least-cost '
        'descending path, drawn the same way for every hotspot: it is the cheapest '
        'route through the velocity field and not, by itself, a detected conduit. '
        'It is dotted inside the search cap around the hotspot, where a descent is '
        'not yet resolved. The red axis follows the slow structure to the depth at '
        'which it ends. Inset globes show the section line and its midpoint.')
    if _A.all_hotspots:
        _cap += (' Each sub-label opens with the track-support group the hotspot '
                 'falls in - deep, upper/mid, weak or disjointed - which is a '
                 'description drawn from where its path sits in the distribution of '
                 '300 ambient paths, not a test; the tested comparison is of the '
                 'whole distributions.'
                 # --sites draws the panels in the order given, which is the order
                 # of the argument and not by group; saying otherwise in the caption
                 # of Figure 5 was false for as long as the figure has existed.
                 + ('' if _A.sites else ' Panels are ordered by group and then '
                    'alphabetically.')
                 + ' Sub-labels then give the corridor width in the '
                 'upper mantle and '
                 'transition zone, the upper lower mantle and the deep lower '
                 'mantle, and whether the descent ends inside a large '
                 'low-shear-velocity province. The corridor is the set of routes '
                 'the data cannot distinguish from the one drawn, so its width is '
                 'the error bar on that line and not a width of anything physical.')

    if _ST is not None and any(_second_for(n) is not None for n, _, _, _ in page):
        _cap += (f' Where a second descent costs within {_A.second_margin:g} per cent '
                 'of the first - a difference of a tenth of a per cent in the mean '
                 'anomaly the route integrates, which the model does not resolve - '
                 'both are drawn, the second dashed. The section plane is fixed by '
                 'the first descent, so the second is drawn faint where it lies more '
                 f'than {SECOND_XT:.0f} km out of the plane and is only projected into '
                 'it.')

    tag_out = out + (f'_p{pno}' if len(PAGES) > 1 else '')
    fig.savefig(tag_out + '.pdf')
    fig.savefig(tag_out + '.png', dpi=400)
    with open(tag_out + '_caption.txt', 'w') as _fh:
        _fh.write(_cap + '\n')
    print('wrote', tag_out, f'({len(page)} panels)', flush=True)
    _check_text(tag_out + '.pdf')
