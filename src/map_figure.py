"""Global map: Courtillot's 49 hotspots, classified across the independent models.

The map shows the consensus rather than any one model, because the spread between
models is part of the result. RevealLO at two lateral samplings is one model and
is counted once.
"""
import argparse, glob, math, os, re, warnings
import numpy as np, pandas as pd, pygmt
warnings.filterwarnings('ignore')

_ap = argparse.ArgumentParser()
_ap.add_argument('--slice', default=None,
                 help='depth slice to draw under the symbols; by default the '
                      'one conduit_depth_slice.py wrote, if there is one')
_ap.add_argument('--transparency', type=float, default=60.0,
                 help='per cent; the background is context, not the message')
_ap.add_argument('--lim', type=float, default=1.8,
                 help='per cent, the same saturation as the cross-sections')
_A, _ = _ap.parse_known_args()

MAP_CM, PLACED_CM = 17.0, 16.0          # drawn width, and the width it is placed at


def pt(p):
    """A point size given at the size the map is printed, not the size it is drawn."""
    return f'{math.ceil(p * MAP_CM / PLACED_CM * 10) / 10:.1f}p'

CMP = 'out/comparison_all_models.csv'
d = pd.read_csv(CMP)
NMOD = int(d.n_models.max())
MAJ = (NMOD + 1) // 2

def cat(r):
    if r.n_deep >= MAJ:
        return 'major'
    if r.n_deep >= 1:
        return 'minor'
    return 'none'

d['c'] = d.apply(cat, axis=1)
N = {k: int((d.c == k).sum()) for k in ('major', 'minor', 'none')}

fig = pygmt.Figure()
pygmt.config(FONT_ANNOT_PRIMARY=pt(8.5), FONT_LABEL=pt(9),
             MAP_FRAME_PEN='0.7p')
proj = 'N150/17c'
fig.basemap(region='d', projection=proj, frame=['af', 'WSne'])

# The tomographic background is drawn first and heavily transparent: it is there
# so a reader can see that the rooted hotspots sit on slow structure, not so that
# it competes with the symbols. Same colour map and same saturation as the
# cross-sections, so a reader can carry one scale between the two figures.
SLICE = _A.slice
if SLICE is None:
    cand = sorted(glob.glob('out/slice_*km.nc')) + sorted(glob.glob('figures/slice_*km.nc'))
    SLICE = cand[0] if cand else None
SLICE_Z = None
if SLICE and os.path.exists(SLICE):
    m = re.search(r'_(\d+)km', os.path.basename(SLICE))
    SLICE_Z = int(m.group(1)) if m else None
    pygmt.makecpt(cmap='roma', series=[-_A.lim, _A.lim, 0.02], continuous=True)
    fig.grdimage(grid=SLICE, region='d', projection=proj, cmap=True,
                 transparency=_A.transparency, nan_transparent=True)
    fig.colorbar(cmap=True, position='JBC+jTC+o0c/2.35c+w7.0c/0.28c+h+e',
                 frame=['xa1f0.5+lShear-velocity anomaly at '
                        f'{SLICE_Z} km (per cent)'])
    fig.coast(shorelines='0.25p,gray45', resolution='l')
else:
    print('  no depth slice found; run conduit_depth_slice.py for the background')
    fig.coast(land='#f2efe9', water='white', shorelines='0.25p,gray55',
              resolution='l')

STY = {'major': ('c0.38c', '#B4442E'),
       'minor': ('c0.30c', '#E8A33D'),
       'none':  ('i0.28c', '#2E5E8E')}
for k in ('none', 'minor', 'major'):
    s = d[d.c == k]
    if len(s):
        fig.plot(x=s.lon_180.values, y=s.lat.values, style=STY[k][0],
                 fill=STY[k][1], pen='0.7p,black')

# Labelled: everything a model found, plus the hotspots most often argued to be
# primary plumes, so that a reader can find them whether or not they were found.
NOTABLE = ['Hawaii', 'Yellowstone', 'Louisville', 'Easter', 'Tristan',
           'Kerguelen', 'Galapagos', 'Azores', 'Marion', 'Bouvet']
PLACE = {
 'Iceland': ('CB', '0c/0.26c'),      'Jan Mayen': ('LM', '0.32c/0.10c'),
 'Pitcairn': ('LM', '0.32c/0.22c'),   'Macdonald': ('RM', '-0.32c/-0.20c'),
 'Tahiti': ('RM', '-0.32c/0c'),       'Marquesas': ('LM', '0.32c/0.10c'),
 'Samoa': ('RM', '-0.32c/0.10c'),     'Caroline': ('LM', '0.32c/0c'),
 'Reunion': ('LM', '0.32c/0c'),       'Comores': ('RM', '-0.32c/0.16c'),
 'Afar': ('LM', '0.32c/0.14c'),       'Darfur': ('RM', '-0.32c/0c'),
 'Cape Verde': ('LM', '0.32c/-0.16c'), 'Canary': ('LM', '0.32c/0.18c'),
 'Marion': ('RM', '-0.32c/-0.18c'),   'Bouvet': ('LM', '0.32c/-0.16c'),
 'Lord Howe': ('LM', '0.34c/0.26c'),  'Tasmanid': ('LM', '0.32c/-0.26c'),
 'Hawaii': ('LM', '0.32c/0c'),        'Kerguelen': ('LM', '0.32c/0c'),
 'Eifel': ('LM', '0.32c/0.14c'),      'Yellowstone': ('LM', '0.32c/0.12c'),
 'Louisville': ('LM', '0.32c/0c'),    'Easter': ('LM', '0.32c/-0.26c'),
 'Tristan': ('RM', '-0.32c/-0.16c'),  'Galapagos': ('LM', '0.32c/0c'),
 'Azores': ('LM', '0.32c/0.16c'),
}
lab = d[(d.c != 'none') |
        d.hotspot.astype(str).str.split('(').str[0].str.split('/').str[0]
         .str.strip().isin(NOTABLE)]
for _, r in lab.iterrows():
    nm = str(r.hotspot).split('(')[0].split('/')[0].strip()[:13]
    just, off = PLACE.get(nm, ('LM', '0.32c/0.14c'))
    fig.text(x=r.lon_180, y=r.lat, text=nm, justify=just, offset=off,
             font=f'{pt(8.5)},Helvetica-Bold,black', fill='white@25',
             clearance='0.04c/0.03c')

leg = 'figures/_legend.txt'
open(leg, 'w').write('\n'.join([
    'N 1',
    f'S 0.22c c 0.34c #B4442E 0.7p,black 0.7c rooted in a '
    f'majority of the {NMOD} models ({N["major"]})',
    f'S 0.22c c 0.28c #E8A33D 0.7p,black 0.7c in at least one ({N["minor"]})',
    f'S 0.22c i 0.26c #2E5E8E 0.7p,black 0.7c in none ({N["none"]})']) + '\n')
fig.legend(spec=leg, position='JBC+jTC+o0c/1.0c+w11.4c',
           box='+gwhite+p0.6p,gray40')
os.makedirs('figures', exist_ok=True)
out = 'figures/fig_map'
fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=400)
print('wrote', out, N, f'background {SLICE_Z} km' if SLICE_Z else 'no background')
