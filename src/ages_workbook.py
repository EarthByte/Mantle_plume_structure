"""Supporting data workbook for the hotspot age compilation.

Three sheets: the 49 hotspots with both longevity measures beside the
connectivity result each is tested against; the five oldest dated samples on
each Pacific chain, so that a reader can see the chain ages are corroborated
rather than resting on a single date; and the column definitions and sources.
The row order of the first sheet is taken from the classification table so
that it cannot drift out of step with Table 1.
"""
import argparse
import collections
import os
import sys

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longevity import CATALOGUE, DOCUMENTED, NOT_FOUND

ap = argparse.ArgumentParser(
    description='Supporting data workbook: the hotspot age compilation, the '
                'Pacific samples each chain age rests on, and their sources.')
ap.add_argument('--comparison', default='out/comparison_all_models.csv')
ap.add_argument('--table', default='out/table1_classification.csv')
ap.add_argument('--ages', default='../../Papers/Chase_Wessel_2022_data/'
                                  'PHT2021_raw_data/PHT2021_pacific_ages.txt',
                help='the raw Pacific age table of Chase and Wessel (2021)')
ap.add_argument('--out', default='out/Data_Set_S1_hotspot_ages.xlsx')
A = ap.parse_args()

ARIAL   = 'Arial'
HDR_F   = Font(name=ARIAL, size=10, bold=True, color='FFFFFF')
HDR_FIL = PatternFill('solid', fgColor='44546A')
BODY_F  = Font(name=ARIAL, size=10)
NOTE_F  = Font(name=ARIAL, size=9, italic=True, color='595959')
TITLE_F = Font(name=ARIAL, size=11, bold=True)
THIN    = Border(bottom=Side(style='thin', color='BFBFBF'))

wb = Workbook(); wb.remove(wb.active)

def header(ws, row, cols):
    for j, h in enumerate(cols, 1):
        c = ws.cell(row=row, column=j, value=h)
        c.font, c.fill = HDR_F, HDR_FIL
        c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    ws.row_dimensions[row].height = 30

def widths(ws, w):
    for j, x in enumerate(w, 1):
        ws.column_dimensions[get_column_letter(j)].width = x

KIND = {'lip': 'flood basalt or oceanic plateau',
        'chain': 'oldest dated edifice in the chain',
        'chain*': 'oldest dated edifice in the chain'}
NOTE = {
 'Macdonald (Cook-Austral)':
   'Describes the Rurutu-Arago trail of the Cook-Austral province, not Macdonald itself: '
   'the present hotspot of that trail is placed at Arago Seamount (Konrad et al., 2018).',
 'Hawaii':
   'Detroit is the oldest dated Emperor edifice in the compilation; Meiji, further north, '
   'is not dated there, so this is a lower bound on the chain. Agrees with Wei et al. (2020).',
}

# ---------------------------------------------------------------- sheet 1
# Row order comes from the classification table, so that this sheet and Table 1
# cannot diverge; the values come from the comparison table, which carries them
# at full precision rather than at the two decimals Table 1 prints.
order = pd.read_csv(A.table)[['hotspot', 'tier']]
cmp_ = pd.read_csv(A.comparison)
cmp_['spread'] = cmp_.f_max - cmp_.f_min
d = order.merge(cmp_, on='hotspot', how='left')
d = d.drop(columns=['n_models']).rename(
    columns={'f_mean': 'root_fraction', 'n_deep': 'n_models', 'lon_180': 'lon'})

ws = wb.create_sheet('Hotspot ages')
ws['A1'] = ('Data Set S1a. Longevity of the surface volcanic record for the 49 hotspots '
            'of Courtillot et al. (2003), with the connectivity result each age is tested against.')
ws['A1'].font = TITLE_F
ws['A2'] = ('In the row order of Table 1: by tier, then by root fraction. No age column is '
            'available to the search. Sources and column definitions are on the '
            '"Sources and notes" sheet.')
ws['A2'].font = NOTE_F
for r in (1, 2):
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=12)
    ws.cell(row=r, column=1).alignment = Alignment(wrap_text=True, vertical='top')
ws.row_dimensions[2].height = 28

COLS = ['Hotspot', 'Latitude (deg)', 'Longitude (deg)', 'Root fraction, mean of six runs',
        'Spread across runs', 'Models rooted (of 5)', 'Tier', 'Catalogue age (Ma)',
        'Dated age (Ma)', '1-sigma (Myr)', 'Kind of evidence', 'Dated feature']
header(ws, 4, COLS)
for i, (_, r) in enumerate(d.iterrows()):
    doc = DOCUMENTED.get(r.hotspot)
    row = 5 + i
    vals = [r.hotspot, round(float(r.lat), 2), round(float(r.lon), 2),
            round(float(r.root_fraction), 3), round(float(r.spread), 3),
            int(r.n_models), r.tier,
            CATALOGUE.get(r.hotspot),
            doc[0] if doc else None,
            doc[3] if doc else None,
            KIND.get(doc[1]) if doc else None,
            doc[2].split(';')[0].strip() if doc else None]
    for j, v in enumerate(vals, 1):
        c = ws.cell(row=row, column=j, value=v)
        c.font, c.border = BODY_F, THIN
        if j in (2, 3):
            c.number_format = '0.00'
        if j in (4, 5):
            c.number_format = '0.000'
        if j in (8, 9, 10):
            c.number_format = {8: '0', 9: '0.0', 10: '0.00'}[j]
        if j >= 2:
            c.alignment = Alignment(horizontal='center')
    if r.hotspot in NOTE:
        ws.cell(row=row, column=13, value=NOTE[r.hotspot]).font = NOTE_F
widths(ws, [26, 13, 14, 15, 14, 12, 7, 12, 11, 11, 26, 34, 70])
ws.freeze_panes = 'B5'

n_cat = sum(h in CATALOGUE for h in d.hotspot)
n_doc = sum(h in DOCUMENTED for h in d.hotspot)
end = 5 + len(d)
ws.cell(row=end + 1, column=1,
        value=(f'Catalogue ages for {n_cat} of 49 hotspots; dated ages for {n_doc}. '
               f'No age of either kind was found for: {", ".join(NOT_FOUND)}.')).font = NOTE_F

# ---------------------------------------------------------------- sheet 2
rows = []
for line in open(A.ages):
    if line.startswith('#') or not line.strip():
        continue
    f = [x.strip() for x in line.rstrip('\n').split('\t') if x.strip()]
    if len(f) != 7:
        continue
    try:
        rows.append((f[5], f[6], f[4], float(f[0]), float(f[1]), float(f[2]), float(f[3])))
    except ValueError:
        continue
by = collections.defaultdict(list)
for r in rows:
    by[r[0]].append(r)
USED = {'HI': 'Hawaii', 'LV': 'Louisville', 'RU': 'Macdonald (Cook-Austral)',
        'CB': 'Juan de Fuca/Cobb', 'KO': 'Bowie', 'CR': 'Caroline', 'SA': 'Samoa',
        'PC': 'Pitcairn', 'MQ': 'Marquesas', 'SO': 'Tahiti/Society'}

ws = wb.create_sheet('Pacific chain samples')
ws['A1'] = ('Data Set S1b. The five oldest dated samples on each Pacific chain used above, '
            'showing that each chain age is corroborated rather than resting on one date.')
ws['A1'].font = TITLE_F
ws['A2'] = ('Extracted from PHT2021_raw_data/PHT2021_pacific_ages.txt in Chase and Wessel (2021), '
            'CC-BY-4.0. That table descends from Clouard and Bonneville (2005, their Table 2) and '
            'carries later revisions. The oldest sample on each chain is the value used, and is '
            'shaded. Foundation is tabulated there but is not one of the 49 hotspots.')
ws['A2'].font = NOTE_F
for r in (1, 2):
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8)
    ws.cell(row=r, column=1).alignment = Alignment(wrap_text=True, vertical='top')
ws.row_dimensions[2].height = 40

header(ws, 4, ['Chain tag', 'Chain', 'Hotspot in this study', 'Sample',
               'Longitude (deg E)', 'Latitude (deg)', 'Age (Ma)', '1-sigma (Myr)'])
SHADE = PatternFill('solid', fgColor='FCE4D6')
row = 5
for tag in sorted(USED):
    for k, s in enumerate(sorted(by[tag], key=lambda r: -r[5])[:5]):
        vals = [tag, s[1], USED[tag] if k == 0 else '', s[2].replace('_', ' '),
                round(s[3], 3), round(s[4], 3), s[5], s[6]]
        for j, v in enumerate(vals, 1):
            c = ws.cell(row=row, column=j, value=v)
            c.font, c.border = BODY_F, THIN
            if k == 0:
                c.fill = SHADE
            if j >= 5:
                c.number_format = '0.000' if j <= 6 else '0.00'
                c.alignment = Alignment(horizontal='center')
        row += 1
widths(ws, [10, 20, 26, 30, 17, 15, 11, 12])
ws.freeze_panes = 'A5'

# ---------------------------------------------------------------- sheet 3
ws = wb.create_sheet('Sources and notes')
ws['A1'] = 'Data Set S1c. Column definitions and sources.'
ws['A1'].font = TITLE_F
ws.column_dimensions['A'].width = 30
ws.column_dimensions['B'].width = 108
row = 3
def block(title, items):
    global row
    c = ws.cell(row=row, column=1, value=title); c.font = Font(name=ARIAL, size=10, bold=True)
    row += 1
    for k, v in items:
        ws.cell(row=row, column=1, value=k).font = BODY_F
        b = ws.cell(row=row, column=2, value=v); b.font = BODY_F
        b.alignment = Alignment(wrap_text=True, vertical='top')
        ws.row_dimensions[row].height = 13 * max(1, (len(v) // 105) + 1)
        row += 1
    row += 1

block('Columns', [
 ('Spread across runs', 'Range of that share across the six runs: the largest value minus the smallest.'),
 ('Root fraction', 'Mean over the six model runs of the share of calibrated configurations in which the cheapest descending path from beneath the hotspot is cheaper than that run\'s own random-location null. Zero means no configuration beat the null.'),
 ('Models rooted', 'Number of the five independently constructed tomographic models in which the hotspot is rooted; RevealLO at two lateral samplings counts once.'),
 ('Tier', 'I, rooted in a majority of the independent models; II, in at least one; IV, in none.'),
 ('Catalogue age', 'Hotspot age of Steinberger (2000) as tabulated by Torsvik et al. (2006, their Table 2). The definition of that column is not restated in the source, and for several hotspots it exceeds the oldest dated feature on the track, so it is best read as a catalogue value for the presumed onset of the hotspot rather than as a dated quantity.'),
 ('Dated age', 'Age of the oldest volcanism dated and attributed to the hotspot: the associated flood basalt or oceanic plateau where Courtillot et al. (2003) make that link, otherwise the oldest dated edifice in the chain.'),
 ('1-sigma', 'Analytical uncertainty on the dated edifice, where the compilation reports one. Province ages carry no single uncertainty and are left blank.'),
])
block('Sources', [
 ('Chase and Wessel (2021)', 'Pacific hotspot trails datasets (Version 1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5576466. CC-BY-4.0. Source of every chain age here.'),
 ('Chase and Wessel (2022)', 'Analysis of Pacific hotspot chains. Geochemistry, Geophysics, Geosystems, 23, e2021GC010225. https://doi.org/10.1029/2021GC010225'),
 ('Clouard and Bonneville (2005)', 'Ages of seamounts, islands, and plateaus on the Pacific plate. Geological Society of America Special Paper, 388, 71-90. https://doi.org/10.1130/0-8137-2388-4.71. The compilation the dataset above descends from.'),
 ('Courtillot et al. (2003)', 'Three distinct types of hotspots in the Earth\'s mantle. Earth and Planetary Science Letters, 205, 295-308. https://doi.org/10.1016/S0012-821X(02)01048-8'),
 ('Courtillot and Renne (2003)', 'On the ages of flood basalt events. Comptes Rendus Geoscience, 335, 113-140. https://doi.org/10.1016/S1631-0713(03)00006-3. Source of every flood basalt and oceanic plateau age here.'),
 ('Konrad et al. (2018)', 'On the relative motions of long-lived Pacific mantle plumes. Nature Communications, 9, 854. https://doi.org/10.1038/s41467-018-03277-x'),
 ('Sleep (1990)', 'Monteregian hotspot track: A long-lived mantle plume. Journal of Geophysical Research, 95, 21983-21990. https://doi.org/10.1029/JB095iB13p21983'),
 ('Steinberger (2000)', 'Plumes in a convecting mantle: Models and observations for individual hotspots. Journal of Geophysical Research, 105, 11127-11152. https://doi.org/10.1029/1999JB900398'),
 ('Torsvik et al. (2006)', 'Global plate motion frames: Toward a unified model. Reviews of Geophysics, 44, RG3001. https://doi.org/10.1029/2005RG000185'),
 ('Wei et al. (2020)', 'Seismic evidence for a plume head beneath the Emperor seamount chain. Nature Communications, 11, 4085. https://doi.org/10.1038/s41467-020-17945-4'),
])
wb.save(A.out)
print(f'wrote {A.out}')
