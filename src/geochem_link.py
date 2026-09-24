"""Is the reported link between hotspot geochemistry and depth of connection a
link to depth, or to sitting above a large low-velocity province?

Konter and Becker (2012) report a correlation between extreme lava compositions
and the depth extent to which a hotspot can be traced seismically, using the
extent of Boschi et al. (2007): the fraction of a conduit, advected through a
flow field, on which the anomaly is below a fixed threshold. That measure is not
compared against a null, so a hotspot standing over broadly slow mantle scores
highly whether or not anything connects it downward.

This script separates the two. The same geochemistry is tested against their
ranking and against the depth reached here, and then against their ranking again
with province membership held constant. Province membership is measured
independently by province.py.

Rank correlation is used throughout: it asks only whether two quantities rise and
fall together, needs no assumption that the relation is a straight line, and is
not carried by one extreme hotspot. The partial correlation removes, from both
quantities, the part that can be predicted from province membership, and asks
whether what remains still moves together.
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, pearsonr
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--oibid', required=True,
                help='Supplementary Dataset 1 of Hardardottir and Jackson (2025), .xlsx')
ap.add_argument('--tag', default='RevealLO')
ap.add_argument('--dir', default='out')
ap.add_argument('--sheet', default='OIB igneous silicate data')
A = ap.parse_args()

# filtered columns of the database, by position: the unfiltered ones retain
# analyses the compilers' own quality criteria reject
COLS = {'3He/4He (R/RA)': 87, '87Sr/86Sr': 33, '143Nd/144Nd': 35,
        '176Hf/177Hf': 37, '206Pb/204Pb': 39, '208Pb/204Pb': 43,
        '187Os/188Os': 81}

names = pd.read_csv(os.path.join(HERE, 'oib_hotspot_names.csv'), comment='#')
names = names.dropna(subset=['oibid_name'])
rank = pd.read_csv(os.path.join(HERE, 'bbs2007_figure14_ranking.csv'), comment='#')
rank['bbs_score'] = len(rank) + 1 - rank['rank']       # higher = deeper in their order

print('reading the database', flush=True)
d = pd.read_excel(A.oibid, sheet_name=A.sheet, header=2)
d['Hotspot'] = d['Hotspot'].astype(str).str.strip()
rows = []
for _, r in names.iterrows():
    sub = d[d.Hotspot == r.oibid_name]
    e = {'hotspot': r.hotspot, 'n_samples': len(sub)}
    for lab, ci in COLS.items():
        v = pd.to_numeric(sub.iloc[:, ci], errors='coerce').dropna()
        e[f'max_{lab}'] = v.max() if len(v) else np.nan
        e[f'med_{lab}'] = v.median() if len(v) else np.nan
        e[f'n_{lab}'] = len(v)
    rows.append(e)
g = pd.DataFrame(rows)
print(f'{len(g)} hotspots matched, {g.n_samples.sum()} samples')

D = A.dir
m = (g.merge(pd.read_csv(f'{D}/tracked_depth_{A.tag}.csv')[['hotspot', 'deepest_km']],
             on='hotspot', how='left')
      .merge(pd.read_csv(f'{D}/comparison_all_models.csv')[['hotspot', 'f_ind_mean', 'n_deep']],
             on='hotspot', how='left')
      .merge(pd.read_csv(f'{D}/province_{A.tag}.csv')[['hotspot', 'dVs_deep']],
             on='hotspot', how='left')
      .merge(rank[['hotspot', 'bbs_score']], on='hotspot', how='left'))
m['deepest_km'] = m.deepest_km.fillna(0.0)


def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return pearsonr(ex, ey)


def show(a, b, lab):
    k = m[a].notna() & m[b].notna()
    if k.sum() < 8:
        print(f'  {lab:46s} only {k.sum()} hotspots, not tested'); return
    r, p = spearmanr(m.loc[k, a], m.loc[k, b])
    print(f'  {lab:46s} rho {r:+.3f}  p = {p:.4f}  n = {k.sum()}')


print('\nWhat each measure of depth tracks:')
show('bbs_score', 'dVs_deep', 'their ranking vs province membership')
show('deepest_km', 'dVs_deep', 'depth reached here vs province membership')
show('f_ind_mean', 'dVs_deep', 'root fraction here vs province membership')
show('bbs_score', 'deepest_km', 'their ranking vs depth reached here')
show('bbs_score', 'f_ind_mean', 'their ranking vs root fraction here')

print('\nGeochemistry against each, and against theirs with province held constant:')
print(f"  {'':22s} {'vs their ranking':>22s} {'vs depth here':>20s} {'province held':>20s}")
for lab in ('3He/4He (R/RA)', '87Sr/86Sr', '206Pb/204Pb', '208Pb/204Pb', '187Os/188Os'):
    c = f'max_{lab}'
    k = m[c].notna() & m.bbs_score.notna() & m.dVs_deep.notna()
    if k.sum() < 10:
        continue
    r1, p1 = spearmanr(m.loc[k, c], m.loc[k, 'bbs_score'])
    k2 = m[c].notna() & m.deepest_km.notna()
    r2, p2 = spearmanr(m.loc[k2, c], m.loc[k2, 'deepest_km'])
    r3, p3 = partial(m.loc[k, c].values, m.loc[k, 'bbs_score'].values,
                     m.loc[k, 'dVs_deep'].values)
    print(f'  max {lab:18s} {r1:+.3f} p={p1:.3f} {r2:+7.3f} p={p2:.3f} '
          f'{r3:+8.3f} p={p3:.3f}  n={k.sum()}')

m.to_csv(os.path.join(D, f'geochem_link_{A.tag}.csv'), index=False)
print(f'\nwrote geochem_link_{A.tag}.csv')
