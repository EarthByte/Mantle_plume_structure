#!/usr/bin/env python3
"""Combine the two cross-model directional tests into one table.

The upper-mantle test is written by apm_crossmodel.py and the lower-mantle test by
lean_confirm.py. Figure 2d and the manuscript audit both read the combined file, so
the figure cannot show one set of numbers while the text quotes another.

  python3 apm_crossmodel.py
  python3 lean_confirm.py --tags RevealLO,REVEAL,GLADM35,SPiRaL,SEMUCB-WM1,RevealLO_30km
  python3 crossmodel_table.py
"""
import os
import pandas as pd
import provenance

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'out')
a = pd.read_csv(os.path.join(D, 'apm_crossmodel.csv'))
b = pd.read_csv(os.path.join(D, 'lean_confirm_crossmodel.csv'))
a['window'] = '300-660'
keep = ['model', 'window', 'n', 'R', 'P']

# The two windows exclude paths under different rules, and carrying one column
# called "vertical" across both said they were the same rule when they are not.
# Upper mantle: no azimuth exists (displacement identically zero), or too few
# nodes. Lower mantle: displacement under 20 km, too few nodes, or too few trench
# points to define a direction to lean away from. Both are reported as one
# "excluded" count with the components kept beside it, and the table caption gives
# the rule for each window.
for _f in ('vertical', 'thin', 'short', 'few_trench'):
    if _f not in a.columns:
        a[_f] = float('nan')
    if _f not in b.columns:
        b[_f] = float('nan')
if 'power' not in b.columns:
    b['power'] = float('nan')
comp = ['vertical', 'thin', 'short', 'few_trench']
a['excluded'] = a[comp].fillna(0).sum(axis=1)
b['excluded'] = b[comp].fillna(0).sum(axis=1)
cols = keep + comp + ['excluded', 'power']
out = pd.concat([a[cols], b[cols]], ignore_index=True)
_cl = os.path.join(D, 'crossmodel_lean.csv')
out.to_csv(_cl, index=False)
provenance.stamp(_cl, n_rows=len(out), models=sorted(out.model.unique().tolist())
                 if 'model' in out else None,
                 inputs=[os.path.join(D, 'lean_confirm_crossmodel.csv')])
print(out.to_string(index=False))
