#!/usr/bin/env python3
"""The independent long-track classification, fixed before any morphology is seen.

The variable this paper compares morphology against must not be derived from
tomography, and it must not be derived from the morphology it is compared with.
Neither of the two longevity measures already in the study is adequate on its own:
the catalogue age is a presumed onset rather than a dated quantity, and the
documented oldest volcanism mixes a flood basalt with the oldest edifice of a
chain and exists for fewer than half the hotspots. This builds the classification
the comparison needs, from six attributes of the surface volcanic record and of
the non-seismic evidence for a deep source.

  1  duration of independently dated volcanism attributed to the hotspot
  2  monotonicity of age with distance along the trail
  3  continuity: no long break in the dated record and no large along-trail gap
  4  number of independent radiometric ages
  5  confidence that older and younger segments belong to one source
  6  independent evidence of a deep source, none of it seismic

Attributes 1 to 4 are COMPUTED for the eleven Pacific chains from the raw age
table of Chase and Wessel (2021) and are CURATED from the literature elsewhere;
5 and 6 are curated everywhere. Every value carries its provenance, and the
computed values override the curated ones where both exist, so that the
compilation cannot silently disagree with the table it came from.

Attribute 6 admits plume-motion modelling, hotspot paleolatitude, high helium
isotope ratios and an associated large igneous province. It admits no seismic
observation of any kind - not a low-velocity anomaly, not a thinned transition
zone measured by tomography, not a receiver-function discontinuity depth used as
a proxy for temperature - because the paper's question is whether independently
established persistence predicts tomographic expression, and an input drawn from
the seismic field would make that question circular. The one discontinuity entry
allowed, beneath Iceland, is retained only because it is a transition-zone
thickness measurement rather than a velocity anomaly, and the classification is
reported with and without it.

THE RULE

  P1  a long, coherent, age-progressive trail with independent deep-source support:
      duration at or above MIN_DURATION, a monotonic progression, one source, at
      least one item of independent deep-source evidence, and either continuity
      or at least MIN_AGES dated samples
  P2  probably long-lived, but with a gap, a disputed correlation or no
      independent deep-source evidence: duration at or above MIN_DURATION and
      either a progression or a single source
  P3  everything else

The thresholds are written into the output so that the classification a result
was computed against can always be recovered from the file, and the file is
hashed so that a later change cannot pass unnoticed.
"""
from __future__ import annotations

import argparse, hashlib, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MIN_DURATION = 40.0        # Myr of dated volcanism
MIN_RHO = 0.90             # Spearman rank correlation of age against distance
MIN_AGES = 8
MAX_AGE_GAP = 25.0         # Myr, largest break in the dated record
MAX_SPACE_GAP = 1500.0     # km, largest along-trail gap between dated edifices

# Chase and Wessel chain -> the hotspot of the Courtillot table it belongs to.
# Rurutu is listed against Macdonald because that is the entry the Courtillot
# table carries for the Cook-Austral province, and the mismatch between the
# dated trail and the named centre is itself recorded as a failure of attribute 5.
CHAIN_TO_HOTSPOT = {
    'Hawaii-Emperor': 'Hawaii', 'Louisville': 'Louisville', 'Samoa': 'Samoa',
    'Rurutu': 'Macdonald (Cook-Austral)', 'Marquesas': 'Marquesas',
    'Society': 'Tahiti/Society', 'Cobb': 'Juan de Fuca/Cobb',
    'Kodiak-Bowie': 'Bowie', 'Pitcairn': 'Pitcairn', 'Caroline': 'Caroline',
}

R_E = 6371.0
DEG = np.pi / 180.0


def gc_km(la1, lo1, la2, lo2):
    p1, p2 = np.asarray(la1) * DEG, np.asarray(la2) * DEG
    dl = (np.asarray(lo2) - np.asarray(lo1)) * DEG
    return R_E * np.arccos(np.clip(np.sin(p1) * np.sin(p2) +
                                   np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1))


def read_ages(path):
    rows = []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        p = [q.strip() for q in line.rstrip('\n').split('\t') if q.strip()]
        if len(p) < 6:
            continue
        try:
            lon, lat, age = float(p[0]), float(p[1]), float(p[2])
        except ValueError:
            continue
        rows.append(dict(lon=lon, lat=lat, age=age, name=p[4], chain=p[-1]))
    return pd.DataFrame(rows)


def chain_attributes(df):
    """Duration, monotonicity, continuity and count, per Pacific chain."""
    from scipy.stats import spearmanr
    out = {}
    for chain, g in df.groupby('chain'):
        g = g[np.isfinite(g.age)].sort_values('age').reset_index(drop=True)
        if len(g) < 3:
            continue
        # distance from the youngest dated edifice, which stands for the present
        # centre: it is the datum the progression is measured against and needs
        # no assumption about where the hotspot is now
        d = gc_km(g.lat.iloc[0], g.lon.iloc[0], g.lat.values, g.lon.values)
        rho = float(spearmanr(g.age.values, d).statistic)
        age_gap = float(np.max(np.diff(g.age.values))) if len(g) > 1 else np.nan
        order = np.argsort(d)
        space_gap = float(np.max(np.diff(np.sort(d)))) if len(g) > 1 else np.nan
        out[chain] = dict(duration_myr=float(g.age.max() - g.age.min()),
                          rho_age_distance=rho, n_ages=int(len(g)),
                          max_age_gap_myr=age_gap, max_space_gap_km=space_gap)
    return out


def classify(r, min_duration, min_ages):
    dur = r['duration_myr']
    long_enough = np.isfinite(dur) and dur >= min_duration
    prog = str(r['progression']) == 'monotonic'
    cont = str(r['gap_ok']) == 'yes'
    many = np.isfinite(r['n_ages']) and r['n_ages'] >= min_ages
    one = str(r['one_source']) == 'yes'
    deep = bool(str(r['deep_evidence']).strip()) and str(r['deep_evidence']) != 'nan'
    if long_enough and prog and one and deep and (cont or many):
        return 'P1'
    if long_enough and (prog or one):
        return 'P2'
    return 'P3'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', default='track_evidence.csv')
    ap.add_argument('--ages', default='../../Papers/Chase_Wessel_2022_data/'
                                      'PHT2021_raw_data/PHT2021_pacific_ages.txt')
    ap.add_argument('--hotspots', default='hotspots_courtillot2003.csv')
    ap.add_argument('--dir', default='out')
    ap.add_argument('--min-duration', type=float, default=MIN_DURATION,
                    dest='min_duration')
    ap.add_argument('--min-ages', type=int, default=MIN_AGES, dest='min_ages')
    ap.add_argument('--no-discontinuity', action='store_true',
                    dest='no_disc',
                    help='drop the one transition-zone thickness entry from the '
                         'deep-source evidence, as a check that the '
                         'classification does not depend on it')
    ap.add_argument('--targets-out', default='morph_targets.txt',
                    dest='targets_out')
    ap.add_argument('--controls', default='Tahiti/Society,Pitcairn,Samoa',
                    help='hotspots added to the target list as morphological '
                         'controls whatever their class. They are named here, '
                         'before any morphology is computed, and they are named '
                         'for what they are: a consensus detection with a short '
                         'trail, a second such detection, and an ambiguous case. '
                         'Choosing controls after seeing the corridors would '
                         'make the contrast unfalsifiable')
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)

    ev = pd.read_csv(a.evidence)
    ev['source_of_1_to_4'] = 'curated'
    comp = {}
    if os.path.exists(a.ages):
        comp = chain_attributes(read_ages(a.ages))
        print(f'computed attributes for {len(comp)} Pacific chains')
    for chain, at in comp.items():
        hs = CHAIN_TO_HOTSPOT.get(chain)
        if hs is None or hs not in set(ev.hotspot):
            continue
        m = ev.hotspot == hs
        ev.loc[m, 'duration_myr'] = at['duration_myr']
        ev.loc[m, 'n_ages'] = at['n_ages']
        ev.loc[m, 'rho_age_distance'] = at['rho_age_distance']
        ev.loc[m, 'max_age_gap_myr'] = at['max_age_gap_myr']
        ev.loc[m, 'max_space_gap_km'] = at['max_space_gap_km']
        ev.loc[m, 'progression'] = ('monotonic' if at['rho_age_distance'] >= MIN_RHO
                                    else ('partial' if at['rho_age_distance'] >= 0.6
                                          else 'none'))
        ev.loc[m, 'gap_ok'] = ('yes' if (at['max_age_gap_myr'] <= MAX_AGE_GAP and
                                         at['max_space_gap_km'] <= MAX_SPACE_GAP)
                               else 'no')
        ev.loc[m, 'source_of_1_to_4'] = f'Chase and Wessel 2021, chain {chain}'

    if a.no_disc:
        ev['deep_evidence'] = ev.deep_evidence.fillna('').apply(
            lambda s: ';'.join([q for q in str(s).split(';')
                                if q and q != 'discontinuity']))
    ev['n_ages'] = pd.to_numeric(ev.n_ages, errors='coerce')
    ev['duration_myr'] = pd.to_numeric(ev.duration_myr, errors='coerce')
    ev['track_class'] = [classify(r, a.min_duration, a.min_ages)
                         for _, r in ev.iterrows()]
    ev['min_duration_myr'] = a.min_duration
    ev['min_ages'] = a.min_ages

    hs = pd.read_csv(a.hotspots)
    missing = sorted(set(hs.hotspot.astype(str)) - set(ev.hotspot.astype(str)))
    if missing:
        print(f'!! {len(missing)} hotspots have no evidence row: '
              f'{", ".join(missing)}')
    out = os.path.join(a.dir, 'track_classes.csv')
    ev.to_csv(out, index=False)
    h = hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16]
    open(os.path.join(a.dir, 'track_classes.sha256'), 'w').write(h + '\n')

    prim = ev[ev.track_class.isin(('P1', 'P2'))].hotspot.tolist()
    ctrl = [c.strip() for c in a.controls.split(',') if c.strip()]
    targets = prim + [c for c in ctrl if c not in prim]
    with open(a.targets_out, 'w') as f:
        f.write('\n'.join(targets) + '\n')
    print(ev.track_class.value_counts().to_string())
    for cls in ('P1', 'P2'):
        print(f'\n{cls}: ' + ', '.join(ev[ev.track_class == cls].hotspot))
    print(f'\nwrote {out} (sha256 {h}) and {a.targets_out} with '
          f'{len(targets)} targets: {len(prim)} of class P1 or P2 and '
          f'{len(targets) - len(prim)} named controls')

    # The classification must not be a restatement of the Courtillot criteria
    # count, which the existing paper already compares against the root
    # fraction: if it were, the new comparison would carry no new information.
    if 'count' in hs.columns:
        from scipy.stats import spearmanr
        m = hs.merge(ev[['hotspot', 'track_class']], on='hotspot', how='inner')
        rank = m.track_class.map({'P1': 3, 'P2': 2, 'P3': 1})
        rho, p = spearmanr(m['count'], rank)
        print(f'\ntrack class against the Courtillot criteria count: '
              f'rho = {rho:+.3f}, p = {p:.4f}, n = {len(m)}')


if __name__ == '__main__':
    main()
