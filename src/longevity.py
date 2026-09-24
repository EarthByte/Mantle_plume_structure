"""Documented longevity of the surface volcanic record, for the 49 hotspots.

Two independent measures are carried, because there is no single compilation that
is both complete and unambiguous.

CATALOGUE AGE. The hotspot ages of Steinberger (2000), as tabulated by Torsvik
et al. (2006, their Table 2). This is the only global per-hotspot age column in
the literature we work from, and it covers 44 of the 49. Its definition is not
restated in that paper, and for several hotspots it substantially exceeds the
oldest radiometrically dated feature on the track - Hawaii 100 Ma against a
Meiji-Detroit age near 81 Ma, Marion 195 Ma, Jan Mayen 210 Ma - so it is best
read as a catalogue value for the presumed onset of the hotspot rather than as a
dated quantity.

DOCUMENTED OLDEST VOLCANISM. The age of the oldest volcanism actually dated and
attributed to the hotspot: the associated flood basalt or oceanic plateau where
Courtillot et al. (2003) make that link, otherwise the oldest radiometrically
dated edifice in the chain. This is defensible hotspot by hotspot but exists for
only 19 of the 49, and mixes two kinds of evidence, which is why it is used as a
check on the catalogue age rather than as the primary variable.

The chain ages are the oldest sample in the Pacific age table released with
Chase & Wessel (2021, Zenodo), file PHT2021_raw_data/PHT2021_pacific_ages.txt.
That table descends from Clouard & Bonneville (2005, their Table 2) and carries
later revisions, among them Sharp & Clague (2006) and O'Connor et al. (2013) for
the Hawaiian-Emperor chain, Koppers et al. (2011) and Heaton & Koppers (2019)
for Louisville and Samoa, and Konrad et al. (2018) for Rurutu. Sample names and
1-sigma analytical uncertainties are carried here so that every value can be
traced to the dated edifice it comes from.

Sources are recorded per entry so that every number in the paper can be traced.
"""
from __future__ import annotations

# hotspot -> (age Ma, source key)
# Steinberger (2000) catalogue, via Torsvik et al. (2006) Table 2
CATALOGUE = {
    'Jan Mayen': 210, 'Marion': 195, 'Darfur': 140, 'Tristan': 125,
    'Louisville': 120, 'Martin/Trindade': 120, 'Macdonald (Cook-Austral)': 120,
    'Great Meteor/New England': 120, 'Meteor': 120, 'Kerguelen(Heard)': 117,
    'St Helena': 100, 'Azores': 100, 'Hawaii': 100, 'Galapagos': 85,
    'Caroline': 80, 'Tibesti': 80, 'Fernando': 70, 'Easter': 68,
    'Reunion': 67, 'Canary': 65, 'Comores': 63, 'Iceland': 60,
    'Australia E': 50, 'Tasmanid (Tasman central)': 50,
    'Lord Howe (Tasman East)': 50, 'Juan de Fuca/Cobb': 43, 'Vema': 40,
    'Eifel': 40, 'Afar': 40, 'Balleny': 36, 'Cameroon': 31,
    'Juan Fernandez': 30, 'San Felix': 30, 'Bowie': 28,
    'Baja/Guadalupe': 25, 'Socorro': 25, 'Raton': 20, 'Hoggar': 20,
    'Cape Verde': 20, 'Yellowstone': 15, 'Samoa': 14, 'Marquesas': 9,
    'Pitcairn': 8, 'Tahiti/Society': 5,
}

# Oldest dated volcanism attributed to the hotspot.
# (age Ma, kind, provenance, 1-sigma Myr or None)
#   lip   = associated flood basalt or oceanic plateau (Courtillot & Renne 2003)
#   chain = oldest dated edifice in the volcanic chain
DOCUMENTED = {
    'Tristan':                  (133, 'lip', 'Parana-Etendeka; Courtillot & Renne 2003', None),
    'Great Meteor/New England': (124, 'chain', 'Monteregian Hills; Sleep 1990', None),
    'Kerguelen(Heard)':         (118, 'lip', 'Rajmahal/Kerguelen; Courtillot & Renne 2003', None),
    'Galapagos':                (89, 'lip', 'Caribbean plateau; Courtillot & Renne 2003', None),
    'Marion':                   (88, 'lip', 'Madagascar traps; Courtillot & Renne 2003', None),
    # Detroit Seamount, the oldest dated Emperor edifice in the table; the value
    # agrees with the 81 Ma of Wei et al. (2020). Meiji, further north, is not
    # dated in the table, so this is a lower bound on the chain.
    'Hawaii':                   (81.2, 'chain', 'Detroit Seamount, Hawaiian-Emperor; Chase & Wessel 2021', 1.3),
    'Louisville':               (78.8, 'chain', 'Osbourn Seamount (SOTW9-58-B), Louisville; Chase & Wessel 2021', 1.3),
    # CAUTION: the Chase & Wessel chain is Rurutu, whose present hotspot is
    # placed at Arago Seamount (Konrad et al. 2018), not Macdonald. Courtillot's
    # entry lumps the Cook-Austral province, so this age describes the province
    # rather than Macdonald itself, and Buff et al. (2021) argue both Macdonald
    # and Arago are long-lived. Flagged rather than dropped, and not to be quoted
    # as a Macdonald age without checking.
    'Macdonald (Cook-Austral)': (72.4, 'chain*', 'Burtaritari Seamount, Rurutu/Arago trail of the Cook-Austral province; Chase & Wessel 2021', 0.5),
    'Reunion':                  (66, 'lip', 'Deccan traps; Courtillot & Renne 2003', None),
    'Iceland':                  (61, 'lip', 'North Atlantic province; Courtillot & Renne 2003', None),
    'Afar':                     (30, 'lip', 'Ethiopian-Yemen traps; Courtillot & Renne 2003', None),
    'Juan de Fuca/Cobb':        (29.3, 'chain', 'Patton Seamount, Cobb; Chase & Wessel 2021', 1.0),
    'Bowie':                    (23.8, 'chain', 'Kodiak Seamount, Kodiak-Bowie; Chase & Wessel 2021', 0.4),
    'Yellowstone':              (16, 'lip', 'Columbia River basalts; Courtillot & Renne 2003', None),
    'Caroline':                 (13.9, 'chain', 'Truk (Fefan), Caroline; Chase & Wessel 2021', 0.3),
    'Samoa':                    (13.2, 'chain', 'Alia 125-04, Samoa; Chase & Wessel 2021', 0.2),
    'Pitcairn':                 (11.1, 'chain', 'Moruroa, Pitcairn; Chase & Wessel 2021', 0.03),
    'Marquesas':                (5.5, 'chain', 'Eiao, Marquesas; Chase & Wessel 2021', 0.1),
    'Tahiti/Society':           (4.2, 'chain', 'Maupiti, Society; Chase & Wessel 2021', 0.03),
}

NOT_FOUND = ['Ascension', 'Bermuda', 'Bouvet', 'Crozet/Pr. Edward', 'Discovery']
