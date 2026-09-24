"""How each model tag is written for a reader. One place, so nothing drifts.

Figure S1's legend said GLADM35 and RevealLO 30km while Table S2 and the text said
GLAD-M35 and RevealLO, 30 km sampling. Both were typed independently and neither was
wrong on its own terms, which is exactly how a reader ends up wondering whether two
names mean two models.

`display` is the full form for a table or a caption; `short` is the compact form for
a legend, where the full form would not fit.
"""
import re

DISPLAY = {
    'RevealLO': 'RevealLO',
    'RevealLO_30km': 'RevealLO, 30 km sampling',
    'REVEAL': 'REVEAL',
    'GLADM35': 'GLAD-M35',
    'SPiRaL': 'SPiRaL',
    'SEMUCB-WM1': 'SEMUCB-WM1',
}

SHORT = dict(DISPLAY, **{'RevealLO_30km': 'RevealLO, 30 km'})


def display(tag):
    """The full name, for a table or a caption."""
    return DISPLAY.get(tag, str(tag).replace('_', ' '))


def short(tag):
    """The compact name, for a legend or an axis."""
    return SHORT.get(tag, str(tag).replace('_', ' '))


def hotspot(name):
    """A hotspot as a reader sees it. The catalogue key is kept everywhere a file is
    joined on it; only printed names change. Courtillot et al. (2003) write
    Kerguelen(Heard) without a space, which in a table or a panel title reads as a
    typo, and the table and the atlas printed it two different ways."""
    return re.sub(r'(?<=\S)\(', ' (', str(name))
