"""The five synthetic operators (Section 4, Table 1).

Each operator is a sending style: speed (wpm), dash-to-dot ratio r, per-element and per-gap timing
jitter in time units (sigma_elem, sigma_gap), and the spread of per-word speed drift (omega). The
table lives in one place, the RP2040 encoder's firmware/encoder/fists.h, and is parsed from there so
the device and the software always render the same operators. The C literals are decimal doubles
that gcc and Python's float() parse to the same IEEE-754 value, so the parse is exact.
"""
import re

from . import FIRMWARE

FISTS_H = FIRMWARE / "encoder" / "fists.h"
FIELDS = ("wpm", "r", "sigma_elem", "sigma_gap", "omega")

_TABLE = re.compile(r"static const enc_fist_t ENC_FISTS\[\]\s*=\s*\{(.*?)\n\};", re.S)
_ROW = re.compile(r'\{\s*"([a-z_0-9]+)"' + r"\s*,\s*([-\d.eE+]+)" * 5 + r"\s*\}")


def parse_fists_h(path=FISTS_H):
    """[(name, (wpm, r, sigma_elem, sigma_gap, omega)), ...] in table order."""
    m = _TABLE.search(path.read_text())
    assert m, f"ENC_FISTS[] not found in {path}"
    rows = [(r[0], tuple(float(x) for x in r[1:])) for r in _ROW.findall(m.group(1))]
    names = [n for n, _ in rows]
    assert rows and len(set(names)) == len(names), f"bad operator table in {path}: {names}"
    return rows


# name -> dict(name, wpm, r, sigma_elem, sigma_gap, omega), in table order. The order matters: a
# corpus renders the operators in this order from one random generator.
OPERATORS = {name: dict(name=name, **dict(zip(FIELDS, vals))) for name, vals in parse_fists_h()}
