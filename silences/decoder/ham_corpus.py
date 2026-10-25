"""The templated ham-traffic corpus, one of the two texts the character language model is trained on.

This corpus is a designed approximation: canonical QSO, contest and ragchew exchanges assembled from
templates that follow standard operating references. Template weights, abbreviation tiers and the
RST and callsign distributions are hand-assigned to be plausible; they are not fitted to real logs.

Sources the templates follow:
  - CW QSO structure and example exchanges:
      https://www.electronics-notes.com/articles/ham_radio/morse_code/cw-qso-morse-contact.php
      https://www.eucw.org/op/en/qso.html
      http://lidscw.org/resources/cq-qso-template
      https://naqcc.info/cw_qsos.html
  - CW abbreviations:
      https://rochesterham.org/rxcwg/CW%20Abbreviations.pdf
      https://www.qsl.net/w3df/cwabbrev.html
      https://www.hamradiobase.com/ham-radio-jargon-prosigns-cw/
  - Contest exchange conventions (5NN cut numbers, zero-padded serials, zones, "TEST <call>"):
      https://www.qsl.net/zs1an/contesting_faq.html
      https://www.galanto.com/en/cw-contests/

Prosigns (AR, BT, SK, KN) are single fused characters on air but are not in the Morse table, so the
decoder can only emit them as letter pairs; the corpus writes them that way (BT where operators
write '=').
"""
import numpy as np

# ---- token pools (tiered frequencies; weights are DESIGNED, see module docstring) ----
US_PFX = ["W", "K", "N", "KA", "KB", "KD", "WA", "WB", "AA", "AC", "AF"]
EU_PFX = ["G", "M", "DL", "DJ", "DK", "F", "I", "IK", "IZ", "IU", "EA", "ON", "PA", "OZ",
          "SM", "OH", "OK", "SP", "HB9", "OE", "LA", "EI", "GM", "GW", "YO", "S5", "9A", "HA"]
DX_PFX = ["JA", "JH", "JR", "VE", "VA", "VK", "ZL", "PY", "LU", "ZS", "4X", "UA", "UR"]
NAMES = ["JOHN", "BOB", "JIM", "MIKE", "DAVE", "TOM", "BILL", "JOE", "AL", "ED", "RON", "KEN",
         "DON", "STEVE", "PETE", "PAUL", "RICK", "DAN", "ART", "HANK", "FRED", "GARY", "CARL",
         "JACK", "LUKE", "WALT", "RAY", "LEE", "SAM", "MAX", "IVAN", "JAN", "LARS", "HANS",
         "PIERRE", "MARIO", "LUIGI", "PEDRO", "JOSE", "YURI"]
QTHS = ["BOSTON", "DENVER", "DALLAS", "SEATTLE", "PHOENIX", "TAMPA", "TUCSON", "AKRON",
        "LEEDS", "YORK", "BRISTOL", "HAMBURG", "MUNICH", "MILANO", "TORINO", "NAPOLI",
        "PARIS", "LYON", "MADRID", "PRAGUE", "OSLO", "BERGEN", "TOKYO", "OSAKA", "SYDNEY",
        "TORONTO", "QUEBEC", "SOFIA", "KRAKOW", "ROMA", "FIRENZE", "GENOVA", "PISA"]
RIGS = ["FT991", "IC7300", "K3", "TS590", "FT817", "IC705", "QRP RIG", "HOMEBREW", "KX3", "TS850"]
ANTS = ["DIPOLE", "VERTICAL", "3EL YAGI", "EFHW", "GP", "LONGWIRE", "LOOP", "2EL QUAD", "INV VEE"]
WX = ["SUNNY", "CLOUDY", "RAIN", "SNOW", "WINDY", "COLD", "HOT", "FB", "CLR", "OVERCAST"]
RSTS = ["599", "599", "599", "579", "589", "569", "559", "549", "479", "339", "459", "578"]
GREET = ["GM", "GE", "GA"]

def _callsign(rng):
    pool, w = (US_PFX, 0.4) if rng.random() < 0.4 else ((EU_PFX, 0.45) if rng.random() < 0.75
                                                        else (DX_PFX, 0.15))
    pfx = pool[rng.integers(0, len(pool))]
    digit = str(rng.integers(0, 10)) if rng.random() < 0.25 else str(rng.integers(1, 8))
    slen = int(rng.choice([1, 2, 2, 3, 3, 3]))
    sfx = "".join(chr(ord("A") + rng.integers(0, 26)) for _ in range(slen))
    return pfx + digit + sfx

def _serial(rng):
    return f"{int(rng.integers(1, 1200)):03d}"

def _pick(rng, pool):
    return pool[rng.integers(0, len(pool))]

# ---- templates (weights DESIGNED; 73/CQ/DE/RST-heavy per the references) ----
def _cq(rng):
    c = _callsign(rng)
    v = rng.random()
    if v < 0.45: return f"CQ CQ CQ DE {c} {c} {c} K"
    if v < 0.70: return f"CQ CQ DE {c} {c} K"
    if v < 0.85: return f"CQ DX CQ DX DE {c} {c} K"
    return f"QRZ? DE {c} K"

def _answer(rng):
    return f"{_callsign(rng)} DE {_callsign(rng)} {_callsign(rng)} KN"

def _exchange1(rng):
    c1, c2 = _callsign(rng), _callsign(rng)
    rst, n, q, g = _pick(rng, RSTS), _pick(rng, NAMES), _pick(rng, QTHS), _pick(rng, GREET)
    return (f"{c1} DE {c2} BT {g} OM ES TNX FER CALL BT UR RST {rst} {rst} BT "
            f"NAME {n} {n} BT QTH {q} {q} BT HW? AR {c1} DE {c2} KN")

def _exchange2(rng):
    c1, c2 = _callsign(rng), _callsign(rng)
    rst, n, q = _pick(rng, RSTS), _pick(rng, NAMES), _pick(rng, QTHS)
    return (f"R R {c1} DE {c2} BT TNX FB RPT DR {n} BT UR RST {rst} BT OP {_pick(rng, NAMES)} ES "
            f"QTH {q} BT RIG {_pick(rng, RIGS)} PWR {_pick(rng, ['100W','50W','5W','10W'])} ES ANT "
            f"{_pick(rng, ANTS)} BT WX HR {_pick(rng, WX)} TEMP {int(rng.integers(20, 95))}F BT HW? BK")

def _contest(rng):
    c1, c2 = _callsign(rng), _callsign(rng)
    v = rng.random()
    if v < 0.30: return f"TEST {c1} {c1}"
    if v < 0.55: return f"{c2} 5NN {int(rng.integers(1, 41))}"
    if v < 0.80: return f"{c2} TU 5NN {_serial(rng)}"
    return f"{c2} DE {c1} 5NN {int(rng.integers(1, 41))} {int(rng.integers(1, 41))} TU"

def _signoff(rng):
    c1, c2, n = _callsign(rng), _callsign(rng), _pick(rng, NAMES)
    v = rng.random()
    if v < 0.4: return f"TNX FER FB QSO {n} BT 73 73 ES CUL BT {c1} DE {c2} SK EE"
    if v < 0.7: return f"OK {n} TU 73 GL ES GUD DX BT {c1} DE {c2} SK"
    if v < 0.85: return f"73 ES HPE CUAGN OM SK EE"
    return f"TU 73 88 {c1} DE {c2} SK"

def _qcode_chatter(rng):
    c = _callsign(rng)
    v = rng.random()
    if v < 0.15: return "QRL? QRL?"
    if v < 0.30: return f"PSE QRS QRS {c} DE {_callsign(rng)}"
    if v < 0.45: return f"QSY {int(rng.integers(7000, 7200))} OK?"
    if v < 0.60: return "QRM HR OM PSE AGN AGN"
    if v < 0.70: return "QSB ON UR SIG PSE RPT"
    if v < 0.80: return f"QSL VIA BURO 73 DE {c}"
    if v < 0.88: return f"QRP HR PWR 5W ONLY HW CPY? DE {c}"
    if v < 0.94: return "QRN VY BAD HR SRI OM"
    return f"QRT NW GM ES TNX 73 DE {c} SK"

TEMPLATES = [(_cq, 0.20), (_answer, 0.10), (_exchange1, 0.20), (_exchange2, 0.15),
             (_contest, 0.15), (_signoff, 0.12), (_qcode_chatter, 0.08)]
_W = np.array([w for _, w in TEMPLATES]); _W = _W / _W.sum()


def generate(n_exchanges=40000, seed=199_100):
    """Return a list of templated exchange strings (pre-normalisation)."""
    rng = np.random.default_rng(seed)
    fns = [f for f, _ in TEMPLATES]
    return [fns[rng.choice(len(fns), p=_W)](rng) for _ in range(n_exchanges)]
