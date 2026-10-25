"""The RP2040 encoder board's renderer, reproduced bit for bit (firmware/encoder).

The software corpora (silences.corpus) draw their jitter from numpy. The encoder board cannot, so
it has its own generator: xoshiro128** seeded through splitmix64, and Gaussians by Box-Muller with a
square root, logarithm and cosine written in basic floating-point operations only, so that the board's
double arithmetic and CPython's agree exactly (both are IEEE-754 with correctly rounded + - * /, and
neither depends on libm's log or cos). This module is the board's specification: given the same
seed, it produces the same mark-completion times, microsecond for microsecond.

The draw order, one stream per sentence: per word, the speed drift (if omega > 0); per element, the
duration jitter; per intra-letter gap and after every letter, the gap jitter. Only the nominal gap
after a word's last letter differs between the single-word and sentence renderers (3 vs 7 time units).
"""
import math

from .morse import MORSE_TABLE

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
PI = 3.141592653589793
TWO_PI = 6.283185307179586
HALF_PI = 1.5707963267948966
LN2 = 0.6931471805599453
T_THRESH = 1.92
INTER_GAP = 3.0
WORD_GAP_TU = 7.0
SPIKE_W_US = 1000              # the width of a completion spike on the wire
CAL_US = 400_000               # the calibration pulse that opens every transmission
POST_CAL_GAP_US = 20_000


def _rotl32(x, k):
    return ((x << k) | (x >> (32 - k))) & M32


class Rng:
    """xoshiro128**, its four words seeded by splitmix64."""
    __slots__ = ("s",)

    def __init__(self, seed):
        z, st = seed & M64, [0, 0, 0, 0]
        for i in range(4):
            z = (z + 0x9E3779B97F4A7C15) & M64
            x = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
            x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
            st[i] = (x ^ (x >> 31)) & M32
        self.s = st

    def next_u32(self):
        s = self.s
        result = (_rotl32((s[1] * 5) & M32, 7) * 9) & M32
        t = (s[1] << 9) & M32
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl32(s[3], 11)
        return result

    def unit(self):
        """Uniform on (0, 1), never 0 or 1."""
        return (float(self.next_u32()) + 0.5) * (1.0 / 4294967296.0)

    def normal(self):
        u1, u2 = self.unit(), self.unit()
        return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2)


def sqrt(x):
    """Six Newton steps on the mantissa in [0.5, 2)."""
    if x <= 0.0:
        return 0.0
    m, e = math.frexp(x)
    if e & 1:
        m *= 2.0
        e -= 1
    y = 1.0
    for _ in range(6):
        y = 0.5 * (y + m / y)
    return math.ldexp(y, e >> 1)


def log(x):
    """log(m 2^e) = e ln 2 + 2 atanh((m - 1)/(m + 1)), mantissa in [0.707, 1.414), nine series terms."""
    m, e = math.frexp(x)
    if m < 0.7071067811865476:
        m *= 2.0
        e -= 1
    f = (m - 1.0) / (m + 1.0)
    f2 = f * f
    term = s = f
    for k in (3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0):
        term *= f2
        s += term / k
    return float(e) * LN2 + 2.0 * s


def cos(theta):
    """Reduced to [0, pi/2] by symmetry, then a degree-18 Taylor polynomial in Horner form."""
    a = theta
    while a >= TWO_PI:
        a -= TWO_PI
    while a < 0.0:
        a += TWO_PI
    if a > PI:
        a = TWO_PI - a
    sign = 1.0
    if a > HALF_PI:
        a = PI - a
        sign = -1.0
    a2 = a * a
    c = 1.0 - a2 * (1.0 / 2.0 - a2 * (1.0 / 24.0 - a2 * (1.0 / 720.0 - a2 * (1.0 / 40320.0 - a2 * (
        1.0 / 3628800.0 - a2 * (1.0 / 479001600.0 - a2 * (1.0 / 87178291200.0 - a2 * (
            1.0 / 20922789888000.0 - a2 * (1.0 / 6402373705728000.0)))))))))
    return sign * c


def clean_text(s):
    """The encoder's input rule: upper-case; keep A-Z and 0-9; anything else becomes a space; collapse."""
    return " ".join("".join(c if ("A" <= c <= "Z") or ("0" <= c <= "9") else " " for c in s.upper()).split())


def render(text, wpm, r, sigma_elem, sigma_gap, omega, seed, sentence=True):
    """Render text from one generator seeded with seed.

    Returns (ticks [us], channels, letter index per mark, word index per mark, words). With
    sentence=False only the first word is rendered and every letter gap is nominally 3 time units
    (the board's single-word mode)."""
    rng = Rng(seed)
    words = [w for w in clean_text(text).split(" ") if w]
    if not sentence:
        words = words[:1]
    t_u_base = 1.2 / wpm
    t_cursor, gletter = 0.0, -1
    ticks, chans, lidx, widx = [], [], [], []
    for wi, word in enumerate(words):
        if omega > 0.0:
            w = 1.0 + omega * rng.normal()
            omega_word = w if w > 0.2 else 0.2
        else:
            omega_word = 1.0
        t_u = t_u_base * omega_word
        t_thresh, floor_dur = T_THRESH * t_u, 0.1 * t_u
        sigma_e, sigma_g = sigma_elem * t_u, sigma_gap * t_u
        for ci, c in enumerate(word):
            els = MORSE_TABLE.get(c)
            if els is None:
                continue
            gletter += 1
            for e in range(len(els)):
                t_dur = ((1.0 * t_u) if els[e] == "." else (r * t_u)) + ((sigma_e * rng.normal()) if sigma_e > 0.0 else 0.0)
                if t_dur < floor_dur:
                    t_dur = floor_dur
                t_spike = t_cursor + t_dur
                ticks.append(round(t_spike * 1e6))         # half to even, as the board's lrint
                chans.append(0 if t_dur < t_thresh else 1)
                lidx.append(gletter)
                widx.append(wi)
                if e < len(els) - 1:
                    g = 1.0 * t_u + ((sigma_g * rng.normal()) if sigma_g > 0.0 else 0.0)
                    t_cursor = t_spike + (g if g >= floor_dur else floor_dur)
                else:
                    t_cursor = t_spike
            noise = (sigma_g * rng.normal()) if sigma_g > 0.0 else 0.0
            gap_nom = WORD_GAP_TU if (sentence and ci == len(word) - 1) else INTER_GAP
            gl = gap_nom * t_u + noise
            t_cursor += gl if gl >= floor_dur else floor_dur
    return ticks, chans, lidx, widx, words


def waveform(ticks, chans, spike_w_us=SPIKE_W_US):
    """The encoder's output as [[rise_us, fall_us, ch], ...]: a 1 ms spike at each mark's completion."""
    return [[int(t), int(t) + int(spike_w_us), int(c)] for t, c in zip(ticks, chans)]
