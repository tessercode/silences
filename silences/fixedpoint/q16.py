"""Q16.16 fixed-point arithmetic, exactly as the RP2040 firmware computes it.

Values are signed integers with 16 fractional bits. Products are formed in 64 bits and shifted back
(Python's >> floors on negative numbers, as an arithmetic shift does in C). Nonlinearities are
lookup tables with integer linear interpolation, clamped at both ends. The tables are built here from
floating-point functions once; everything at inference is integer. Every function has a vectorised
numpy form, bit-identical to the scalar one, because evaluation runs it over large corpora.
"""
import math

import numpy as np

FRAC = 16
ONE = 1 << FRAC
HALF = ONE >> 1
INT32_MIN = -(2 ** 31)


def to_q(x):
    """Nearest Q16.16 integer (Python round: half to even)."""
    return int(round(x * ONE))


def fxmul(a, b):
    """Q16.16 product, truncated toward minus infinity."""
    return (a * b) >> FRAC


def fxmul_round(a, b):
    """Q16.16 product, rounded half up."""
    return ((a * b) + HALF) >> FRAC


class LUT:
    """func tabulated on [lo, hi] in n equal steps; lookups interpolate linearly in integers."""

    def __init__(self, func, lo, hi, n):
        self.lo_q, self.hi_q, self.n = to_q(lo), to_q(hi), n
        self.step_q = (self.hi_q - self.lo_q) // n
        assert self.lo_q + n * self.step_q == self.hi_q, "table span must be a whole number of steps"
        self.table = [to_q(func(lo + (hi - lo) * i / n)) for i in range(n + 1)]
        self._t = np.array(self.table, dtype=np.int64)

    def __call__(self, x_q):
        """Scalar lookup (Python ints)."""
        if x_q <= self.lo_q:
            return self.table[0]
        if x_q >= self.hi_q:
            return self.table[-1]
        off = x_q - self.lo_q
        idx = off // self.step_q
        frac = ((off - idx * self.step_q) << FRAC) // self.step_q
        t0, t1 = self.table[idx], self.table[idx + 1]
        return t0 + (((t1 - t0) * frac) >> FRAC)

    def vec(self, x_q):
        """Vectorised lookup over an int64 array."""
        x = np.asarray(x_q, dtype=np.int64)
        off = np.clip(x - self.lo_q, 0, self.n * self.step_q)
        idx = np.minimum(off // self.step_q, self.n - 1)
        frac = ((off - idx * self.step_q) << FRAC) // self.step_q
        t0, t1 = self._t[idx], self._t[idx + 1]
        out = t0 + (((t1 - t0) * frac) >> FRAC)
        return np.where(x <= self.lo_q, self._t[0], np.where(x >= self.hi_q, self._t[-1], out))


SIGMOID = LUT(lambda x: 1.0 / (1.0 + math.exp(-x)), -16.0, 16.0, 4096)
TANH = LUT(math.tanh, -8.0, 8.0, 4096)
EXP = LUT(math.exp, -16.0, 0.0, 4096)                # reaches -16 so the tail rounds to exactly 0
LOG_MANTISSA = LUT(math.log, 1.0, 2.0, 1024)
LN2_Q = to_q(math.log(2.0))


def fxlog(p_q):
    """log of a Q16.16 probability: exponent by shifting into [1, 2), mantissa from the table."""
    x, e = max(p_q, 1), 0
    while x < ONE:
        x <<= 1
        e -= 1
    while x >= (ONE << 1):
        x >>= 1
        e += 1
    return e * LN2_Q + LOG_MANTISSA(x)


def fxlog_vec(p_q):
    """Vectorised fxlog."""
    x = np.maximum(np.asarray(p_q, dtype=np.int64), 1).copy()
    e = np.zeros_like(x)
    lo = x < ONE
    while lo.any():
        x, e = np.where(lo, x << 1, x), np.where(lo, e - 1, e)
        lo = x < ONE
    hi = x >= (ONE << 1)
    while hi.any():
        x, e = np.where(hi, x >> 1, x), np.where(hi, e + 1, e)
        hi = x >= (ONE << 1)
    return e * LN2_Q + LOG_MANTISSA.vec(x)


def softmax_lp3(logits):
    """(G, 3) Q16.16 logits -> (G, 3) Q16.16 log-posteriors, as the segmenter boards compute them:
    subtract the max, exp from the table (exp(0) exact), divide by the sum, log."""
    d = logits - logits.max(1, keepdims=True)
    e = np.where(d == 0, np.int64(ONE), EXP.vec(d))
    return fxlog_vec((e << FRAC) // np.maximum(e.sum(1, keepdims=True), 1))


def quantize_lp3(logp):
    """Float log-posteriors -> Q16.16 rows the beam takes (strictly negative, int32 range)."""
    q = np.clip(np.rint(np.asarray(logp) * ONE).astype(np.int64), INT32_MIN, -1)
    return [[int(v) for v in row] for row in q]


def scale_lp3(lp3, T_q):
    """The decoder weight T applied as a device would: ((T_q * lp) + 2^15) >> 16 on every value.

    No renormalisation is needed: the beam adds one row per gap, so a per-gap constant cancels.
    T_q = ONE is the identity."""
    if T_q == ONE or len(lp3) == 0:
        return lp3
    return [[((T_q * int(v)) + HALF) >> FRAC for v in row] for row in lp3]


def grid_q(x):
    """Q16.16 of a grid value (a decoder weight); asserts it is exact."""
    q = int(round(x * ONE))
    assert abs(q - x * ONE) < 1e-9, f"{x} is not exact in Q16.16"
    return q
