"""The threshold readout (Section 6.4): the form every adaptive-threshold Morse recognizer takes.

Each interval, in units of a time-unit estimate, meets two thresholds chosen by the channel bit of the mark
that closes it (a low and a high one for dots, another pair for dashes). Two sigmoids of steepness k give
graded "past the low threshold" and "past the high threshold" responses s_lo and s_hi, and an ordinal
readout turns them into the three class probabilities the beam consumes:

    intra-letter 1 - s_lo,   inter-letter s_lo - s_hi,   inter-word s_hi.

Four forms are scored (evaluation/nonlearned.py): the fixed thresholds of the earlier rule-based reader at
the front end's estimate and at the estimate they were designed for (timing.estimate_tu_us); the same five
parameters fitted by per-gap cross-entropy on a training corpus (fit); and an adaptive variant that first
subtracts a per-transmission estimate of the dot and dash on-times (lp3_adaptive).
"""
import numpy as np

ONE = 65536
EPS = 1e-6
INT32_MIN = -(2 ** 31)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def normalized_intervals(ticks_us, tu_us):
    """Interval / time unit per gap; tu_us is one value per transmission or one per gap."""
    return np.diff(np.asarray(ticks_us, dtype=np.float64)) / np.asarray(tu_us, dtype=np.float64)


class Params:
    """theta_dot and theta_dash are (low, high) thresholds in time units; k is the shared steepness."""

    def __init__(self, theta_dot=(3.0, 6.0), theta_dash=(5.0, 8.0), k=1.5, temperature=1.0):
        self.theta_dot = tuple(theta_dot)
        self.theta_dash = tuple(theta_dash)
        self.k = float(k)
        self.temperature = float(temperature)


def class_probs(x, bit, p):
    """(n, 3) probabilities [intra-letter, inter-letter, inter-word]."""
    x = np.asarray(x, dtype=np.float64)
    bit = np.asarray(bit, dtype=np.float64)
    lo = np.where(bit >= 0.5, p.theta_dash[0], p.theta_dot[0])
    hi = np.where(bit >= 0.5, p.theta_dash[1], p.theta_dot[1])
    s_lo, s_hi = _sigmoid(p.k * (x - lo)), _sigmoid(p.k * (x - hi))
    P = np.stack([1.0 - s_lo, np.clip(s_lo - s_hi, 0.0, None), s_hi], axis=-1)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def _q16(P, temperature):
    q = np.clip(np.rint(temperature * np.log(P + EPS) * ONE).astype(np.int64), INT32_MIN, -1)
    return [[int(v) for v in row] for row in q]


def lp3(x, bit, p):
    """Per-gap Q16.16 log-score rows for the beam."""
    return _q16(class_probs(x, bit, p), p.temperature)


def auto_onsets(x, bit, plow=8.0):
    """Per-channel element on-time, per transmission: a channel's shortest intervals are one unit of gap plus
    the on-time, so the on-time is a low percentile of that channel's intervals minus one."""
    x = np.asarray(x, float); bit = np.asarray(bit, float)

    def onset(b, dflt):
        m = bit == b
        if m.sum() < 2:
            return dflt
        return float(np.clip(np.percentile(x[m], plow) - 1.0, 0.3, 8.0))
    return onset(0, 1.0), onset(1, 3.0)


def class_probs_adaptive(x, bit, onsets, k, t_lo=2.0, t_hi=5.0):
    """The ordinal readout on the gap estimate: the per-channel on-time subtracted, then two shared thresholds."""
    x = np.asarray(x, float); bit = np.asarray(bit, float)
    gap = x - np.where(bit >= 0.5, onsets[1], onsets[0])
    s_lo, s_hi = _sigmoid(k * (gap - t_lo)), _sigmoid(k * (gap - t_hi))
    P = np.stack([1.0 - s_lo, np.clip(s_lo - s_hi, 0.0, None), s_hi], axis=-1)
    return P / np.clip(P.sum(axis=-1, keepdims=True), EPS, None)


def lp3_adaptive(x, bit, k=3.0, t_lo=2.0, t_hi=5.0, temperature=1.0, plow=8.0):
    return _q16(class_probs_adaptive(x, bit, auto_onsets(x, bit, plow), k, t_lo, t_hi), temperature)


# ---- fitting the five parameters ----
def cross_entropy(x, bit, labels, p):
    P = class_probs(x, bit, p)
    return float(-np.mean(np.log(np.clip(P[np.arange(len(labels)), labels], EPS, None))))


def _unpack(z):
    """[dot low, log dot width, dash low, log dash width, log k] -> Params, each clipped to a sane range."""
    dot_lo = float(np.clip(z[0], 1.0, 9.0))
    dot_hi = dot_lo + float(np.clip(np.exp(z[1]), 0.5, 8.0))
    dash_lo = float(np.clip(z[2], 2.0, 11.0))
    dash_hi = dash_lo + float(np.clip(np.exp(z[3]), 0.5, 8.0))
    return Params((dot_lo, dot_hi), (dash_lo, dash_hi), float(np.clip(np.exp(z[4]), 0.8, 6.0)), 1.0)


def fit(x, bit, labels, init):
    """Per-gap cross-entropy over the five parameters: Nelder-Mead from init, then a Powell polish."""
    from scipy.optimize import minimize
    x = np.asarray(x, dtype=np.float64); bit = np.asarray(bit, dtype=np.float64); labels = np.asarray(labels, dtype=np.int64)
    z0 = np.array([init.theta_dot[0], np.log(max(init.theta_dot[1] - init.theta_dot[0], 0.5)),
                   init.theta_dash[0], np.log(max(init.theta_dash[1] - init.theta_dash[0], 0.5)), np.log(init.k)])

    def obj(z):
        return cross_entropy(x, bit, labels, _unpack(z))
    best = minimize(obj, z0, method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-4, fatol=1e-7))
    polished = minimize(obj, best.x, method="Powell", options=dict(maxiter=6000))
    if polished.fun < best.fun:
        best = polished
    return _unpack(best.x)
