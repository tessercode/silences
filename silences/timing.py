"""Time-unit estimates other than the front end's (Sections 5.2, 6.4 and 10.3).

The pipeline of the paper normalises every interval by the front end's single estimate per transmission
(features.fe_tu_us). Two experiments need others:

    estimate_tu_us   the resonator-bank estimate the earlier rule-based reader was designed for: a bank of
                     damped resonators driven by the completion spikes locks onto the 2-unit cadence of
                     dot-plus-gap, and the unit is half the winning period. It sits at the true unit, where
                     the front end's estimate sits at about 0.85 of it.
    local_tu_us      the same estimate from a short window of marks around each gap (centred, or trailing
                     for a streaming estimate): one unit per gap, which follows an operator's drift.

The per-word oracle of Section 10.3 is not an estimate; it is the renderer's true unit (corpus tx["tu_words_us"]).
"""
import numpy as np

from .features import fe_tu_us


def _anchor_us(ticks_us, pctile=5.0):
    """A coarse 2-unit anchor from a low percentile of the intervals; it only centres the resonator band."""
    isis = np.diff(np.asarray(ticks_us, dtype=np.float64))
    isis = isis[isis > 0]
    if isis.size == 0:
        return None
    return float(np.percentile(isis, pctile))


def resonator_energy(ticks_us, periods_us, Q=4.0):
    """Drive each resonator with a unit impulse at every spike; the summed response magnitude per period."""
    t = np.asarray(ticks_us, dtype=np.float64)
    P = np.asarray(periods_us, dtype=np.float64)
    w = 2.0 * np.pi / P
    gamma = w / (2.0 * Q)                     # Q cycles of memory
    z = np.zeros(len(P), dtype=np.complex128)
    energy = np.zeros(len(P), dtype=np.float64)
    tprev = t[0]
    for tj in t:
        z = z * np.exp((1j * w - gamma) * (tj - tprev))
        z += 1.0
        energy += np.abs(z)
        tprev = tj
    return energy


def _parabolic_peak(logP, E, k):
    if k <= 0 or k >= len(E) - 1:
        return logP[k]
    y0, y1, y2 = E[k - 1], E[k], E[k + 1]
    denom = (y0 - 2 * y1 + y2)
    if denom == 0:
        return logP[k]
    return logP[k] + 0.5 * (y0 - y2) / denom * (logP[k + 1] - logP[k])


def estimate_tu_us(ticks_us, Q=4.0, band=(0.6, 1.7), n=20):
    """The resonator-bank time unit in microseconds, one per transmission (the front end's with under three marks)."""
    t = np.asarray(ticks_us, dtype=np.float64)
    if len(t) < 3:
        return float(fe_tu_us(ticks_us))
    anchor = _anchor_us(t)
    if anchor is None or anchor <= 0:
        return float(fe_tu_us(ticks_us))
    P = np.geomspace(anchor * band[0], anchor * band[1], n)
    E = resonator_energy(t, P, Q)
    logP = np.log(P)
    return float(np.exp(_parabolic_peak(logP, E, int(np.argmax(E)))) / 2.0)


def local_tu_us(ticks_us, half_window=5, Q=4.0, band=(0.6, 1.7), n=16, causal=False):
    """One resonator-bank estimate per gap, from the marks within half_window of it (causal: only those before)."""
    t = np.asarray(ticks_us, dtype=np.float64)
    N = len(t)
    if N < 3:
        return np.full(max(N - 1, 0), float(fe_tu_us(ticks_us)))
    glob = estimate_tu_us(t, Q=Q, band=band, n=n)
    out = np.empty(N - 1, dtype=np.float64)
    for i in range(N - 1):
        a = max(0, i - half_window)
        b = (i + 2) if causal else min(N, i + 2 + half_window)
        w = t[a:b]
        out[i] = estimate_tu_us(w, Q=Q, band=band, n=n) if len(w) >= 3 else glob
    return out
