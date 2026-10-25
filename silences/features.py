"""The front end's time-unit estimate and the wide-window gap features (Sections 5 and 7).

The front end sees only mark-completion times. It estimates one time unit per transmission from the
shortest gaps, in integer arithmetic exactly as the RP2040 front end does, and every gap interval is
normalised by that estimate.

The feedforward segmenters see, for each gap i, a window of the gaps i-W..i+W: the log of each
normalised interval and the channel bit of the mark that ends it. With W = 10 that is 42 inputs.
The window is padded at the edges with a long interval and a dot.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

W = 10                         # half-width of the gap window
PAD_LOGX = np.log(12.0)        # edge padding: a long interval


def fe_tu_us(ticks_us):
    """The front end's time-unit estimate in microseconds, or -1 with fewer than two marks.

    The tenth-percentile positive interval (linearly interpolated in tenths) times 9/20, rounded
    half up, in integers only."""
    end = np.asarray(ticks_us, dtype=np.int64)
    if len(end) < 2:
        return -1
    isis = sorted(int(d) for d in np.diff(end) if d > 0)
    if not isis:
        return -1
    m = len(isis)
    pos = m - 1
    k, r = pos // 10, pos - 10 * (pos // 10)
    if r == 0 or k + 1 >= m:
        num, den = isis[k], 1
    else:
        num, den = 10 * isis[k] + r * (isis[k + 1] - isis[k]), 10
    return (num * 9 + (den * 20) // 2) // (den * 20)


def window_features(ticks_us, ch, tu_us, W=W):
    """(n_gaps, 2(2W+1)) features: the window of log(interval / tu) followed by the window of bits."""
    ticks = np.asarray(ticks_us, dtype=np.float64)
    ch = np.asarray(ch, dtype=np.int64)
    x = np.diff(ticks) / float(tu_us)
    logx = np.log(np.clip(x, 1e-3, None))
    bit = ch[1:].astype(np.float64)
    if len(x) == 0:
        return np.zeros((0, 2 * (2 * W + 1)))
    xp = np.concatenate([np.full(W, PAD_LOGX), logx, np.full(W, PAD_LOGX)])
    bp = np.concatenate([np.zeros(W), bit, np.zeros(W)])
    return np.concatenate([sliding_window_view(xp, 2 * W + 1), sliding_window_view(bp, 2 * W + 1)], axis=1)


def transmission_features(tx, W=W):
    """Window features of one corpus transmission, normalised by the front end's estimate."""
    return window_features(tx["ticks"], tx["ch"], fe_tu_us(tx["ticks"]), W)
