"""The front-end board (firmware/front_end), reproduced byte for byte.

The front end watches the encoder's two channel lines, timestamps each mark's rise and fall in
microseconds, and splits the stream into transmissions at any silence longer than 1.75 s. For each
transmission it sends one 0x06 frame: the marks' completion times relative to the first rise, their
channel bits, and its time-unit estimate (features.fe_tu_us). A transmission of more than 256 marks
is replaced by an overflow frame.
"""
import numpy as np

from . import frames
from .features import fe_tu_us

IDLE_SENTENCE_US = 1_750_000       # silence that ends a transmission


def split_transmissions(rise_us, fall_us, g_idle=IDLE_SENTENCE_US):
    """[(first mark, last mark), ...] for each transmission, and the boundaries between them.

    A silence longer than g_idle ends a transmission; so does a mark held longer than g_idle (it
    is dropped, together with any capture it interrupts the start of)."""
    n = len(fall_us)
    assert (np.diff(fall_us) > 0).all(), "mark falls must be strictly increasing"
    cap_start, cap_n = 0, 0
    captures, boundaries = [], []
    n_midmark = n_orphan = 0
    for j in range(n):
        if cap_n > 0:
            s = rise_us[j] - fall_us[j - 1]
            if s > g_idle:
                captures.append((cap_start, j - 1))
                boundaries.append(dict(last=j - 1, next=j, silence_us=int(s), midmark=False))
                cap_start, cap_n = j, 0
        d = fall_us[j] - rise_us[j]
        if d > g_idle:
            if cap_n > 0:
                captures.append((cap_start, j - 1))
                boundaries.append(dict(last=j - 1, next=j + 1, silence_us=int(d), midmark=True))
                n_midmark += 1
            else:
                n_orphan += 1
            cap_start, cap_n = j + 1, 0
            continue
        if cap_n == 0:
            cap_start = j
        cap_n += 1
    if cap_n > 0:
        captures.append((cap_start, n - 1))
    return captures, boundaries, n_midmark, n_orphan


def frame_for_capture(seq, rise_us, fall_us, chans, i0, i1):
    """The 0x06 frame for marks i0..i1, and a record of what it carries (None for an overflow)."""
    n_obs = i1 - i0 + 1
    if n_obs > frames.N_MARKS_MAX:
        return frames.frame(frames.MODULE_FE, frames.overflow_payload(seq, n_obs)), None
    t0 = int(rise_us[i0])
    marks = [(int(fall_us[i]) - t0, int(chans[i])) for i in range(i0, i1 + 1)]
    tu = fe_tu_us([m[0] for m in marks])
    return (frames.frame(frames.MODULE_FE, frames.fe_payload(seq, tu, marks)),
            dict(seq=seq, tu_us=int(tu), n_marks=len(marks), marks=marks))
