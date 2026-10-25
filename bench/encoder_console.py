#!/usr/bin/env python3
"""e2_encoder.py — E.2 helpers: console driver, DI capture of GP6/GP7, CAL-normalized tick
extraction, and the waveform comparator. The DAQ now LISTENS (transmitter measurement)."""
import os
import sys
import time
from pathlib import Path

import numpy as np
import serial

HERE = Path(__file__).resolve().parent
RATE = 2_000_000                 # 0.5 us/sample; after CAL normalization the floor is +/-0.25 us
CAL_US = 400_000                 # encoder CAL_TICKS
GP6, GP7 = 0, 1                  # DI bit0 <- GP6 (dot), bit1 <- GP7 (dash)
PORT = os.environ.get("ENCODER_PORT", "COM9")


# ---------------- console ----------------
def console():
    s = serial.Serial(PORT, 115200, timeout=0.3)
    time.sleep(0.3)
    s.reset_input_buffer()
    return s


def cmd(s, line, wait=0.35):
    s.write((line + "\r").encode())
    s.flush()
    time.sleep(wait)
    return s.read(s.in_waiting or 1).decode(errors="replace")


# ---------------- DI capture (finite, sized to the emit) ----------------
def capture_di(window_s, started=None):
    import nidaqmx
    from nidaqmx.constants import LineGrouping, AcquisitionType
    from nidaqmx.stream_readers import DigitalSingleChannelReader
    n = int(window_s * RATE)
    with nidaqmx.Task() as di:
        di.di_channels.add_di_chan("Dev1/port1/line0:1", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        di.timing.cfg_samp_clk_timing(RATE, sample_mode=AcquisitionType.FINITE, samps_per_chan=n)
        rdr = DigitalSingleChannelReader(di.in_stream)
        buf = np.zeros(n, dtype=np.uint32)
        di.start()
        if started is not None:
            started.set()          # acquisition is now live -> caller may trigger the emit
        rdr.read_many_sample_port_uint32(buf, number_of_samples_per_channel=n,
                                         timeout=window_s + 15)
    return buf


def capture_emit(s, text, window_s):
    """Arm the DI, wait until it is acquiring, trigger the emit by typing `text`, return the capture."""
    import threading
    started = threading.Event()
    result = {}

    def run():
        result["cap"] = capture_di(window_s, started)
    t = threading.Thread(target=run)
    t.start()
    started.wait(timeout=10)
    time.sleep(0.10)               # small margin past di.start()
    s.write((text + "\r").encode())
    s.flush()
    t.join()
    return result.get("cap")


def rising_edges(bit_line):
    """sample indices where the line goes 0->1."""
    x = bit_line.astype(np.int8)
    return np.flatnonzero((x[1:] == 1) & (x[:-1] == 0)) + 1


def falling_edges(bit_line):
    x = bit_line.astype(np.int8)
    return np.flatnonzero((x[1:] == 0) & (x[:-1] == 1)) + 1


def extract(cap):
    """From a raw DI capture: the CAL (first long GP6 pulse), then the completion spikes.
    Returns dict(cal_samples, cal_us_per_sample, marks=[(tick_us, ch)], first_rise_sample)."""
    g6 = (cap >> GP6) & 1
    g7 = (cap >> GP7) & 1
    r6, f6 = rising_edges(g6), falling_edges(g6)
    r7, f7 = rising_edges(g7), falling_edges(g7)
    if len(r6) == 0:
        return None
    # CAL = the first GP6 pulse whose width ~ 400 ms (>> the 1 ms spikes)
    cal_i = None
    for rs in r6:
        fs = f6[f6 > rs]
        if len(fs) == 0:
            continue
        w = fs[0] - rs
        if w > 0.3 * RATE:                       # > 300 ms -> the CAL, not a 1 ms spike
            cal_i = (rs, fs[0], w)
            break
    if cal_i is None:
        return None
    cal_rs, cal_fs, cal_w = cal_i
    us_per = CAL_US / cal_w                       # CAL-normalized: cancels DAQ-vs-encoder clock offset
    # spikes = all GP6/GP7 rising edges AFTER the CAL falls (skip the CAL rise itself)
    dot = [(int(rs), 0) for rs in r6 if rs > cal_fs]
    dash = [(int(rs), 1) for rs in r7 if rs > cal_fs]
    marks = sorted(dot + dash)
    if not marks:
        return None
    first = marks[0][0]
    out = [((s - first) * us_per, ch) for s, ch in marks]   # tick relative to first mark's rise
    return dict(cal_w=int(cal_w), cal_us_per=us_per, cal_ms=cal_w / RATE * 1000,
                marks=out, first_rise=first, n=len(out))


def compare(dev_marks, exp_ticks, exp_ch):
    """dev_marks = [(tick_us, ch)] relative to first mark; exp from cases.txt (absolute-from-start).
    Normalize exp to relative-to-first-mark, then per-mark tick error + channel check."""
    if len(dev_marks) != len(exp_ticks):
        return None, None, None
    e0 = exp_ticks[0]
    exp_rel = [t - e0 for t in exp_ticks]
    dev_t = [m[0] for m in dev_marks]
    dev_c = [m[1] for m in dev_marks]
    ch_ok = dev_c == list(exp_ch)
    errs = [dev_t[i] - exp_rel[i] for i in range(len(dev_t))]
    return ch_ok, errs, exp_rel


# ---------------- cases.txt ----------------
def load_cases():
    lines = (HERE.parent / "firmware" / "encoder" / "host" / "cases.txt").read_text().splitlines()
    n = int(lines[0]); i = 1; out = []
    for _ in range(n):
        hdr = lines[i].split()
        text = lines[i].split("|")[1]
        ticks = [int(x) for x in lines[i + 1].split()]
        ch = [int(x) for x in lines[i + 2].split()]
        i += 3
        out.append(dict(vid=hdr[0], tag=hdr[2], cfist=int(hdr[4]), seed=int(hdr[5]),
                        nm=int(hdr[6]), text=text, ticks=ticks, ch=ch))
    return out


def key_entry(s, c):
    cmd(s, f":fist {c['cfist']}")
    cmd(s, ":jit 1")
    cmd(s, f":seed {c['seed']}")


def emit_window_s(c):
    return 0.6 + c["ticks"][-1] / 1e6 + 1.2      # console latency + CAL + span + margin
