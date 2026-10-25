#!/usr/bin/env python3
"""daq_uart.py — B2.2 rig: bit-bang a 9600 8N1 UART on NI-6535 DO lines, capture on DI, decode back.

B1.2's DAQ drove GPIO *edges*; B2.2 drives a *UART* into a real receiver. 8N1: idle-high, one
start bit (low), 8 data bits LSB-first, one stop bit (high).

⚠️ RATE: at 1 MS/s a bit is 1e6/9600 = 104.167 samples — non-integer, so byte boundaries drift
~1.7 µs/byte (0.16%). **960 kS/s gives exactly 100 samples/bit** — chosen to remove that drift
(well inside a UART's ±3%, but free to avoid).

DO bit map (generate_capture drives port0/line0:1): bit0 -> GP2, bit1 -> GP3.
DI bit map (captures port1/line0:1):               bit0 -> GP4, bit1 -> GP5.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uart_frame as uf          # decode_8n1 (wire-validated)

RATE = 960_000                   # 100 samples/bit at 9600 — exact, no boundary drift
BAUD = 9600
SPB = RATE // BAUD               # 100


def uart_bits(data: bytes):
    """Yield the line level (0/1) sample-by-sample for `data` as 8N1, idle-high framing."""
    for byte in data:
        yield from (0,) * SPB                        # start bit
        for i in range(8):                           # 8 data bits, LSB first
            yield from ((byte >> i) & 1,) * SPB
        yield from (1,) * SPB                        # stop bit


def build_do(streams, idle_lead_us=2000, idle_tail_us=2000, other_low=True):
    """Build a DO uint32 sample array. `streams` = list of (do_bit, bytes). Each stream's bytes are
    UART-framed on its DO bit; all bits idle HIGH between/around bytes EXCEPT bits not in any stream,
    which idle LOW when other_low (so GP2=GO stays low -> no accidental replay). Streams are laid
    out concurrently from t=0 (for the router's two-line drive); pad shorter ones with idle-high."""
    per = []
    maxlen = 0
    for bit, data in streams:
        arr = np.array(list(uart_bits(data)), dtype=np.uint8)
        per.append((bit, arr))
        maxlen = max(maxlen, len(arr))
    lead = int(idle_lead_us * RATE // 1_000_000)
    tail = int(idle_tail_us * RATE // 1_000_000)
    n = lead + maxlen + tail
    do = np.zeros(n, dtype=np.uint32)
    used = {b for b, _ in streams}
    # idle-high for every stream bit across the whole window; then overlay framed bytes
    for bit, _ in streams:
        do[:] |= (0)  # noop; explicit idle set below
    for bit, arr in per:
        line = np.ones(n, dtype=np.uint8)            # idle high
        line[lead:lead + len(arr)] = arr
        do |= (line.astype(np.uint32) << bit)
    return do


def bytes_to_do_single(data, do_bit, idle_lead_us=2000, idle_tail_us=2000):
    return build_do([(do_bit, data)], idle_lead_us, idle_tail_us)


def selftest():
    """Computational round-trip: encode -> samples -> decode_8n1 -> compare. Proves the codec +
    sample construction independent of hardware (the hardware loopback is gap9 vector 0, per the
    task: 'gap9 proves the rig')."""
    msg = bytes(range(0, 256)) + b"\xA5\x5A\x06\x00\x09" + bytes([0xFF, 0x00, 0x7F, 0x80])
    do = bytes_to_do_single(msg, do_bit=1)           # GP3, the gap9 0x06 line
    line = (do >> 1) & 1                              # what a DI on that node would sample
    got = bytes(uf.decode_8n1(line.astype(np.int8), RATE, BAUD))
    ok = got == msg
    print(f"codec round-trip: {len(got)}/{len(msg)} bytes, {'OK' if ok else 'FAIL'}")
    # also prove GP2 (bit0) stays low throughout (GO must not fire)
    print(f"GP2(bit0) max level over window: {int((do & 1).max())} (must be 0)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
