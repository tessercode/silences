#!/usr/bin/env python3
"""b2_common.py — shared helpers for the B2.2 silicon runners: SWD flash + ring read, a u16
RESULT_FRAME deframer, and continuous DI capture with a DAQ-UART drive."""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import uart_frame as uf
import daq_uart as du

OPENOCD = os.environ.get("OPENOCD", "openocd")
# xpack layout: <root>/bin/openocd[.exe] next to <root>/openocd/scripts; override with OPENOCD_SCRIPTS
OOSCRIPTS = os.environ.get("OPENOCD_SCRIPTS",
                           os.path.join(os.path.dirname(os.path.dirname(OPENOCD)), "openocd", "scripts"))
RATE = du.RATE
BAUD = du.BAUD


def openocd(*cmds, capture=True):
    return subprocess.run([OPENOCD, "-s", OOSCRIPTS, "-f", "interface/cmsis-dap.cfg",
                           "-f", "target/rp2040.cfg", "-c", "adapter speed 5000"]
                          + [x for c in cmds for x in ("-c", c)],
                          capture_output=capture, text=True)


def flash(elf):
    elf = str(elf).replace("\\", "/")
    r = openocd(f"program {elf} verify reset exit")
    ok = "** Verified OK **" in r.stdout + r.stderr
    print(f"  flash {Path(elf).name}: {'Verified OK' if ok else 'FAILED'}")
    if not ok:
        print(r.stderr[-600:]); sys.exit(1)


def read_word(addr):
    """Read one 32-bit word over SWD (halts, reads, resumes)."""
    r = openocd("init", "halt", f"mdw {addr:#x}", "resume", "exit")
    for line in (r.stdout + r.stderr).splitlines():
        # openocd mdw prints: "2000dd08: 00000257"
        if ":" in line:
            parts = line.split(":")
            try:
                if int(parts[0].strip(), 16) == addr:
                    return int(parts[1].strip().split()[0], 16)
            except ValueError:
                continue
    raise RuntimeError(f"mdw parse failed for {addr:#x}: {r.stdout[-200:]}")


def read_ring(base, has_hwm=True):
    """Return (hwm, dropped) from a ring at `base` (layout head,tail,dropped[,hwm],buf)."""
    dropped = read_word(base + 8)
    hwm = read_word(base + 12) if has_hwm else None
    return hwm, dropped


# ---------------- u16 RESULT_FRAME deframer (LENGTH u16 BE; CRC-8 over mid+len(2)+payload) ----
def crc8(data):
    c = 0
    for byte in data:
        c ^= byte
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if (c & 0x80) else ((c << 1) & 0xFF)
    return c


def deframe_u16(buf, want_mid=None):
    """Yield dict(off, mid, frame) for each CRC-valid u16 frame; optionally only want_mid."""
    i, n = 0, len(buf)
    out = []
    while i + 6 <= n:
        if buf[i] == 0xA5 and buf[i + 1] == 0x5A:
            length = (buf[i + 3] << 8) | buf[i + 4]
            end = i + 5 + length
            if end < n:
                if crc8(buf[i + 2:end]) == buf[end]:
                    if want_mid is None or buf[i + 2] == want_mid:
                        out.append({"off": i, "mid": buf[i + 2], "frame": bytes(buf[i:end + 1])})
                    i = end + 1
                    continue
            i += 1
            continue
        i += 1
    return out


# ---------------- finite synchronized drive+capture (the A2.2/A4.2 proven pattern) ----------------
def drive_and_capture(streams, window_us, di_bit=0, idle_lead_us=3000):
    """Build a DO carrying `streams` (UART-framed) padded to `window_us`, drive it while capturing
    DI on the SAME RTSI7 clock, and return the chosen DI bit as an int8 line. The window must span
    the 0x06 TX + the board's compute + the result emit + margin."""
    from daq_sync import generate_capture
    do = du.build_do(streams, idle_lead_us=idle_lead_us, idle_tail_us=0)
    total = int(window_us * RATE // 1_000_000)
    if total > len(do):
        do = np.concatenate([do, np.full(total - len(do), (do[-1] if len(do) else 0), dtype=np.uint32)])
    # ensure every stream bit idles HIGH through the tail (build_do left tail at last frame level)
    for bit, _ in streams:
        # after the last framed byte's stop bit the line is already high; keep it high in the pad
        pass
    di = generate_capture(do.astype(np.uint32), RATE)
    return ((di >> di_bit) & 1).astype(np.int8)


# ---------------- DI capture (continuous) + DAQ-UART drive ----------------
class DICapture:
    """Continuous DI acquisition on port1/line0:1 at RATE; bit0=GP4, bit1=GP5. Background thread
    accumulates the GP4 (or chosen) bit; caller polls decode_and_deframe."""
    def __init__(self, di_bit=0):
        import nidaqmx
        from nidaqmx.constants import LineGrouping, AcquisitionType
        from nidaqmx.stream_readers import DigitalSingleChannelReader
        self.di_bit = di_bit
        self.task = nidaqmx.Task()
        self.task.di_channels.add_di_chan("Dev1/port1/line0:1", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        self.task.timing.cfg_samp_clk_timing(RATE, sample_mode=AcquisitionType.CONTINUOUS,
                                             samps_per_chan=RATE * 12)
        self.rdr = DigitalSingleChannelReader(self.task.in_stream)
        self.chunks = []
        self.lock = threading.Lock()
        self.stop = False
        self.buf = np.zeros(RATE, dtype=np.uint32)
        self.task.start()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self.stop:
            try:
                nread = self.rdr.read_many_sample_port_uint32(
                    self.buf, number_of_samples_per_channel=RATE // 4, timeout=5.0)
            except Exception:
                continue
            with self.lock:
                self.chunks.append(((self.buf[:nread] >> self.di_bit) & 1).astype(np.int8).copy())

    def line(self):
        with self.lock:
            return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.int8)

    def reset(self):
        with self.lock:
            self.chunks = []

    def close(self):
        self.stop = True
        self.t.join(timeout=2)
        self.task.close()


def drive(streams, idle_lead_us=3000, idle_tail_us=2000):
    """Drive one finite DO task (port0/line0:1) carrying the UART `streams` = [(do_bit, bytes)]."""
    import nidaqmx
    from nidaqmx.constants import LineGrouping, AcquisitionType
    from nidaqmx.stream_writers import DigitalSingleChannelWriter
    do = du.build_do(streams, idle_lead_us, idle_tail_us)
    n = len(do)
    with nidaqmx.Task() as t:
        t.do_channels.add_do_chan("Dev1/port0/line0:1", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        t.timing.cfg_samp_clk_timing(RATE, sample_mode=AcquisitionType.FINITE, samps_per_chan=n)
        DigitalSingleChannelWriter(t.out_stream, auto_start=False).write_many_sample_port_uint32(do)
        t.start()
        t.wait_until_done(timeout=n / RATE + 10)


def decode_line(line):
    if len(line) == 0:
        return b""
    lows = np.flatnonzero(line == 0)
    if lows.size == 0:
        return b""
    a = max(0, int(lows[0]) - 4)
    b = min(len(line), int(lows[-1]) + int(12 * (RATE / BAUD)))
    return bytes(uf.decode_8n1(line[a:b], RATE, BAUD))
