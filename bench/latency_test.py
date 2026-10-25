#!/usr/bin/env python3
"""clf_a2_silicon.py — Test A (compute + latency, secondary) for spiking_gap9_{lif,mlp} on silicon.

Mirrors firmware/gap9_3c/a2_2_run.py: flash+verify the clf_bench_gp4 image over SWD, arm a 6535 DI
capture on the GP4 tap (Dev1/port1/line0, 1 MS/s), reset boot-fresh, capture the bench's stdio-UART
(uart1 TX=GP4 @115200), burst-wise 8N1 decode skipping >5 ms compute gaps, compare "vid ng lp..."
lines bit-exact to the golden, and measure per-vector latency = the compute silence BEFORE each
vector's UART burst (clf_bench_main.c: `clf_run(v)` [silent compute] then `printf` [burst]).

Usage: python clf_a2_silicon.py <lif|mlp> <clf_bench_gp4.elf> [window_s]
"""
import os
import hashlib
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]                     # the repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))
import uart_frame as uf                                          # noqa: E402  decode_8n1

OPENOCD = os.environ.get("OPENOCD", "openocd")
# xpack layout: <root>/bin/openocd[.exe] next to <root>/openocd/scripts; override with OPENOCD_SCRIPTS
OOSCRIPTS = os.environ.get("OPENOCD_SCRIPTS",
                           os.path.join(os.path.dirname(os.path.dirname(OPENOCD)), "openocd", "scripts"))
RATE, BAUD = 1_000_000, 115200
GAP_SAMPLES = 5000                                              # >5 ms silence = compute gap / burst edge


def openocd(*cmds):
    return subprocess.run([OPENOCD, "-s", OOSCRIPTS, "-f", "interface/cmsis-dap.cfg",
                           "-f", "target/rp2040.cfg", "-c", "adapter speed 5000"]
                          + [x for c in cmds for x in ("-c", c)], capture_output=True, text=True)


def flash(elf):
    r = openocd(f"program {str(elf).replace(chr(92), '/')} verify reset exit")
    ok = "** Verified OK **" in r.stdout + r.stderr
    print(f"  flash {Path(elf).name}: {'Verified OK' if ok else 'FAILED'}")
    if not ok:
        print(r.stderr[-500:]); sys.exit(1)


def capture_run(window_s):
    import nidaqmx
    from nidaqmx.constants import LineGrouping, AcquisitionType
    from nidaqmx.stream_readers import DigitalSingleChannelReader
    chunks = []
    with nidaqmx.Task("clf_a2") as di:
        di.di_channels.add_di_chan("Dev1/port1/line0:1", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        di.timing.cfg_samp_clk_timing(RATE, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=10_000_000)
        rdr = DigitalSingleChannelReader(di.in_stream)
        di.start()
        rst = subprocess.Popen([OPENOCD, "-s", OOSCRIPTS, "-f", "interface/cmsis-dap.cfg",
                                "-f", "target/rp2040.cfg", "-c", "adapter speed 5000",
                                "-c", "init", "-c", "reset run", "-c", "exit"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time(); buf = np.zeros(1_000_000, dtype=np.uint32)
        while time.time() - t0 < window_s:
            n = rdr.read_many_sample_port_uint32(buf, number_of_samples_per_channel=1_000_000, timeout=10.0)
            chunks.append((buf[:n] & 1).astype(np.int8).copy())
        rst.wait(timeout=10)
    return np.concatenate(chunks)


def split_bursts(line):
    """Return [(start_sample, end_sample)] for each active burst (bounded by >5 ms silence)."""
    lows = np.flatnonzero(line == 0)
    if lows.size == 0:
        return []
    splits = np.flatnonzero(np.diff(lows) > GAP_SAMPLES)
    starts = np.concatenate(([0], splits + 1))
    ends = np.concatenate((splits, [lows.size - 1]))
    return [(int(lows[s]), int(lows[e])) for s, e in zip(starts, ends)]


def decode_burst(line, a, b):
    spb = RATE / BAUD
    lo = max(0, a - 4); hi = min(len(line), b + int(12 * spb))
    return bytes(uf.decode_8n1(line[lo:hi], RATE, BAUD)).decode(errors="replace")


def fit(x, y_ms):
    x = np.asarray(x, float); y = np.asarray(y_ms, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (m, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ [m, c]; s2 = float(np.sum((y - yhat) ** 2)) / max(1, len(x) - 2)
    sxx = float(np.sum((x - x.mean()) ** 2))
    return m, c, 1.96 * (s2 / sxx) ** 0.5


def main():
    module, elf = sys.argv[1], sys.argv[2]
    window_s = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    ref = __import__("json").load(open(ROOT / "firmware" / "golden" / f"segmenter_{module}.json"))
    gold = {int(r["vid"]): r for r in ref["vectors"]}
    print(f"[{module}] clf_bench_gp4.elf sha256 {hashlib.sha256(Path(elf).read_bytes()).hexdigest()}")
    flash(elf)
    line = capture_run(window_s)
    bursts = split_bursts(line)

    # decode each burst; a burst may carry >1 line (printf back-to-back if compute<5ms) — parse all
    rows = []                          # (vid, ng, burst_start_sample, prev_burst_end_sample)
    prev_end = None
    ok_vids = set(); bad_vids = set(); n_val = 0; mism = None
    for (a, b) in bursts:
        txt = decode_burst(line, a, b)
        for ln in txt.splitlines():
            p = ln.split()
            if not p or not p[0].lstrip("-").isdigit():
                continue
            vid, ng = int(p[0]), int(p[1]); got = [int(x) for x in p[2:]]
            if vid not in gold:
                continue
            exp = [v for row in gold[vid]["lp3"] for v in row]
            if ng == gold[vid]["n_gaps"] and got == exp:
                if vid not in ok_vids:
                    n_val += len(got)
                ok_vids.add(vid)                        # dedupe: a line can decode twice at a burst edge
            else:
                bad_vids.add(vid)
                if mism is None:
                    mism = (vid, ng, got[:6], exp[:6])
            rows.append((vid, ng, a, prev_end))
        prev_end = b
    n_ok = len(ok_vids)
    print(f"[{module}] BIT-EXACT: {n_ok}/{len(gold)} vectors compute-correct, {n_val} lp3 values"
          + (f"  MISMATCH {mism}" if mism else ""))

    # latency = compute silence before each vector's burst = burst_start - prev_burst_end (skip the
    # first burst: its lead-in includes boot + sleep_ms(200), not a clean compute).
    lat = []                            # (n_gaps, latency_ms)
    seen = set()
    for (vid, ng, start, pend) in rows:
        if pend is None or vid in seen:
            continue
        seen.add(vid)
        lat.append((ng, (start - pend) / RATE * 1000.0))
    res = {"module": module, "compute_ok": n_ok, "nvec": len(gold), "latency": lat}
    if len(lat) >= 3:
        xs = [g for g, _ in lat]; ys = [ms for _, ms in lat]
        m, c, ci = fit(xs, ys)
        mx = max(lat, key=lambda t: t[0])
        print(f"[{module}] latency: {m:.3f} ± {ci:.3f} ms/gap + {c:.2f} ms fixed  "
              f"(max {mx[0]} gaps -> {mx[1]:.1f} ms; n={len(lat)} pts)")
        res.update(slope_ms_per_gap=m, intercept_ms=c, ci_slope=ci,
                   max_gap=mx[0], max_gap_ms=mx[1])
    (ROOT / "outputs" / f"latency_{module}.json").write_text(__import__("json").dumps(res, indent=1))
    print(f"[{module}] === TEST A {'PASS' if n_ok == len(gold) else 'REVIEW'} "
          f"({n_ok}/{len(gold)} lp3 bit-exact) ===")


if __name__ == "__main__":
    main()
