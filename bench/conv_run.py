#!/usr/bin/env python3
"""conv_run.py — CONVERGENCE (seam 6): the 5-board chain, end-to-end.

Type a sentence into the encoder (COM9); read the decode off the decode board (DEC line on its
USB-CDC). Between, the DAQ is a PASSIVE listener on five DI taps — it drives NOTHING (the chain
drives itself). For each golden vector we key the encoder and compare:

  (Task 3 mechanical gate, per hop, seq-masked bit-exact vs the FROZEN golden)
    FE GP4  0x06  == expected_0x06_hex   (encoder -> FE)
    gap9    0x03  == expected_0x03_hex   (FE -> gap9)
    router  0x05  == expected_0x05_hex   (gap9 -> router)
  (Task 4 acceptance, vid-independent)
    decode  DEC   == expected_decode {text, seg, final_score_q16, beam_died}

WHY seq-masked: the FE stamps a per-transmission seq counter (0 at boot, seq++ each transmission)
as the frames' vector_id (payload[1:3] of every hop). Keyed in vid order from a boot-fresh FE with
the sole silent entry (vid 63, over-cap) last, seq==vid and the frames match byte-exact. The mask
(splice the golden vid into payload[1:3], recompute CRC) makes the *chain-correctness* verdict robust
to any single silent/dropped transmission that would otherwise desync seq for all that follow; seq==vid
is reported separately as the ordering check.

torch is unavailable on this host, so option-1's "compose the CAPTURED waveform -> expected decode"
(the A2/lp3 step needs the gap9 torch checkpoints) cannot run here. The frozen golden already carries
every hop's expected frame + decode (SHA-verified), so comparing the live chain's real output against
it IS the mechanical gate, torch-free, and localizes a break to a hop. The GP6/GP7 taps give the
chain input (drift/ISI) for the record; E.2 already characterized emission fully.

Run:  python -u conv_run.py --bringup            # Task 2: one word, walk every hop
      python -u conv_run.py --gate               # Task 3+4: all 63 emitting vectors
      python -u conv_run.py --gate --only 0,5,50 # a subset
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import serial

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_common as B                       # deframe_u16, crc8, decode_line, RATE(960k), BAUD, uf
import uart_frame as uf

import encoder_console as E                       # console(COM9), cmd, load_cases, key_entry

RATE = B.RATE                                # 960 kS/s — 100 samples/bit at 9600
BAUD = B.BAUD
DECODE_PORT = os.environ.get("DECODE_PORT", "COM8")   # decode board USB-CDC (B2.2 rig)
GOLDEN = HERE.parent / "firmware" / "golden" / "chain_bigru_n5.json"

# DI tap map (Dev1/port1/line0:4) — passive listeners
BIT_GP6, BIT_GP7 = 0, 1                      # encoder dot / dash (chain input; liveness + coarse drift)
BIT_0x06, BIT_0x03, BIT_0x05 = 2, 3, 4       # FE / gap9 / router forward frames


# ------------------------------------------------------------------ per-entry FINITE DI capture
# The 6535's continuous-mode DI would not acquire here (silent); a FINITE task with READ_ALL_AVAILABLE
# polling — the E.2-proven pattern — captures all five taps reliably and lets us stop early once the
# decode board's DEC line lands (so long sentences don't each burn the full max window).
def capture_entry(enc, dec, ecase, maxwin=50):
    """Start a FINITE DI capture on port1/line0:4, key the encoder, poll DI + the decode USB until the
    DEC line arrives (or maxwin), then stop. Returns (raw uint32 port stream, dec_line, latency_s)."""
    import nidaqmx
    from nidaqmx.constants import LineGrouping, AcquisitionType
    from nidaqmx.stream_readers import DigitalSingleChannelReader
    n = int(maxwin * RATE)
    CHUNK = RATE // 10                                       # 0.1 s per read
    cbuf = np.zeros(CHUNK, dtype=np.uint32)
    chunks = []
    stop = threading.Event()

    with nidaqmx.Task() as di:
        di.di_channels.add_di_chan("Dev1/port1/line0:4", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        di.timing.cfg_samp_clk_timing(RATE, sample_mode=AcquisitionType.FINITE, samps_per_chan=n)
        rdr = DigitalSingleChannelReader(di.in_stream)

        def reader():                                        # dedicated thread: read fixed chunks in real
            while not stop.is_set():                         # time so nothing lags and the buffer never
                try:                                         # backs up (the corrupt-timeline failure mode)
                    got = rdr.read_many_sample_port_uint32(cbuf, number_of_samples_per_channel=CHUNK,
                                                           timeout=1.0)
                except Exception:
                    break
                if got > 0:
                    chunks.append(cbuf[:got].copy())

        di.start()
        th = threading.Thread(target=reader, daemon=True); th.start()
        E.key_entry(enc, ecase); enc.reset_input_buffer(); dec.reset_input_buffer()
        t0 = time.time()
        enc.write((ecase["text"] + "\r").encode()); enc.flush()
        dec_line, acc = None, b""
        while time.time() - t0 < maxwin:
            acc += dec.read(dec.in_waiting or 1)
            while b"\n" in acc:
                ln, acc = acc.split(b"\n", 1)
                s = ln.decode(errors="replace").strip()
                if s.startswith("DEC "):
                    dec_line = s
                    break
            if dec_line:
                break
            time.sleep(0.02)
        latency = time.time() - t0
        time.sleep(0.7)                                     # let the reader capture the final 0x05 tail
        #   (0x05 completes before DEC — decode needs the whole frame — so this is margin, not the fix;
        #    the window derivation in run_entry is what guarantees DEC is reached at all, incl. vid 62)
        stop.set(); th.join(timeout=3)
        di.stop()
    raw = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint32)
    return raw, dec_line, latency


def bit_line(raw, bit):
    return ((raw >> bit) & 1).astype(np.int8)


def decode_frames(raw, bit, mid):
    """Extract one DI bit, 8N1-decode the whole line, return the LAST CRC-valid frame of `mid`
    (a hop line may also carry 0x01 debug frames; deframe filters by mid). None if absent."""
    line = bit_line(raw, bit)
    lows = np.flatnonzero(line == 0)
    if lows.size == 0:
        return None
    a = max(0, int(lows[0]) - 4)
    b = min(len(line), int(lows[-1]) + int(12 * (RATE / BAUD)))
    buf = bytes(uf.decode_8n1(line[a:b], RATE, BAUD))
    frames = B.deframe_u16(buf, want_mid=mid)
    return frames[-1]["frame"] if frames else None


def enc_marks(raw):
    """Count encoder spikes on GP6/GP7 after the CAL (>300 ms GP6 pulse) — chain-input liveness."""
    g6 = bit_line(raw, BIT_GP6)
    r6 = E.rising_edges(g6); f6 = E.falling_edges(g6)
    cal_fs = None
    for rs in r6:
        fs = f6[f6 > rs]
        if len(fs) and (fs[0] - rs) > 0.3 * RATE:
            cal_fs = fs[0]; break
    if cal_fs is None:
        return 0, False
    r7 = E.rising_edges(bit_line(raw, BIT_GP7))
    n = int((r6 > cal_fs).sum() + (r7 > cal_fs).sum())
    return n, True


# ------------------------------------------------------------------ frame comparison (seq-masked)
def splice_vid_crc(frame, vid):
    """Return `frame` with payload[1:3] (the u16 vid/seq, BE) replaced by `vid` and the CRC-8
    recomputed over mid+LENGTH(2)+payload. Frame = A5 5A mid len_hi len_lo payload... crc."""
    f = bytearray(frame)
    length = (f[3] << 8) | f[4]
    p = 5                                   # payload start
    f[p + 1] = (vid >> 8) & 0xFF            # payload[1..2] = seq/vid, BE
    f[p + 2] = vid & 0xFF
    f[2 + 1 + 2 + length] = B.crc8(bytes(f[2:5 + length]))  # crc at index 5+length
    return bytes(f)


def frame_match(captured, golden_hex, golden_vid):
    """Compare a captured hop frame to the frozen golden. The golden was composed from the IDEAL
    rendered waveform; the live chain runs the REAL encoder, whose emit drift (E.2: <~256 us/mark,
    ISI <= 5.6 us) perturbs the per-mark tick LSBs -> the frame cannot be byte-identical. Verdicts:
      exact : identical except the per-boot seq counter (would hold with zero drift)
      drift : same length + CRC-valid + n_marks/n_gaps intact; the only differences are seq + the
              bounded drift bytes (the physically-correct outcome)
      DIFF  : length/structure mismatch -> a real chain fault
    Returns present, verdict, ndiff (bytes differing after seq-masking), seq, len_ok."""
    if captured is None:
        return dict(present=False, verdict="----", ndiff=None, seq=None, len_ok=False, hex=None)
    golden = bytes.fromhex(golden_hex)
    seq = (captured[6] << 8) | captured[7]
    len_ok = (len(captured) == len(golden))
    if len_ok:
        masked = splice_vid_crc(captured, golden_vid)
        ndiff = sum(1 for i in range(len(golden)) if masked[i] != golden[i])
        # n_marks (0x06 frame[12:14]) / n_gaps (0x03/0x05 frame[8:10]) are drift-invariant headers;
        # both survive inside a byte-identical header prefix, so ndiff localized to value+crc bytes.
        verdict = "exact" if ndiff == 0 else "drift"
    else:
        ndiff, verdict = None, "DIFF"
    # PERSIST the captured bytes (Option-1 re-capture): the analysis host recomposes each hop from these exact
    # FE-reconstructed bytes and byte-compares against them, same-boot. seq is the boot counter each
    # frame carried (payload[1:3]) — threaded as an INPUT into build_0x03/0x05, not masked.
    return dict(present=True, verdict=verdict, ndiff=ndiff, seq=seq, len_ok=len_ok, hex=captured.hex())


# ------------------------------------------------------------------ DEC line (decode board USB)
def parse_dec(line):
    """DEC <vid> <n_gaps> <died>[ <text> <seg> <score>]  (died=1 -> no text/seg/score)."""
    t = line.split()
    if len(t) < 4 or t[0] != "DEC":
        return None
    d = dict(vid=int(t[1]), n_gaps=int(t[2]), died=(t[3] == "1"))
    if not d["died"] and len(t) >= 7:
        # text may be empty-string words? seg is 0/1 run of length n_gaps; score is the last token.
        d["score"] = int(t[-1]); d["seg"] = t[-2]; d["text"] = " ".join(t[4:-2])
    else:
        d["text"], d["seg"], d["score"] = None, None, None
    return d


def wait_dec(dec, timeout=40):
    dec.reset_input_buffer()
    end = time.time() + timeout
    acc = b""
    while time.time() < end:
        acc += dec.read(dec.in_waiting or 1)
        while b"\n" in acc:
            ln, acc = acc.split(b"\n", 1)
            s = ln.decode(errors="replace").strip()
            if s.startswith("DEC "):
                return s
    return None


def dec_match(dec_line, gvec):
    """Compare the chain's DEC line to the golden's expected_decode. text+seg+died are drift-ROBUST
    (they reproduce the golden's segmentation and letters exactly); the score is drift-SENSITIVE
    (the beam sums the drift-perturbed lp3), so it is reported as a delta, not a pass/fail. The
    convergence gate is repro_ok = text and seg and died."""
    g = gvec["expected_decode"]
    p = parse_dec(dec_line) if dec_line else None
    if p is None:
        return dict(got=dec_line, text_ok=False, seg_ok=False, died_ok=False, repro_ok=False,
                    score_delta=None)
    died_ok = (p["died"] == g["beam_died"])
    if p["died"]:
        return dict(got=dec_line, died_ok=died_ok, text_ok=died_ok, seg_ok=died_ok,
                    repro_ok=(died_ok and g["beam_died"]), score_delta=None, parsed=p)
    text_ok = (p["text"] == g["text"])
    seg_ok = (p["seg"] == "".join(str(x) for x in g["seg"]))
    score_delta = (p["score"] - g["final_score_q16"]) if p["score"] is not None else None
    return dict(got=dec_line, died_ok=died_ok, text_ok=text_ok, seg_ok=seg_ok,
                repro_ok=(died_ok and text_ok and seg_ok), score_delta=score_delta, parsed=p)


# ------------------------------------------------------------------ CER / WER
def cer(ref, hyp):
    ref, hyp = ref or "", hyp or ""
    m, n = len(ref), len(hyp)
    if m == 0:
        return float(n > 0)
    d = list(range(n + 1))
    for i in range(1, m + 1):
        prev, d[0] = d[0], i
        for j in range(1, n + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return d[n] / m


def wer(ref, hyp):
    r, h = (ref or "").split(), (hyp or "").split()
    m, n = len(r), len(h)
    if m == 0:
        return float(n > 0)
    d = list(range(n + 1))
    for i in range(1, m + 1):
        prev, d[0] = d[0], i
        for j in range(1, n + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[n] / m


# ------------------------------------------------------------------ per-entry chain run
def run_entry(enc, dec, gvec, ecase, window=None):
    # Window derived from the chain, not the median: end-to-end latency = theta + FE emit + gap9 C(ng)
    # + 0x03 emit + 0x05 emit + margin, all scaling with mark count. Measured ~0.32-0.35 s/mark
    # (vid62/240 marks -> 76.6 s), so 0.45*n_marks + 25 gives headroom over the worst case (240 ->
    # 133 s cap) while capture_entry still stops early at DEC + tail for short vectors. This is the fix
    # for the truncated 0x05: the router's forward must complete inside the window every time.
    if window is None:
        window = max(40.0, 0.45 * gvec["n_marks"] + 25.0)
    raw, dec_line, latency = capture_entry(enc, dec, ecase, maxwin=window)
    n_in, cal_ok = enc_marks(raw)
    f06 = decode_frames(raw, BIT_0x06, 0x06)
    f03 = decode_frames(raw, BIT_0x03, 0x03)
    f05 = decode_frames(raw, BIT_0x05, 0x05)
    return dict(
        vid=gvec["vid"], text=ecase["text"], latency_s=round(latency, 2),
        enc_in_marks=n_in, cal_ok=cal_ok, gold_marks=gvec["n_marks"],
        h06=frame_match(f06, gvec["expected_0x06_hex"], gvec["vid"]),
        h03=frame_match(f03, gvec["expected_0x03_hex"], gvec["vid"]),
        h05=frame_match(f05, gvec["expected_0x05_hex"], gvec["vid"]),
        dec=dec_match(dec_line, gvec),
    )


def fmt_hop(h):
    if not h["present"]:
        return "----"
    if h["verdict"] == "exact":
        return "exact"
    if h["verdict"] == "drift":
        return f"d{h['ndiff']}"           # drift: N bytes differ (seq + per-mark tick LSBs)
    return "DIFF"


def run_burst(enc, dec, gv, cases, vids, t_itx=3.5):
    """Task 5 back-to-back: key each transmission t_itx seconds after the previous FINISHES emitting
    (the FE's 1.75 s delimiter + margin), WITHOUT waiting for the full chain to drain — so multiple
    transmissions are in flight and the router/decode rings see the worst backlog. Collect every DEC
    line that lands, then the caller reads the rings via conv_rings.py (probe moved per board)."""
    dec.reset_input_buffer()
    keyed = []
    for v in vids:
        c = cases[v]
        E.key_entry(enc, c)
        enc.reset_input_buffer()
        enc.write((c["text"] + "\r").encode()); enc.flush()
        keyed.append((v, time.time()))
        emit_s = 0.6 + (c["ticks"][-1] / 1e6 if c["ticks"] else 0)   # encoder emit duration
        time.sleep(emit_s + t_itx)                                    # idle for the FE to delimit + T_ITX
    # drain remaining DEC lines, IN ARRIVAL ORDER (the chain is FIFO, so the k-th DEC corresponds to
    # the k-th keyed vid; the DEC's own vid field is the FE seq, not the golden vid).
    end = time.time() + 60
    order = []
    acc = b""
    while time.time() < end and len(order) < len(vids):
        acc += dec.read(dec.in_waiting or 1)
        while b"\n" in acc:
            ln, acc = acc.split(b"\n", 1)
            s = ln.decode(errors="replace").strip()
            if s.startswith("DEC "):
                order.append(s)
    print(f"BURST: keyed {len(vids)} at T_ITX={t_itx}s, received {len(order)} DEC lines")
    ok = 0
    for i, v in enumerate(vids):
        line = order[i] if i < len(order) else None
        dm = dec_match(line, gv[v]) if line else {"repro_ok": False}
        ok += int(dm["repro_ok"])
        print(f"  keyed vid {v:>3}: {'OK ' if dm['repro_ok'] else 'X  '} "
              f"{gv[v]['text_truth']!r} -> {(dm.get('parsed') or {}).get('text')!r}")
    print(f"BURST decode-repro {ok}/{len(vids)} | delimiting: {len(order)}/{len(vids)} decodes "
          f"delivered (one per keyed transmission). Read rings: python conv_rings.py gap9|router|decode")
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bringup", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--burst", action="store_true")
    ap.add_argument("--t-itx", type=float, default=3.5)
    ap.add_argument("--only", default="")
    ap.add_argument("--window", type=int, default=None, help="override the per-entry derived window (s)")
    ap.add_argument("--golden", default=str(GOLDEN),
                    help="golden JSON (default gap9's convergence_golden_n5; pass a clf chain_golden_*.json)")
    ap.add_argument("--out", default=str(HERE.parent / "outputs" / "chain_result.json"))
    args = ap.parse_args()

    golden = json.loads(Path(args.golden).read_text())
    gv = {int(v["vid"]): v for v in golden["vectors"]}
    cases = {int(c["vid"]): c for c in E.load_cases()}

    # ---- normalize the clf chain_golden_*.json shape to the fields the rest of conv_run reads ----
    # The clf chain golden carries expected_decode as a STRING (+ separate expected_seg/expected_score)
    # and n_gaps rather than n_marks; the gap9 convergence golden carries expected_decode as a dict and
    # n_marks. Map the former onto the latter (beam_died=False: every emitted chain sentence decodes).
    for v in gv.values():
        if isinstance(v.get("expected_decode"), str):
            v["expected_decode"] = {"text": v["expected_decode"], "seg": v.get("expected_seg", []),
                                    "final_score_q16": v.get("expected_score", 0), "beam_died": False}
        if "n_marks" not in v:
            v["n_marks"] = len(cases[v["vid"]]["ticks"]) if v["vid"] in cases else v.get("n_gaps", 0) + 1
        v.setdefault("text_truth", v.get("text", ""))

    only = [int(x) for x in args.only.split(",") if x.strip() != ""]
    if args.bringup and not only:
        only = [0]                                   # one short vector, all hops
    if only:
        vids = [v for v in only if v in gv and v in cases]
    else:                                            # emitting entries only: vid 63 (over-cap) stays silent
        vids = [v for v in sorted(gv) if v in cases and gv[v]["n_marks"] <= 256]

    enc = E.console()
    dec = serial.Serial(DECODE_PORT, 115200, timeout=0.3); time.sleep(0.3); dec.reset_input_buffer()

    if args.burst:
        run_burst(enc, dec, gv, cases, vids, t_itx=args.t_itx)
        enc.close(); dec.close()
        return

    print(f"CONVERGENCE run: {len(vids)} vectors | taps GP6/GP7 + 0x06/0x03/0x05 | decode {DECODE_PORT}")
    print("hops: exact | dN = drift (N bytes: seq + per-mark tick LSBs) | DIFF = structural fault\n")
    print(f"{'vid':>4} {'in':>4} {'0x06':>6} {'0x03':>6} {'0x05':>6} {'txt':>4} {'seg':>4} "
          f"{'dScore':>8} {'lat':>5}  truth -> decode")

    results = []
    for v in vids:
        r = run_entry(enc, dec, gv[v], cases[v], window=args.window)
        d = r["dec"]
        ds = "" if d["score_delta"] is None else str(d["score_delta"])
        print(f"{v:>4} {r['enc_in_marks']:>4} {fmt_hop(r['h06']):>6} {fmt_hop(r['h03']):>6} "
              f"{fmt_hop(r['h05']):>6} {'ok' if d['text_ok'] else 'X':>4} {'ok' if d['seg_ok'] else 'X':>4} "
              f"{ds:>8} {r['latency_s']:>5}  {gv[v]['text_truth']!r} -> {(d.get('parsed') or {}).get('text')!r}")
        results.append(r)

    enc.close(); dec.close()

    # ---- summary ----
    def hopcount(key, verdicts):
        return sum(r[key]["verdict"] in verdicts for r in results)
    N = len(results)
    dec_ok = sum(r["dec"]["repro_ok"] for r in results)    # reproduces golden text+seg+died (drift-robust)
    # acceptance (Task 4): CER/WER of the chain's decode vs the KEYED TRUTH (text_truth), the decode
    # quality metric that compares to the E.0 baseline. Distinct from dec_ok (== golden's prediction).
    cers, wers, exact_txt, ntxt = [], [], 0, 0
    for r in results:
        g = gv[r["vid"]]
        truth = g["text_truth"]
        hyp = (r["dec"].get("parsed") or {}).get("text") or ""
        cers.append(cer(truth, hyp)); wers.append(wer(truth, hyp))
        exact_txt += int(hyp == truth); ntxt += 1
    print(f"\nSUMMARY  N={N}")
    for key, mid in (("h06", "0x06"), ("h03", "0x03"), ("h05", "0x05")):
        ok = hopcount(key, ("exact", "drift"))             # structurally intact (present, CRC, len, hdr)
        exact = hopcount(key, ("exact",))
        diff = hopcount(key, ("DIFF",))
        absent = sum(not r[key]["present"] for r in results)
        print(f"  hop {mid}: {ok}/{N} match-modulo-drift ({exact} zero-drift exact) | {diff} DIFF | {absent} absent")
    print(f"  reproduces golden decode (text+seg+died, drift-robust): {dec_ok}/{N}")
    deltas = [abs(r["dec"]["score_delta"]) for r in results if r["dec"]["score_delta"] is not None]
    if deltas:
        print(f"  score delta (drift readout): mean |dScore| {np.mean(deltas):.0f} q16 "
              f"({np.mean(deltas)/65536:.4f}), {sum(d == 0 for d in deltas)}/{len(deltas)} zero")
    if ntxt:
        print(f"  acceptance vs KEYED TRUTH: CER {np.mean(cers):.4f} mean, WER {np.mean(wers):.4f}, "
              f"{exact_txt}/{ntxt} exact-text")
        base = golden.get("cer_wer_baseline", {})
        print(f"  E.0 baseline: {base}")
    Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
