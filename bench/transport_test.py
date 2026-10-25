#!/usr/bin/env python3
"""clf_b2_silicon.py — Test B (transport, primary) for spiking_gap9_{lif,mlp} on silicon.

Mirrors firmware/gap9_3c/b2_gap9_silicon.py EXACTLY in method (drive each 0x06 on GP3, finite
synchronized DI capture of the 0x03 on GP4, byte-exact compare; then the burst: cap frame + a real
follower ~into compute; ring HWM/dropped over SWD). The only adaptations forced by the CLF golden:

  * The golden is reference_clf_<m>.json = {vectors:[{vid,tu,ticks,ch,n_gaps,lp3}]} — NOT pre-baked
    hex — so each 0x06 is built with frames_b2.build_0x06_frame(vid,tu,ticks,ch) and each expected
    0x03 with frames_b2.build_0x03_frame(vid,lp3) (the exact builders the pipeline uses).
  * G_RING is RE-DERIVED per built ELF (passed in from `nm ... | grep g_ring`), never gap9's 0x2000dcfc.

Runs on Windows python (nidaqmx + openocd, via b2_common). Usage:
    python clf_b2_silicon.py <lif|mlp> <clf_hw.elf> <g_ring_hex> <per_gap_us> [burst_delay_ms]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]                     # the repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))         # b2_common, daq_uart, daq_sync
sys.path.insert(0, str(ROOT))
import b2_common as bc                                           # noqa: E402
import daq_uart as du                                            # noqa: E402
from silences import frames                                       # noqa: E402

GP3 = 1     # 0x06 in on GP3 = DO bit1 (gap9 convention, confirmed on this rig: GP2/GO=port0/line0=bit0,
            # GP3=port0/line1=bit1). GO(bit0) held low by build_do. 0x03 out on GP4 = DI bit0.
            # (Driving bit0 instead triggers GO -> run_pass -> the {0x02} end-of-pass sentinel, not compute.)


def window_us(f06, exp03, ng, per_gap):
    return (len(f06) + len(exp03)) * 1100 + ng * per_gap + 1_500_000


def run(module, elf, g_ring, per_gap, burst_delay_ms):
    ref = json.load(open(ROOT / "firmware" / "golden" / f"segmenter_{module}.json"))
    vecs = ref["vectors"]
    sha = hashlib.sha256(Path(elf).read_bytes()).hexdigest()
    print(f"[{module}] clf_hw.elf sha256 {sha}")
    print(f"[{module}] G_RING {g_ring:#x} (re-derived; NOT gap9's 0x2000dcfc)")
    bc.flash(elf)                                                # openocd program verify reset exit
    import time; time.sleep(1.5)                                 # let clf_hw fully boot before the first drive

    # ---- 50/50 per-vector 0x06 -> 0x03 byte-exact ----
    npass, fails = 0, []
    for v in vecs:
        f06 = frames.fe_frame(v["vid"], v["tu"], v["ticks"], v["ch"])
        exp03 = frames.segmenter_frame(v["vid"], v["lp3"])
        line = bc.drive_and_capture([(GP3, f06)], window_us(f06, exp03, v["n_gaps"], per_gap), di_bit=0)
        fr = bc.deframe_u16(bc.decode_line(line), want_mid=0x03)
        got = fr[-1]["frame"] if fr else b""
        if got == exp03:
            npass += 1
        else:
            fails.append((v["vid"], v["n_gaps"], len(got), len(exp03)))
            if len(fails) <= 5:
                print(f"  FAIL vid {v['vid']} ng {v['n_gaps']}: got {len(got)}B want {len(exp03)}B")
    print(f"[{module}] correctness: {npass}/{len(vecs)} 0x03 frames byte-exact")

    # ---- BURST: cap frame (largest ng) + a real follower delivered burst_delay into its compute ----
    cap_v = max(vecs, key=lambda v: v["n_gaps"])                 # vid 104, 30 gaps
    fol_v = next(v for v in vecs if 9 <= v["n_gaps"] <= 14 and v["vid"] != cap_v["vid"])
    fa = frames.fe_frame(cap_v["vid"], cap_v["tu"], cap_v["ticks"], cap_v["ch"])
    fb_ = frames.fe_frame(fol_v["vid"], fol_v["tu"], fol_v["ticks"], fol_v["ch"])
    bitsA = np.array(list(du.uart_bits(fa)), dtype=np.uint8)
    bitsB = np.array(list(du.uart_bits(fb_)), dtype=np.uint8)
    lead = np.ones(int(0.003 * du.RATE), dtype=np.uint8)
    gap = np.ones(int(burst_delay_ms / 1000.0 * du.RATE), dtype=np.uint8)   # idle AFTER frame A -> lands in compute
    win = int(window_us(fa, frames.segmenter_frame(cap_v["vid"], cap_v["lp3"]), cap_v["n_gaps"], per_gap)
              / 1e6 * du.RATE) + len(gap) + len(bitsB)
    seq = np.concatenate([lead, bitsA, gap, bitsB])
    line1 = np.ones(max(win, len(seq) + du.RATE), dtype=np.uint8)
    line1[:len(seq)] = seq
    do = (line1.astype(np.uint32) << GP3)
    from daq_sync import generate_capture
    di = generate_capture(do, du.RATE)
    both = bc.deframe_u16(bc.decode_line(((di >> 0) & 1).astype(np.int8)), want_mid=0x03)
    exp_cap = frames.segmenter_frame(cap_v["vid"], cap_v["lp3"])
    exp_fol = frames.segmenter_frame(fol_v["vid"], fol_v["lp3"])
    frames = {f["frame"] for f in both}
    burst_ok = exp_cap in frames and exp_fol in frames
    print(f"[{module}] BURST: {len(both)} 0x03 frames; cap({cap_v['vid']},{cap_v['n_gaps']}g) + "
          f"follower({fol_v['vid']},{fol_v['n_gaps']}g @ +{burst_delay_ms}ms) both exact: {burst_ok}")

    hwm = bc.read_word(g_ring + 12)
    dropped = bc.read_word(g_ring + 8)
    print(f"[{module}] ring: HWM {hwm} B (ring 1024), dropped {dropped}")
    res = dict(module=module, sha256=sha, g_ring=hex(g_ring), correct=(npass == len(vecs)),
               npass=npass, nvec=len(vecs), burst=burst_ok, hwm=hwm, dropped=dropped,
               cap_vid=cap_v["vid"], fol_vid=fol_v["vid"], fails=fails)
    (ROOT / "outputs" / f"transport_{module}.json").write_text(json.dumps(res, indent=1))
    print(f"[{module}] === TEST B {'PASS' if res['correct'] and burst_ok and dropped == 0 else 'REVIEW'} "
          f"({npass}/{len(vecs)}, burst {burst_ok}, dropped {dropped}, HWM {hwm}) ===")
    return res


if __name__ == "__main__":
    module = sys.argv[1]
    elf = sys.argv[2]
    g_ring = int(sys.argv[3], 16)
    per_gap = int(sys.argv[4]) if len(sys.argv) > 4 else 50000
    burst_delay_ms = float(sys.argv[5]) if len(sys.argv) > 5 else 40.0
    run(module, elf, g_ring, per_gap, burst_delay_ms)
