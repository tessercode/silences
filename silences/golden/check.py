"""Check every firmware golden against the software reference.

    python -m silences.golden.check

The goldens in firmware/golden/ are what the firmware's host and emulator tests compare the boards'
output with, byte for byte. This recomputes every expected value in every golden from the software
in this package and fails unless all of them match, so the firmware is checked against the same
arithmetic that produced the paper's numbers.
"""
import json
import struct
import subprocess
import sys

import numpy as np

from .. import DATA, FIRMWARE, OUTPUTS
from .. import encoder_ref, frames, front_end
from ..decoder.beam import decode_tx
from ..fixedpoint.window_net import FixedWindowNet
from ..pipeline import compose
from ..segmenters import bigru
from ..segmenters.lif import LIF, SpikeFedLIF
from ..segmenters.mlp import MLP
from ..evaluation.producers import DEPLOYED

G = FIRMWARE / "golden"
OPERATORS = ("fast_normalr", "slow_normalr", "high_r_bug", "cootie_high_jitter", "cootie_corrected")


def _load(name):
    return json.loads((G / name).read_text())


def _vectors():
    return {int(v["vid"]): v for v in json.loads((DATA / "golden_inputs" / "segmenter_vectors.json").read_text())["vectors"]}


def check_rng():
    """The generator against the published reference code, and the Box-Muller freeze."""
    K = _load("rng_kat.json")
    exe = OUTPUTS / "kat_driver"
    OUTPUTS.mkdir(exist_ok=True)
    kat = FIRMWARE / "tests" / "kat"
    subprocess.run(["cc", "-O2", "-I", str(kat), "-o", str(exe), str(kat / "kat_driver.c")], check=True)
    lines = subprocess.run([str(exe)] + [str(c["seed"]) for c in K["seeds"]], capture_output=True, text=True,
                           check=True).stdout.splitlines()
    ok = 0
    for c, line in zip(K["seeds"], lines):
        f = line.split()
        state, stream, fold = [int(v) for v in f[2:6]], [int(v) for v in f[7:7 + K["n_stream"]]], [int(v) for v in f[-2:]]
        r = encoder_ref.Rng(c["seed"])
        ours = [r.next_u32() for _ in range(K["n_stream"])]
        r = encoder_ref.Rng(c["seed"])
        fs = fx = 0
        for _ in range(K["n_fold"]):
            v = r.next_u32()
            fs, fx = (fs + v) & 0xFFFFFFFF, fx ^ v
        r = encoder_ref.Rng(c["seed"])
        bits = [struct.unpack("<Q", struct.pack("<d", r.normal()))[0] for _ in range(K["n_normal"])]
        ok += (state == c["state"] == list(encoder_ref.Rng(c["seed"]).s) and stream == c["stream"] == ours
               and fold == [c["fold_sum"], c["fold_xor"]] == [fs, fx] and bits == c["normal_bits"])
    r_tree = encoder_ref.sqrt(-2.0 * encoder_ref.log(0.5 / 2 ** 32))
    assert abs(r_tree - K["rmax"]) <= 2 * np.spacing(K["rmax"]), "the Box-Muller support bound moved"
    return ok, len(K["seeds"])


def check_front_end():
    ok = n = 0
    for op in OPERATORS:
        for t in _load(f"fe_{op}.json")["transmissions"]:
            w = np.array(t["waveform_us"], dtype=np.int64)
            caps, _, _, _ = front_end.split_transmissions(w[:, 0], w[:, 1])
            fr, rec = front_end.frame_for_capture(t["replay_seq"], w[:, 0], w[:, 1], w[:, 2], *caps[0])
            ok += len(caps) == 1 and fr.hex() == t["expected_frame_hex"] and rec["tu_us"] == t["tu_shortest_isi_us"]
            n += 1
    o = _load("fe_overflow.json")
    w = np.array(o["waveform_us"], dtype=np.int64)
    caps, _, _, _ = front_end.split_transmissions(w[:, 0], w[:, 1])
    ok += front_end.frame_for_capture(0, w[:, 0], w[:, 1], w[:, 2], *caps[0])[0].hex() == o["expected_frame_hex"]
    return ok, n + 1


def check_segmenter_bigru():
    ok = n = 0
    vec = _vectors()
    for k in (1, 5):
        seeds = bigru.deployed(k)
        for r in _load(f"segmenter_bigru_n{k}.json")["vectors"]:
            v = vec[int(r["vector_id"])]
            ok += bigru.lp3_ticks(seeds, v["ticks"], v["ch"], v["tu"]) == r["log_p3_q16"]
            n += 1
    return ok, n


def _lp3_inputs(k):
    """(vid -> (ticks, ch, tu, lp3)) for the beam and frame goldens: the 97 segmenter vectors and the 20
    long vectors that exercise the 255-gap cap."""
    vec, ref = _vectors(), {int(r["vector_id"]): r["log_p3_q16"] for r in _load(f"segmenter_bigru_n{k}.json")["vectors"]}
    out = {vid: (v["ticks"], v["ch"], v["tu"], ref[vid]) for vid, v in vec.items()}
    for v in _load("beam_cap_vectors.json")["vectors"]:
        out[int(v["vid"])] = (v["ticks"], v["ch"], v["tu"], v[f"lp3_n{k}"])
    return out


def check_beam():
    ok = n = 0
    for k in (1, 5):
        ins = _lp3_inputs(k)
        for g in _load(f"beam_n{k}.json")["vectors"]:
            _, ch, _, lp3 = ins[int(g["vector_id"])]
            text, seg, score = decode_tx(ch, lp3)
            ok += (text is None) == g["beam_died"] and (text is None or (text, seg, score) == (g["decoded_text"], g["seg"], g["final_score_q16"]))
            n += 1
    return ok, n


def check_frames():
    ok = n = 0
    for k in (1, 5):
        ins = _lp3_inputs(k)
        for g in _load(f"frames_n{k}.json")["vectors"]:
            vid = int(g["vector_id"])
            ticks, ch, tu, lp3 = ins[vid]
            ok += (frames.fe_frame(vid, tu, ticks, ch).hex() == g["expected_0x06_hex"]
                   and frames.segmenter_frame(vid, lp3).hex() == g["expected_0x03_hex"]
                   and frames.router_frame(vid, ch, lp3).hex() == g["expected_0x05_hex"])
            n += 1
    return ok, n


def _chain_matches(g, c, decode_fields):
    if c["hex_0x06"] != g["expected_0x06_hex"] or g.get("overflow", False) != c["overflow"]:
        return False
    if c["overflow"]:
        return True
    return (c["tu_us"] == g["tu_us"] and c["lp3_q16"] == g["lp3_q16"] and c["hex_0x03"] == g["expected_0x03_hex"]
            and c["hex_0x05"] == g["expected_0x05_hex"] and decode_fields(g) == (c["decoded_text"], c["seg"], c["final_score_q16"]))


def check_chain_bigru():
    ok = n = 0
    for k in (1, 5):
        seeds = bigru.deployed(k)
        seg = lambda t, ch, tu: bigru.lp3_ticks(seeds, t, ch, tu)          # noqa: E731
        for g in _load(f"chain_bigru_n{k}.json")["vectors"]:
            c = compose(g["text"], int(g["fist"]), int(g["seed"]), seg, seq=int(g["vid"]))
            ok += (c["waveform_us"] == g["expected_waveform_us"] and
                   _chain_matches(g, c, lambda g: (g["expected_decode"]["text"], g["expected_decode"]["seg"],
                                                   g["expected_decode"]["final_score_q16"])))
            n += 1
    return ok, n


def _window_nets():
    return dict(spk=FixedWindowNet.from_spk(SpikeFedLIF.load(DEPLOYED["spk"])),
                lif=FixedWindowNet.from_lif(LIF.load(DEPLOYED["lif"])), mlp=FixedWindowNet.from_mlp(MLP.load(DEPLOYED["mlp"])))


def _window_decode(g):
    """The golden's expected decode, or the beam's death triple: the spike-fed chain golden carries one sentence
    (vid 58) on which the software beam dies at the default decoder weights, and the board dies the same way."""
    if g["expected_decode"] is None:
        return (None, None, None)
    return (g["expected_decode"], g["expected_seg"], g["expected_score"])


def check_segmenter_window():
    ok = n = 0
    for name, net in _window_nets().items():
        for v in _load(f"segmenter_{name}.json")["vectors"]:
            ok += net.lp3_ticks(v["ticks"], v["ch"], v["tu"]) == v["lp3"]
            n += 1
    return ok, n


def check_chain_window():
    ok = n = 0
    for name, net in _window_nets().items():
        for g in _load(f"chain_{name}.json")["vectors"]:
            c = compose(g["text"], int(g["fist"]), int(g["seed"]), net.lp3_ticks, seq=int(g["vid"]))
            ok += _chain_matches(g, c, _window_decode)
            n += 1
    return ok, n


CHECKS = dict(rng=check_rng, front_end=check_front_end, segmenter_bigru=check_segmenter_bigru, beam=check_beam,
              frames=check_frames, chain_bigru=check_chain_bigru, segmenter_window=check_segmenter_window,
              chain_window=check_chain_window)


def main(names=None):
    bad = 0
    for name in names or CHECKS:
        ok, n = CHECKS[name]()
        bad += ok != n
        print(f"  {name:18s} {ok}/{n} {'match' if ok == n else 'MISMATCH'}", flush=True)
    print("every golden reproduced by the software reference" if not bad else f"{bad} golden families differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
