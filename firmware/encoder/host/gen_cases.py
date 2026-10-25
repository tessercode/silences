#!/usr/bin/env python3
"""Write the encoder host gate's case files from the chain golden.

    cases.txt          every chain-golden sentence rendered by the software reference (silences.encoder_ref),
                       which must reproduce the golden's waveform exactly
    anchor_cases.txt   single-word renders (the board's single-word mode)
    rng_cases.txt      the generator's raw draws: u32, uniform and normal bit patterns

The golden is checked against firmware/golden/manifest.json, and the operator table against fists.h.
"""
import hashlib
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GOLDEN = os.path.join(REPO, "firmware", "golden")
sys.path.insert(0, REPO)

from silences import encoder_ref as S          # noqa: E402  the encoder's specification
from silences.operators import parse_fists_h   # noqa: E402

FISTS = parse_fists_h()                        # index i <-> console ":fist i+1" <-> ENC_FISTS[i]


def load_golden():
    man = json.load(open(os.path.join(GOLDEN, "manifest.json")))["files"]
    raw = open(os.path.join(GOLDEN, "chain_bigru_n5.json"), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == man["chain_bigru_n5.json"], "chain golden does not match the manifest"
    return json.loads(raw)


def d2b(x):
    """IEEE-754 bit pattern, as the board's rng dump prints it (integers only, so the comparison is bitwise)."""
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def main():
    by_name = dict(FISTS)
    idx_of = {n: i for i, (n, _) in enumerate(FISTS)}
    g = load_golden()
    out = []
    for r in g["vectors"]:
        p = by_name[r["fist_name"]]
        assert tuple(g["fists"][str(r["fist"])]["params"]) == p, f"vid {r['vid']}: golden operator differs from fists.h"
        t, c, _, _, _ = S.render(r["text"], *p, r["seed"])
        assert S.waveform(t, c) == [[int(x) for x in w] for w in r["expected_waveform_us"]], f"vid {r['vid']}: reference != golden"
        out.append(dict(vid=r["vid"], kind=r["kind"], fist_index=idx_of[r["fist_name"]], seed=r["seed"], text=r["text"],
                        ticks=t, ch=c))
    with open(os.path.join(HERE, "cases.txt"), "w") as f:
        f.write(f"{len(out)}\n")
        for c in out:            # vid kind tag fist_index console_fist seed n |text|
            f.write(f"{c['vid']} {c['kind']} GOLDEN {c['fist_index']} {c['fist_index'] + 1} {c['seed']} {len(c['ticks'])} |{c['text']}|\n")
            f.write(" ".join(str(x) for x in c["ticks"]) + "\n")
            f.write(" ".join(str(x) for x in c["ch"]) + "\n")
    print(f"  wrote cases.txt ({len(out)} sentences; the reference reproduces every golden waveform)")

    anc = []
    for word in ("LUKE", "CQ", "DAME", "K7QX", "MOON"):
        for sd in (1, 7, 42, 12345):
            tk, ch = S.render(word, *by_name["cootie_corrected"], sd, sentence=False)[:2]
            anc.append((word, sd, list(tk), list(ch)))
    with open(os.path.join(HERE, "anchor_cases.txt"), "w") as f:
        f.write(f"{len(anc)} {idx_of['cootie_corrected']}\n")
        for word, sd, tk, ch in anc:
            f.write(f"{word} {sd} {len(tk)}\n")
            f.write(" ".join(str(x) for x in tk) + "\n")
            f.write(" ".join(str(x) for x in ch) + "\n")
    print(f"  wrote anchor_cases.txt ({len(anc)} single-word renders)")

    seeds, N = [1, 7, 42, 12345, 0xDEADBEEF], 64
    with open(os.path.join(HERE, "rng_cases.txt"), "w") as f:
        f.write(f"{len(seeds)} {N}\n")
        for sd in seeds:
            f.write(f"{sd}\n")
            rng = S.Rng(sd)
            f.write(" ".join(str(rng.next_u32()) for _ in range(N)) + "\n")
            rng = S.Rng(sd)
            f.write(" ".join(f"{d2b(rng.unit()):016x}" for _ in range(N)) + "\n")
            rng = S.Rng(sd)
            f.write(" ".join(f"{d2b(rng.normal()):016x}" for _ in range(N)) + "\n")
    print(f"  wrote rng_cases.txt ({len(seeds)} seeds x {N} draws)")


if __name__ == "__main__":
    main()
