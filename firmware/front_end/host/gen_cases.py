#!/usr/bin/env python3
"""Write the front-end host gate's case files from the golden waveforms (fe_<operator>.json, fe_overflow.json),
after checking each against firmware/golden/manifest.json.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLD = HERE.parent.parent / "golden"
OUT = HERE / "build" / "cases"

FISTS = ["fast_normalr", "slow_normalr", "high_r_bug", "cootie_high_jitter", "cootie_corrected"]


def verify_and_load(name: str, want_sha: str):
    raw = (GOLD / name).read_bytes()
    got = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if got != want_sha:
        sys.exit(f"GOLDEN INTEGRITY FAILURE: {name} sha {got} != manifest {want_sha}")
    return json.loads(raw.decode("utf-8"))


def write_case(path: Path, name: str, txs):
    with open(path, "w", newline="\n") as f:
        f.write(f"NAME {name} NTX {len(txs)}\n")
        for t in txs:
            seq = t.get("replay_seq", 0)
            f.write(f"TX {seq} {t['n_marks']} {t['pacing_after_us']}\n")
            for rise, fall, ch in t["waveform_us"]:
                f.write(f"{rise} {fall} {ch}\n")
            f.write(f"FRAME {t['expected_frame_hex']}\n")


def main():
    manifest = json.loads((GOLD / "manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    OUT.mkdir(parents=True, exist_ok=True)

    total = 0
    for fist in FISTS:
        g = verify_and_load(f"fe_{fist}.json", files[f"fe_{fist}.json"])
        assert g["fist"] == fist and g["n_transmissions"] == len(g["transmissions"])
        write_case(OUT / f"{fist}.txt", fist, g["transmissions"])
        total += len(g["transmissions"])

    o = verify_and_load("fe_overflow.json", files["fe_overflow.json"])
    otx = dict(replay_seq=0, n_marks=o["n_marks"], pacing_after_us=o["pacing_after_us"],
               waveform_us=o["waveform_us"], expected_frame_hex=o["expected_frame_hex"])
    write_case(OUT / "overflow_marker_case.txt", "overflow_marker_case", [otx])

    print(f"cases written to {OUT}: {total} golden transmissions + overflow case (all SHA-verified)")


if __name__ == "__main__":
    main()
