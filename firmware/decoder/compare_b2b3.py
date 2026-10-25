#!/usr/bin/env python3
"""Check the decoder test image's result lines (decoded text, gap classes, final score) against the beam golden.
"""
import hashlib, json, os, sys

def main():
    gpath, rpath = sys.argv[1], sys.argv[2]
    gdir = os.path.dirname(os.path.abspath(gpath))
    m = json.load(open(os.path.join(gdir, "manifest.json")))
    got = hashlib.sha256(open(gpath, "rb").read()).hexdigest()
    assert got == m["files"][os.path.basename(gpath)], "golden sha != manifest — refuse to compare"
    g = json.load(open(gpath))["vectors"]

    raw = open(rpath, "rb").read().replace(b"\r\n", b"\n").decode("utf-8", "replace")
    lines = [l for l in raw.split("\n") if l.startswith("DEC ") or l == "DONE"]
    if "DONE" not in lines:
        print("FAIL: no DONE marker — the runner did not finish")
        return 1
    dec = [l for l in lines if l.startswith("DEC ")]

    want = []
    for v in g:
        vid, ng = int(v["vector_id"]), int(v["n_gaps"])
        if v["beam_died"]:
            want.append(f"DEC {vid} {ng} 1")
        else:
            seg = "".join(str(x) for x in v["seg"])
            want.append(f"DEC {vid} {ng} 0 {v['decoded_text']} {seg} {v['final_score_q16']}")
    if len(dec) != len(want):
        print(f"FAIL: {len(dec)} result lines, want {len(want)}")
        return 1
    for i, (a, b) in enumerate(zip(dec, want)):
        if a != b:
            print(f"FAIL vec {g[i]['vector_id']}:\n  got  {a}\n  want {b}")
            return 1
    n_died = sum(1 for v in g if v["beam_died"])
    print(f"PASS: {len(dec)}/117 decoded result lines IDENTICAL to {os.path.basename(gpath)} "
          f"({n_died} beam death(s)); the EOP sentinel produced no result line (R3 rejection)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
