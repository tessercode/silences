"""
Bit-exact comparator for the BiGRU segmenter's host build: parse its framed output and check it
reproduces firmware/golden/segmenter_bigru_n1.json (or _n5) exactly, every per-gap [lp0, lp1, lp2]
Q16.16 triple. Exit 0 iff all vectors, gaps and values match; otherwise print the first diverging
(vector, gap, class, C value, golden value) and exit 1.
"""
import sys
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "..", "golden")


def parse_frames(text):
    out = {}
    cur = None
    saw_done = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("VEC "):
            _, vid, ng = line.split()
            cur = {"vid": int(vid), "n_gaps": int(ng), "lp3": []}
        elif line.startswith("G "):
            _, t, a, b, c = line.split()
            if cur is not None:
                cur["lp3"].append((int(t), [int(a), int(b), int(c)]))
        elif line.startswith("EVEC "):
            if cur is not None:
                out[cur["vid"]] = cur
                cur = None
        elif line.startswith("DONE "):
            saw_done = int(line.split()[1])
    return out, saw_done


def main():
    text = sys.stdin.read()
    ref_name = sys.argv[1] if len(sys.argv) > 1 else "segmenter_bigru_n1.json"
    ref = json.load(open(os.path.join(GOLD, ref_name)))
    ref_by_id = {v["vector_id"]: v for v in ref["vectors"]}
    got, done = parse_frames(text)
    if done is None or done != len(ref_by_id):
        print(f"FAIL: DONE {done} != {len(ref_by_id)} vectors")
        return 1
    n_gaps = n_vals = 0
    for vid, rv in sorted(ref_by_id.items()):
        gv = got.get(vid)
        if gv is None or gv["n_gaps"] != rv["n_gaps"] or len(gv["lp3"]) != rv["n_gaps"]:
            print(f"FAIL: vector {vid} missing/short (got {gv and gv['n_gaps']}, ref {rv['n_gaps']})")
            return 1
        for (t, triple), ref_triple in zip(gv["lp3"], rv["log_p3_q16"]):
            for c in range(3):
                if triple[c] != ref_triple[c]:
                    print(f"FAIL: vector {vid} gap {t} class {c}: C {triple[c]} != golden {ref_triple[c]}")
                    return 1
            n_vals += 3
            n_gaps += 1
    print(f"BIT-EXACT PASS: {len(ref_by_id)}/{len(ref_by_id)} vectors, {n_gaps} gaps, "
          f"{n_vals} log-posterior values exact vs {ref_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
