"""Check the LIF or MLP classifier's output lines ('vid ng lp0 lp1 lp2 ...' on stdin, from the host build or
Renode) against its golden, bit for bit.  Usage: <output> | python clf_compare.py golden/segmenter_<net>.json
"""
import json
import sys

ref = {int(r["vid"]): r for r in json.load(open(sys.argv[1]))["vectors"]}
n_ok = n_val = 0
for line in sys.stdin:
    p = line.split()
    if not p or not p[0].lstrip("-").isdigit():
        continue                                    # skip boot noise / the DONE sentinel
    vid, ng = int(p[0]), int(p[1])
    got = [int(x) for x in p[2:]]
    exp = [int(v) for row in ref[vid]["lp3"] for v in row]
    assert ng == ref[vid]["n_gaps"], f"vid {vid}: n_gaps {ng} != {ref[vid]['n_gaps']}"
    assert got == exp, f"vid {vid}: lp3 MISMATCH\n got {got[:9]}\n exp {exp[:9]}"
    n_ok += 1; n_val += len(got)
assert n_ok == len(ref), f"only {n_ok}/{len(ref)} vectors in the capture (truncated run?)"
print(f"BIT-EXACT: {n_ok}/{len(ref)} vectors, {n_val} lp3 values match golden")
