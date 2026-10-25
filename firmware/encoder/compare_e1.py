#!/usr/bin/env python3
"""Check the encoder test image's tick/channel stream (emulated or from the bench) against the cases the host
gate generated from the encoder reference.
"""
import re
import sys


def load_cases(p):
    lines = open(p).read().split("\n")
    n = int(lines[0])
    out = {}
    li = 1
    for _ in range(n):
        h = lines[li].split("|")
        f = h[0].split()
        vid, kind, tag, fi, cf, seed, cnt = int(f[0]), f[1], f[2], int(f[3]), int(f[4]), int(f[5]), int(f[6])
        ticks = [int(x) for x in lines[li + 1].split()] if lines[li + 1].strip() else []
        chs = [int(x) for x in lines[li + 2].split()] if lines[li + 2].strip() else []
        out[vid] = dict(kind=kind, tag=tag, n=cnt, ticks=ticks, ch=chs)
        li += 3
    assert len(out) == n
    return out


def parse_stream(p):
    b = open(p, "rb").read()
    txt = b.decode("ascii", "replace")
    if "E1BEGIN" not in txt or "E1END" not in txt:
        sys.exit("FAIL: stream is missing its E1BEGIN/E1END sentinels")
    body = txt[txt.index("E1BEGIN") + 7: txt.index("E1END")]
    got, cur = {}, None
    for line in body.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^V (\d+) REJECT (\d+)$", line)
        if m:
            got[int(m.group(1))] = dict(reject=True, nsp=int(m.group(2)), ticks=[], ch=[])
            cur = None
            continue
        m = re.match(r"^V (\d+) (\d+)$", line)
        if m:
            cur = int(m.group(1))
            got[cur] = dict(reject=False, n=int(m.group(2)), ticks=[], ch=[])
            continue
        m = re.match(r"^(\d+) ([01])$", line)
        if m and cur is not None:
            got[cur]["ticks"].append(int(m.group(1)))
            got[cur]["ch"].append(int(m.group(2)))
    return got


def main():
    cases = load_cases(sys.argv[1])
    got = parse_stream(sys.argv[2])
    missing = set(cases) - set(got)
    if missing:
        sys.exit(f"FAIL: device produced no record for vids {sorted(missing)}")

    stats = {"GOLDEN": [0, 0], "SPEC": [0, 0]}
    fails = []
    for vid, c in sorted(cases.items()):
        g = got[vid]
        s = stats[c["tag"]]
        s[1] += 1
        if c["n"] > 256:                                  # the over-cap entry: expect a REJECT
            ok = g["reject"] and g["nsp"] == c["n"]
        else:
            ok = (not g["reject"] and g["ticks"] == c["ticks"] and g["ch"] == c["ch"])
        if ok:
            s[0] += 1
        else:
            fails.append((vid, c["kind"], c["tag"]))
    print(f"  GOLDEN (chain-golden waveform)    : {stats['GOLDEN'][0]}/{stats['GOLDEN'][1]}")
    print(f"  SPEC   (re-derived, e1_port.md §3): {stats['SPEC'][0]}/{stats['SPEC'][1]}")
    tot_ok = stats["GOLDEN"][0] + stats["SPEC"][0]
    tot = stats["GOLDEN"][1] + stats["SPEC"][1]
    nvals = sum(len(c["ticks"]) for c in cases.values())
    print(f"  TOTAL  : {tot_ok}/{tot} vectors, {nvals} (tick,channel) pairs compared")
    if fails:
        for vid, kind, tag in fails[:10]:
            print(f"    FAIL vid {vid} {kind} {tag}")
        sys.exit(1)
    print("[e1] PASS — emulated ARM reproduces the host render EXACTLY (ticks + channels).")


if __name__ == "__main__":
    main()
