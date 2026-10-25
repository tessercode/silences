#!/usr/bin/env python3
"""Check the router test image's 0x05 result line against the golden frames.
"""
import hashlib, json, os, sys

def main():
    gpath, rpath = sys.argv[1], sys.argv[2]
    gdir = os.path.dirname(os.path.abspath(gpath))
    m = json.load(open(os.path.join(gdir, "manifest.json")))
    got = hashlib.sha256(open(gpath, "rb").read()).hexdigest()
    assert got == m["files"][os.path.basename(gpath)], "golden sha != manifest — refuse to compare"
    g = json.load(open(gpath))
    raw = open(rpath, "rb").read()

    # in-protocol terminator: the 0x05 line's own end-of-pass sentinel (KIND 0x02)
    EOP = bytes.fromhex("a55a0500010255")
    if not raw.endswith(EOP):
        print(f"FAIL: stream does not end with the end-of-pass sentinel {EOP.hex()}")
        return 1
    raw = raw[:-len(EOP)]

    want = b"".join(bytes.fromhex(v["expected_0x05_hex"]) for v in g["vectors"])
    if raw != want:
        o = 0
        for v in g["vectors"]:
            e = bytes.fromhex(v["expected_0x05_hex"])
            gs = raw[o:o + len(e)]
            if gs != e:
                bad = next((i for i in range(min(len(e), len(gs))) if e[i] != gs[i]), len(gs))
                print(f"FAIL vec {v['vector_id']} (n_spikes={v['n_spikes']}): first differing byte "
                      f"{bad} of {len(e)}; got {gs[bad:bad+1].hex() or '<eof>'} want {e[bad:bad+1].hex()}")
                return 1
            o += len(e)
        print(f"FAIL: stream length {len(raw)} != expected {len(want)}")
        return 1
    print(f"PASS: {len(g['vectors'])}/117 0x05 frames BYTE-IDENTICAL to {os.path.basename(gpath)} "
          f"({len(raw)} bytes) + end-of-pass sentinel verified")
    return 0

if __name__ == "__main__":
    sys.exit(main())
