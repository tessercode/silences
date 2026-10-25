#!/usr/bin/env python3
"""Check the BiGRU segmenter test image's 0x03 result line against the golden frames.
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

    # The runner terminates IN-PROTOCOL with the wrapper's own end-of-pass sentinel (KIND 0x02,
    # gap9_hw.c run_pass()): [A5 5A][03][0001][02][CRC]. Verify and strip it, then the remainder
    # must be exactly the 117 golden frames.
    EOP = bytes.fromhex("a55a0300010221")
    if not raw.endswith(EOP):
        print(f"FAIL: stream does not end with the end-of-pass sentinel {EOP.hex()}")
        return 1
    raw = raw[:-len(EOP)]

    want = b"".join(bytes.fromhex(v["expected_0x03_hex"]) for v in g["vectors"])
    if raw != want:
        # localize the first divergence to a vector
        o = 0
        for v in g["vectors"]:
            e = bytes.fromhex(v["expected_0x03_hex"])
            got_slice = raw[o:o + len(e)]
            if got_slice != e:
                bad = next((i for i in range(min(len(e), len(got_slice)))
                            if e[i] != got_slice[i]), len(got_slice))
                print(f"FAIL vec {v['vector_id']} (n_gaps={v['n_gaps']}): first differing byte {bad} "
                      f"of {len(e)}; got {got_slice[bad:bad+1].hex() or '<eof>'} want {e[bad:bad+1].hex()}")
                return 1
            o += len(e)
        print(f"FAIL: stream length {len(raw)} != expected {len(want)}")
        return 1
    print(f"PASS: {len(g['vectors'])}/117 0x03 frames BYTE-IDENTICAL to "
          f"{os.path.basename(gpath)} ({len(raw)} bytes) + end-of-pass sentinel verified")
    return 0

if __name__ == "__main__":
    sys.exit(main())
