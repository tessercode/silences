#!/usr/bin/env python3
"""conv_rings.py — CONVERGENCE Task 5: read each board's RX-ring invariants over SWD.

ONE probe exists, so this runs per-board: move the probe onto a board, run `python conv_rings.py
<board>`, read (hwm, dropped) for each of its rings, and compare against the b2a_sizing.py derived
worst-line bounds. A HWM at/over the 1024 ring size, or any non-zero dropped, is a chain finding.

  gap9   : one ring (FE 0x06 in)                         bound 595
  router : two rings  ring[0] / ring[1] (0x06 / 0x03 in) bounds 630 / 746
  decode : one ring (router 0x05 in)                     bound 602

rx_ring_t = head,tail,dropped,hwm (16 B) + buf[1024]; sizeof = 1040. dropped@+8, hwm@+12.
Bases are the flashed ELFs' symbols (verified 2026-07): gap9 0x2000dcfc, router 0x20008a24,
decode 0x20010820. Read the base from the ELF with --base auto if a rebuild ever moves them."""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_common as B

SIZEOF = 1040
BOARDS = {
    "gap9":   dict(base=0x2000dcfc, rings=[("in", 595)]),
    "router": dict(base=0x20008a24, rings=[("ring0(0x06)", 630), ("ring1(0x03)", 746)]),
    "decode": dict(base=0x20010820, rings=[("in", 602)]),
}
# router_t R @ 0x20000780: slots[8] (8*3336 = 0x6840) then int paired, mismatches, dropped_full,
# hwm(slot-occupancy), evictions. Task-5 pairing invariants — evictions/mismatches must be 0.
ROUTER_R = 0x20000780 + 0x6840
ROUTER_R_FIELDS = ["paired", "mismatches", "dropped_full", "slot_hwm", "evictions"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("board", choices=list(BOARDS))
    ap.add_argument("--base", default=None, help="override ring base (hex), e.g. 0x2000dcfc")
    args = ap.parse_args()
    b = BOARDS[args.board]
    base = int(args.base, 16) if args.base else b["base"]
    print(f"== {args.board}: ring base {base:#010x}, ring size 1024 ==")
    worst = 0
    for i, (name, bound) in enumerate(b["rings"]):
        ring = base + i * SIZEOF
        hwm, dropped = B.read_ring(ring, has_hwm=True)
        verdict = "OK" if (hwm < 1024 and dropped == 0 and hwm <= bound) else "OVER-BOUND" if hwm > bound else "DROPS"
        if hwm > bound or dropped:
            verdict = "!! " + ("DROPS" if dropped else "OVER-BOUND")
        print(f"  {name:>12} @ {ring:#010x}: hwm {hwm:>5} / bound {bound:<4} | dropped {dropped} -> {verdict}")
        worst = max(worst, hwm)
    print(f"  worst HWM {worst} (ring cap 1024)")
    if args.board == "router":
        vals = {f: B.read_word(ROUTER_R + 4 * i) for i, f in enumerate(ROUTER_R_FIELDS)}
        inv = "OK" if (vals["mismatches"] == 0 and vals["evictions"] == 0 and vals["dropped_full"] == 0) else "!! INVARIANT"
        print(f"  R pairing: paired={vals['paired']} mismatches={vals['mismatches']} "
              f"evictions={vals['evictions']} dropped_full={vals['dropped_full']} "
              f"slot_hwm={vals['slot_hwm']} -> {inv}")


if __name__ == "__main__":
    main()
