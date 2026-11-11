#!/usr/bin/env bash
# Every board's host gate: each firmware module compiled with the host gcc (address and undefined-behaviour
# sanitizers on) and checked against its golden. Needs no Pico SDK and no emulator.
#   PY=<python with this package's requirements> ./check_host.sh
set -euo pipefail
cd "$(dirname "$0")"
FW="$PWD"
export PY="${PY:-python3}"
B="$FW/../outputs/host"; mkdir -p "$B"
CC="gcc -O2 -fsanitize=address,undefined -fno-sanitize-recover=all"
step() { echo "== $*"; }

step "encoder: render vs the encoder reference"; bash encoder/host/build_and_run.sh | tail -1
step "front end: capture and 0x06 frames";        bash front_end/host/build_and_run.sh | tail -1
for m in spk lif mlp; do
    step "segmenter_$m: Q16.16 classifier core"
    (cd segmenter_$m && $CC -I. -I../common ../common/clf_host_main.c clf_forward.c -o "$B/clf_$m" -lm)
    "$B/clf_$m" | "$PY" common/clf_compare.py golden/segmenter_$m.json
done
for n in 1 5; do
    step "segmenter_bigru: model core, $n seed(s)"
    (cd segmenter_bigru && $CC -g -DGAP9_3C_N_SEEDS=$n -o "$B/bigru_n$n" host_main.c gap9_3c.c)
    "$B/bigru_n$n" | "$PY" segmenter_bigru/compare_output.py segmenter_bigru_n$n.json
    step "segmenter_bigru: 0x06 in -> 0x03 out, $n seed(s)"
    (cd segmenter_bigru && $CC -Ihost/mock -I. -I../common -DGAP9_3C_N_SEEDS=$n -o "$B/b2b_n$n" host/b2b_host_check.c)
    "$B/b2b_n$n" golden/frames_n$n.json | tail -1
done
(cd router && $CC -I../common -o "$B/b2b2" host/b2b2_host_check.c)
for n in n1 n5; do
    step "router: 0x06 + 0x03 in -> 0x05 out ($n)"; "$B/b2b2" golden/frames_$n.json | tail -1
    step "decoder: 0x05 in -> decoded text ($n)"
    "$PY" decoder/gen_b2b3_vectors.py $n >/dev/null
    (cd decoder && $CC -Ihost/mock -I. -I../common -DCHARLM_EMBED=1 -o "$B/b2b3_$n" host/b2b3_host_check.c)
    "$B/b2b3_$n" | tail -1
done
step "common: UART receive ring"
(cd common/host && $CC -I.. -o "$B/rx_ring" rx_ring_test.c && "$B/rx_ring" | tail -1)
step "known answers: the encoder's generator vs the published xoshiro128** and SplitMix64"
(cd .. && "$PY" -m silences.golden.check rng | tail -1)
echo "ALL HOST GATES PASS"
