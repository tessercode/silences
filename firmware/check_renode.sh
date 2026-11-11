#!/usr/bin/env bash
# The boards' firmware on Renode's emulated RP2040, each checked against its golden. Needs the Pico SDK,
# arm-none-eabi-gcc, Renode and Renode_RP2040 (third_party/, see README.md). Each run takes minutes.
#   PYTHON=<python with this package's requirements> ./check_renode.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHON="${PYTHON:-python3}"
export PICO_SDK_PATH="${PICO_SDK_PATH:-$PWD/third_party/pico-sdk}"
for m in encoder router decoder segmenter_bigru segmenter_spk segmenter_lif segmenter_mlp; do
    [[ -f "$m/build/CMakeCache.txt" ]] || cmake -S "$m" -B "$m/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
done
bash encoder/run_e1_renode.sh 2>&1 | grep -E "PASS|FAIL"
bash router/run_rtr_renode.sh 2>&1 | grep -E "PASS|FAIL"
bash decoder/run_b2b3_renode.sh 2>&1 | grep -E "PASS|FAIL"
bash segmenter_bigru/run_b2b_renode.sh 2>&1 | grep -E "PASS|FAIL"
for m in spk lif mlp; do bash common/run_clf_renode.sh $m 2>&1 | grep -E "BIT-EXACT|FAIL|Error"; done
echo "ALL RENODE GATES PASS"
