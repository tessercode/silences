#!/usr/bin/env bash
# Front-end host gate: the harness compiles the firmware in the same translation unit (ASan + UBSan) and
# runs it on the golden waveforms; every 0x06 frame must match byte for byte. Exit 0 = pass.
set -euo pipefail
cd "$(dirname "$0")"

"${PY:-python3}" gen_cases.py

mkdir -p build
gcc -std=c11 -O1 -g -Wall -Wextra -Wno-unused-parameter \
    -fsanitize=address,undefined -fno-sanitize-recover=all \
    -I mock host_harness.c -o build/fe_host_test

./build/fe_host_test build/cases
