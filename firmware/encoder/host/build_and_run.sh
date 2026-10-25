#!/usr/bin/env bash
# Encoder host gate: the firmware's renderer compiled for the host (ASan + UBSan, no recovery) against the
# software reference (silences.encoder_ref) on every chain-golden sentence.
set -euo pipefail
cd "$(dirname "$0")"
"${PY:-python3}" gen_cases.py
gcc -O2 -std=c11 -Wall -Wextra \
    -fsanitize=address,undefined -fno-sanitize-recover=all \
    -I mock -I .. \
    enc_host_check.c -lm -o enc_host_check
./enc_host_check
