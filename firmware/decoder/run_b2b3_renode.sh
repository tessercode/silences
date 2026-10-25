#!/usr/bin/env bash
# Decoder on the emulated RP2040: for each ensemble, bake its golden 0x05 frames, rebuild the test image,
# boot it headless, capture the decoded result lines on uart0 and check them with the host gate's comparator.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD="$(cd "$HERE/.." && pwd)/third_party"
RENODE="${RENODE:-$THIRD/renode/renode}"
VENV="${PYTHON:-python3}"
export PICO_SDK_PATH="${PICO_SDK_PATH:-$THIRD/pico-sdk}"
A4="$HERE/../golden"
[[ -x "$RENODE" ]] || { echo "ERROR: Renode not found at $RENODE" >&2; exit 3; }
for ns in n1 n5; do
    echo "[b2b3_${ns}] baking the ${ns} goldens + rebuilding the runner..."
    "$VENV" "$HERE/gen_b2b3_vectors.py" "$ns" >/dev/null
    ( cd "$HERE/build" && make b2b3_test -j"$(nproc)" >/dev/null 2>&1 )
    ELF="$HERE/build/b2b3_test.elf"; RESLOG="$HERE/result_b2b3_${ns}.log"
    rm -f "$RESLOG"
    RUNRESC="$HERE/.b2b3_run_${ns}.resc"
    cat > "$RUNRESC" <<EOR
\$fw=@$ELF
\$reslog=@$RESLOG
include @$HERE/verify_b2b3.resc
emulation RunFor "60"
quit
EOR
    echo "[b2b3_${ns}] booting emulated RP2040 (headless)..."
    timeout 2400 "$RENODE" --disable-gui --console "$RUNRESC" > "$HERE/.renode_b2b3_${ns}.log" 2>&1 || true
    rm -f "$RUNRESC"
    grep -q "DONE" "$RESLOG" 2>/dev/null || {
        echo "[b2b3_${ns}] FAIL — no DONE marker (see .renode_b2b3_${ns}.log)" >&2; exit 2; }
    echo "[b2b3_${ns}] comparing the uart0 result lines to beam_${ns}.json..."
    "$VENV" "$HERE/compare_b2b3.py" "$A4/beam_${ns}.json" "$RESLOG"
done
echo "[b2b3] PASS — both ensembles decode the golden 0x05 frames to the golden text on emulated ARM."
