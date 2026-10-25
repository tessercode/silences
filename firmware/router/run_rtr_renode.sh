#!/usr/bin/env bash
# Router on the emulated RP2040: for each ensemble, bake its golden (0x06, 0x03) frame pairs, rebuild the
# test image, boot it headless, capture the 0x05 result line and check it with the host gate's comparator. RunFor 200 (virtual s): the ~70 s of 0x05 wire time at 9600 + margin.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD="$(cd "$HERE/.." && pwd)/third_party"
RENODE="${RENODE:-$THIRD/renode/renode}"
VENV="${PYTHON:-python3}"
export PICO_SDK_PATH="${PICO_SDK_PATH:-$THIRD/pico-sdk}"
G="$HERE/../golden"
[[ -x "$RENODE" ]] || { echo "ERROR: Renode not found at $RENODE" >&2; exit 3; }
for ns in n1 n5; do
    echo "[rtr_${ns}] baking ${ns} input pairs + rebuilding the runner..."
    "$VENV" "$HERE/gen_rtr_vectors.py" "$ns" >/dev/null
    ( cd "$HERE/build" && make rtr_test -j"$(nproc)" >/dev/null 2>&1 )
    ELF="$HERE/build/rtr_test.elf"; RESLOG="$HERE/result_rtr_${ns}.bin"
    rm -f "$RESLOG"
    RUNRESC="$HERE/.rtr_run_${ns}.resc"
    cat > "$RUNRESC" <<EOR
\$fw=@$ELF
\$reslog=@$RESLOG
include @$HERE/verify_rtr.resc
emulation RunFor "200"
quit
EOR
    echo "[rtr_${ns}] booting emulated RP2040 (headless)..."
    timeout 2400 "$RENODE" --disable-gui --console "$RUNRESC" > "$HERE/.renode_rtr_${ns}.log" 2>&1 || true
    rm -f "$RUNRESC"
    tail -c 7 "$RESLOG" 2>/dev/null | xxd -p | grep -q "a55a0500010255" || {
        echo "[rtr_${ns}] FAIL — no end-of-pass sentinel (see .renode_rtr_${ns}.log)" >&2; exit 2; }
    echo "[rtr_${ns}] comparing the uart0 capture to frames_${ns}.json..."
    "$VENV" "$HERE/compare_rtr.py" "$G/frames_${ns}.json" "$RESLOG"
done
echo "[rtr] PASS — both builds reproduce the golden 0x05 frames bit-exactly on emulated ARM."
