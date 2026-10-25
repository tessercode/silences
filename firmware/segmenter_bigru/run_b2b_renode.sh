#!/usr/bin/env bash
# BiGRU segmenter on the emulated RP2040: boot both builds headless, capture the 0x03 result line and
# check each against its golden with the host gate's comparator.
# RunFor 120 (virtual s): the 67,462 B stream is ~70 s of 9600-baud wire time + ~3 s of compute.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD="$(cd "$HERE/.." && pwd)/third_party"
RENODE="${RENODE:-$THIRD/renode/renode}"
VENV="${PYTHON:-python3}"
export PICO_SDK_PATH="${PICO_SDK_PATH:-$THIRD/pico-sdk}"
G="$HERE/../golden"
[[ -x "$RENODE" ]] || { echo "ERROR: Renode not found at $RENODE" >&2; exit 3; }
for ns in 1 5; do
    ELF="$HERE/build/b2b_test_n${ns}.elf"
    RESLOG="$HERE/result_b2b_n${ns}.bin"
    ( cd "$HERE/build" && make b2b_test_n${ns} -j"$(nproc)" >/dev/null 2>&1 )
    [[ -f "$ELF" ]] || { echo "ERROR: $ELF not built" >&2; exit 3; }
    rm -f "$RESLOG"
    RUNRESC="$HERE/.b2b_run_n${ns}.resc"
    cat > "$RUNRESC" <<EOR
\$fw=@$ELF
\$reslog=@$RESLOG
include @$HERE/verify_b2b.resc
emulation RunFor "120"
quit
EOR
    echo "[b2b_n${ns}] booting emulated RP2040 (headless)..."
    timeout 1800 "$RENODE" --disable-gui --console "$RUNRESC" > "$HERE/.renode_b2b_n${ns}.log" 2>&1 || true
    rm -f "$RUNRESC"
    # Termination is IN-PROTOCOL: the wrapper's own end-of-pass sentinel (KIND 0x02) = a55a0300010221
    tail -c 7 "$RESLOG" 2>/dev/null | xxd -p | grep -q "a55a0300010221" || {
        echo "[b2b_n${ns}] FAIL — no end-of-pass sentinel (see .renode_b2b_n${ns}.log)" >&2; exit 2; }
    echo "[b2b_n${ns}] comparing the uart0 capture to frames_n${ns}.json..."
    "$VENV" "$HERE/compare_b2b.py" "$G/frames_n${ns}.json" "$RESLOG"
done
echo "[b2b] PASS — both builds reproduce the golden 0x03 frames bit-exactly on emulated ARM."
