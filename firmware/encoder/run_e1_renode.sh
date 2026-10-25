#!/usr/bin/env bash
# Encoder on the emulated RP2040: boot the test image headless, capture the computed tick/channel stream
# on uart0 and check it with the host gate's comparator.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD="$(cd "$HERE/.." && pwd)/third_party"
RENODE="${RENODE:-$THIRD/renode/renode}"
VENV="${PYTHON:-python3}"
export PICO_SDK_PATH="${PICO_SDK_PATH:-$THIRD/pico-sdk}"
[[ -x "$RENODE" ]] || { echo "ERROR: Renode not found at $RENODE" >&2; exit 3; }
"$VENV" "$HERE/gen_e1_vectors.py"
( cd "$HERE/build" && make e1_test -j"$(nproc)" >/dev/null 2>&1 )
RESLOG="$HERE/result_e1.bin"; rm -f "$RESLOG"
RUNRESC="$HERE/.e1_run.resc"
cat > "$RUNRESC" <<EOR
\$fw=@$HERE/build/e1_test.elf
\$reslog=@$RESLOG
include @$HERE/verify_e1.resc
emulation RunFor "240"
quit
EOR
echo "[e1] booting emulated RP2040 (headless)..."
timeout 3600 "$RENODE" --disable-gui --console "$RUNRESC" > "$HERE/.renode_e1.log" 2>&1 || true
rm -f "$RUNRESC"
grep -q "E1END" "$RESLOG" || { echo "[e1] FAIL — no E1END sentinel (see .renode_e1.log)" >&2; exit 2; }
"$VENV" "$HERE/compare_e1.py" "$HERE/host/cases.txt" "$RESLOG"
