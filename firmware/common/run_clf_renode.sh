#!/usr/bin/env bash
# Renode gate for the LIF / MLP gap classifier: build clf_bench, boot it headless on the emulated
# RP2040 through verify.resc, and diff its uart0 output against the Q16.16 golden with the SAME
# comparator the host gate uses. RunFor 150 (virtual s) covers the 50 vectors with margin.
#   usage: run_clf_renode.sh lif|mlp
set -euo pipefail
NET="${1:?usage: run_clf_renode.sh lif|mlp}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD="$(cd "$HERE/../segmenter_${NET}" && pwd)"
THIRD="$(cd "$HERE/.." && pwd)/third_party"
RENODE="${RENODE:-$THIRD/renode/renode}"
VENV="${PYTHON:-python3}"
GOLDEN="$HERE/../golden/segmenter_${NET}.json"
export PICO_SDK_PATH="${PICO_SDK_PATH:-$THIRD/pico-sdk}"
[[ -x "$RENODE" ]] || { echo "ERROR: Renode not found at $RENODE" >&2; exit 3; }

echo "[clf_${NET}] building clf_bench..."
mkdir -p "$MOD/build"
( cd "$MOD/build" && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make clf_bench -j"$(nproc)" >/dev/null )
UARTLOG="$MOD/build/uart_capture_renode.log"
rm -f "$UARTLOG"
RUNRESC="$MOD/build/.clf_run.resc"
cat > "$RUNRESC" <<EOR
\$fw=@$MOD/build/clf_bench.elf
\$uartlog=@$UARTLOG
include @$HERE/verify.resc
emulation RunFor "150"
quit
EOR
echo "[clf_${NET}] booting emulated RP2040 (headless)..."
timeout 1800 "$RENODE" --disable-gui --console "$RUNRESC" > "$MOD/build/.renode_clf.log" 2>&1 || true
rm -f "$RUNRESC"
[[ -s "$UARTLOG" ]] || { echo "[clf_${NET}] FAIL — no uart0 capture (see build/.renode_clf.log)" >&2; exit 2; }
echo "[clf_${NET}] comparing the uart0 capture to reference_clf_${NET}.json..."
tr -d '\r' < "$UARTLOG" | "$VENV" "$HERE/clf_compare.py" "$GOLDEN"
