#!/usr/bin/env bash
# The deployed segmenter image's footprint (Section 9): rebuild clf_hw and write arm-none-eabi-size and the
# largest -fstack-usage frames to records/deployment/footprint_<net>.txt, which make_figures.py reads.
# Rebuilding with the pinned toolchain leaves that record unchanged (git diff shows any change).
#   usage: measure_footprint.sh lif|mlp
set -euo pipefail
NET="${1:?usage: measure_footprint.sh lif|mlp}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD="$(cd "$HERE/../segmenter_${NET}" && pwd)"
export PICO_SDK_PATH="${PICO_SDK_PATH:-$(cd "$HERE/.." && pwd)/third_party/pico-sdk}"
mkdir -p "$MOD/build"
( cd "$MOD/build" && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 && make clf_hw -j"$(nproc)" >/dev/null 2>&1 )
OUT="$(cd "$HERE/../.." && pwd)/records/deployment/footprint_${NET}.txt"
{
    echo "# clf_hw.elf footprint ($NET), regenerate with firmware/common/measure_footprint.sh $NET"
    echo "# $(arm-none-eabi-gcc --version | head -1); pico-sdk $(git -C "$PICO_SDK_PATH" describe --tags 2>/dev/null || echo unknown)"
    arm-none-eabi-size "$MOD/build/clf_hw.elf" | sed "s|$MOD/build/||"
    echo
    arm-none-eabi-size -A "$MOD/build/clf_hw.elf" | sed "s|$MOD/build/||"
    echo "# largest stack frames (-fstack-usage)"
    cat "$MOD"/build/CMakeFiles/clf_hw.dir/*.su "$MOD"/build/CMakeFiles/clf_hw.dir/*/*.su 2>/dev/null \
        | sed "s|^.*/||" | sort -t$'\t' -k2 -n -r | head -8 || true
} > "$OUT"
read -r TEXT DATA BSS _ < <(arm-none-eabi-size "$MOD/build/clf_hw.elf" | tail -1)
FLASH_KB=$(( (TEXT + DATA + 500) / 1000 )); SRAM_KB=$(( (DATA + BSS + 500) / 1000 ))
echo "[$NET] measured: flash ${FLASH_KB} KB (text+data $((TEXT + DATA)) B), SRAM ${SRAM_KB} KB (data+bss $((DATA + BSS)) B) -> $OUT"
