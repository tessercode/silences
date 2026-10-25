#!/usr/bin/env bash
# Fetch the toolchain pieces the emulator gates need into firmware/third_party/ (not committed):
# the Pico SDK 2.2.0, Renode 1.16.1 (portable, Linux x86-64) and Renode_RP2040 at the commit used.
# arm-none-eabi-gcc comes from your distribution (the images of record used 13.2.1).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p third_party && cd third_party
[[ -d pico-sdk ]] || { git clone -b 2.2.0 --depth 1 https://github.com/raspberrypi/pico-sdk.git && git -C pico-sdk submodule update --init; }
if [[ ! -x renode/renode ]]; then
    curl -L -o renode.tar.gz https://github.com/renode/renode/releases/download/v1.16.1/renode-1.16.1.linux-portable-dotnet.tar.gz
    mkdir -p renode && tar -xzf renode.tar.gz -C renode --strip-components=1 && rm renode.tar.gz
fi
[[ -d Renode_RP2040 ]] || { git clone --recursive https://github.com/matgla/Renode_RP2040.git && git -C Renode_RP2040 checkout 205a5e4b25440582008a4292074bb07f80a72328; }
echo "third_party/ ready: pico-sdk $(git -C pico-sdk describe --tags), $(renode/renode --version 2>/dev/null | head -1)"
