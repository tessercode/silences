# Firmware: the five-board RP2040 sentence decoder (Section 9)

```
encoder ──waveform──> front_end ──0x06──> segmenter ──0x03──> router ──0x05──> decoder ──USB──> text
                                    └──────────────0x06──────────────┘
```

| Board | Directory | Image of record | What it does |
|---|---|---|---|
| Encoder | `encoder/` | `encoder_gpio` | Keys a sentence typed on its console as TEMPO completion spikes on two GPIO lines, rendered by the generator in `enc_rng.h` and the operators in `fists.h` |
| Front end | `front_end/` | `fe_sentence_b1` | Captures the two lines, splits transmissions at 1.75 s of silence, sends each as a 0x06 frame with its time-unit estimate |
| Segmenter | `segmenter_lif/`, `segmenter_mlp/`, `segmenter_bigru/` | `clf_hw` (LIF, MLP); `gap9_hw_3c_n5` (BiGRU) | Turns the marks of a 0x06 frame into per-gap log-posteriors (a 0x03 frame); one image per segmenter, the same board |
| Router | `router/` | `router_hw` | Pairs each 0x03 frame with its 0x06 frame's channels and forwards a 0x05 frame |
| Decoder | `decoder/` | `decode_board` | Beam search with the character language model; decoded text over USB |

`common/` holds what the boards share: the classifier core (`clf_core.h`) behind the LIF and MLP
segmenters, the router core, the UART receive ring and PIO programs, the lookup tables. The frame formats
are documented in `silences/frames.py`.

## The software reference and the goldens

Every board's arithmetic has a reference implementation in the Python package: `silences.encoder_ref`
(the encoder's generator and renderer), `silences.front_end`, `silences.fixedpoint` and
`silences.segmenters.bigru` (the segmenters in Q16.16), `silences.frames`, and `silences.decoder.beam`.
The goldens in `golden/` are those references' outputs on fixed inputs, and every check below compares
the firmware against them bit for bit. `python -m silences.golden.check` recomputes every golden from the
package; `python -m silences.golden.headers` regenerates the generated headers (weights, lookup tables,
the language-model table, test vectors) and checks them.

| Golden | Checked by |
|---|---|
| `rng_kat.json` | the encoder's generator against the published xoshiro128** and SplitMix64 (`tests/kat/`) |
| `fe_*.json` | the front end: waveform in, 0x06 frame out |
| `segmenter_{lif,mlp}.json`, `segmenter_bigru_n{1,5}.json` | each segmenter's posteriors on the 97 (BiGRU) or 50 (LIF, MLP) test vectors |
| `frames_n{1,5}.json` | the BiGRU segmenter's 0x03 and the router's 0x05 frames |
| `beam_n{1,5}.json`, `beam_cap_vectors.json` | the decoder's text, gap classes and final score, including vectors at the 255-gap cap |
| `chain_bigru_n{1,5}.json`, `chain_{lif,mlp}.json` | the whole chain, sentence by sentence (what the hardware runs were compared with) |

## Running the checks

```bash
PY=../.venv/bin/python ./check_host.sh          # every board on the host (gcc, sanitizers), no SDK needed
./third_party_fetch.sh                          # once: Pico SDK 2.2.0, Renode 1.16.1, Renode_RP2040
PYTHON=../.venv/bin/python ./check_renode.sh    # every board on the emulated RP2040 (about 30 minutes)
```

Renode's RP2040 models only uart0 and cannot drive or observe GPIO or the PIO receive lines, so each
emulator test feeds its board's inputs from a baked header and reports on uart0; the GPIO and PIO paths
themselves were checked on the bench (`bench/`).

## Building the images

```bash
export PICO_SDK_PATH=$PWD/third_party/pico-sdk
cmake -S segmenter_lif -B segmenter_lif/build -DCMAKE_BUILD_TYPE=Release && make -C segmenter_lif/build clf_hw
```

The chain's decoder image was built with `-DB2_SILICON_HWM=ON` (receive-ring occupancy readable over
SWD). Rebuilt images match the source of the images flashed for the paper's runs, but are not expected to
hash-match those flashed binaries: compiler and build paths change the bytes.
