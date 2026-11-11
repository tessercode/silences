# Firmware: the five-board RP2040 sentence decoder (Section 9)

```
encoder ──waveform──> front_end ──0x06──> segmenter ──0x03──> router ──0x05──> decoder ──USB──> text
                                    └──────────────0x06──────────────┘
```

| Board | Directory | Image of record | What it does |
|---|---|---|---|
| Encoder | `encoder/` | `encoder_gpio` | Keys a sentence typed on its console as TEMPO completion spikes on two GPIO lines, rendered by the generator in `enc_rng.h` and the operators in `fists.h` |
| Front end | `front_end/` | `fe_sentence_b1` | Captures the two lines, splits transmissions at 1.75 s of silence, sends each as a 0x06 frame with its time-unit estimate |
| Segmenter | `segmenter_spk/`, `segmenter_lif/`, `segmenter_mlp/`, `segmenter_bigru/` | `clf_hw` (the LIF fed spikes, the image of record since v1.2.0; the LIF fed currents; the MLP); `gap9_hw_3c_n5` (BiGRU) | Turns the marks of a 0x06 frame into per-gap log-posteriors (a 0x03 frame); one image per segmenter, the same board |
| Router | `router/` | `router_hw` | Pairs each 0x03 frame with its 0x06 frame's channels and forwards a 0x05 frame |
| Decoder | `decoder/` | `decode_board` | Beam search with the character language model; decoded text over USB |

`common/` holds what the boards share: the classifier core (`clf_core.h`) behind the LIF and MLP
segmenters (the spike-fed LIF adds its input encoder in its own `clf_forward.c`), the router core, the UART receive ring and PIO programs, the lookup tables. The frame formats
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
| `segmenter_{spk,lif,mlp}.json`, `segmenter_bigru_n{1,5}.json` | each segmenter's posteriors on the 97 (BiGRU) or 50 (spike-fed LIF, LIF, MLP) test vectors |
| `frames_n{1,5}.json` | the BiGRU segmenter's 0x03 and the router's 0x05 frames |
| `beam_n{1,5}.json`, `beam_cap_vectors.json` | the decoder's text, gap classes and final score, including vectors at the 255-gap cap |
| `chain_bigru_n{1,5}.json`, `chain_{spk,lif,mlp}.json` | the whole chain, sentence by sentence (what the hardware runs were compared with; on one of `chain_spk.json`'s 63 sentences the beam dies at the default decoder weights, in software and on the board alike) |

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
cmake -S segmenter_spk -B segmenter_spk/build -DCMAKE_BUILD_TYPE=Release && make -C segmenter_spk/build clf_hw
```

The chain's decoder image was built with `-DB2_SILICON_HWM=ON` (receive-ring occupancy readable over
SWD). Rebuilt images match the source of the images flashed for the paper's runs, but are not expected to
hash-match those flashed binaries: compiler and build paths change the bytes.

## Reading the source comments

The firmware was written in stages, and its comments name those stages. The labels are historical: they
say which step introduced or validated a piece of code, and they have no meaning outside these comments.
Comments that cite a `*.md` note or a `deploy/` or `phase2*` path refer to design notes and earlier
firmware in the research repository, which is not part of this release.

| Label | Meaning |
|---|---|
| `A<n>` | A per-board port of one software module to the RP2040: `A2` the BiGRU segmenter (one seed), `A3` the same code built as the five-seed ensemble, `A4` the beam decoder and its Q16.16 language-model table. `A1` is the earlier word-scoped decoder these boards were developed from |
| `B<n>` | The chain's transport and framing: `B1` the front end and its 0x06 frame, `B2` the 0x03 and 0x05 frames (`B2.0` the frame goldens, `B2a` the interrupt-driven receive rings, `B2b.1` the segmenter board's receive and transmit wrapper, `B2b.2` the router, `B2b.3` the decoder's receive path), `B3` the router's slot-eviction rule |
| `E.<n>` | The encoder board: `E.0` the chain's frozen reference, `E.1` the port of the encoder's generator and renderer, `E.2` its timing measured on the bench |
| `<stage>.0`, `.0b`, `.1`, `.2` | Within a stage: the inventory of what existed, the golden reference, the port checked on the host and on Renode, and the check on physical boards |
| `D<n>` | A design decision recorded during a stage, for example `D1` (the segmenter runs in Q16.16, not int8) and `D4` (the 0x03 frame carries posteriors only, with no decision bits) |
| `G<n>`, `T<n>` | A numbered gate or test within the stage the comment belongs to |
| `gap9`, `gap9_3c` | The BiGRU baseline segmenter (nine features per gap), and its three-class form, which is the one in the paper |
| `phase2a`, `phase2b`, `phase2c` | The earlier word-scoped firmware: its floating-point reference vectors, its Q16.16 datapath (reused here), and an int8 variant that was dropped |
| sentence tree | These boards, as opposed to the earlier word-scoped firmware |
