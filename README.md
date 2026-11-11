# Playing the Silences: code, models and records

This repository reproduces *Playing the Silences: A Feedforward Spiking Gap Detector for Word-Level
Decoding of TEMPO Morse* (Hindman and Cantley, 2026).

The paper's question is whether a strictly feedforward spiking network can find the gaps in TEMPO
Morse well enough to decode text. Every gap between two marks is either intra-letter, inter-letter or
inter-word. A segmenter assigns each gap a posterior over those three classes, and a beam search with
a character language model turns the posteriors into text. This repository holds:

- the synthetic operators and the corpus renderer (`silences/corpus.py`), the front end's
  time-unit estimate and the gap-window features;
- the segmenters: the feedforward LIF network in its two forms (fed the gap window as spike trains,
  the network of record since v1.2.0, and fed it as a constant current, the ladder's middle rung), its
  tanh MLP control and the bidirectional-GRU baseline, with every trained model the paper reports: the
  families of record, trained on 12,000 operator transmissions per seed (ten seeds of each LIF form,
  ten MLP seeds, the eight-seed BiGRU pool with its final phase on that corpus), the spike-fed and the
  current-fed network at T = 80, the rest of the model-by-corpus grid of Table 4 (the same recipes on
  the generic corpus and on the original 5,000-transmission corpus), and the deployed networks, which
  are the first seed of each feedforward family;
- the decoder (integer beam search and the Q16.16 character language model), the per-model fit of the
  decoder weights, and the evaluation and statistics (`silences/evaluation/`);
- the result records the paper's numbers come from (`records/`), and the script that turns them into
  the paper's tables, figures and numbers (`paper/make_figures.py`);
- the firmware for the five-board RP2040 decoder (`firmware/`), the bench scripts that measured it
  (`bench/`), and the goldens the firmware is checked against, bit for bit.

## Interactive demo

[tessercode.com/silences-demo](https://tessercode.com/silences-demo/) runs this release's encoder,
front end, segmenters and beam decoder in the browser (Pyodide), with the same models and fixed-point
arithmetic as the RP2040 chain. Its source is at
[tessercode/silences-demo](https://github.com/tessercode/silences-demo).

## Quick start

```bash
make setup     # Python 3.12 venv with the pinned requirements, plus the NLTK word lists
make paper     # every table, figure and cited number, regenerated from the records (seconds)
make check     # unit tests, every firmware golden, every generated header (seconds)
```

`make paper` ends by checking that its output is byte-identical to the committed `paper/` directory.
[REPRODUCE.md](REPRODUCE.md) goes further: regenerating the records from the shipped models (about ten
minutes), retraining the models (bit-identical), and checking the firmware on the host and on an
emulated RP2040.

## Where each result comes from

| In the paper | Produced by | From |
|---|---|---|
| Table 1, the five operators | `paper/fists_table.tex` | `firmware/encoder/fists.h` |
| Figure 1 and its numbers, the 4.0 T_u collision | `paper/figures/fig_collision.pdf`, `records/collision.json` | `python -m silences.collision` |
| Figure 2, style coverage | `paper/figures/fig_coverage.pdf` | `data/operator_style_fits.csv`, `fists.h` |
| Table 2, the training recipes | `paper/training_table.tex` | `protocol/`, `silences/training/`, `records/training/` |
| Table 3, English error rates of the deployed models | `paper/cerwer_table_eng.tex` | `records/spike_fed/`, `records/matched/` |
| Table 4, model by training corpus | `paper/grid_table.tex` | `records/matched/` (the grid block) |
| Table 5, training-seed distributions | `paper/seed_table_dw.tex` | `records/spike_fed/`, `records/matched/` |
| Table 6, the ladder from the control to the spiking network | `paper/ladder_table.tex` | `records/spike_fed/`, `records/matched/` |
| Figure 7, error by operator | `paper/figures/fig_cerwer_eng.pdf` | `records/spike_fed/`, `records/matched/` |
| Figure 8, the decoder weights | `paper/figures/fig_decoder_weights.pdf` | `records/spike_fed/fit.json`, `records/matched/fit.json` |
| Section 8.6, what the spikes carry | `paper/numbers.tex` | `records/spike_fed/spikes_*.json` |
| Random-letter results and the fixed-point fidelity quoted in the text | `paper/numbers.tex` | `records/spike_fed/`, `records/matched/` |
| Section 9, the RP2040 chain and bench | `paper/numbers.tex` | `records/deployment/` (the 2026-11 runs) |

Every number the prose cites is a macro in `paper/numbers.tex`; none is typed by hand. The records are
produced by `make evaluate-spike-fed`, `make evaluate-matched` and `make evaluate` (see REPRODUCE.md), and
`records/EQUIVALENCE.md` shows they equal, value for value, the records the paper was written from.

## What changed in v1.2.0

The paper's spiking network is now the LIF network fed the gap window as spike trains (review item M2:
the current-fed form read its window as a constant current and so did not process spike timing). Each of
the 42 standardised window values is split into an ON and an OFF channel driving a regular spike train
whose rate is proportional to the value and saturates at three standard deviations; the 84 trains enter
the first layer as spikes (`silences.segmenters.lif.SpikeFedLIF`). Trained with the LIF recipe unchanged
on the matched op12 corpora, ten seeds at T = 40 (`models/spk40/op12/`) and ten at T = 80
(`models/spk80/op12/`), beside the current-fed network at T = 80 (`models/lif80/op12/`), it is scored
against the matched record in `records/spike_fed/` (`protocol/spike_fed.json`): on English text it is not
separated from the baseline seeds, and the paper presents the ladder MLP, LIF fed currents, LIF fed spikes
(Table 6). The spike-character diagnostic of both LIF forms (`records/spike_fed/spikes_*.json`,
`silences.evaluation.spike_character`) measures what the spikes carry. Seed 0 of the spike-fed family is
the deployed network: its Q16.16 reference (`FixedWindowNet.from_spk`, with the input encoder in integer
arithmetic), firmware (`firmware/segmenter_spk/`), goldens (`segmenter_spk.json`, `chain_spk.json`) and
bench runs of 2026-11-10 (`records/deployment/*_spk_20261110.json`) are added; the current-fed network's
firmware and records stay under their names as the ladder's middle rung.

## What changed in v1.1.0

The paper's revision for review compares the three segmenters at matched training data. Tracing the
training data (review item M1) showed that the first release's baseline had been trained on a generic
operator distribution and the feedforward networks on 5,000 transmissions from the five evaluation
operators. The baseline was retrained with its final phase on 12,000 operator transmissions per seed,
the feedforward networks on the same kind of corpus at the same budget (and, as a control, on the
baseline's generic corpus), and the comparison of record moved to that row of the grid. The deployed
LIF and MLP networks are now the first training seed of each family of record; the RP2040 chain was
re-validated with them (2026-11-06) and the bench test repeated (2026-11-08). The superseded models,
records and goldens stay in the tree under their original names (`models/*/seed*.pkl` on the original
corpus, `models/*/deployed_5k.pkl`, `records/decoder_weights/`, `records/random_letters/`,
`records/deployment/*_20261014.json` and the undated files) and the paper cites them as such.

## Layout

| Path | Contents |
|---|---|
| `silences/` | The package: `corpus`, `features`, `segmenters/`, `fixedpoint/`, `decoder/`, `evaluation/`, `training/`, `golden/`, plus the device-exact `encoder_ref`, `front_end`, `frames` and `pipeline` |
| `protocol/` | The pre-registered evaluation protocols, verbatim |
| `models/` | Every trained model: `spk40/op12/` (the LIF fed spikes, the family of record; seed 0 deployed), `spk80/op12/` and `lif80/op12/` (both forms at T = 80), `lif/`, `mlp/` (`seed*.pkl` on the original corpus; `op12/` the matched families, `gen/` the generic-corpus control; `deployed_5k.pkl` the first deployed networks), `bigru/` (`pool/` the original pool; `fist/` the baseline of record, `rep/`, `shared/`; each also as Q16.16 weights); the language-model table |
| `records/` | The result records (`spike_fed/` is the paper's, against `matched/`; `decoder_weights/` and `random_letters/` the superseded ones), the training records, the deployment measurements |
| `paper/` | `make_figures.py` and what it writes |
| `firmware/` | The five boards' firmware, their host and emulator tests, and the goldens |
| `bench/` | The scripts that ran the hardware measurements |
| `data/` | The operator-style fits of the real recordings, and the firmware test inputs |
| `tests/`, `tools/` | Fast checks, and the record comparer |

## Licence and citation

MIT (see `LICENSE`). Third-party material is listed in [THIRD_PARTY.md](THIRD_PARTY.md). Please cite the
paper and this release: see [CITATION.cff](CITATION.cff). The TEMPO benchmark itself is described in
the TEMPO paper; its library is at [tessercode/tempo](https://github.com/tessercode/tempo).
