# Playing the Silences: code, models and records

This repository reproduces *Playing the Silences: A Feedforward Spiking Gap Detector for Word-Level
Decoding of TEMPO Morse* (Hindman and Cantley, 2026).

The paper's question is whether a strictly feedforward spiking network can find the gaps in TEMPO
Morse well enough to decode text. Every gap between two marks is either intra-letter, inter-letter or
inter-word. A segmenter assigns each gap a posterior over those three classes, and a beam search with
a character language model turns the posteriors into text. This repository holds:

- the synthetic operators and the corpus renderer (`silences/corpus.py`), the front end's
  time-unit estimate and the gap-window features;
- the three segmenters: the feedforward LIF network, its tanh MLP control and the bidirectional-GRU
  baseline, with every trained model the paper reports: the families of record, trained on 12,000
  operator transmissions per seed (ten LIF seeds, ten MLP seeds, the eight-seed BiGRU pool with its
  final phase on that corpus), the rest of the model-by-corpus grid of Table 4 (the same recipes on
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
| Table 3, English error rates of the deployed models | `paper/cerwer_table_eng.tex` | `records/matched/` |
| Table 4, model by training corpus | `paper/grid_table.tex` | `records/matched/` (the grid block) |
| Table 5, training-seed distributions | `paper/seed_table_dw.tex` | `records/matched/` |
| Figure 7, error by operator | `paper/figures/fig_cerwer_eng.pdf` | `records/matched/` |
| Figure 8, the decoder weights | `paper/figures/fig_decoder_weights.pdf` | `records/matched/fit.json` |
| Random-letter results and the fixed-point fidelity quoted in the text | `paper/numbers.tex` | `records/matched/` |
| Section 9, the RP2040 chain and bench | `paper/numbers.tex` | `records/deployment/` (the 2026-11 runs) |

Every number the prose cites is a macro in `paper/numbers.tex`; none is typed by hand. The records are
produced by `make evaluate-matched` and `make evaluate` (see REPRODUCE.md), and `records/EQUIVALENCE.md`
shows they equal, value for value, the records the paper was written from.

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
| `protocol/` | The two pre-registered evaluation protocols, verbatim |
| `models/` | Every trained model: `lif/`, `mlp/` (`seed*.pkl` on the original corpus; `op12/` the families of record, `gen/` the generic-corpus control; `deployed_5k.pkl` the first deployed networks), `bigru/` (`pool/` the original pool; `fist/` the baseline of record, `rep/`, `shared/`; each also as Q16.16 weights); the language-model table |
| `records/` | The result records (`matched/` is the paper's; `decoder_weights/` and `random_letters/` the superseded ones), the training records, the deployment measurements |
| `paper/` | `make_figures.py` and what it writes |
| `firmware/` | The five boards' firmware, their host and emulator tests, and the goldens |
| `bench/` | The scripts that ran the hardware measurements |
| `data/` | The operator-style fits of the real recordings, and the firmware test inputs |
| `tests/`, `tools/` | Fast checks, and the record comparer |

## Licence and citation

MIT (see `LICENSE`). Third-party material is listed in [THIRD_PARTY.md](THIRD_PARTY.md). Please cite the
paper and this release: see [CITATION.cff](CITATION.cff). The TEMPO benchmark itself is described in
the TEMPO paper; its library is at [tessercode/tempo](https://github.com/tessercode/tempo).
