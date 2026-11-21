# Reproducing the results

Each level checks itself against what is committed and says so. The times are on a 24-core machine.
Start with `make setup` (a Python 3.12 venv with the versions in `requirements.txt`, and the NLTK
`words` and `brown` corpora).

## Which results are bit-identical where

Level 1 (the paper from the records) and every firmware check use integer arithmetic or fixed inputs,
and reproduce exactly on any machine. Levels 2 and 3 run floating-point code (numpy's log, exp and tanh,
BLAS matrix products, torch) whose last bits depend on the CPU's vector instructions. They were verified
bit for bit on x86-64 with AVX2 (an Intel Core Ultra 9 285K, and a GitHub-hosted AMD EPYC 7763). On a CPU
whose kernels differ, for example one with AVX-512, the training matrix can differ in its last bits
(the training scripts then say so: it still equals the original to 9 decimals); retrained models will
then not be byte-identical, and the evaluation records may differ in rare values. Rounding to Q16.16
before decoding absorbs almost all of such differences; `make evaluate` lists any that remain.

## 1. The paper from the records (seconds)

```bash
make paper
```

`paper/make_figures.py` reads the records, checks each against its manifest's hashes and protocol
hash, and writes `paper/numbers.tex` (the numbers the prose and the proposal cite, listed in
`paper/consumed_macros.txt`), the six tables and the four data figures. The PDFs carry no creation date, so the whole `paper/` directory must come out
byte-identical; `make paper` ends with `git diff --exit-code` on it.

## 2. The records from the shipped models

```bash
make evaluate-matched     # the matched-data record, about 40 minutes
make evaluate-spike-fed   # the record of the paper: the spike-fed families against it, about 25 minutes
make spike-character      # what the spikes carry: both LIF forms, about 40 minutes
make evaluate-width       # the window width and the spiking networks at half-width 5, about 25 minutes (needs torch)
make evaluate-tempo       # every family under a per-gap and a per-word time unit, about 20 minutes
make evaluate-nonlearned  # the threshold readouts, multimon-ng and the checks on the fits, about 30 minutes
make evaluate             # the superseded 5,000-transmission records, about 10 minutes
```

`make evaluate-spike-fed` runs `silences.evaluation.spike_fed`: the thirty spike-fed and T = 80 models
(and the deployed spike-fed network in Q16.16) on the matched record's eighteen corpora, whose other
rows are copied from `outputs/tables_matched/` (so `make evaluate-matched` runs first); the anchors
(the harness reproduces the matched record's `lif_op12_seed0` row exactly; no training transmission
is in any evaluation corpus), the (T, γ) fits, the evaluation at the fitted and the default weights,
criterion 1A on the spike-fed and current-fed families (anchored on the matched record's values), the
deployed network's fixed-point fidelity over every gap, and the statistics of
`records/spike_fed/results.json`. `make spike-character` then runs the timing-destroying controls
and the spike statistics of `records/spike_fed/spikes_{spk40,lif}.json`; its `torch` step
(the training-side forward against the inference forward) needs torch and is run explicitly:
`python -m silences.evaluation.spike_character spk40 torch`. `check_train.json` (the feature-matrix
hashes and the forward check of every model) is likewise `python -m silences.evaluation.spike_fed check-train`.

The four records added in v1.3.0 are chained to these and run after them, in this order.

`make evaluate-width` runs `silences.evaluation.width`, then `silences.evaluation.width_lif`. The first adds to
the matched record's eighteen corpora the control at half-widths 2, 5, 15 and 20, the baseline's recipe on the
window's two raw values, and the baseline of record through a floating-point path (these two need torch); its
anchors are that the harness reproduces the matched record's `mlp_op12_seed0` row, that no training
transmission is in an evaluation corpus, that the op12 feature matrices at every width hash to the ones the
shipped models were trained on, and that the float baseline agrees with its fixed-point rows (to 0.0001 pp of
character error on English). It also writes per-gap accuracy, the inter-spike interval, the window's lookahead
and the renderer's channel error (`records/width/gapacc.json`), which Sections 7.1 and 9.5 and Appendix A cite.
The second scores the spike-fed and current-fed networks at half-width 5 beside both records.

`make evaluate-tempo` runs `silences.evaluation.tempo`: the spike-fed LIF, the current-fed LIF, the control and
the baseline on the eight record corpora with the time unit supplied four ways (the front end's single
estimate, a centred and a trailing per-gap estimate, the renderer's true unit per word), every model at its
weights and decoder weights of record. Under the single estimate each row must reproduce the record's stored
row exactly, and does. The same module re-runs the earlier rule-based reader whose figures Section 5.2 quotes.

`make evaluate-nonlearned` runs `silences.evaluation.nonlearned`: the threshold readout's five parameters fitted
on each training seed's corpus, the four readout forms scored like networks, multimon-ng on audio rendered
from the record corpora, every learned family refitted to word error and on an extended grid, and the readouts
refitted over negative insertion bonuses. The audio step needs the `multimon-ng` program on the PATH (the
record was made with version 1.3.0; Debian and Ubuntu package it); run the other steps one at a time
(`python -m silences.evaluation.nonlearned fitrule`, `tables`, ...) to skip it, in which case `check` and
`stats` use the committed `audio.json`.

`make evaluate-matched` runs `silences.evaluation.matched`: every segmenter of the grid (the original
rows, the three BiGRU arms with their ensembles, the op12 and gen feedforward families, the deployed
networks in Q16.16) on eighteen corpora, the ten of the decoder-weight protocol and four fresh
evaluation corpora per text (seeds 1106, 2212, 3318, 4424); then the (T, γ) fit of every row on the
calibration corpora, the evaluation at the fitted and the default weights, the fixed-point fidelity,
criterion 1A's reference means, and the statistics of `records/matched/results.json`. Rows already in
`outputs/tables/` are copied, not recomputed. `tools/compare_records.py --git records/matched` then
compares every value with the committed record.

`make evaluate` regenerates the two earlier records:

| Step | What it does | Time |
|---|---|---|
| `silences.evaluation.tables` | Every segmenter's per-gap posteriors on the ten corpora: for random letters and for English, a calibration corpus (seeds 4242, 4243) and four evaluation corpora (999, 777, 555, 321), 250 transmissions per operator each | 2 min |
| `silences.evaluation.random_letters` | Random letters at the default insertion bonus: the posterior scale T fitted per model on seed 4242, the evaluation at that T and at T = 1, the Welch statistics, the joint (T, γ) diagnostic | 1 min |
| `silences.evaluation.decoder_weights` | Both texts: (T, γ) fitted per model and text on the calibration corpora, the evaluation at the fitted and the default weights, the statistics, the γ-grid edge check | 6 min |

The records are rewritten in place, then `tools/compare_records.py --git` compares every value with
the committed version, ignoring only provenance (timestamps, commit and environment details, file
hashes and paths). The posterior tables are cached in `outputs/tables/`; a second run skips that step.

The protocols the records follow are pre-registered in `protocol/`. They were written in the research
repository's vocabulary; the glossary in `records/README.md` maps their terms to this package.

## 3. The models (minutes to hours)

```bash
make train-spk40 SEED=0 CORPUS=op12 # the network of record, the LIF fed spikes: about 3 hours per seed, CPU
make train-spk80 SEED=0 CORPUS=op12 # the same at T = 80, about 6 hours; train-lif80 the current-fed form at T = 80
make train-lif SEED=0 CORPUS=op12   # the current-fed LIF of the matched record: about 70 minutes per seed, CPU
make train-mlp SEED=0 CORPUS=op12   # about 2 minutes per seed, CPU
make train-lif SEED=0 CORPUS=gen    # the generic-corpus control (the BiGRU recipe's final-phase corpus)
make train-lif SEED=0               # the original 5,000-transmission corpus, about 18 minutes per seed
make train-mlp SEED=0               # about 30 seconds per seed
make train-width FAMILY=mlp_w5 SEED=0   # a width family: mlp_w2, mlp_w5, mlp_w15, mlp_w20 (about 3 minutes each),
                                        # lif_w5 and spk40_w5 (as their record forms above)
make train-bigru ARM=fist           # the baseline of record, about 5 minutes on a CUDA GPU
make train-bigru ARM=raw            # the same recipe on the window's two raw values per gap
make train-bigru                    # the original eight-seed pool; ARM=rep and ARM=shared are the other arms
```

LIF and MLP training is deterministic on the CPU: the retrained pickle (in `outputs/retrain/`) must be
byte-identical to `models/<family>/seed<N>.pkl` (or `models/<family>/<corpus>/seed<N>.pkl`), and the
command says whether it is. Seeds run 0 to 9. The op12 and gen corpora are generated per seed and
their feature matrices are checked against the hashes in `records/training/window_nets_grid.json`
before training starts. The deployed networks are `models/{spk40,lif,mlp}/op12/seed0.pkl`, so
`CORPUS=op12 SEED=0` regenerates what runs on the boards (the spike-fed network since 2026-11-10).

The BiGRU pool trains on the GPU in bfloat16 with CUDA graphs. On the GPU and software it was trained
with (an RTX 5080, torch 2.11.0 with CUDA 12.8) every retrained weight equals the shipped
checkpoint's; elsewhere expect the retrain to agree statistically, which the command checks against
`records/training/bigru_pool.json` (each seed's selection accuracy within the shipped pool's range
widened by its own spread, the ensemble above the 93.9% floor).

The width families were trained with two threads, not six, and `make train-width` retrains them that way; the
thread count is part of what makes a retrain byte-identical.

The BiGRU arms (`ARM=rep|fist|shared`) swap only the final phase's corpus; `ARM=raw` keeps the fist arm's
corpora and swaps the nine features for the window's two raw values (its checkpoints are `models/bigru/raw/`,
its training record `records/width/research/exp215_train_raw.json`); on the GPU and software
above each arm retrains bit-identical, and `records/training/bigru_arms.json` carries the statistical
check elsewhere. `models/lif/deployed_5k.pkl` and `models/mlp/deployed_5k.pkl`, the networks first
deployed (the `lif_committed` and `mlp_committed` rows), were trained by an earlier version of the
training script and are shipped as artifacts of record.

```bash
make lm                   # rebuild the character language model from NLTK Brown and the templated ham corpus
make collision            # rerun the collision study behind Figure 1
```

Both must reproduce their committed files byte for byte.

## 4. The firmware

```bash
make check                # includes every firmware golden recomputed from the package, and every generated header
make firmware-host        # every board built for the host with gcc (sanitizers on) and run against its golden
make firmware-renode      # every board built for the RP2040 and run on Renode's emulated Cortex-M0+
```

The goldens in `firmware/golden/` are the boards' expected outputs: the encoder's waveforms, the front
end's 0x06 frames, each segmenter's posteriors, the router's 0x05 frames and the decoder's text.
`silences.golden.check` recomputes every value in every golden with this package's reference code.
`firmware/check_host.sh` and `firmware/check_renode.sh` run the firmware itself against the same
goldens. The emulator gates need the Pico SDK 2.2.0, arm-none-eabi-gcc (the images of record used
13.2.1), Renode 1.16.1 and Renode_RP2040; see `firmware/README.md`.

The hardware measurements in Section 9 (the five-board chain, per-gap latency, transport, footprint)
were made on the bench with `bench/`; see `bench/README.md`. Their results are in `records/deployment/`
(the spike-fed network's runs of 2026-11-10 are the paper's; the current-fed network's of 2026-11-06/08
stay as the ladder's middle rung).
