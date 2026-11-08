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
`paper/consumed_macros.txt`), the five tables and the four data figures. The PDFs carry no creation date, so the whole `paper/` directory must come out
byte-identical; `make paper` ends with `git diff --exit-code` on it.

## 2. The records from the shipped models

```bash
make evaluate-matched     # the record of the paper, about 40 minutes
make evaluate             # the superseded 5,000-transmission records, about 10 minutes
```

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
make train-lif SEED=0 CORPUS=op12   # the family of record: about 70 minutes per seed, CPU
make train-mlp SEED=0 CORPUS=op12   # about 2 minutes per seed, CPU
make train-lif SEED=0 CORPUS=gen    # the generic-corpus control (the BiGRU recipe's final-phase corpus)
make train-lif SEED=0               # the original 5,000-transmission corpus, about 18 minutes per seed
make train-mlp SEED=0               # about 30 seconds per seed
make train-bigru ARM=fist           # the baseline of record, about 5 minutes on a CUDA GPU
make train-bigru                    # the original eight-seed pool; ARM=rep and ARM=shared are the other arms
```

LIF and MLP training is deterministic on the CPU: the retrained pickle (in `outputs/retrain/`) must be
byte-identical to `models/<family>/seed<N>.pkl` (or `models/<family>/<corpus>/seed<N>.pkl`), and the
command says whether it is. Seeds run 0 to 9. The op12 and gen corpora are generated per seed and
their feature matrices are checked against the hashes in `records/training/window_nets_grid.json`
before training starts. The deployed networks are `models/{lif,mlp}/op12/seed0.pkl`, so
`CORPUS=op12 SEED=0` regenerates what runs on the boards.

The BiGRU pool trains on the GPU in bfloat16 with CUDA graphs. On the GPU and software it was trained
with (an RTX 5080, torch 2.11.0 with CUDA 12.8) every retrained weight equals the shipped
checkpoint's; elsewhere expect the retrain to agree statistically, which the command checks against
`records/training/bigru_pool.json` (each seed's selection accuracy within the shipped pool's range
widened by its own spread, the ensemble above the 93.9% floor).

The BiGRU arms (`ARM=rep|fist|shared`) swap only the final phase's corpus; on the GPU and software
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
were made on the bench with `bench/`; see `bench/README.md`. Their results are in `records/deployment/`.
