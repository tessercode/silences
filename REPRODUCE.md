# Reproducing the results

Each level checks itself against what is committed and says so. The times are on a 24-core machine.
Start with `make setup` (a Python 3.12 venv with the versions in `requirements.txt`, and the NLTK
`words` and `brown` corpora).

## 1. The paper from the records (seconds)

```bash
make paper
```

`paper/make_figures.py` reads the records, checks each against its manifest's hashes and protocol
hash, and writes `paper/numbers.tex` (the 64 numbers the prose cites), the three tables and the four
data figures. The PDFs carry no creation date, so the whole `paper/` directory must come out
byte-identical; `make paper` ends with `git diff --exit-code` on it.

## 2. The records from the shipped models (about 10 minutes)

```bash
make evaluate
```

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
make train-lif SEED=0     # about 18 minutes per seed, CPU
make train-mlp SEED=0     # about 30 seconds per seed, CPU
make train-bigru          # the eight-seed BiGRU pool, about 5 minutes on a CUDA GPU
```

LIF and MLP training is deterministic on the CPU: the retrained pickle (in `outputs/retrain/`) must be
byte-identical to `models/<family>/seed<N>.pkl`, and the command says whether it is. Seeds run 0 to 9.

The BiGRU pool trains on the GPU in bfloat16 with CUDA graphs. On the GPU and software it was trained
with (an RTX 5080, torch 2.11.0 with CUDA 12.8) every retrained weight equals the shipped
checkpoint's; elsewhere expect the retrain to agree statistically, which the command checks against
`records/training/bigru_pool.json` (each seed's selection accuracy within the shipped pool's range
widened by its own spread, the ensemble above the 93.9% floor).

`models/lif/committed.pkl` and `models/mlp/committed.pkl`, the models deployed on hardware, were
trained by an earlier version of the training script and are shipped as artifacts of record.

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
