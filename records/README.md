# Records

| Directory or file | What it holds | Made by |
|---|---|---|
| `random_letters/` | Random letters: each model's posterior scale T fitted on seed 4242 (`calibration.json`, `calibration_curves.json`); every model on the four evaluation corpora at its own T and at T = 1, with the family statistics and Welch comparisons (`results.json`); the joint (T, γ) diagnostic (`diag_joint_beta.json`); the oracle at confidence 8 (`oracle_conf8.json`) | `python -m silences.evaluation.random_letters` |
| `decoder_weights/` | English and random letters: (T, γ) fitted per model and text (`fit.json`), every model at its fitted and at the default weights (`eval.json`), seed means, statistics and anchors (`results.json`), the γ-grid edge check (`diag_beta_edge.json`) | `python -m silences.evaluation.decoder_weights` |
| `collision.json` | The 4.0 time-unit collision study | `python -m silences.collision` |
| `training/` | The LIF determinism check and the BiGRU pool's training and selection record | `silences.training` |
| `deployment/` | The RP2040 measurements: the chain runs, latency, transport, footprint, and the one bench figure no other record carries (`silicon.json`) | `bench/`, `firmware/common/measure_footprint.sh` |
| `EQUIVALENCE.md`, `equivalence.json` | The proof that these records equal the records the paper was written from | `tools/` in the research repository |

Each evaluation directory has a `manifest.json` with the sha256 of every file, the commit that produced
it and whether the tree was clean; `paper/make_figures.py` refuses a record that does not match.

## Glossary

The protocols (`protocol/`) were registered before this package existed, and the records keep their
key names so they can be compared with the originals. Their vocabulary maps to the paper and this
package as follows.

| In the records and protocols | In the paper and this package |
|---|---|
| `exp207_temperature`, the Exp 207 protocol | the random-letter evaluation, `random_letters/` |
| `exp209_corpus_decoder_weights`, the Exp 209 protocol | the decoder-weight evaluation, `decoder_weights/` |
| `exp206_seed_sweep` (model sidecars) | the LIF and MLP training seeds, `models/{lif,mlp}/seed*.pkl` |
| `exp197` | the BiGRU pool, `models/bigru/pool/`, `training/bigru_pool.json` |
| `beta` (key), `beta_grid` | the insertion bonus γ and its grid (β in the paper is the LIF decay) |
| `deployed` (condition), `T = 1, beta = 2.5` | the default decoder weights |
| `own_fit` | each model at its own fitted decoder weights |
| `rand`, `eng` | random letters, English words |
| `lif_committed`, `mlp_committed`; `snn_fetu.pkl`, `ff_fetu.pkl` | the deployed models, `models/lif/committed.pkl`, `models/mlp/committed.pkl` |
| `*_committed_fx` | the same models in the boards' Q16.16 arithmetic |
| `bigru_deployed`, gap9, gap9_3c | the five-seed BiGRU ensemble the chain ran |
| `perfect_gaps` | the oracle |
| fist | operator |
| `_corpus.LETTERS`, `spiking_gap9` | `silences.morse.LETTERS`, `silences` |
