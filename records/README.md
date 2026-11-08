# Records

| Directory or file | What it holds | Made by |
|---|---|---|
| `matched/` | **The record of the paper (v1.1.0).** Every segmenter at matched training data, both texts: (T, γ) fitted per row and text for every row of the grid (`fit.json`), every row on the four record and four fresh evaluation corpora at its fitted and the default weights (`eval.json`), the deployed networks' Q16.16 decisions against their float twins on every gap (`fidelity.json`), the random-letter reference means at the fitted posterior scale (`criterion_1a.json`), and `results.json`: seed means, Welch comparisons and singles for the families of record (`texts`, `fresh`), the grid block (`grid`: every family and the grid's comparisons), the deployed networks, the anchors and the ruling on the fixed-point bound. `research/` holds the original experiment records of the two grid experiments and the record's report, verbatim | `python -m silences.evaluation.matched` |
| `random_letters/` | Superseded since v1.1.0 (the 5,000-transmission networks; the paper cites its protocol facts). Random letters: each model's posterior scale T fitted on seed 4242 (`calibration.json`, `calibration_curves.json`); every model on the four evaluation corpora at its own T and at T = 1, with the family statistics and Welch comparisons (`results.json`); the joint (T, γ) diagnostic (`diag_joint_beta.json`); the oracle at confidence 8 (`oracle_conf8.json`) | `python -m silences.evaluation.random_letters` |
| `decoder_weights/` | Superseded since v1.1.0 (the paper cites it as the earlier record and for the γ-grid edge check). English and random letters: (T, γ) fitted per model and text (`fit.json`), every model at its fitted and at the default weights (`eval.json`), seed means, statistics and anchors (`results.json`), the γ-grid edge check (`diag_beta_edge.json`) | `python -m silences.evaluation.decoder_weights` |
| `collision.json` | The 4.0 time-unit collision study | `python -m silences.collision` |
| `training/` | The LIF determinism check and the BiGRU pool's training and selection record; `bigru_arms.json` (the three retrained arms: selection accuracies, chosen ensembles, checkpoint hashes) and `window_nets_grid.json` (the op12 and gen corpora: sizes, class shares, feature-matrix hashes) | `silences.training` |
| `deployment/` | The RP2040 measurements. Of the deployed networks: `chain_*_20261106.json` (the five-board chain, 63 sentences each), `latency_*_20261108.json` (the bench test: 50 vectors bit-exact, per-gap compute), `footprint_*.txt`. Of the first deployed networks: `chain_*.json` (2026-07-30), `chain_*_20261014.json`, `latency_*.json`, `transport_*.json`. `silicon.json` carries the one bench figure no other record has | `bench/`, `firmware/common/measure_footprint.sh` |
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
| `exp210_m1_controls`, the Exp 210 protocol | the BiGRU arms, `protocol/grid_bigru.json`; arms `rep`, `fist`, `shared`, rows `bigru_<arm>_seed*`, `bigru_<arm>_ens5` |
| `exp211_reverse_control`, the Exp 211 protocol | the feedforward arms, `protocol/grid_window.json`; arms `op12` (the families of record), `gen`; rows `lif_op12_seed*`, `mlp_gen_seed*`, ... |
| `exp212_matched_record`, the Exp 212 protocol | the record of the paper, `protocol/matched.json`, `matched/` |
| `lif_op12_seed0`, `mlp_op12_seed0` (the `lif_committed` / `mlp_committed` aliases of `matched/results.json`'s `singles`) | the deployed networks, `models/{lif,mlp}/op12/seed0.pkl` |
| `*_fx` rows of `matched/` | the deployed networks in the boards' Q16.16 arithmetic |
| `bigru_matched_ens5`, `bigru_fist_ens5` | the five-seed ensemble of the baseline of record (Table 3) |
| `record` / `fresh` (evaluation seeds) | the four evaluation corpora of record (999, 777, 555, 321) and four fresh ones (1106, 2212, 3318, 4424) |
| `exp209_corpus_decoder_weights`, the Exp 209 protocol | the decoder-weight evaluation, `decoder_weights/` |
| `exp206_seed_sweep` (model sidecars) | the LIF and MLP training seeds, `models/{lif,mlp}/seed*.pkl` |
| `exp197` | the BiGRU pool, `models/bigru/pool/`, `training/bigru_pool.json` |
| `beta` (key), `beta_grid` | the insertion bonus γ and its grid (β in the paper is the LIF decay) |
| `deployed` (condition), `T = 1, beta = 2.5` | the default decoder weights |
| `own_fit` | each model at its own fitted decoder weights |
| `rand`, `eng` | random letters, English words |
| `lif_committed`, `mlp_committed` as rows of `decoder_weights/`, `random_letters/` and `matched/`'s tables; `snn_fetu.pkl`, `ff_fetu.pkl` before 2026-11-06 | the networks first deployed, `models/lif/deployed_5k.pkl`, `models/mlp/deployed_5k.pkl` (superseded) |
| `*_committed_fx` | the same models in the boards' Q16.16 arithmetic |
| `bigru_deployed`, gap9, gap9_3c | the five-seed BiGRU ensemble the chain ran |
| `perfect_gaps` | the oracle |
| fist | operator |
| `_corpus.LETTERS`, `spiking_gap9` | `silences.morse.LETTERS`, `silences` |
