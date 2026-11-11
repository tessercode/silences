Protocol 76de3765e2be3d62 (base Exp 211 c041694763a2f430); finished 2026-11-10T12:07:38-07:00.
Families present: bigru_fist (8), mlp_op12 (10), lif_op12 (10), spk40_op12 (10), spk80_op12 (10), lif80_op12 (10).

### Anchors

- C1 features hash to Exp 211's: PASS; C2 (reported) min argmax agreement of inference against training forward: 0.99365234375
- C3 harness reproduces Exp 211's lif_op12_seed0 row: PASS; C4 disjointness: PASS (120000 training fingerprints)
- C5 reference families equal Exp 211's: max |diff| 0.0e+00 -> PASS

**Escalation trigger (English, own-fit, record seeds): True**

### Seed means, own_fit, record evaluation seeds (CER / WER %, mean ± sd over training seeds)

| text | bigru_fist (8) | mlp_op12 (10) | lif_op12 (10) | lif80_op12 (10) | spk40_op12 (10) | spk80_op12 (10) |
|---|---|---|---|---|---|---|
| ENG | 14.37 ± 0.31 / 32.24 ± 0.91 | 14.39 ± 0.19 / 32.19 ± 0.45 | 14.77 ± 0.12 / 33.00 ± 0.32 | 14.82 ± 0.24 / 33.11 ± 0.51 | 14.58 ± 0.24 / 32.28 ± 0.65 | 14.59 ± 0.19 / 32.09 ± 0.47 |
| RAND | 17.13 ± 1.15 / 36.21 ± 3.02 | 12.53 ± 0.15 / 29.24 ± 0.47 | 13.01 ± 0.12 / 30.45 ± 0.41 | 13.04 ± 0.17 / 30.53 ± 0.34 | 13.18 ± 0.22 / 30.39 ± 0.46 | 13.00 ± 0.23 / 29.98 ± 0.54 |

### Seed means, own_fit, fresh evaluation seeds (CER / WER %, mean ± sd over training seeds)

| text | bigru_fist (8) | mlp_op12 (10) | lif_op12 (10) | lif80_op12 (10) | spk40_op12 (10) | spk80_op12 (10) |
|---|---|---|---|---|---|---|
| ENG | 14.52 ± 0.29 / 32.83 ± 0.83 | 14.40 ± 0.15 / 32.54 ± 0.39 | 14.82 ± 0.09 / 33.40 ± 0.31 | 14.88 ± 0.21 / 33.46 ± 0.39 | 14.71 ± 0.23 / 32.79 ± 0.61 | 14.66 ± 0.25 / 32.65 ± 0.50 |
| RAND | 17.30 ± 1.21 / 36.13 ± 2.98 | 12.84 ± 0.18 / 29.71 ± 0.42 | 13.31 ± 0.16 / 30.90 ± 0.36 | 13.35 ± 0.16 / 30.91 ± 0.40 | 13.37 ± 0.20 / 30.76 ± 0.32 | 13.18 ± 0.25 / 30.31 ± 0.63 |

### Seed means, deployed, record evaluation seeds (CER / WER %, mean ± sd over training seeds)

| text | bigru_fist (8) | mlp_op12 (10) | lif_op12 (10) | lif80_op12 (10) | spk40_op12 (10) | spk80_op12 (10) |
|---|---|---|---|---|---|---|
| ENG | 18.40 ± 0.24 / 38.78 ± 0.94 | 15.65 ± 0.15 / 35.37 ± 0.33 | 15.85 ± 0.22 / 35.50 ± 0.51 | 15.90 ± 0.32 / 35.59 ± 0.63 | 15.51 ± 0.26 / 34.85 ± 0.89 | 15.57 ± 0.27 / 34.71 ± 0.87 |
| RAND | 42.69 ± 1.37 / 75.20 ± 2.49 | 28.12 ± 1.27 / 58.54 ± 1.57 | 27.88 ± 0.96 / 57.43 ± 1.52 | 27.76 ± 1.06 / 57.24 ± 1.73 | 27.32 ± 1.05 / 56.36 ± 2.38 | 27.11 ± 1.36 / 55.89 ± 2.60 |

### Welch comparisons, own-fit, record seeds (D in pp, positive = first-named better)

| comparison | ENG CER | ENG WER | ENG | RAND CER | RAND WER | RAND |
|---|---|---|---|---|---|---|
| spk40_op12 vs lif_op12 | +0.19 [+0.00, +0.37] | +0.73 [+0.23, +1.23] | HOLDS | -0.17 [-0.33, +0.00] | +0.06 [-0.35, +0.47] | UNRESOLVED |
| spk80_op12 vs lif_op12 | +0.18 [+0.03, +0.33] | +0.92 [+0.53, +1.30] | HOLDS | +0.02 [-0.16, +0.19] | +0.47 [+0.02, +0.92] | UNRESOLVED |
| spk80_op12 vs spk40_op12 | -0.01 [-0.22, +0.19] | +0.19 [-0.35, +0.73] | UNRESOLVED | +0.18 [-0.02, +0.39] | +0.41 [-0.06, +0.88] | UNRESOLVED |
| lif80_op12 vs lif_op12 | -0.05 [-0.23, +0.13] | -0.11 [-0.51, +0.29] | UNRESOLVED | -0.03 [-0.17, +0.11] | -0.08 [-0.44, +0.27] | UNRESOLVED |
| spk80_op12 vs lif80_op12 | +0.23 [+0.02, +0.43] | +1.03 [+0.57, +1.49] | HOLDS | +0.05 [-0.14, +0.23] | +0.55 [+0.13, +0.98] | UNRESOLVED |
| spk40_op12 vs bigru_fist | -0.21 [-0.49, +0.08] | -0.03 [-0.86, +0.79] | UNRESOLVED | +3.95 [+2.99, +4.91] | +5.83 [+3.30, +8.36] | HOLDS |
| spk80_op12 vs bigru_fist | -0.22 [-0.49, +0.06] | +0.16 [-0.63, +0.94] | UNRESOLVED | +4.13 [+3.17, +5.10] | +6.24 [+3.71, +8.77] | HOLDS |
| lif80_op12 vs bigru_fist | -0.45 [-0.73, -0.16] | -0.87 [-1.66, -0.08] | REVERSES | +4.09 [+3.12, +5.05] | +5.69 [+3.16, +8.21] | HOLDS |
| spk40_op12 vs mlp_op12 | -0.19 [-0.40, +0.01] | -0.08 [-0.61, +0.45] | UNRESOLVED | -0.65 [-0.83, -0.48] | -1.15 [-1.59, -0.71] | REVERSES |
| spk80_op12 vs mlp_op12 | -0.20 [-0.38, -0.03] | +0.11 [-0.33, +0.54] | REVERSES | -0.47 [-0.65, -0.29] | -0.74 [-1.21, -0.27] | REVERSES |
| lif80_op12 vs mlp_op12 | -0.43 [-0.64, -0.23] | -0.92 [-1.37, -0.47] | REVERSES | -0.52 [-0.66, -0.37] | -1.29 [-1.68, -0.91] | REVERSES |

### Welch comparisons, own-fit, fresh seeds (D in pp, positive = first-named better)

| comparison | ENG CER | ENG WER | ENG | RAND CER | RAND WER | RAND |
|---|---|---|---|---|---|---|
| spk40_op12 vs lif_op12 | +0.11 [-0.06, +0.29] | +0.61 [+0.14, +1.08] | UNRESOLVED | -0.06 [-0.23, +0.12] | +0.14 [-0.18, +0.46] | UNRESOLVED |
| spk80_op12 vs lif_op12 | +0.16 [-0.02, +0.35] | +0.75 [+0.35, +1.15] | UNRESOLVED | +0.13 [-0.07, +0.33] | +0.59 [+0.10, +1.08] | UNRESOLVED |
| spk80_op12 vs spk40_op12 | +0.05 [-0.18, +0.28] | +0.14 [-0.38, +0.67] | UNRESOLVED | +0.19 [-0.03, +0.40] | +0.45 [-0.03, +0.93] | UNRESOLVED |
| lif80_op12 vs lif_op12 | -0.06 [-0.21, +0.10] | -0.06 [-0.39, +0.28] | UNRESOLVED | -0.04 [-0.19, +0.11] | -0.01 [-0.36, +0.35] | UNRESOLVED |
| spk80_op12 vs lif80_op12 | +0.22 [+0.01, +0.44] | +0.81 [+0.38, +1.24] | HOLDS | +0.17 [-0.03, +0.37] | +0.60 [+0.10, +1.09] | UNRESOLVED |
| spk40_op12 vs bigru_fist | -0.19 [-0.46, +0.08] | +0.04 [-0.72, +0.80] | UNRESOLVED | +3.93 [+2.91, +4.94] | +5.37 [+2.87, +7.86] | HOLDS |
| spk80_op12 vs bigru_fist | -0.14 [-0.42, +0.13] | +0.18 [-0.55, +0.92] | UNRESOLVED | +4.11 [+3.10, +5.13] | +5.82 [+3.31, +8.32] | HOLDS |
| lif80_op12 vs bigru_fist | -0.37 [-0.63, -0.10] | -0.63 [-1.34, +0.09] | REVERSES | +3.94 [+2.93, +4.95] | +5.22 [+2.72, +7.72] | HOLDS |
| spk40_op12 vs mlp_op12 | -0.31 [-0.49, -0.12] | -0.25 [-0.74, +0.24] | REVERSES | -0.53 [-0.71, -0.35] | -1.05 [-1.40, -0.70] | REVERSES |
| spk80_op12 vs mlp_op12 | -0.26 [-0.45, -0.06] | -0.11 [-0.53, +0.32] | REVERSES | -0.34 [-0.55, -0.13] | -0.60 [-1.10, -0.09] | REVERSES |
| lif80_op12 vs mlp_op12 | -0.48 [-0.65, -0.31] | -0.92 [-1.29, -0.55] | REVERSES | -0.51 [-0.68, -0.35] | -1.19 [-1.58, -0.81] | REVERSES |

### Pre-registered labels (own-fit)

| question | ENG record | ENG fresh | RAND record | RAND fresh |
|---|---|---|---|---|
| S1/S3 spk40_op12_vs_lif_op12 | GAINS | UNRESOLVED | FREE | FREE |
| S1/S3 spk80_op12_vs_lif_op12 | GAINS | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| S1/S3 spk80_op12_vs_lif80_op12 | GAINS | GAINS | UNRESOLVED | UNRESOLVED |
| S2/S3 spk80_op12_vs_spk40_op12 | NO EFFECT | NO EFFECT | NO EFFECT | NO EFFECT |
| S2/S3 lif80_op12_vs_lif_op12 | NO EFFECT | NO EFFECT | NO EFFECT | NO EFFECT |
| S4 spk40_op12_vs_bigru_fist | TIE | TIE | LEADS | LEADS |
| S4 spk40_op12_vs_mlp_op12 | TIE | TRAILS | TRAILS | TRAILS |
| S4 spk80_op12_vs_bigru_fist | TIE | TIE | LEADS | LEADS |
| S4 spk80_op12_vs_mlp_op12 | TRAILS | TRAILS | TRAILS | TRAILS |
| S4 lif80_op12_vs_bigru_fist | TRAILS | TRAILS | LEADS | LEADS |
| S4 lif80_op12_vs_mlp_op12 | TRAILS | TRAILS | TRAILS | TRAILS |

### Per operator, ENG own-fit, record seeds (family mean CER %)

| operator | bigru_fist | mlp_op12 | lif_op12 | lif80_op12 | spk40_op12 | spk80_op12 |
|---|---|---|---|---|---|---|
| fast_normalr | 3.7 | 4.1 | 4.9 | 5.0 | 3.9 | 3.9 |
| slow_normalr | 3.3 | 3.8 | 4.6 | 4.6 | 3.6 | 3.6 |
| high_r_bug | 12.3 | 12.8 | 12.8 | 13.0 | 12.9 | 12.7 |
| cootie_high_jitter | 27.9 | 26.2 | 26.3 | 26.4 | 27.3 | 27.4 |
| cootie_corrected | 24.6 | 25.0 | 25.1 | 25.0 | 25.2 | 25.3 |

### Per operator, RAND own-fit, record seeds (family mean CER %)

| operator | bigru_fist | mlp_op12 | lif_op12 | lif80_op12 | spk40_op12 | spk80_op12 |
|---|---|---|---|---|---|---|
| fast_normalr | 1.7 | 2.7 | 3.3 | 3.4 | 2.6 | 2.7 |
| slow_normalr | 1.7 | 2.5 | 3.1 | 3.1 | 2.4 | 2.4 |
| high_r_bug | 13.4 | 8.8 | 9.4 | 9.6 | 9.8 | 9.5 |
| cootie_high_jitter | 36.2 | 24.4 | 24.8 | 24.6 | 26.0 | 25.7 |
| cootie_corrected | 32.3 | 24.1 | 24.2 | 24.2 | 24.8 | 24.5 |

### Fitted decoder weights (T, γ), median over seeds

| family | ENG | RAND |
|---|---|---|
| bigru_fist | (2, 1.5) | (5, 0) |
| mlp_op12 | (2, 4) | (12, 0) |
| lif_op12 | (2, 4) | (10, 0) |
| lif80_op12 | (2, 4) | (8, 0) |
| spk40_op12 | (2, 4) | (8, 0) |
| spk80_op12 | (2, 4) | (12, 0) |

### Training wall time per model (s), and C2 argmax agreement (inference vs training forward)

- spk40_op12: n = 10, 7636–11448 s; C2 min 0.9961, mean 0.9974
- spk80_op12: n = 10, 11137–16199 s; C2 min 0.9971, mean 0.9982
- lif80_op12: n = 10, 6218–8359 s; C2 min 0.9937, mean 0.9974
