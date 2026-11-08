Protocol c041694763a2f430 (base Exp 210 cafa08a4f02b598e); finished 2026-11-06T15:50:54-07:00.
Families present: lif (10), mlp (10), bigru (8), bigru_rep (8), bigru_fist (8), bigru_shared (8), mlp_op12 (10), mlp_gen (10), lif_op12 (10), lif_gen (10).

### Anchors

- B1 harness: seed 0 retrained on the record corpus, max |weight diff| against Exp 206: MLP 0.0, LIF 0.0 -> PASS
- B2 generic corpora equal the notebook builder's: PASS. B3 tempo estimators agree within 9.2e-05 relative: PASS
- B4 disjointness: PASS (240000 training fingerprints)
- B5 existing families equal Exp 210's: max |diff| 0.0e+00 -> PASS

**Pattern (English, own-fit, record seeds): MIXED**

### The grid, own_fit, record evaluation seeds (CER / WER %, mean ± sd over training seeds)

| text | training corpus | BiGRU (8 seeds) | MLP (10) | LIF (10) |
|---|---|---|---|---|
| ENG | generic, 12,000 per seed | 16.37 ± 0.39 / 35.81 ± 1.25 | 16.67 ± 0.24 / 38.89 ± 0.84 | 16.19 ± 0.30 / 36.26 ± 0.59 |
| ENG | operators, 12,000 per seed | 14.37 ± 0.31 / 32.24 ± 0.91 | 14.39 ± 0.19 / 32.19 ± 0.45 | 14.77 ± 0.12 / 33.00 ± 0.32 |
| ENG | operators, 5,000 shared | 15.16 ± 0.35 / 34.41 ± 1.19 | 15.33 ± 0.10 / 34.88 ± 0.30 | 16.37 ± 0.29 / 36.49 ± 0.48 |
| RAND | generic, 12,000 per seed | 18.54 ± 0.45 / 37.99 ± 1.13 | 17.62 ± 0.47 / 40.24 ± 0.85 | 16.87 ± 0.60 / 37.19 ± 0.83 |
| RAND | operators, 12,000 per seed | 17.13 ± 1.15 / 36.21 ± 3.02 | 12.53 ± 0.15 / 29.24 ± 0.47 | 13.01 ± 0.12 / 30.45 ± 0.41 |
| RAND | operators, 5,000 shared | 19.41 ± 1.16 / 40.59 ± 3.04 | 13.59 ± 0.15 / 31.82 ± 0.45 | 14.73 ± 0.23 / 33.99 ± 0.79 |

### The grid, own_fit, fresh evaluation seeds (CER / WER %, mean ± sd over training seeds)

| text | training corpus | BiGRU (8 seeds) | MLP (10) | LIF (10) |
|---|---|---|---|---|
| ENG | generic, 12,000 per seed | 16.37 ± 0.43 / 36.12 ± 1.39 | 16.92 ± 0.24 / 39.60 ± 0.66 | 16.44 ± 0.26 / 36.96 ± 0.65 |
| ENG | operators, 12,000 per seed | 14.52 ± 0.29 / 32.83 ± 0.83 | 14.40 ± 0.15 / 32.54 ± 0.39 | 14.82 ± 0.09 / 33.40 ± 0.31 |
| ENG | operators, 5,000 shared | 15.23 ± 0.39 / 34.93 ± 1.20 | 15.39 ± 0.17 / 35.37 ± 0.36 | 16.46 ± 0.28 / 36.81 ± 0.48 |
| RAND | generic, 12,000 per seed | 18.59 ± 0.49 / 37.59 ± 1.22 | 17.66 ± 0.43 / 40.16 ± 0.66 | 16.99 ± 0.54 / 37.48 ± 0.77 |
| RAND | operators, 12,000 per seed | 17.30 ± 1.21 / 36.13 ± 2.98 | 12.84 ± 0.18 / 29.71 ± 0.42 | 13.31 ± 0.16 / 30.90 ± 0.36 |
| RAND | operators, 5,000 shared | 19.50 ± 1.15 / 40.40 ± 2.93 | 13.85 ± 0.17 / 32.11 ± 0.46 | 14.96 ± 0.24 / 34.40 ± 0.80 |

### The grid, deployed, record evaluation seeds (CER / WER %, mean ± sd over training seeds)

| text | training corpus | BiGRU (8 seeds) | MLP (10) | LIF (10) |
|---|---|---|---|---|
| ENG | generic, 12,000 per seed | 16.62 ± 0.29 / 36.45 ± 0.82 | 16.65 ± 0.23 / 39.24 ± 0.63 | 16.12 ± 0.30 / 36.49 ± 0.59 |
| ENG | operators, 12,000 per seed | 18.40 ± 0.24 / 38.78 ± 0.94 | 15.65 ± 0.15 / 35.37 ± 0.33 | 15.85 ± 0.22 / 35.50 ± 0.51 |
| ENG | operators, 5,000 shared | 19.61 ± 0.38 / 41.40 ± 1.11 | 16.39 ± 0.14 / 37.66 ± 0.32 | 17.02 ± 0.36 / 38.16 ± 0.71 |
| RAND | generic, 12,000 per seed | 29.77 ± 0.53 / 58.07 ± 0.81 | 26.18 ± 0.94 / 57.13 ± 1.22 | 24.74 ± 1.02 / 51.11 ± 1.45 |
| RAND | operators, 12,000 per seed | 42.69 ± 1.37 / 75.20 ± 2.49 | 28.12 ± 1.27 / 58.54 ± 1.57 | 27.88 ± 0.96 / 57.43 ± 1.52 |
| RAND | operators, 5,000 shared | 46.10 ± 1.52 / 80.00 ± 2.81 | 29.14 ± 0.71 / 61.39 ± 1.11 | 29.42 ± 1.54 / 59.82 ± 2.23 |

### Welch comparisons, own-fit, record seeds (D in pp, positive = first-named better)

| comparison | ENG CER | ENG WER | ENG | RAND CER | RAND WER | RAND |
|---|---|---|---|---|---|---|
| mlp_op12 vs mlp | +0.95 [+0.80, +1.09] | +2.68 [+2.32, +3.05] | HOLDS | +1.07 [+0.93, +1.21] | +2.58 [+2.15, +3.02] | HOLDS |
| lif_op12 vs lif | +1.61 [+1.39, +1.83] | +3.49 [+3.10, +3.88] | HOLDS | +1.72 [+1.54, +1.89] | +3.54 [+2.93, +4.15] | HOLDS |
| lif_op12 vs bigru_fist | -0.39 [-0.66, -0.13] | -0.76 [-1.53, +0.01] | REVERSES | +4.11 [+3.15, +5.08] | +5.77 [+3.24, +8.30] | HOLDS |
| mlp_op12 vs bigru_fist | -0.01 [-0.29, +0.26] | +0.05 [-0.73, +0.83] | UNRESOLVED | +4.60 [+3.64, +5.57] | +6.98 [+4.45, +9.51] | HOLDS |
| lif_gen vs bigru | +0.18 [-0.18, +0.54] | -0.44 [-1.52, +0.63] | UNRESOLVED | +1.67 [+1.14, +2.20] | +0.79 [-0.24, +1.83] | UNRESOLVED |
| mlp_gen vs bigru | -0.30 [-0.65, +0.05] | -3.08 [-4.20, -1.95] | REVERSES | +0.92 [+0.46, +1.39] | -2.25 [-3.29, -1.20] | REVERSES |
| lif vs bigru_shared | -1.21 [-1.55, -0.88] | -2.08 [-3.09, -1.07] | REVERSES | +4.68 [+3.71, +5.65] | +6.61 [+4.05, +9.17] | HOLDS |
| mlp vs bigru_shared | -0.18 [-0.47, +0.12] | -0.46 [-1.46, +0.53] | UNRESOLVED | +5.81 [+4.84, +6.78] | +8.78 [+6.23, +11.32] | HOLDS |
| mlp_op12 vs mlp_gen | +2.28 [+2.08, +2.49] | +6.69 [+6.05, +7.34] | HOLDS | +5.09 [+4.74, +5.44] | +11.00 [+10.34, +11.66] | HOLDS |
| lif_op12 vs lif_gen | +1.42 [+1.20, +1.64] | +3.25 [+2.79, +3.71] | HOLDS | +3.86 [+3.42, +4.29] | +6.75 [+6.12, +7.38] | HOLDS |
| bigru_fist vs bigru | +2.00 [+1.61, +2.38] | +3.57 [+2.39, +4.75] | HOLDS | +1.41 [+0.43, +2.40] | +1.77 [-0.81, +4.36] | UNRESOLVED |
| mlp_op12 vs lif_op12 | +0.38 [+0.23, +0.53] | +0.81 [+0.44, +1.18] | HOLDS | +0.49 [+0.36, +0.62] | +1.21 [+0.80, +1.63] | HOLDS |
| mlp_gen vs lif_gen | -0.48 [-0.74, -0.23] | -2.63 [-3.32, -1.94] | REVERSES | -0.75 [-1.26, -0.24] | -3.04 [-3.83, -2.25] | REVERSES |
| mlp vs lif | +1.04 [+0.82, +1.25] | +1.62 [+1.23, +2.00] | HOLDS | +1.14 [+0.95, +1.32] | +2.17 [+1.55, +2.79] | HOLDS |
| lif vs bigru | -0.00 [-0.37, +0.36] | -0.68 [-1.74, +0.38] | UNRESOLVED | +3.81 [+3.42, +4.20] | +4.00 [+2.97, +5.03] | HOLDS |
| mlp vs bigru | +1.03 [+0.70, +1.36] | +0.93 [-0.11, +1.98] | UNRESOLVED | +4.95 [+4.56, +5.33] | +6.17 [+5.21, +7.14] | HOLDS |
| lif vs bigru_fist | -2.00 [-2.31, -1.69] | -4.25 [-5.04, -3.46] | REVERSES | +2.40 [+1.43, +3.36] | +2.23 [-0.31, +4.77] | UNRESOLVED |
| mlp vs bigru_fist | -0.96 [-1.22, -0.70] | -2.63 [-3.40, -1.87] | REVERSES | +3.53 [+2.57, +4.50] | +4.40 [+1.87, +6.92] | HOLDS |

### Welch comparisons, own-fit, fresh seeds (D in pp, positive = first-named better)

| comparison | ENG CER | ENG WER | ENG | RAND CER | RAND WER | RAND |
|---|---|---|---|---|---|---|
| mlp_op12 vs mlp | +0.99 [+0.84, +1.14] | +2.83 [+2.47, +3.18] | HOLDS | +1.00 [+0.84, +1.17] | +2.40 [+1.98, +2.81] | HOLDS |
| lif_op12 vs lif | +1.64 [+1.43, +1.84] | +3.41 [+3.02, +3.79] | HOLDS | +1.64 [+1.44, +1.84] | +3.50 [+2.90, +4.10] | HOLDS |
| lif_op12 vs bigru_fist | -0.31 [-0.55, -0.06] | -0.57 [-1.28, +0.14] | REVERSES | +3.98 [+2.97, +4.99] | +5.23 [+2.73, +7.73] | HOLDS |
| mlp_op12 vs bigru_fist | +0.12 [-0.14, +0.37] | +0.29 [-0.43, +1.01] | UNRESOLVED | +4.45 [+3.44, +5.46] | +6.41 [+3.92, +8.91] | HOLDS |
| lif_gen vs bigru | -0.07 [-0.45, +0.31] | -0.84 [-2.04, +0.36] | UNRESOLVED | +1.60 [+1.09, +2.12] | +0.11 [-0.98, +1.20] | UNRESOLVED |
| mlp_gen vs bigru | -0.55 [-0.93, -0.17] | -3.48 [-4.68, -2.28] | REVERSES | +0.94 [+0.47, +1.41] | -2.58 [-3.64, -1.51] | REVERSES |
| lif vs bigru_shared | -1.23 [-1.58, -0.87] | -1.88 [-2.90, -0.85] | REVERSES | +4.54 [+3.58, +5.51] | +6.00 [+3.53, +8.47] | HOLDS |
| mlp vs bigru_shared | -0.16 [-0.49, +0.17] | -0.44 [-1.45, +0.58] | UNRESOLVED | +5.65 [+4.69, +6.61] | +8.29 [+5.83, +10.74] | HOLDS |
| mlp_op12 vs mlp_gen | +2.52 [+2.32, +2.71] | +7.06 [+6.54, +7.57] | HOLDS | +4.81 [+4.49, +5.13] | +10.45 [+9.92, +10.98] | HOLDS |
| lif_op12 vs lif_gen | +1.61 [+1.42, +1.81] | +3.56 [+3.06, +4.05] | HOLDS | +3.68 [+3.28, +4.07] | +6.58 [+5.99, +7.16] | HOLDS |
| bigru_fist vs bigru | +1.85 [+1.45, +2.25] | +3.29 [+2.03, +4.55] | HOLDS | +1.30 [+0.26, +2.34] | +1.46 [-1.11, +4.03] | UNRESOLVED |
| mlp_op12 vs lif_op12 | +0.42 [+0.30, +0.54] | +0.86 [+0.53, +1.20] | HOLDS | +0.47 [+0.31, +0.64] | +1.19 [+0.82, +1.55] | HOLDS |
| mlp_gen vs lif_gen | -0.48 [-0.72, -0.24] | -2.64 [-3.25, -2.03] | REVERSES | -0.66 [-1.12, -0.20] | -2.69 [-3.36, -2.01] | REVERSES |
| mlp vs lif | +1.07 [+0.85, +1.29] | +1.44 [+1.04, +1.84] | HOLDS | +1.11 [+0.91, +1.31] | +2.29 [+1.66, +2.91] | HOLDS |
| lif vs bigru | -0.09 [-0.48, +0.29] | -0.68 [-1.87, +0.50] | UNRESOLVED | +3.64 [+3.22, +4.06] | +3.19 [+2.10, +4.29] | HOLDS |
| mlp vs bigru | +0.98 [+0.61, +1.34] | +0.75 [-0.42, +1.93] | UNRESOLVED | +4.75 [+4.34, +5.16] | +5.48 [+4.44, +6.52] | HOLDS |
| lif vs bigru_fist | -1.94 [-2.23, -1.65] | -3.98 [-4.71, -3.24] | REVERSES | +2.34 [+1.33, +3.35] | +1.73 [-0.78, +4.24] | UNRESOLVED |
| mlp vs bigru_fist | -0.87 [-1.13, -0.62] | -2.54 [-3.25, -1.82] | REVERSES | +3.45 [+2.44, +4.46] | +4.02 [+1.52, +6.52] | HOLDS |

### Pre-registered labels (own-fit)

| question | ENG record | ENG fresh | RAND record | RAND fresh |
|---|---|---|---|---|
| R1 mlp_op12_vs_mlp | HELPS | HELPS | HELPS | HELPS |
| R1 lif_op12_vs_lif | HELPS | HELPS | HELPS | HELPS |
| R2 lif_op12_vs_bigru_fist | TRAILS | TRAILS | LEADS | LEADS |
| R2 mlp_op12_vs_bigru_fist | TIE | TIE | LEADS | LEADS |
| R3 lif_gen_vs_bigru | TIE | TIE | LEADS ON ONE | LEADS ON ONE |
| R3 mlp_gen_vs_bigru | TRAILS | TRAILS | TRAILS | TRAILS |
| 5k lif_vs_bigru_shared | TRAILS | TRAILS | LEADS | LEADS |
| 5k mlp_vs_bigru_shared | TIE | TIE | LEADS | LEADS |
| R6 mlp_op12_vs_lif_op12 | HOLDS | HOLDS | HOLDS | HOLDS |
| R6 mlp_gen_vs_lif_gen | REVERSES | REVERSES | REVERSES | REVERSES |

### Per operator, ENG own-fit, record seeds (family mean CER %)

| operator | bigru | mlp_gen | lif_gen | bigru_fist | mlp_op12 | lif_op12 | bigru_shared | mlp | lif |
|---|---|---|---|---|---|---|---|---|---|
| fast_normalr | 4.1 | 5.3 | 5.1 | 3.7 | 4.1 | 4.9 | 4.5 | 5.2 | 5.9 |
| slow_normalr | 3.7 | 4.9 | 4.8 | 3.3 | 3.8 | 4.6 | 4.0 | 4.8 | 5.7 |
| high_r_bug | 14.3 | 15.5 | 15.4 | 12.3 | 12.8 | 12.8 | 13.2 | 13.6 | 15.0 |
| cootie_high_jitter | 33.9 | 31.4 | 29.0 | 27.9 | 26.2 | 26.3 | 28.8 | 27.1 | 28.4 |
| cootie_corrected | 25.8 | 26.2 | 26.4 | 24.6 | 25.0 | 25.1 | 25.2 | 25.9 | 26.8 |

Matched comparisons per operator, ENG CER (D with interval):

| operator | lif_op12_vs_bigru_fist | mlp_op12_vs_bigru_fist | lif_gen_vs_bigru | mlp_gen_vs_bigru |
|---|---|---|---|---|
| fast_normalr | -1.23 [-1.64, -0.82] | -0.40 [-0.75, -0.06] | -1.05 [-1.70, -0.40] | -1.26 [-1.90, -0.62] |
| slow_normalr | -1.32 [-1.72, -0.91] | -0.55 [-0.88, -0.22] | -1.12 [-1.80, -0.43] | -1.13 [-1.79, -0.47] |
| high_r_bug | -0.51 [-0.99, -0.04] | -0.42 [-0.90, +0.06] | -1.18 [-1.72, -0.65] | -1.19 [-1.91, -0.48] |
| cootie_high_jitter | +1.57 [+0.76, +2.38] | +1.69 [+0.91, +2.46] | +4.82 [+3.92, +5.72] | +2.44 [+1.53, +3.35] |
| cootie_corrected | -0.50 [-1.22, +0.22] | -0.40 [-1.08, +0.28] | -0.60 [-1.06, -0.13] | -0.39 [-0.77, -0.01] |

### Per operator, RAND own-fit, record seeds (family mean CER %)

| operator | bigru | mlp_gen | lif_gen | bigru_fist | mlp_op12 | lif_op12 | bigru_shared | mlp | lif |
|---|---|---|---|---|---|---|---|---|---|
| fast_normalr | 2.3 | 3.8 | 3.7 | 1.7 | 2.7 | 3.3 | 2.1 | 3.5 | 3.8 |
| slow_normalr | 2.2 | 3.4 | 3.4 | 1.7 | 2.5 | 3.1 | 2.2 | 3.2 | 3.7 |
| high_r_bug | 14.5 | 14.2 | 14.4 | 13.4 | 8.8 | 9.4 | 16.6 | 9.8 | 11.6 |
| cootie_high_jitter | 42.1 | 36.5 | 33.0 | 36.2 | 24.4 | 24.8 | 40.4 | 25.9 | 27.5 |
| cootie_corrected | 31.4 | 29.9 | 29.6 | 32.3 | 24.1 | 24.2 | 35.4 | 25.3 | 26.8 |

Matched comparisons per operator, RAND CER (D with interval):

| operator | lif_op12_vs_bigru_fist | mlp_op12_vs_bigru_fist | lif_gen_vs_bigru | mlp_gen_vs_bigru |
|---|---|---|---|---|
| fast_normalr | -1.62 [-1.98, -1.25] | -0.98 [-1.30, -0.66] | -1.46 [-1.80, -1.11] | -1.50 [-1.94, -1.06] |
| slow_normalr | -1.35 [-1.77, -0.93] | -0.74 [-1.11, -0.38] | -1.18 [-1.51, -0.84] | -1.23 [-1.67, -0.79] |
| high_r_bug | +3.96 [+2.22, +5.70] | +4.55 [+2.81, +6.29] | +0.09 [-0.73, +0.91] | +0.25 [-0.47, +0.97] |
| cootie_high_jitter | +11.45 [+9.81, +13.08] | +11.88 [+10.25, +13.50] | +9.05 [+7.55, +10.55] | +5.58 [+4.06, +7.11] |
| cootie_corrected | +8.02 [+6.44, +9.59] | +8.19 [+6.63, +9.76] | +1.79 [+1.07, +2.50] | +1.48 [+0.64, +2.32] |

### Fitted decoder weights (T, γ), median over seeds

| family | ENG | RAND |
|---|---|---|
| lif | (1.75, 4) | (8, 0) |
| mlp | (2, 4) | (12, 0) |
| bigru | (1.5, 2) | (3, 0) |
| bigru_rep | (1.5, 2) | (3, 0) |
| bigru_fist | (2, 1.5) | (5, 0) |
| bigru_shared | (2, 1) | (4, 0) |
| mlp_op12 | (2, 4) | (12, 0) |
| mlp_gen | (1, 3) | (4, 0) |
| lif_op12 | (2, 4) | (10, 0) |
| lif_gen | (1, 3) | (6, 0) |

### Training wall time per model (s)

- mlp_op12: n = 10, 87–98
- mlp_gen: n = 10, 89–100
- lif_op12: n = 10, 3731–4361
- lif_gen: n = 10, 3944–4523
