Protocol cafa08a4f02b598e (base a7f80cf1ace9639d); finished 2026-11-06T09:10:44-07:00.

### Anchors

- A1 decode path: 6968 values against the Exp 209 record, max |diff| 0.00e+00 -> PASS
- A2 corpora reproduce: PASS (asserted at table build). A5 disjointness: PASS ({'fist': 96000, 'shared': 5000})
- A3 harness: max |weight diff| of `rep` against the Exp 197 pool 0 (bit-identical: True); SELECT balanced accuracy 94.42% vs record 94.42%; top-5 [47, 42, 48, 49, 43] vs record [47, 42, 48, 49, 43]; rand record−rep CER +0.00 [-0.49, +0.49], WER +0.00 [-1.21, +1.21]; eng record−rep CER +0.00 [-0.42, +0.42], WER +0.00 [-1.34, +1.34] -> PASS
- A4 fixed point: PASS
- A6 gate: rep 8/8, fist 8/8, shared 8/8

### Family seed means, record evaluation seeds (CER / WER %, mean ± sd over training seeds)

| text, decoder | lif (10) | mlp (10) | bigru (8) | bigru_rep (8) | bigru_fist (8) | bigru_shared (8) |
|---|---|---|---|---|---|---|
| ENG own_fit | 16.37 ± 0.29 / 36.49 ± 0.48 | 15.33 ± 0.10 / 34.88 ± 0.30 | 16.37 ± 0.39 / 35.81 ± 1.25 | 16.37 ± 0.39 / 35.81 ± 1.25 | 14.37 ± 0.31 / 32.24 ± 0.91 | 15.16 ± 0.35 / 34.41 ± 1.19 |
| ENG deployed | 17.02 ± 0.36 / 38.16 ± 0.71 | 16.39 ± 0.14 / 37.66 ± 0.32 | 16.62 ± 0.29 / 36.45 ± 0.82 | 16.62 ± 0.29 / 36.45 ± 0.82 | 18.40 ± 0.24 / 38.78 ± 0.94 | 19.61 ± 0.38 / 41.40 ± 1.11 |
| RAND own_fit | 14.73 ± 0.23 / 33.99 ± 0.79 | 13.59 ± 0.15 / 31.82 ± 0.45 | 18.54 ± 0.45 / 37.99 ± 1.13 | 18.54 ± 0.45 / 37.99 ± 1.13 | 17.13 ± 1.15 / 36.21 ± 3.02 | 19.41 ± 1.16 / 40.59 ± 3.04 |
| RAND deployed | 29.42 ± 1.54 / 59.82 ± 2.23 | 29.14 ± 0.71 / 61.39 ± 1.11 | 29.77 ± 0.53 / 58.07 ± 0.81 | 29.77 ± 0.53 / 58.07 ± 0.81 | 42.69 ± 1.37 / 75.20 ± 2.49 | 46.10 ± 1.52 / 80.00 ± 2.81 |

### Family seed means, fresh evaluation seeds (CER / WER %, mean ± sd over training seeds)

| text, decoder | lif (10) | mlp (10) | bigru (8) | bigru_rep (8) | bigru_fist (8) | bigru_shared (8) |
|---|---|---|---|---|---|---|
| ENG own_fit | 16.46 ± 0.28 / 36.81 ± 0.48 | 15.39 ± 0.17 / 35.37 ± 0.36 | 16.37 ± 0.43 / 36.12 ± 1.39 | 16.37 ± 0.43 / 36.12 ± 1.39 | 14.52 ± 0.29 / 32.83 ± 0.83 | 15.23 ± 0.39 / 34.93 ± 1.20 |
| ENG deployed | 17.09 ± 0.41 / 38.53 ± 0.70 | 16.53 ± 0.21 / 38.08 ± 0.38 | 16.76 ± 0.31 / 36.97 ± 0.86 | 16.76 ± 0.31 / 36.97 ± 0.86 | 18.60 ± 0.26 / 39.56 ± 0.85 | 19.87 ± 0.45 / 42.17 ± 1.19 |
| RAND own_fit | 14.96 ± 0.24 / 34.40 ± 0.80 | 13.85 ± 0.17 / 32.11 ± 0.46 | 18.59 ± 0.49 / 37.59 ± 1.22 | 18.59 ± 0.49 / 37.59 ± 1.22 | 17.30 ± 1.21 / 36.13 ± 2.98 | 19.50 ± 1.15 / 40.40 ± 2.93 |
| RAND deployed | 29.61 ± 1.44 / 60.03 ± 2.05 | 29.41 ± 0.77 / 61.69 ± 1.17 | 30.13 ± 0.55 / 58.39 ± 0.91 | 30.13 ± 0.55 / 58.39 ± 0.91 | 42.69 ± 1.28 / 74.82 ± 2.41 | 46.14 ± 1.45 / 79.78 ± 2.97 |

### Welch comparisons, own-fit, record seeds (D in pp, positive = first-named better)

| comparison | ENG CER | ENG WER | ENG | RAND CER | RAND WER | RAND |
|---|---|---|---|---|---|---|
| lif vs bigru | -0.00 [-0.37, +0.36] | -0.68 [-1.74, +0.38] | UNRESOLVED | +3.81 [+3.42, +4.20] | +4.00 [+2.97, +5.03] | HOLDS |
| mlp vs bigru | +1.03 [+0.70, +1.36] | +0.93 [-0.11, +1.98] | UNRESOLVED | +4.95 [+4.56, +5.33] | +6.17 [+5.21, +7.14] | HOLDS |
| mlp vs lif | +1.04 [+0.82, +1.25] | +1.62 [+1.23, +2.00] | HOLDS | +1.14 [+0.95, +1.32] | +2.17 [+1.55, +2.79] | HOLDS |
| lif vs bigru_rep | -0.00 [-0.37, +0.36] | -0.68 [-1.74, +0.38] | UNRESOLVED | +3.81 [+3.42, +4.20] | +4.00 [+2.97, +5.03] | HOLDS |
| mlp vs bigru_rep | +1.03 [+0.70, +1.36] | +0.93 [-0.11, +1.98] | UNRESOLVED | +4.95 [+4.56, +5.33] | +6.17 [+5.21, +7.14] | HOLDS |
| bigru_rep vs bigru | +0.00 [-0.42, +0.42] | +0.00 [-1.34, +1.34] | UNRESOLVED | +0.00 [-0.49, +0.49] | +0.00 [-1.21, +1.21] | UNRESOLVED |
| lif vs bigru_fist | -2.00 [-2.31, -1.69] | -4.25 [-5.04, -3.46] | REVERSES | +2.40 [+1.43, +3.36] | +2.23 [-0.31, +4.77] | UNRESOLVED |
| mlp vs bigru_fist | -0.96 [-1.22, -0.70] | -2.63 [-3.40, -1.87] | REVERSES | +3.53 [+2.57, +4.50] | +4.40 [+1.87, +6.92] | HOLDS |
| bigru_fist vs bigru | +2.00 [+1.61, +2.38] | +3.57 [+2.39, +4.75] | HOLDS | +1.41 [+0.43, +2.40] | +1.77 [-0.81, +4.36] | UNRESOLVED |
| lif vs bigru_shared | -1.21 [-1.55, -0.88] | -2.08 [-3.09, -1.07] | REVERSES | +4.68 [+3.71, +5.65] | +6.61 [+4.05, +9.17] | HOLDS |
| mlp vs bigru_shared | -0.18 [-0.47, +0.12] | -0.46 [-1.46, +0.53] | UNRESOLVED | +5.81 [+4.84, +6.78] | +8.78 [+6.23, +11.32] | HOLDS |
| bigru_shared vs bigru | +1.21 [+0.81, +1.61] | +1.40 [+0.09, +2.70] | HOLDS | -0.87 [-1.86, +0.13] | -2.60 [-5.20, -0.00] | REVERSES |
| bigru_fist vs bigru_rep | +2.00 [+1.61, +2.38] | +3.57 [+2.39, +4.75] | HOLDS | +1.41 [+0.43, +2.40] | +1.77 [-0.81, +4.36] | UNRESOLVED |
| bigru_shared vs bigru_rep | +1.21 [+0.81, +1.61] | +1.40 [+0.09, +2.70] | HOLDS | -0.87 [-1.86, +0.13] | -2.60 [-5.20, -0.00] | REVERSES |
| bigru_fist vs bigru_shared | +0.79 [+0.43, +1.14] | +2.17 [+1.03, +3.31] | HOLDS | +2.28 [+1.04, +3.52] | +4.38 [+1.13, +7.63] | HOLDS |

### Welch comparisons, own-fit, fresh seeds (D in pp, positive = first-named better)

| comparison | ENG CER | ENG WER | ENG | RAND CER | RAND WER | RAND |
|---|---|---|---|---|---|---|
| lif vs bigru | -0.09 [-0.48, +0.29] | -0.68 [-1.87, +0.50] | UNRESOLVED | +3.64 [+3.22, +4.06] | +3.19 [+2.10, +4.29] | HOLDS |
| mlp vs bigru | +0.98 [+0.61, +1.34] | +0.75 [-0.42, +1.93] | UNRESOLVED | +4.75 [+4.34, +5.16] | +5.48 [+4.44, +6.52] | HOLDS |
| mlp vs lif | +1.07 [+0.85, +1.29] | +1.44 [+1.04, +1.84] | HOLDS | +1.11 [+0.91, +1.31] | +2.29 [+1.66, +2.91] | HOLDS |
| lif vs bigru_rep | -0.09 [-0.48, +0.29] | -0.68 [-1.87, +0.50] | UNRESOLVED | +3.64 [+3.22, +4.06] | +3.19 [+2.10, +4.29] | HOLDS |
| mlp vs bigru_rep | +0.98 [+0.61, +1.34] | +0.75 [-0.42, +1.93] | UNRESOLVED | +4.75 [+4.34, +5.16] | +5.48 [+4.44, +6.52] | HOLDS |
| bigru_rep vs bigru | +0.00 [-0.46, +0.46] | +0.00 [-1.49, +1.49] | UNRESOLVED | +0.00 [-0.52, +0.52] | +0.00 [-1.31, +1.31] | UNRESOLVED |
| lif vs bigru_fist | -1.94 [-2.23, -1.65] | -3.98 [-4.71, -3.24] | REVERSES | +2.34 [+1.33, +3.35] | +1.73 [-0.78, +4.24] | UNRESOLVED |
| mlp vs bigru_fist | -0.87 [-1.13, -0.62] | -2.54 [-3.25, -1.82] | REVERSES | +3.45 [+2.44, +4.46] | +4.02 [+1.52, +6.52] | HOLDS |
| bigru_fist vs bigru | +1.85 [+1.45, +2.25] | +3.29 [+2.03, +4.55] | HOLDS | +1.30 [+0.26, +2.34] | +1.46 [-1.11, +4.03] | UNRESOLVED |
| lif vs bigru_shared | -1.23 [-1.58, -0.87] | -1.88 [-2.90, -0.85] | REVERSES | +4.54 [+3.58, +5.51] | +6.00 [+3.53, +8.47] | HOLDS |
| mlp vs bigru_shared | -0.16 [-0.49, +0.17] | -0.44 [-1.45, +0.58] | UNRESOLVED | +5.65 [+4.69, +6.61] | +8.29 [+5.83, +10.74] | HOLDS |
| bigru_shared vs bigru | +1.14 [+0.70, +1.57] | +1.19 [-0.21, +2.59] | UNRESOLVED | -0.90 [-1.90, +0.09] | -2.81 [-5.33, -0.28] | REVERSES |
| bigru_fist vs bigru_rep | +1.85 [+1.45, +2.25] | +3.29 [+2.03, +4.55] | HOLDS | +1.30 [+0.26, +2.34] | +1.46 [-1.11, +4.03] | UNRESOLVED |
| bigru_shared vs bigru_rep | +1.14 [+0.70, +1.57] | +1.19 [-0.21, +2.59] | UNRESOLVED | -0.90 [-1.90, +0.09] | -2.81 [-5.33, -0.28] | REVERSES |
| bigru_fist vs bigru_shared | +0.72 [+0.35, +1.08] | +2.10 [+0.98, +3.22] | HOLDS | +2.20 [+0.94, +3.47] | +4.27 [+1.10, +7.44] | HOLDS |

### Part A pre-registered outcomes (record seeds, own-fit)

- `fist`: Q1 HELPS; Q2 LIF TRAILS; Q3 MLP TRAILS; Q4 PARTIAL
- `shared`: Q1 HELPS; Q2 LIF TRAILS; Q3 CER LEAD LOST; Q4 LEAD STANDS

### Per operator, ENG own-fit, record seeds (family mean CER / WER; then D with interval)

| operator | LIF | MLP | BiGRU record | BiGRU fist | BiGRU shared | LIF vs record CER | LIF vs fist CER | LIF vs fist WER | fist vs rep CER |
|---|---|---|---|---|---|---|---|---|---|
| fast_normalr | 5.9 / 14.0 | 5.2 / 13.2 | 4.1 / 9.5 | 3.7 / 8.7 | 4.5 / 10.5 | -1.85 [-2.54, -1.16] | -2.20 [-2.69, -1.71] | -5.35 [-6.40, -4.30] | +0.35 [-0.28, +0.98] |
| slow_normalr | 5.7 / 13.5 | 4.8 / 11.9 | 3.7 / 8.5 | 3.3 / 7.6 | 4.0 / 9.2 | -1.92 [-2.61, -1.24] | -2.40 [-2.88, -1.93] | -5.90 [-6.90, -4.90] | +0.48 [-0.15, +1.12] |
| high_r_bug | 15.0 / 34.8 | 13.6 / 32.9 | 14.3 / 34.1 | 12.3 / 28.2 | 13.2 / 30.5 | -0.72 [-1.26, -0.17] | -2.65 [-3.14, -2.16] | -6.58 [-7.76, -5.41] | +1.93 [+1.35, +2.51] |
| cootie_high_jitter | 28.4 / 61.4 | 27.1 / 59.4 | 33.9 / 70.8 | 27.9 / 62.6 | 28.8 / 65.6 | +5.46 [+4.60, +6.32] | -0.55 [-1.44, +0.34] | +1.12 [-1.51, +3.74] | +6.01 [+5.09, +6.93] |
| cootie_corrected | 26.8 / 58.7 | 25.9 / 57.0 | 25.8 / 56.2 | 24.6 / 54.2 | 25.2 / 56.3 | -1.01 [-1.53, -0.49] | -2.22 [-2.96, -1.48] | -4.53 [-6.28, -2.78] | +1.21 [+0.52, +1.91] |

### Per operator, RAND own-fit, record seeds (family mean CER / WER; then D with interval)

| operator | LIF | MLP | BiGRU record | BiGRU fist | BiGRU shared | LIF vs record CER | LIF vs fist CER | LIF vs fist WER | fist vs rep CER |
|---|---|---|---|---|---|---|---|---|---|
| fast_normalr | 3.8 / 9.0 | 3.5 / 8.5 | 2.3 / 5.3 | 1.7 / 4.1 | 2.1 / 4.9 | -1.57 [-1.96, -1.17] | -2.12 [-2.46, -1.77] | -4.93 [-5.71, -4.15] | +0.55 [+0.17, +0.93] |
| slow_normalr | 3.7 / 8.5 | 3.2 / 7.6 | 2.2 / 5.0 | 1.7 / 4.0 | 2.2 / 5.1 | -1.49 [-1.88, -1.10] | -1.96 [-2.35, -1.57] | -4.55 [-5.41, -3.69] | +0.47 [+0.06, +0.88] |
| high_r_bug | 11.6 / 28.6 | 9.8 / 25.1 | 14.5 / 34.7 | 13.4 / 31.1 | 16.6 / 37.9 | +2.88 [+2.20, +3.55] | +1.77 [+0.03, +3.52] | +2.56 [-1.45, +6.56] | +1.10 [-0.67, +2.88] |
| cootie_high_jitter | 27.5 / 62.7 | 25.9 / 59.5 | 42.1 / 79.0 | 36.2 / 75.1 | 40.4 / 81.9 | +14.57 [+13.47, +15.67] | +8.77 [+7.14, +10.40] | +12.32 [+7.36, +17.29] | +5.81 [+4.02, +7.59] |
| cootie_corrected | 26.8 / 60.7 | 25.3 / 58.1 | 31.4 / 65.6 | 32.3 / 66.4 | 35.4 / 72.6 | +4.55 [+3.87, +5.23] | +5.43 [+3.86, +7.01] | +5.64 [+1.66, +9.62] | -0.88 [-2.48, +0.72] |

### Single rows, own-fit (CER / WER %)

| row | ENG record | ENG fresh | RAND record | RAND fresh |
|---|---|---|---|---|
| bigru_deployed | 15.54 / 34.05 | 15.50 / 34.27 | 17.82 / 36.04 | 17.96 / 35.95 |
| bigru_fist_ens5 | 13.58 / 30.80 | 13.81 / 31.63 | 16.54 / 34.33 | 16.79 / 34.28 |
| bigru_rep_ens5 | 15.54 / 34.05 | 15.50 / 34.27 | 17.82 / 36.04 | 17.96 / 35.95 |
| bigru_shared_ens5 | 14.32 / 31.63 | 14.37 / 32.44 | 18.38 / 38.79 | 18.40 / 38.27 |
| lif_committed | 16.71 / 36.90 | 16.68 / 37.17 | 14.82 / 34.07 | 14.80 / 33.97 |
| lif_committed_fx | 16.71 / 36.92 | 16.63 / 37.13 | 14.82 / 34.11 | 14.81 / 34.03 |
| mlp_committed | 15.48 / 35.45 | 15.72 / 36.06 | 13.42 / 31.81 | 13.73 / 32.19 |
| mlp_committed_fx | 15.49 / 35.46 | 15.73 / 36.07 | 13.42 / 31.83 | 13.73 / 32.21 |

### Fitted decoder weights (T, γ) of the new rows

- ENG bigru: (1.5, 2.0), (1.5, 2.5), (1.5, 2.5), (1, 2.0), (1.5, 2.5), (1.5, 1.0), (1.5, 1.0), (1.5, 2.0)
- ENG bigru_rep: (1.5, 2.0), (1.5, 2.5), (1.5, 2.5), (1, 2.0), (1.5, 2.5), (1.5, 1.0), (1.5, 1.0), (1.5, 2.0)
- ENG bigru_fist: (2, 0.0), (2, 2.0), (2, 2.0), (2, 2.0), (2, 1.0), (2, 1.0), (2, 2.0), (2, 0.0)
- ENG bigru_shared: (2, 0.0), (2, 1.0), (2, 1.0), (2, 1.0), (3, 1.0), (2, 1.0), (2, 0.0), (2, 0.0)
- RAND bigru: (3, 0.0), (4, 0.0), (3, 0.0), (3, 0.0), (4, 0.0), (3, 0.0), (4, 0.0), (3, 0.0)
- RAND bigru_rep: (3, 0.0), (4, 0.0), (3, 0.0), (3, 0.0), (4, 0.0), (3, 0.0), (4, 0.0), (3, 0.0)
- RAND bigru_fist: (4, 0.0), (6, 0.0), (8, 0.0), (6, 0.0), (6, 0.0), (3, 0.0), (4, 0.0), (3, 0.0)
- RAND bigru_shared: (3, 0.0), (6, 0.0), (6, 0.0), (4, 0.0), (4, 0.0), (4, 0.0), (3, 0.0), (4, 0.0)

### Part B: replication of the comparison outcomes on fresh seeds (own-fit)

| text | comparison | record seeds | fresh seeds |
|---|---|---|---|
| RAND | lif_vs_bigru | HOLDS | HOLDS |
| RAND | mlp_vs_bigru | HOLDS | HOLDS |
| RAND | mlp_vs_lif | HOLDS | HOLDS |
| RAND | lif_vs_bigru_fist | UNRESOLVED | UNRESOLVED |
| RAND | mlp_vs_bigru_fist | HOLDS | HOLDS |
| ENG | lif_vs_bigru | UNRESOLVED | UNRESOLVED |
| ENG | mlp_vs_bigru | UNRESOLVED | UNRESOLVED |
| ENG | mlp_vs_lif | HOLDS | HOLDS |
| ENG | lif_vs_bigru_fist | REVERSES | REVERSES |
| ENG | mlp_vs_bigru_fist | REVERSES | REVERSES |

### Part B: per-corpus gap, baseline family mean − candidate family mean (pp, own-fit)

| text | gap | metric | record corpora (999, 777, 555, 321) | fresh corpora (1106, 2212, 3318, 4424) | mean record − mean fresh | label |
|---|---|---|---|---|---|---|
| ENG | bigru_minus_lif | CER | +0.12, -0.18, +0.05, -0.01 | +0.03, -0.16, -0.03, -0.20 | +0.09 [-0.12, +0.29] | NO DETECTABLE EFFECT |
| ENG | bigru_minus_lif | WER | -0.33, -0.99, -0.83, -0.56 | -0.40, -1.11, -0.51, -0.71 | +0.00 [-0.52, +0.53] | NO DETECTABLE EFFECT |
| ENG | bigru_minus_mlp | CER | +1.24, +0.96, +1.00, +0.93 | +1.12, +0.76, +1.13, +0.92 | +0.05 [-0.23, +0.33] | NO DETECTABLE EFFECT |
| ENG | bigru_minus_mlp | WER | +1.30, +1.03, +0.78, +0.63 | +1.05, +0.04, +1.34, +0.60 | +0.18 [-0.67, +1.03] | NO DETECTABLE EFFECT |
| ENG | bigru_fist_minus_lif | CER | -2.03, -2.14, -1.86, -1.97 | -2.09, -1.89, -1.77, -2.03 | -0.06 [-0.29, +0.17] | NO DETECTABLE EFFECT |
| ENG | bigru_fist_minus_lif | WER | -4.12, -4.63, -4.24, -4.00 | -4.21, -3.96, -3.50, -4.23 | -0.27 [-0.82, +0.27] | NO DETECTABLE EFFECT |
| ENG | bigru_fist_minus_mlp | CER | -0.91, -1.00, -0.91, -1.03 | -1.00, -0.97, -0.61, -0.91 | -0.09 [-0.36, +0.18] | NO DETECTABLE EFFECT |
| ENG | bigru_fist_minus_mlp | WER | -2.49, -2.61, -2.63, -2.81 | -2.77, -2.81, -1.65, -2.92 | -0.10 [-1.02, +0.83] | NO DETECTABLE EFFECT |
| RAND | bigru_minus_lif | CER | +4.16, +3.91, +3.46, +3.70 | +3.73, +3.88, +3.17, +3.78 | +0.17 [-0.37, +0.71] | NO DETECTABLE EFFECT |
| RAND | bigru_minus_lif | WER | +4.27, +4.30, +4.06, +3.39 | +3.08, +3.57, +2.60, +3.52 | +0.81 [+0.05, +1.57] | INFLATED |
| RAND | bigru_minus_mlp | CER | +5.25, +5.14, +4.67, +4.73 | +4.94, +4.93, +4.33, +4.79 | +0.20 [-0.30, +0.70] | NO DETECTABLE EFFECT |
| RAND | bigru_minus_mlp | WER | +6.37, +6.93, +5.95, +5.43 | +5.39, +5.64, +5.06, +5.82 | +0.69 [-0.26, +1.65] | NO DETECTABLE EFFECT |
| RAND | bigru_fist_minus_lif | CER | +3.10, +2.08, +2.17, +2.24 | +2.29, +2.75, +1.92, +2.40 | +0.06 [-0.68, +0.79] | NO DETECTABLE EFFECT |
| RAND | bigru_fist_minus_lif | WER | +3.19, +1.64, +2.08, +2.00 | +1.54, +2.21, +1.11, +2.06 | +0.50 [-0.55, +1.54] | NO DETECTABLE EFFECT |
| RAND | bigru_fist_minus_mlp | CER | +4.19, +3.30, +3.38, +3.27 | +3.50, +3.81, +3.08, +3.41 | +0.09 [-0.59, +0.76] | NO DETECTABLE EFFECT |
| RAND | bigru_fist_minus_mlp | WER | +5.29, +4.28, +3.98, +4.04 | +3.85, +4.29, +3.57, +4.36 | +0.38 [-0.54, +1.30] | NO DETECTABLE EFFECT |

### Part B: family mean on fresh seeds minus record seeds (pp, own-fit), with the evaluation-seed spread

| text | family | CER record → fresh (shift; spread record, fresh) | WER record → fresh (shift; spread record, fresh) |
|---|---|---|---|
| ENG | lif | 16.37 → 16.46 (+0.09; 0.88, 0.30) | 36.49 → 36.81 (+0.32; 1.62, 0.53) |
| ENG | mlp | 15.33 → 15.39 (+0.06; 1.08, 0.36) | 34.88 → 35.37 (+0.49; 2.45, 0.89) |
| ENG | bigru | 16.37 → 16.37 (+0.00; 1.04, 0.41) | 35.81 → 36.12 (+0.31; 2.05, 0.94) |
| ENG | bigru_rep | 16.37 → 16.37 (+0.00; 1.04, 0.41) | 35.81 → 36.12 (+0.31; 2.05, 0.94) |
| ENG | bigru_fist | 14.37 → 14.52 (+0.15; 1.05, 0.12) | 32.24 → 32.83 (+0.59; 2.25, 0.66) |
| ENG | bigru_shared | 15.16 → 15.23 (+0.08; 1.07, 0.19) | 34.41 → 34.93 (+0.52; 2.33, 0.68) |
| RAND | lif | 14.73 → 14.96 (+0.22; 0.65, 0.76) | 33.99 → 34.40 (+0.41; 0.98, 1.29) |
| RAND | mlp | 13.59 → 13.85 (+0.25; 0.48, 0.61) | 31.82 → 32.11 (+0.29; 1.00, 1.49) |
| RAND | bigru | 18.54 → 18.59 (+0.05; 0.84, 0.74) | 37.99 → 37.59 (-0.40; 1.86, 1.74) |
| RAND | bigru_rep | 18.54 → 18.59 (+0.05; 0.84, 0.74) | 37.99 → 37.59 (-0.40; 1.86, 1.74) |
| RAND | bigru_fist | 17.13 → 17.30 (+0.17; 1.24, 1.00) | 36.21 → 36.13 (-0.09; 2.17, 1.93) |
| RAND | bigru_shared | 19.41 → 19.50 (+0.09; 1.09, 1.09) | 40.59 → 40.40 (-0.20; 2.33, 2.09) |
