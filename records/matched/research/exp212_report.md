Exp 212 protocol 58c31eb368faa229 (base Exp 211 c041694763a2f430); finished 2026-11-08T07:14:40-07:00.

### Anchors

- C1 model identity: PASS lif: deploy/rp2040_sentence_decoder/spiking_gap9/snn_fetu.pkl = outputs/exp211_reverse_control/models/lif_op12_seed0.pkl (d39dc2a54d1fae04); mlp: deploy/rp2040_sentence_decoder/spiking_gap9/ff_fetu.pkl = outputs/exp211_reverse_control/models/mlp_op12_seed0.pkl (093406ee66d55cff)
- C4 family means equal Exp 211's: max |diff| 0.0e+00 pp -> PASS
- H1 Q16.16 fidelity: decisions lif 99.993%, mlp 100.000% -> PASS; CER/WER max |Q16.16 - float| at the same decoder weights 0.20 pp -> FAIL (each at its own fit: 0.71 pp, which includes grid selection)
- anchors (pre-registered gating): FAIL; after the 2026-11-08 ruling (h1_fx_pp_pass reported, not gating): PASS
- escalation flags: {'beats_mlp_control_eng': False, 'lif_beats_bigru_eng': False}

### record evaluation seeds, own_fit (CER / WER %, seed mean ± sd)

| text | BiGRU (8) | MLP (10) | LIF (10) | LIF vs BiGRU | MLP vs BiGRU | MLP vs LIF |
|---|---|---|---|---|---|---|
| RAND | 17.13 ± 1.15 / 36.21 ± 3.02 | 12.53 ± 0.15 / 29.24 ± 0.47 | 13.01 ± 0.12 / 30.45 ± 0.41 | HOLDS: +4.11 [+3.15, +5.08] / +5.77 [+3.24, +8.30] | HOLDS: +4.60 [+3.64, +5.57] / +6.98 [+4.45, +9.51] | HOLDS: +0.49 [+0.36, +0.62] / +1.21 [+0.80, +1.63] |
| ENG | 14.37 ± 0.31 / 32.24 ± 0.91 | 14.39 ± 0.19 / 32.19 ± 0.45 | 14.77 ± 0.12 / 33.00 ± 0.32 | REVERSES: -0.39 [-0.66, -0.13] / -0.76 [-1.53, +0.01] | UNRESOLVED: -0.01 [-0.29, +0.26] / +0.05 [-0.73, +0.83] | HOLDS: +0.38 [+0.23, +0.53] / +0.81 [+0.44, +1.18] |

| text | lif_committed | mlp_committed | lif_committed_fx | mlp_committed_fx | bigru_deployed | bigru_matched_ens5 | perfect_gaps |
|---|---|---|---|---|---|---|---|
| RAND | 12.85 / 30.44 | 12.49 / 29.36 | 12.87 / 30.47 | 12.50 / 29.38 | 17.82 / 36.04 | 16.54 / 34.33 | - |
| ENG | 14.70 / 32.92 | 14.33 / 32.08 | 14.44 / 32.21 | 14.33 / 32.08 | 15.54 / 34.05 | 13.58 / 30.80 | - |

### record evaluation seeds, deployed (CER / WER %, seed mean ± sd)

| text | BiGRU (8) | MLP (10) | LIF (10) | LIF vs BiGRU | MLP vs BiGRU | MLP vs LIF |
|---|---|---|---|---|---|---|
| RAND | 42.69 ± 1.37 / 75.20 ± 2.49 | 28.12 ± 1.27 / 58.54 ± 1.57 | 27.88 ± 0.96 / 57.43 ± 1.52 | HOLDS: +14.80 [+13.56, +16.04] / +17.76 [+15.56, +19.97] | HOLDS: +14.56 [+13.22, +15.90] / +16.65 [+14.44, +18.87] | UNRESOLVED: -0.24 [-1.30, +0.82] / -1.11 [-2.56, +0.34] |
| ENG | 18.40 ± 0.24 / 38.78 ± 0.94 | 15.65 ± 0.15 / 35.37 ± 0.33 | 15.85 ± 0.22 / 35.50 ± 0.51 | HOLDS: +2.55 [+2.31, +2.79] / +3.28 [+2.46, +4.11] | HOLDS: +2.74 [+2.53, +2.96] / +3.41 [+2.61, +4.21] | UNRESOLVED: +0.19 [+0.01, +0.37] / +0.13 [-0.28, +0.54] |

| text | lif_committed | mlp_committed | lif_committed_fx | mlp_committed_fx | bigru_deployed | bigru_matched_ens5 | perfect_gaps |
|---|---|---|---|---|---|---|---|
| RAND | 27.28 / 56.09 | 27.56 / 58.15 | 27.31 / 56.28 | 27.53 / 58.26 | 30.66 / 60.05 | 43.90 / 77.32 | 3.29 / 10.85 |
| ENG | 15.66 / 34.94 | 15.74 / 35.51 | 15.68 / 35.01 | 15.77 / 35.63 | 16.12 / 35.54 | 18.40 / 38.84 | 2.48 / 9.34 |

### fresh evaluation seeds, own_fit (CER / WER %, seed mean ± sd)

| text | BiGRU (8) | MLP (10) | LIF (10) | LIF vs BiGRU | MLP vs BiGRU | MLP vs LIF |
|---|---|---|---|---|---|---|
| RAND | 17.30 ± 1.21 / 36.13 ± 2.98 | 12.84 ± 0.18 / 29.71 ± 0.42 | 13.31 ± 0.16 / 30.90 ± 0.36 | HOLDS: +3.98 [+2.97, +4.99] / +5.23 [+2.73, +7.73] | HOLDS: +4.45 [+3.44, +5.46] / +6.41 [+3.92, +8.91] | HOLDS: +0.47 [+0.31, +0.64] / +1.19 [+0.82, +1.55] |
| ENG | 14.52 ± 0.29 / 32.83 ± 0.83 | 14.40 ± 0.15 / 32.54 ± 0.39 | 14.82 ± 0.09 / 33.40 ± 0.31 | REVERSES: -0.31 [-0.55, -0.06] / -0.57 [-1.28, +0.14] | UNRESOLVED: +0.12 [-0.14, +0.37] / +0.29 [-0.43, +1.01] | HOLDS: +0.42 [+0.30, +0.54] / +0.86 [+0.53, +1.20] |

| text | lif_committed | mlp_committed | lif_committed_fx | mlp_committed_fx | bigru_deployed | bigru_matched_ens5 | perfect_gaps |
|---|---|---|---|---|---|---|---|
| RAND | 13.26 / 31.00 | 12.79 / 29.93 | 13.25 / 30.97 | 12.79 / 29.94 | 17.96 / 35.95 | 16.79 / 34.28 | - |
| ENG | 14.76 / 33.36 | 14.38 / 32.35 | 14.60 / 32.90 | 14.38 / 32.35 | 15.50 / 34.27 | 13.81 / 31.63 | - |

### fresh evaluation seeds, deployed (CER / WER %, seed mean ± sd)

| text | BiGRU (8) | MLP (10) | LIF (10) | LIF vs BiGRU | MLP vs BiGRU | MLP vs LIF |
|---|---|---|---|---|---|---|
| RAND | 42.69 ± 1.28 / 74.82 ± 2.41 | 28.28 ± 1.23 / 58.75 ± 1.58 | 28.08 ± 0.90 / 57.67 ± 1.38 | HOLDS: +14.61 [+13.45, +15.77] / +17.14 [+15.03, +19.26] | HOLDS: +14.41 [+13.14, +15.68] / +16.07 [+13.91, +18.23] | UNRESOLVED: -0.20 [-1.22, +0.82] / -1.08 [-2.47, +0.32] |
| ENG | 18.60 ± 0.26 / 39.56 ± 0.85 | 15.82 ± 0.16 / 35.86 ± 0.35 | 16.05 ± 0.22 / 36.17 ± 0.49 | HOLDS: +2.55 [+2.30, +2.79] / +3.40 [+2.65, +4.14] | HOLDS: +2.78 [+2.55, +3.01] / +3.70 [+2.97, +4.43] | UNRESOLVED: +0.24 [+0.05, +0.42] / +0.30 [-0.10, +0.71] |

| text | lif_committed | mlp_committed | lif_committed_fx | mlp_committed_fx | bigru_deployed | bigru_matched_ens5 | perfect_gaps |
|---|---|---|---|---|---|---|---|
| RAND | 27.50 / 56.37 | 27.69 / 58.10 | 27.61 / 56.73 | 27.68 / 58.21 | 30.85 / 59.77 | 43.76 / 76.74 | 3.10 / 10.33 |
| ENG | 15.80 / 35.63 | 15.96 / 36.05 | 15.81 / 35.70 | 15.95 / 36.05 | 16.36 / 36.15 | 18.62 / 39.65 | 2.46 / 9.31 |

### Per operator, record seeds, own-fit (family mean CER %, and the matched comparisons)

**RAND**

| operator | bigru | mlp | lif | lif vs bigru | mlp vs bigru | mlp vs lif |
|---|---|---|---|---|---|---|
| fast_normalr | 1.7 | 2.7 | 3.3 | -1.62 [-1.98, -1.25] | -0.98 [-1.30, -0.66] | +0.64 [+0.28, +1.00] |
| slow_normalr | 1.7 | 2.5 | 3.1 | -1.35 [-1.77, -0.93] | -0.74 [-1.11, -0.38] | +0.61 [+0.23, +0.98] |
| high_r_bug | 13.4 | 8.8 | 9.4 | +3.96 [+2.22, +5.70] | +4.55 [+2.81, +6.29] | +0.59 [+0.33, +0.85] |
| cootie_high_jitter | 36.2 | 24.4 | 24.8 | +11.45 [+9.81, +13.08] | +11.88 [+10.25, +13.50] | +0.43 [-0.07, +0.93] |
| cootie_corrected | 32.3 | 24.1 | 24.2 | +8.02 [+6.44, +9.59] | +8.19 [+6.63, +9.76] | +0.18 [-0.29, +0.64] |

**ENG**

| operator | bigru | mlp | lif | lif vs bigru | mlp vs bigru | mlp vs lif |
|---|---|---|---|---|---|---|
| fast_normalr | 3.7 | 4.1 | 4.9 | -1.23 [-1.64, -0.82] | -0.40 [-0.75, -0.06] | +0.83 [+0.43, +1.22] |
| slow_normalr | 3.3 | 3.8 | 4.6 | -1.32 [-1.72, -0.91] | -0.55 [-0.88, -0.22] | +0.77 [+0.39, +1.14] |
| high_r_bug | 12.3 | 12.8 | 12.8 | -0.51 [-0.99, -0.04] | -0.42 [-0.90, +0.06] | +0.09 [-0.31, +0.49] |
| cootie_high_jitter | 27.9 | 26.2 | 26.3 | +1.57 [+0.76, +2.38] | +1.69 [+0.91, +2.46] | +0.11 [-0.43, +0.66] |
| cootie_corrected | 24.6 | 25.0 | 25.1 | -0.50 [-1.22, +0.22] | -0.40 [-1.08, +0.28] | +0.10 [-0.35, +0.55] |

### Fitted decoder weights (T, γ)

| row | ENG | RAND |
|---|---|---|
| lif (median over seeds) | (10, 0) | (2, 4) |
| mlp (median over seeds) | (12, 0) | (2, 4) |
| bigru (median over seeds) | (5, 0) | (2, 1.5) |
| lif_committed | (6, 0) | (1.5, 3) |
| mlp_committed | (16, 0) | (2, 4) |
| lif_committed_fx | (6, 0) | (2, 4) |
| mlp_committed_fx | (16, 0) | (2, 4) |
| bigru_deployed | (4, 0) | (1.5, 2) |
| bigru_matched_ens5 | (6, 0) | (2, 1) |

### The grid (Exp 211 c041694763a2f430), pattern MIXED

record seeds, own-fit:

| text | corpus | BiGRU | MLP | LIF |
|---|---|---|---|---|
| RAND | generic, 12,000 per seed | 18.54 / 37.99 | 17.62 / 40.24 | 16.87 / 37.19 |
| RAND | operators, 12,000 per seed | 17.13 / 36.21 | 12.53 / 29.24 | 13.01 / 30.45 |
| RAND | operators, 5,000 shared | 19.41 / 40.59 | 13.59 / 31.82 | 14.73 / 33.99 |
| ENG | generic, 12,000 per seed | 16.37 / 35.81 | 16.67 / 38.89 | 16.19 / 36.26 |
| ENG | operators, 12,000 per seed | 14.37 / 32.24 | 14.39 / 32.19 | 14.77 / 33.00 |
| ENG | operators, 5,000 shared | 15.16 / 34.41 | 15.33 / 34.88 | 16.37 / 36.49 |

fresh seeds, own-fit:

| text | corpus | BiGRU | MLP | LIF |
|---|---|---|---|---|
| RAND | generic, 12,000 per seed | 18.59 / 37.59 | 17.66 / 40.16 | 16.99 / 37.48 |
| RAND | operators, 12,000 per seed | 17.30 / 36.13 | 12.84 / 29.71 | 13.31 / 30.90 |
| RAND | operators, 5,000 shared | 19.50 / 40.40 | 13.85 / 32.11 | 14.96 / 34.40 |
| ENG | generic, 12,000 per seed | 16.37 / 36.12 | 16.92 / 39.60 | 16.44 / 36.96 |
| ENG | operators, 12,000 per seed | 14.52 / 32.83 | 14.40 / 32.54 | 14.82 / 33.40 |
| ENG | operators, 5,000 shared | 15.23 / 34.93 | 15.39 / 35.37 | 16.46 / 36.81 |

