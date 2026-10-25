"""The 4.0 time-unit collision (Section 3, Figure 1).

    python -m silences.collision        # writes records/collision.json (seconds)

TEMPO emits one spike when each mark completes, so the interval between two spikes is the gap plus
the next mark. An intra-letter gap followed by a dash and an inter-letter gap followed by a dot both
give 4 time units. This study measures how much of that ambiguity timing alone can resolve.

Words of 3 to 6 letters (letters drawn with English unigram frequencies) are rendered with per-letter
speed drift (a linear ramp of width D) and timing jitter (sigma, in time units, a fraction of whose
variance is shared within a letter). For every gap the study records the summed interval, the next
mark's duration, and whether the gap ends a letter. On the gaps whose nominal interval is 4 time units
it compares three readings, each a single threshold fitted on half the words and tested on the rest:
the majority class, the summed interval alone, and the next mark's duration (what the channel bit
tells a decoder). The production cell is D = 0.5, sigma = 0.39, within-letter fraction 0.936.
"""
import json
import math

import numpy as np

from . import RECORDS
from .morse import MORSE_TABLE

N_WORDS = 6000
WORD_LEN_RANGE = [3, 4, 5, 6]
OMEGA_DRIFTS = [0.0, 0.15, 0.30, 0.50]
SIGMA_GAPS = [0.20, 0.39, 0.55, 0.70]
D_PROD, SIGMA_PROD = 0.50, 0.39
FRAC_REAL = 0.936             # within-letter share of the jitter variance, from real recordings
T_U_REF_MS = 60.0
FLAG_BAR_PP = 5.0
TRAIN_FRAC = 0.5
SAMPLE_SEED = 115
USE_ENGLISH_FREQ = True

LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
_ENG = {'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75, 'S': 6.33, 'H': 6.09,
        'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23,
        'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.49, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15,
        'Q': 0.10, 'Z': 0.07}
_w = np.array([_ENG[c] for c in LETTERS], dtype=float) if USE_ENGLISH_FREQ else np.ones(len(LETTERS))
LETTER_P = _w / _w.sum()
DUR = {'.': 1.0, '-': 3.0}
GAP_INTRA, GAP_INTER = 1.0, 3.0
LETTER_ELEMS = {c: [DUR[s] for s in MORSE_TABLE[c]] for c in LETTERS}


def gen_word_isis(rng, D, sigma, frac):
    """One word's gaps: (summed interval ms, next-mark duration ms, ends-a-letter, nominal interval)."""
    L = rng.choice(WORD_LEN_RANGE)
    letters = rng.choice(len(LETTERS), size=L, p=LETTER_P)
    if D > 0:
        ramp = np.linspace(1.0 - D / 2.0, 1.0 + D / 2.0, L)
        omega = ramp[::-1] if rng.random() < 0.5 else ramp
    else:
        omega = np.ones(L)
    tu_eff = T_U_REF_MS / omega
    elem_dur, elem_letter = [], []
    for li, lc in enumerate(letters):
        for d in LETTER_ELEMS[LETTERS[lc]]:
            elem_dur.append(d)
            elem_letter.append(li)
    elem_dur, elem_letter = np.array(elem_dur), np.array(elem_letter)
    if len(elem_dur) < 2:
        return None
    comp_std = sigma / math.sqrt(2.0)
    rng.normal(0, math.sqrt(frac)) * comp_std            # two unused draws, kept so the random stream matches the record
    rng.normal(0, math.sqrt(frac)) * comp_std
    shg_letter = rng.normal(0, math.sqrt(frac) * comp_std, size=L)
    she_letter = rng.normal(0, math.sqrt(frac) * comp_std, size=L)
    indep_scale = math.sqrt(1.0 - frac) * comp_std
    f_isi, f_nextdur, label, nominal = [], [], [], []
    for i in range(len(elem_dur) - 1):
        owner_next = elem_letter[i + 1]
        same_letter = elem_letter[i] == owner_next
        gap_nom = GAP_INTRA if same_letter else GAP_INTER
        nxt_nom = elem_dur[i + 1]
        tu = tu_eff[owner_next]
        n_gap = (shg_letter[owner_next] + rng.normal(0, indep_scale)) * tu
        n_el = (she_letter[owner_next] + rng.normal(0, indep_scale)) * tu
        el_obs = nxt_nom * tu + n_el
        f_isi.append(gap_nom * tu + n_gap + el_obs)
        f_nextdur.append(el_obs)
        label.append(0 if same_letter else 1)
        nominal.append(round(gap_nom + nxt_nom))
    return np.array(f_isi), np.array(f_nextdur), np.array(label, dtype=int), np.array(nominal, dtype=int)


def build_dataset(D, sigma, frac, n_words, seed):
    rng = np.random.default_rng(seed)
    F_ISI, F_ND, Y, NOM, WID = [], [], [], [], []
    wid = 0
    for _ in range(n_words):
        out = gen_word_isis(rng, D, sigma, frac)
        if out is None:
            continue
        fi, fn, y, nom = out
        F_ISI.append(fi); F_ND.append(fn); Y.append(y); NOM.append(nom)
        WID.append(np.full(len(fi), wid))
        wid += 1
    return (np.concatenate(F_ISI), np.concatenate(F_ND), np.concatenate(Y), np.concatenate(NOM),
            np.concatenate(WID), wid)


def best_threshold(x, y):
    """The training-accuracy-maximising threshold and direction on a one-dimensional feature."""
    xs = np.sort(np.unique(x))
    if len(xs) > 400:
        xs = xs[np.linspace(0, len(xs) - 1, 400).astype(int)]
    cands = (xs[:-1] + xs[1:]) / 2.0 if len(xs) > 1 else np.array([xs[0]])
    best_t, best_acc, best_sign = cands[0], -1.0, 1
    for t in cands:
        for sign in (1, -1):
            acc = (((x > t) if sign == 1 else (x < t)).astype(int) == y).mean()
            if acc > best_acc:
                best_acc, best_t, best_sign = acc, t, sign
    return best_t, best_sign


def apply_threshold(x, t, sign):
    return (x > t).astype(int) if sign == 1 else (x < t).astype(int)


def eval_cell(D, sigma, frac, seed):
    f_isi, f_nd, y, nom, wid, n_words = build_dataset(D, sigma, frac, N_WORDS, seed)
    is_train_word = np.random.default_rng(seed + 7).random(n_words) < TRAIN_FRAC
    tr = is_train_word[wid]
    te = ~tr
    t, s = best_threshold(f_isi[tr], y[tr])
    pred_isi = apply_threshold(f_isi, t, s)
    full_acc = (pred_isi[te] == y[te]).mean()
    correct = (pred_isi == y).astype(bool)
    okmap = {w: bool(correct[(wid == w) & te].all()) for w in np.unique(wid[te])}
    full_pp = float(np.mean(list(okmap.values()))) if okmap else float("nan")
    sub = nom == 4
    sub_te, sub_tr = sub & te, sub & tr
    y_sub = y[sub_te]
    base = max(y_sub.mean(), 1 - y_sub.mean()) if len(y_sub) else float("nan")
    if sub_tr.sum() > 10 and sub_te.sum() > 10:
        ts, ss = best_threshold(f_isi[sub_tr], y[sub_tr])
        ind_sub_acc = (apply_threshold(f_isi[sub_te], ts, ss) == y_sub).mean()
        tn, sn = best_threshold(f_nd[sub_tr], y[sub_tr])
        ctx_sub_acc = (apply_threshold(f_nd[sub_te], tn, sn) == y_sub).mean()
    else:
        ind_sub_acc = ctx_sub_acc = float("nan")
    return {"n_isi_test": int(te.sum()), "n_words": int(n_words),
            "full_indicator_acc": float(full_acc), "full_indicator_pp": float(full_pp),
            "subset_n_test": int(sub_te.sum()), "subset_majority_base": float(base),
            "subset_indicator_acc": float(ind_sub_acc), "subset_context_acc": float(ctx_sub_acc),
            "subset_indicator_over_base_pp": float((ind_sub_acc - base) * 100.0),
            "subset_context_over_base_pp": float((ctx_sub_acc - base) * 100.0),
            "isi_threshold_ms": float(t)}


def run():
    results = {"verdict": "DIAGNOSTIC", "flag_bar_pp": FLAG_BAR_PP, "frac_real": FRAC_REAL,
               "params": {"N_WORDS": N_WORDS, "WORD_LEN_RANGE": WORD_LEN_RANGE, "OMEGA_DRIFTS": OMEGA_DRIFTS,
                          "SIGMA_GAPS": SIGMA_GAPS, "D_PROD": D_PROD, "SIGMA_PROD": SIGMA_PROD, "TRAIN_FRAC": TRAIN_FRAC,
                          "USE_ENGLISH_FREQ": USE_ENGLISH_FREQ, "SAMPLE_SEED": SAMPLE_SEED},
               "grid": {}}
    for D in OMEGA_DRIFTS:
        for sg in SIGMA_GAPS:
            results["grid"]["D%.2f_s%.2f" % (D, sg)] = eval_cell(D, sg, FRAC_REAL, SAMPLE_SEED + int(D * 100) + int(sg * 100))
    prod = results["grid"]["D%.2f_s%.2f" % (D_PROD, SIGMA_PROD)]
    results["prod_cell"] = {"D": D_PROD, "sigma": SIGMA_PROD, "frac": FRAC_REAL, **prod}
    results["frac_sweep_prod"] = {"%.3f" % fr: eval_cell(D_PROD, SIGMA_PROD, fr, SAMPLE_SEED + 900 + int(fr * 100))
                                  for fr in [0.0, 0.5, FRAC_REAL, 1.0]}
    results["flag_indicator_resolves_subset"] = bool(prod["subset_indicator_over_base_pp"] > FLAG_BAR_PP)
    results["flag_margin_pp"] = float(prod["subset_indicator_over_base_pp"])
    results["context_resolves_subset_pp"] = float(prod["subset_context_over_base_pp"])
    return results


if __name__ == "__main__":
    R = run()
    (RECORDS / "collision.json").write_text(json.dumps(R, indent=2))
    p = R["prod_cell"]
    print(f"on the 4.0 time-unit gaps: majority {100 * p['subset_majority_base']:.1f}%, interval alone "
          f"{100 * p['subset_indicator_acc']:.1f}%, next-mark duration {100 * p['subset_context_acc']:.1f}%")
