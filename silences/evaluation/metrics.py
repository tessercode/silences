"""Decoded-text error rates and the Welch comparison of training-seed samples (Section 8)."""
import numpy as np
from scipy import stats

from ..decoder.beam import decode_tx
from ..fixedpoint.q16 import scale_lp3


def levenshtein(a, b):
    """Edit distance between two sequences (strings for CER, word lists for WER)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[m]


def cer_wer(truth, decoded):
    """(character edits, true characters, word edits, true words) for one transmission; a decode
    that died (None) counts as empty text."""
    true_s = " ".join(truth.split())
    dec_s = " ".join(decoded.split()) if decoded else ""
    tw, dw = true_s.split(" "), (dec_s.split(" ") if dec_s else [])
    return levenshtein(true_s, dec_s), len(true_s), levenshtein(tw, dw), len(tw)


def edit_sums(table, row, T_q, gamma_q, per_fist=False):
    """Decode one row of a posterior table at decoder weights (T_q, gamma_q).

    Returns the corpus sums [char edits, chars, word edits, words, exact, transmissions] and, with
    per_fist, the same sums per operator."""
    tot, pf = [0] * 6, {}
    for ch, truth, fist, lp in zip(table["ch"], table["truth"], table["fist"], table["rows"][row]):
        dec = decode_tx(ch, scale_lp3(lp, T_q), gamma_q)[0]
        ce, nc, we, nw = cer_wer(truth, dec)
        for i, v in enumerate((ce, nc, we, nw, int((dec or "").strip() == truth.strip()), 1)):
            tot[i] += v
            if per_fist:
                pf.setdefault(fist, [0] * 6)[i] += v
    return tot, pf


def welch(candidate, baseline, ci=0.95):
    """D = mean(baseline) - mean(candidate) in percentage points (positive: the candidate is better),
    with Welch's interval. A win (or loss) is claimed only when the interval excludes zero."""
    x, y = np.asarray(candidate, float), np.asarray(baseline, float)
    d = float(y.mean() - x.mean())
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = float(np.sqrt(vx + vy))
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    tcrit = float(stats.t.ppf(0.5 + ci / 2, df))
    t, p = stats.ttest_ind(y, x, equal_var=False)
    return dict(d=d, lo=d - tcrit * se, hi=d + tcrit * se, se=se, t=float(t), df=float(df), p=float(p),
                n_candidate=len(x), n_baseline=len(y), ci=ci,
                win=bool(d - tcrit * se > 0), loss=bool(d + tcrit * se < 0))
