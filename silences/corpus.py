"""Synthetic TEMPO transmissions with per-gap truth (Section 4).

A transmission is a few words rendered by one operator into a train of mark-completion spikes, one
per dot or dash, each with its channel bit (0 = dot, 1 = dash, decided against the threshold
1.92 time units, as on the device). Every gap between consecutive spikes is labelled intra-letter
(0), inter-letter (1) or inter-word (2); that label is what a segmenter predicts.

Two texts are used. Random letters (word length uniform on 2..6, letters uniform) give the language
model nothing to read. English draws the word length the same way, then a uniform English word of
that length from the NLTK word list.

The rendering code consumes random draws in a fixed order that the corpora of record depend on: do
not reorder, vectorise or skip draws in render_sentence, random_words or english_words.
"""
import numpy as np

from .morse import LETTERS, MORSE_TABLE
from .operators import OPERATORS

T_THRESH = 1.92          # dot/dash channel threshold, in time units (the device value)
WORD_GAP_TU = 7.0        # nominal inter-word gap, in time units
WORD_LENGTHS = (2, 6)    # letters per word, both texts
WORDS_PER_TX = (2, 5)    # words per transmission


def render_sentence(words, operator, rng, word_gap_tu=WORD_GAP_TU):
    """Render words as one spike train.

    Each word draws its own speed drift omega ~ N(1, operator omega), floored at 0.2. Marks are 1
    (dot) or r (dash) time units, gaps 1 (intra-letter), 3 (inter-letter) or word_gap_tu
    (inter-word) time units, each with Gaussian jitter and floored at 0.1 time units.
    Returns (spikes [(t_s, ch)], letter index per spike, word index per spike, true time unit per
    word in seconds)."""
    t_u_base = 1.2 / operator["wpm"]
    r_val = operator["r"]
    spikes, lidx, widx, tu_words = [], [], [], []
    t_cursor, gletter = 0.0, -1
    for wi, word in enumerate(words):
        omega_word = max(0.2, rng.normal(1.0, operator["omega"]))
        t_u = t_u_base * omega_word
        tu_words.append(t_u)
        t_thresh = T_THRESH * t_u
        sigma_e = operator["sigma_elem"] * t_u
        sigma_g = operator["sigma_gap"] * t_u
        for ci, ch in enumerate(word):
            if ch not in MORSE_TABLE:
                continue
            gletter += 1
            elements = MORSE_TABLE[ch]
            for eidx, el in enumerate(elements):
                t_ideal = (1.0 if el == "." else r_val) * t_u
                noise = rng.normal(0.0, sigma_e) if sigma_e > 0 else 0.0
                t_dur = max(t_ideal + noise, 0.1 * t_u)
                t_spike = t_cursor + t_dur
                spikes.append((t_spike, 0 if t_dur < t_thresh else 1))
                lidx.append(gletter)
                widx.append(wi)
                if eidx < len(elements) - 1:
                    g = rng.normal(0.0, sigma_g) if sigma_g > 0 else 0.0
                    t_cursor = t_spike + max(1.0 * t_u + g, 0.1 * t_u)
                else:
                    t_cursor = t_spike
            # the gap after a letter: inter-letter, or inter-word after a word's last letter
            g = rng.normal(0.0, sigma_g) if sigma_g > 0 else 0.0
            gap_nom = word_gap_tu if ci == len(word) - 1 else 3.0
            t_cursor += max(gap_nom * t_u + g, 0.1 * t_u)
    return spikes, np.array(lidx), np.array(widx), np.array(tu_words)


def random_words(rng, n_words, wlen_range=WORD_LENGTHS):
    """Random-letter text: length uniform on wlen_range, letters uniform over LETTERS."""
    out = []
    for _ in range(n_words):
        L = int(rng.integers(wlen_range[0], wlen_range[1] + 1))
        out.append("".join(LETTERS[int(rng.integers(0, len(LETTERS)))] for _ in range(L)))
    return out


_ENGLISH = None


def english_vocabulary():
    """{length: sorted words} from the NLTK word list: alphabetic, upper-cased, lengths 2..6."""
    global _ENGLISH
    if _ENGLISH is None:
        from nltk.corpus import words as nltk_words
        lo, hi = WORD_LENGTHS
        v = sorted({w.upper() for w in nltk_words.words()
                    if w.isalpha() and lo <= len(w) <= hi and all(c in LETTERS for c in w.upper())})
        _ENGLISH = {L: [w for w in v if len(w) == L] for L in range(lo, hi + 1)}
        assert all(_ENGLISH.values()), "an English length bucket is empty (is the NLTK 'words' corpus installed?)"
    return _ENGLISH


def english_words(rng, n_words):
    """English text with random_words' length law: L uniform on 2..6, then a uniform word of length L."""
    V = english_vocabulary()
    lo, hi = WORD_LENGTHS
    out = []
    for _ in range(n_words):
        L = int(rng.integers(lo, hi + 1))
        out.append(V[L][int(rng.integers(0, len(V[L])))])
    return out


TEXTS = dict(rand=random_words, eng=english_words)


def render_transmission(operator, rng, words_fn=random_words, n_words_range=WORDS_PER_TX):
    """One transmission by the named operator, or None if it has fewer than three spikes.

    Returns dict(fist, ticks [us], ch, widx, lidx, labels per gap, tu_words_us, true_tu_us, words)."""
    op = OPERATORS[operator]
    nw = int(rng.integers(n_words_range[0], n_words_range[1] + 1))
    words = words_fn(rng, nw)
    spikes, lidx, widx, tu_words = render_sentence(words, op, rng)
    if len(spikes) < 3:
        return None
    ticks = np.array([round(s[0] * 1e6) for s in spikes], dtype=np.int64)
    ch = np.array([s[1] for s in spikes], dtype=np.int64)
    labels = np.where(lidx[1:] == lidx[:-1], 0, np.where(widx[1:] == widx[:-1], 1, 2)).astype(np.int64)
    tu_words_us = np.asarray(tu_words, dtype=np.float64) * 1e6
    return dict(fist=operator, ticks=ticks, ch=ch, widx=widx, lidx=lidx, labels=labels,
                tu_words_us=tu_words_us, true_tu_us=float(np.mean(tu_words_us)), words=words)


def corpus(per_fist, seed, text="rand", operators=None, rng=None):
    """per_fist transmissions from each operator in turn, from one generator seeded with seed.

    A rendering with fewer than three spikes is discarded and redrawn; its draws still count."""
    rng = np.random.default_rng(seed) if rng is None else rng
    words_fn = TEXTS[text]
    out = []
    for f in operators or list(OPERATORS):
        made = 0
        while made < per_fist:
            tx = render_transmission(f, rng, words_fn)
            if tx is not None:
                out.append(tx)
                made += 1
    return out
