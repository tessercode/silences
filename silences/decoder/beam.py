"""The sentence decoder (Section 5): a beam search over the segmenter's per-gap posteriors.

A hypothesis is (emitted text, the Morse pattern of the letter in progress, score, gap classes).
At each gap it can extend the current letter (an intra-letter gap) if the longer pattern is still a
prefix of some letter, or close the letter (an inter-letter or inter-word gap) if the pattern is a
complete letter. Closing adds the character language model's log-probability plus the insertion
bonus gamma. Scores are integers in Q16.16 and only ever added: the score of a decode is

    S = sum over gaps of T * log p(class)  +  sum over letters of [log P_LM(letter | two before) + gamma]

with T already applied to the posteriors (fixedpoint.q16.scale_lp3). The beam keeps the 20 best
hypotheses after each gap (a stable sort, so ties keep generation order), and the first best complete
hypothesis wins. The RP2040 decode board runs this same algorithm bit for bit.
"""
from ..fixedpoint.q16 import grid_q
from ..morse import MORSE_TABLE
from .charlm import logp_q16

MORSE_INVERSE = {v: k for k, v in MORSE_TABLE.items()}
MORSE_PREFIXES = frozenset(pat[:i] for pat in MORSE_TABLE.values() for i in range(1, len(pat) + 1))
BEAM_WIDTH = 20
GAMMA = 2.5                               # the default insertion bonus
GAMMA_Q = grid_q(GAMMA)                   # 0x00028000


def decode(channels, lp3, gamma_q=GAMMA_Q, beam_width=BEAM_WIDTH):
    """channels: one '.' or '-' per mark; lp3: one [intra, inter-letter, inter-word] row per gap.

    Returns (text, gap classes, final score), or (None, None, None) if no hypothesis survives."""
    N = len(channels)
    if N == 0:
        return None, None, None
    beam = [("", channels[0], 0, ())]
    for i in range(N - 1):
        lp_in, lp_il, lp_iw = int(lp3[i][0]), int(lp3[i][1]), int(lp3[i][2])
        c_next, next_beam = channels[i + 1], []
        for emitted, current, score, seg in beam:
            if current + c_next in MORSE_PREFIXES:
                next_beam.append((emitted, current + c_next, score + lp_in, seg + (0,)))
            if current in MORSE_INVERSE:
                ch = MORSE_INVERSE[current]
                lm = logp_q16(ch, emitted) + gamma_q            # one LM call serves both closes
                next_beam.append((emitted + ch, c_next, score + lp_il + lm, seg + (1,)))
                next_beam.append((emitted + ch + " ", c_next, score + lp_iw + lm, seg + (2,)))
        next_beam.sort(key=lambda s: s[2], reverse=True)
        beam = next_beam[:beam_width]
        if not beam:
            return None, None, None
    final = [(emitted + MORSE_INVERSE[current], score + logp_q16(MORSE_INVERSE[current], emitted) + gamma_q, seg)
             for emitted, current, score, seg in beam if current in MORSE_INVERSE]
    if not final:
        return None, None, None
    best = max(final, key=lambda s: s[1])
    return best[0].strip(), list(best[2]), int(best[1])


def decode_tx(ch, lp3, gamma_q=GAMMA_Q):
    """decode() from integer channel bits (0 = dot, 1 = dash)."""
    return decode(["." if int(c) == 0 else "-" for c in ch], lp3, gamma_q)
