"""The character language model (Section 5): a trigram over the 37 symbols the decoder can emit
(space, 0-9, A-Z), mixing English (the NLTK Brown corpus) and templated ham traffic at lambda = 0.75.

The decoder and the RP2040 decode board read one table of 37^3 log-probabilities in Q16.16,
models/charlm_mix_q16.bin, indexed (37 c1 + c2) 37 + c3 with the context left-padded by spaces.
`python -m silences.decoder.charlm build` retrains it from the two corpora and checks the result is
byte-identical to the shipped table.

Each corpus is normalised the same way: upper-case, keep the 36 Morse characters, turn everything
else into a space, collapse runs of spaces, strip. Each trigram is add-k smoothed (k = 0.5) with a
space-padded start; the last tenth of each corpus is held out (it is not used in the table).
"""
import hashlib
import sys

import numpy as np

from .. import MODELS
from ..morse import MORSE_TABLE

ALPHABET = " " + "".join(sorted(MORSE_TABLE))           # space, then 0-9, then A-Z
C2I = {c: i for i, c in enumerate(ALPHABET)}
V = len(ALPHABET)
LAMBDA = 0.75                                            # weight of the English model
TABLE = MODELS / "charlm_mix_q16.bin"
TABLE_SHA256 = "b15d10d0391707bd3ce7430d1ec7618831c88d080c62cdf5d9fc5f1a88bd525b"
ONE = 65536

_LMQ = None


def table():
    """The Q16.16 table as int64, refusing a file whose hash is not the shipped one."""
    global _LMQ
    if _LMQ is None:
        raw = TABLE.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == TABLE_SHA256, f"{TABLE} is not the shipped LM table"
        _LMQ = np.frombuffer(raw, dtype="<i4").astype(np.int64)
        assert len(_LMQ) == V ** 3
    return _LMQ


def logp_q16(ch, emitted):
    """log P(ch | last two emitted characters), Q16.16 nats; the context is left-padded with spaces."""
    h = ("  " + emitted)[-2:]
    return int(table()[(V * C2I.get(h[0], 0) + C2I.get(h[1], 0)) * V + C2I.get(ch, 0)])


# ---- building the table ----
def normalize(text):
    out, prev_space = [], True
    for ch in text.upper():
        if ch in C2I and ch != " ":
            out.append(ch)
            prev_space = False
        else:
            if not prev_space:
                out.append(" ")
            prev_space = True
    return "".join(out).strip()


def fit_trigram(text, add_k=0.5):
    """Dense add-k trigram log-probabilities, float32, shape (V, V, V)."""
    counts = np.zeros((V, V, V), dtype=np.float64)
    a, b = C2I[" "], C2I[" "]
    for ch in text:
        i = C2I[ch]
        counts[a, b, i] += 1.0
        a, b = b, i
    counts += add_k
    return np.log(counts / counts.sum(axis=-1, keepdims=True)).astype(np.float32)


def training_streams():
    """(English, ham) training text: each corpus normalised, its last tenth held out, joined by spaces."""
    from nltk.corpus import brown
    from . import ham_corpus
    eng = [t for t in (normalize(" ".join(s)) for s in brown.sents()) if t]
    ham = [t for t in (normalize(x) for x in ham_corpus.generate(n_exchanges=40000, seed=199_100)) if t]
    return " ".join(eng[:-(len(eng) // 10)]), " ".join(ham[:-(len(ham) // 10)])


def build():
    """The table bytes: round-half-even of the float64 mixture log(lambda P_eng + (1 - lambda) P_ham)."""
    eng_text, ham_text = training_streams()
    eng, ham = fit_trigram(eng_text), fit_trigram(ham_text)
    x = np.logaddexp(np.log(LAMBDA) + eng.astype(np.float64), np.log(1.0 - LAMBDA) + ham.astype(np.float64))
    return np.round(x * ONE).astype(np.int64).reshape(-1).astype("<i4").tobytes()


if __name__ == "__main__":
    assert sys.argv[1:] == ["build"], "usage: python -m silences.decoder.charlm build"
    got = hashlib.sha256(build()).hexdigest()
    print(f"rebuilt table sha256 {got[:16]}: {'IDENTICAL to' if got == TABLE_SHA256 else 'DIFFERS from'} the shipped table")
    sys.exit(0 if got == TABLE_SHA256 else 1)
