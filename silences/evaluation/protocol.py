"""The pre-registered evaluation protocols, loaded verbatim from protocol/*.json.

random_letters.json    the evaluation on random letters, one posterior scale T per model fitted on a
                       reserved corpus, insertion bonus fixed at 2.5 (registered 2026-09-16/17)
decoder_weights.json   the evaluation on English words and random letters, (T, gamma) fitted per model
                       and per text (registered 2026-10-14; chained to the first by its hash)

Each protocol is identified by the first 16 hex digits of the sha256 of its JSON with sorted keys.
The records carry that hash, so a result can never be mistaken for one made under another protocol.
The texts use the research repository's names; the README's glossary maps them to this package.
"""
import hashlib
import json

from .. import PROTOCOL


def _load(name, expected_hash):
    p = json.loads((PROTOCOL / f"{name}.json").read_text())
    h = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]
    assert h == expected_hash, f"protocol/{name}.json has changed (hash {h}, registered {expected_hash})"
    return p, h


RANDOM_LETTERS, RANDOM_LETTERS_HASH = _load("random_letters", "f3b85d0c9db49b96")
DECODER_WEIGHTS, DECODER_WEIGHTS_HASH = _load("decoder_weights", "a7f80cf1ace9639d")
assert DECODER_WEIGHTS["base_protocol_hash"] == RANDOM_LETTERS_HASH

EVAL = RANDOM_LETTERS["eval"]                      # per_fist 250, evaluation seeds 999/777/555/321
CALIB = RANDOM_LETTERS["calib"]                    # random-letter calibration: seed 4242, T grid, CER
TRAIN = RANDOM_LETTERS["train"]                    # LIF/MLP training corpus: per_fist 1000, seed 13
GAMMA_PINNED = RANDOM_LETTERS["beta_pinned"]       # 2.5: the insertion bonus of the random-letter protocol
TEXTS = DECODER_WEIGHTS["texts"]                   # ["rand", "eng"]
CALIB_SEED = DECODER_WEIGHTS["calib_seed"]         # rand 4242, eng 4243
T_GRID = DECODER_WEIGHTS["T_grid"]
GAMMA_GRID = DECODER_WEIGHTS["beta_grid"]
DEFAULT_WEIGHTS = DECODER_WEIGHTS["deployed"]      # T = 1, gamma = 2.5
OPERATORS = ("fast_normalr", "slow_normalr", "high_r_bug", "cootie_high_jitter", "cootie_corrected")
