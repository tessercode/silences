"""The pre-registered evaluation protocols, loaded verbatim from protocol/*.json.

random_letters.json    the evaluation on random letters, one posterior scale T per model fitted on a
                       reserved corpus, insertion bonus fixed at 2.5 (registered 2026-09-16/17)
decoder_weights.json   the evaluation on English words and random letters, (T, gamma) fitted per model
                       and per text (registered 2026-10-14; chained to the first by its hash)
grid_bigru.json        the BiGRU recipe retrained on three final-phase corpora, scored on the record
                       and on four fresh evaluation seeds (registered 2026-11-06; chained)
grid_window.json       the feedforward recipe trained on the BiGRU's two corpora (2026-11-06; chained)
matched.json           the record of the paper since v1.1.0: every family at matched training data,
                       the deployed seed-0 networks, their fixed-point fidelity (2026-11-06; chained)

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
GRID_BIGRU, GRID_BIGRU_HASH = _load("grid_bigru", "cafa08a4f02b598e")
GRID_WINDOW, GRID_WINDOW_HASH = _load("grid_window", "c041694763a2f430")
MATCHED, MATCHED_HASH = _load("matched", "58c31eb368faa229")
assert DECODER_WEIGHTS["base_protocol_hash"] == RANDOM_LETTERS_HASH
assert GRID_BIGRU["base_protocol_hash"] == DECODER_WEIGHTS_HASH
assert GRID_WINDOW["base_protocol_hash"] == GRID_BIGRU_HASH
assert MATCHED["base_protocol_hash"] == GRID_WINDOW_HASH

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
FRESH_SEEDS = GRID_BIGRU["part_b"]["fresh_eval_seeds"]      # 1106, 2212, 3318, 4424
GRID_N_TX = GRID_WINDOW["n_tx"]                             # 12,000 transmissions per training seed
GRID_CORPUS_SEED0 = GRID_WINDOW["corpus_seed0"]             # op12 210442, gen 197442 (+ training seed)
BIGRU_ARMS = tuple(GRID_BIGRU["part_a"]["arms"])            # rep, fist, shared
BIGRU_FIST = GRID_BIGRU["part_a"]["fist"]                   # per_fist 2400, seed_base 210400
BIGRU_SHARED = GRID_BIGRU["part_a"]["shared"]               # per_fist 1000, seed 13
MATCHED_FAMILIES = MATCHED["families"]                      # lif -> lif_op12, mlp -> mlp_op12, bigru -> bigru_fist
MATCHED_SINGLES = MATCHED["singles"]
DEPLOYED_SEED = MATCHED["deployed_seed"]                    # 0
# Ruling (Luke, 2026-11-08): the fixed-point CER/WER bound of protocol/matched.json (fidelity.max_abs_pp) was
# missed by one cell (0.20 pp, LIF random-letter WER at the default weights; CER within 0.04 pp) while the
# decision criterion (fidelity.min_agreement) held at 99.993%. The record stands on the decision criterion
# and the bound is reported, not gating. The protocol and its hash are left as registered.
MATCHED_RULING = dict(date="2026-11-08", by="Luke", gating=["c1_model_identity", "c4_pass", "h1_fx_agreement_pass"],
                      reported_not_gating=["h1_fx_pp_pass"],
                      reason="decision criterion met at 99.993%; pp bound missed by one WER cell (0.20 pp), CER within 0.04 pp")
