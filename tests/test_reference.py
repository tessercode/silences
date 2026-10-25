"""Fast checks of the software reference: seconds, no training, no evaluation corpus.

The full checks are `python -m silences.golden.check` (every firmware golden) and `make evaluate`
(every record); these catch a broken change before those run.
"""
import json
import random

import numpy as np

from silences import FIRMWARE, encoder_ref
from silences.decoder.beam import decode_tx
from silences.evaluation import protocol
from silences.fixedpoint import q16
from silences.fixedpoint.window_net import FixedWindowNet
from silences.operators import OPERATORS
from silences.segmenters import bigru
from silences.segmenters.lif import LIF
from silences.evaluation.producers import COMMITTED
from silences.training.window_nets import training_matrix

G = FIRMWARE / "golden"


def _golden(name):
    return json.loads((G / name).read_text())


def test_protocols_are_the_registered_ones():
    assert protocol.RANDOM_LETTERS_HASH == "f3b85d0c9db49b96"
    assert protocol.DECODER_WEIGHTS_HASH == "a7f80cf1ace9639d"


def test_operators_in_table_order():
    assert list(OPERATORS) == ["fast_normalr", "slow_normalr", "high_r_bug", "cootie_high_jitter", "cootie_corrected"]
    assert OPERATORS["cootie_corrected"]["omega"] == 0.3774


def test_q16_scalar_and_vector_agree():
    rng = random.Random(0)
    xs = [rng.randint(-20 * q16.ONE, 20 * q16.ONE) for _ in range(5000)]
    for lut in (q16.SIGMOID, q16.TANH, q16.EXP):
        assert [lut(x) for x in xs] == list(lut.vec(np.array(xs)))
    ps = [rng.randint(1, q16.ONE) for _ in range(5000)]
    assert [q16.fxlog(p) for p in ps] == list(q16.fxlog_vec(np.array(ps)))
    assert q16.scale_lp3([[-5, -7]], q16.ONE) == [[-5, -7]]
    assert q16.scale_lp3([[-65536]], q16.grid_q(1.5)) == [[-98304]]


def test_training_matrix_is_the_one_the_models_were_trained_on():
    assert training_matrix()[2].startswith("119dbc64")      # also asserted inside


def test_generator_matches_the_published_stream():
    K = _golden("rng_kat.json")["seeds"][3]                  # seed 42
    r = encoder_ref.Rng(K["seed"])
    assert [r.next_u32() for _ in range(16)] == K["stream"]


def test_beam_decodes_golden_vectors():
    cap = {v["vid"]: v for v in _golden("beam_cap_vectors.json")["vectors"]}
    for g in _golden("beam_n5.json")["vectors"]:
        if g["source"] == "cap_scale_new" and int(g["vector_id"]) in list(cap)[:3]:
            v = cap[int(g["vector_id"])]
            assert decode_tx(v["ch"], v["lp3_n5"]) == (g["decoded_text"], g["seg"], g["final_score_q16"])


def test_bigru_and_lif_fixed_point_match_their_goldens():
    vec = {v["vid"]: v for v in json.loads((FIRMWARE.parent / "data" / "golden_inputs" / "segmenter_vectors.json").read_text())["vectors"]}
    seeds = bigru.deployed(5)
    for r in _golden("segmenter_bigru_n5.json")["vectors"][:3]:
        v = vec[int(r["vector_id"])]
        assert bigru.lp3_ticks(seeds, v["ticks"], v["ch"], v["tu"]) == r["log_p3_q16"]
    net = FixedWindowNet.from_lif(LIF.load(COMMITTED["lif"]))
    for v in _golden("segmenter_lif.json")["vectors"][:3]:
        assert net.lp3_ticks(v["ticks"], v["ch"], v["tu"]) == v["lp3"]
