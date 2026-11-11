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
from silences.segmenters.lif import LIF, SpikeFedLIF
from silences.evaluation.producers import DEPLOYED
from silences.training.window_nets import training_matrix

G = FIRMWARE / "golden"


def _golden(name):
    return json.loads((G / name).read_text())


def test_protocols_are_the_registered_ones():
    assert protocol.RANDOM_LETTERS_HASH == "f3b85d0c9db49b96"
    assert protocol.DECODER_WEIGHTS_HASH == "a7f80cf1ace9639d"
    assert protocol.MATCHED_HASH == "58c31eb368faa229"
    assert protocol.SPIKE_FED_HASH == "76de3765e2be3d62"


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
    training_matrix()          # raises unless the matrix equals the training one, exactly or to 9 decimals


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
    net = FixedWindowNet.from_lif(LIF.load(DEPLOYED["lif"]))
    for v in _golden("segmenter_lif.json")["vectors"][:3]:
        assert net.lp3_ticks(v["ticks"], v["ch"], v["tu"]) == v["lp3"]


def test_spike_fed_network_fixed_point_matches_its_golden():
    """The deployed network of record (the LIF fed spikes) in Q16.16, encoder included, against the firmware golden."""
    m = SpikeFedLIF.load(DEPLOYED["spk"])
    assert m.scale == 3.0 and m.T == 40 and m.fcs[0][0].shape[0] == 84
    net = FixedWindowNet.from_spk(m)
    for v in _golden("segmenter_spk.json")["vectors"][:3]:
        assert net.lp3_ticks(v["ticks"], v["ch"], v["tu"]) == v["lp3"]


def test_spike_encoder_fires_floor_of_rate_times_step():
    """Each ON/OFF channel fires floor(t * rate) spikes by step t, rate = clip(+-z / scale, 0, 1)."""
    m = SpikeFedLIF.load(DEPLOYED["spk"])
    F = np.tile(m.mu, (3, 1)) + np.array([[0.0], [1.5], [-4.0]]) * m.sd        # z = 0, +1.5 sd, -4 sd on every feature
    spk = m.encode(F)
    assert spk.shape == (m.T, 3, 84) and set(np.unique(spk)) <= {0.0, 1.0}
    cum = spk.cumsum(0)
    assert cum[-1, 0].sum() == 0                                                 # z = 0: no spikes on either channel
    assert np.all(cum[:, 1, :42] == np.floor(np.arange(1, m.T + 1)[:, None] * 0.5)) and cum[-1, 1, 42:].sum() == 0
    assert np.all(cum[:, 2, 42:] == np.arange(1, m.T + 1)[:, None]) and cum[-1, 2, :42].sum() == 0   # saturated OFF channel


def test_matched_record_names_the_deployed_models():
    """The record of the paper hashes the two deployed networks, and its protocol chain reaches the first one."""
    import hashlib
    from silences import PROTOCOL, RECORDS
    from silences.evaluation import protocol as P
    R = json.loads((RECORDS / "matched" / "results.json").read_text())
    assert R["protocol_hash"] == P.MATCHED_HASH and R["anchors_pass"]
    for fam in ("lif", "mlp"):
        p = RECORDS.parent / R["deployed"][fam]["installed"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == R["deployed"][fam]["pickle_sha256"]
    assert P.MATCHED["base_protocol_hash"] == P.GRID_WINDOW_HASH and P.GRID_WINDOW["base_protocol_hash"] == P.GRID_BIGRU_HASH
    assert P.GRID_BIGRU["base_protocol_hash"] == P.DECODER_WEIGHTS_HASH and (PROTOCOL / "matched.json").exists()


def test_spike_fed_record_names_the_deployed_model():
    """The record of the paper since v1.2.0: the spike-fed family's seed 0 is the deployed model, the ruling that
    adopted it names that file, and the port gate's fidelity was measured on it."""
    import hashlib
    from silences import RECORDS
    from silences.evaluation import protocol as P
    R = json.loads((RECORDS / "spike_fed" / "results.json").read_text())
    D = json.loads((RECORDS / "spike_fed" / "research" / "decision.json").read_text())
    F = json.loads((RECORDS / "spike_fed" / "fidelity.json").read_text())
    assert R["protocol_hash"] == P.SPIKE_FED_HASH and P.SPIKE_FED["base_protocol_hash"] == P.GRID_WINDOW_HASH
    assert R["anchors"]["c3_pass"] and R["anchors"]["c4"]["c4_disjoint"] and R["anchors"]["c5"]["passed"]
    sha = hashlib.sha256(DEPLOYED["spk"].read_bytes()).hexdigest()
    assert D["target"]["family"] == "spk40_op12" and D["target"]["sha256"] == sha
    assert F["g1"]["g1_pass"] and F["g1"]["installed_sha256"] == sha and F["g2_agreement_pass"]
