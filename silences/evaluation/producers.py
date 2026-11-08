"""Every segmenter the paper evaluates, as a named producer tx -> lp3.

Row names are the keys of the result records:
    lif_seed0..9, mlp_seed0..9      the ten training seeds of each feedforward family on the original
                                    5,000-transmission corpus
    bigru_seed42..49                the baseline's eight training seeds (generic corpus, the original pool)
    bigru_deployed                  the five-seed ensemble of that pool, the one on the gap9 board's firmware
    lif_committed, mlp_committed    the feedforward networks first deployed (5,000 transmissions; superseded
                                    rows since v1.1.0, never seed-sample members)
    lif_committed_fx, mlp_committed_fx   the same two in the firmware's Q16.16 arithmetic
    perfect_gaps                    the oracle
The matched-data record (v1.1.0) adds, through grid_producers():
    lif_op12_seed0..9, mlp_op12_seed0..9   the families of record: 12,000 operator transmissions per seed
    lif_gen_seed0..9, mlp_gen_seed0..9     the same recipe on the BiGRU's generic corpus
    bigru_{rep,fist,shared}_seed42..49     the BiGRU recipe with its final phase on each corpus
                                           (fist is the baseline of record), and bigru_{arm}_ens5
    lif_op12_seed0_fx, mlp_op12_seed0_fx   the deployed networks in the firmware's Q16.16 arithmetic
"""
import json

from .. import MODELS
from ..fixedpoint.window_net import FixedWindowNet
from ..segmenters import bigru, oracle
from ..segmenters.lif import LIF
from ..segmenters.mlp import MLP

SEEDS = list(range(10))
FAMILY_ROWS = dict(lif=[f"lif_seed{s}" for s in SEEDS], mlp=[f"mlp_seed{s}" for s in SEEDS],
                   bigru=[f"bigru_seed{s}" for s in bigru.POOL_SEEDS])


def model_path(family, seed):
    return MODELS / family / f"seed{seed}.pkl"


COMMITTED = dict(lif=MODELS / "lif" / "deployed_5k.pkl", mlp=MODELS / "mlp" / "deployed_5k.pkl")   # the superseded rows
DEPLOYED = dict(lif=MODELS / "lif" / "op12" / "seed0.pkl", mlp=MODELS / "mlp" / "op12" / "seed0.pkl")  # on the boards since v1.1.0
_LOAD = dict(lif=LIF.load, mlp=MLP.load)
GRID_ARMS = ("op12", "gen")


def grid_model_path(family, arm, seed):
    return MODELS / family / arm / f"seed{seed}.pkl"


def bigru_arm_training():
    return json.loads((MODELS.parent / "records" / "training" / "bigru_arms.json").read_text())["arms"]


def grid_family_rows():
    """{family: [rows]} for the grid families, in the records' order."""
    F = {}
    for arm in bigru.ARMS:
        F[f"bigru_{arm}"] = [f"bigru_{arm}_seed{s}" for s in bigru_arm_training()[arm]["alive"]]
    for fam in ("mlp", "lif"):
        for arm in GRID_ARMS:
            F[f"{fam}_{arm}"] = [f"{fam}_{arm}_seed{s}" for s in SEEDS]
    return F


def grid_producers(fixed_point=True):
    """{row: producer}, {row: model file} for the rows the matched-data record adds."""
    prods, sources = {}, {}
    T = bigru_arm_training()
    for arm in bigru.ARMS:
        for s in T[arm]["alive"]:
            prods[f"bigru_{arm}_seed{s}"] = bigru.producer([bigru.load_q16(s, bigru.arm_q16(arm))])
            sources[f"bigru_{arm}_seed{s}"] = bigru.arm_dir(arm) / f"bigru3_seed{s}.pt"
        prods[f"bigru_{arm}_ens5"] = bigru.producer([bigru.load_q16(s, bigru.arm_q16(arm)) for s in T[arm]["selected_top"]])
        sources[f"bigru_{arm}_ens5"] = bigru.arm_q16(arm)
    for fam in ("mlp", "lif"):
        for arm in GRID_ARMS:
            for s in SEEDS:
                p = grid_model_path(fam, arm, s)
                prods[f"{fam}_{arm}_seed{s}"], sources[f"{fam}_{arm}_seed{s}"] = _LOAD[fam](p).lp3, p
    if fixed_point:
        prods["lif_op12_seed0_fx"] = FixedWindowNet.from_lif(LIF.load(DEPLOYED["lif"])).lp3
        prods["mlp_op12_seed0_fx"] = FixedWindowNet.from_mlp(MLP.load(DEPLOYED["mlp"])).lp3
        sources["lif_op12_seed0_fx"], sources["mlp_op12_seed0_fx"] = DEPLOYED["lif"], DEPLOYED["mlp"]
    return prods, sources


def all_producers(oracle_conf=oracle.CONF, fixed_point=True):
    """{row: producer} in the records' row order, and {row: model file}."""
    prods, sources = {}, {}
    for fam in ("lif", "mlp"):
        for s in SEEDS:
            p = model_path(fam, s)
            prods[f"{fam}_seed{s}"], sources[f"{fam}_seed{s}"] = _LOAD[fam](p).lp3, p
    for s in bigru.POOL_SEEDS:
        prods[f"bigru_seed{s}"] = bigru.producer([bigru.load_q16(s)])
        sources[f"bigru_seed{s}"] = bigru.POOL_DIR / f"bigru3_seed{s}.pt"
    prods["bigru_deployed"], sources["bigru_deployed"] = bigru.producer(bigru.deployed()), bigru.POOL_Q16
    for fam in ("lif", "mlp"):
        prods[f"{fam}_committed"], sources[f"{fam}_committed"] = _LOAD[fam](COMMITTED[fam]).lp3, COMMITTED[fam]
    prods["perfect_gaps"], sources["perfect_gaps"] = oracle.producer(oracle_conf), None
    if fixed_point:
        prods["lif_committed_fx"] = FixedWindowNet.from_lif(LIF.load(COMMITTED["lif"])).lp3
        prods["mlp_committed_fx"] = FixedWindowNet.from_mlp(MLP.load(COMMITTED["mlp"])).lp3
        sources["lif_committed_fx"], sources["mlp_committed_fx"] = COMMITTED["lif"], COMMITTED["mlp"]
    return prods, sources
