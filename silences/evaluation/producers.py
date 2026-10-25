"""Every segmenter the paper evaluates, as a named producer tx -> lp3.

Row names are the keys of the result records:
    lif_seed0..9, mlp_seed0..9      the ten training seeds of each feedforward family
    bigru_seed42..49                the baseline's eight training seeds
    bigru_deployed                  the five-seed ensemble the RP2040 chain runs
    lif_committed, mlp_committed    the models committed for deployment (rows, never seed-sample members)
    lif_committed_fx, mlp_committed_fx   the same two in the firmware's Q16.16 arithmetic
    perfect_gaps                    the oracle
"""
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


COMMITTED = dict(lif=MODELS / "lif" / "committed.pkl", mlp=MODELS / "mlp" / "committed.pkl")
_LOAD = dict(lif=LIF.load, mlp=MLP.load)


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
