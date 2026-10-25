"""The evaluation on random letters, one posterior scale per model (protocol/random_letters.json).

    python -m silences.evaluation.random_letters

With the insertion bonus fixed at 2.5, each model's scale T is chosen from the grid as the value
that minimises decoded CER on the random-letter calibration corpus (seed 4242; ties go to the
smaller T). Every model is then evaluated on the four evaluation corpora at its own T, and again at
T = 1. Comparisons follow the protocol's rule: per training seed, the mean over the evaluation
seeds; between families, Welch's interval on the difference of seed means.

Records written to records/random_letters/:
    calibration.json, calibration_curves.json   the fitted T per model and CER/WER along the grid
    results.json                                every row at its fitted T and at T = 1, the statistics
    diag_joint_beta.json                        how much refitting the bonus jointly with T would buy
                                                (the committed LIF only; reported, never the claim)
    oracle_conf8.json                           the oracle at the confidence the T = 1 record used
    manifest.json                               hashes of the above, the models and the environment
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json          # noqa: E402
import time          # noqa: E402
from multiprocessing import get_context   # noqa: E402

import numpy as np   # noqa: E402

from .. import RECORDS, ROOT   # noqa: E402
from ..decoder.beam import GAMMA_Q   # noqa: E402
from ..fixedpoint.q16 import ONE, grid_q, scale_lp3   # noqa: E402
from ..segmenters import bigru, oracle   # noqa: E402
from . import protocol as P, tables   # noqa: E402
from .metrics import edit_sums, welch   # noqa: E402
from .producers import FAMILY_ROWS, all_producers, model_path   # noqa: E402
from .provenance import env_info, git_info, now, sha256_file   # noqa: E402

OUT = RECORDS / "random_letters"
OPS = P.OPERATORS
WORKERS = tables.WORKERS


def _frac(tot):
    """Overall CER, WER and exact-match as fractions, as the random-letter record computes them."""
    CE, NC, WE, NW, EX, N = tot
    return dict(cer=CE / NC if NC else 0, wer=WE / NW if NW else 0, exact=EX / N if N else 0)


# ---- calibration: one T per row on seed 4242 ----
_TAB = None


def _fit(row):
    curve, best = {}, None
    for T in P.CALIB["grid"]:
        r = _frac(edit_sums(_TAB, row, grid_q(T), GAMMA_Q)[0])
        curve[str(T)] = dict(cer=r["cer"] * 100, wer=r["wer"] * 100)
        if best is None or curve[str(T)][P.CALIB["criterion"]] < best[1]:
            best = (T, curve[str(T)][P.CALIB["criterion"]])
    T = best[0]
    return row, dict(T=T, T_q=grid_q(T), cer=curve[str(T)]["cer"], wer=curve[str(T)]["wer"],
                     at_grid_edge=(T == P.CALIB["grid"][-1]), curve=curve)


def _family(row):
    return row.rsplit("_", 1)[0] if row.startswith(("lif_seed", "mlp_seed", "bigru_seed")) else "row"


def calibrate(rows, sources):
    global _TAB
    _TAB = tables.load("rand", P.CALIB["seed"])
    with get_context("fork").Pool(WORKERS) as pool:
        fits = dict(pool.imap_unordered(_fit, rows))
    out = {}
    for r in rows:
        f = fits[r]
        f.update(family=_family(r), source=str(sources[r].relative_to(ROOT)), source_sha256=sha256_file(sources[r]))
        out[r] = f
    return dict(experiment="exp207_temperature", protocol_hash=P.RANDOM_LETTERS_HASH,
                corpus=dict(seed=P.CALIB["seed"], per_fist=P.CALIB["per_fist"], n_tx=len(_TAB["truth"])),
                grid=P.CALIB["grid"], criterion=P.CALIB["criterion"], beta_pinned=P.GAMMA_PINNED,
                rows=out, finished=now(), git=git_info(), env=env_info())


# ---- evaluation on the four evaluation corpora ----
def _eval(args):
    """One row on one evaluation corpus at Q16.16 scale T_q: overall and per-operator fractions."""
    seed, row, T_q = args
    tot, pf = edit_sums(_TABS[seed], row, T_q, GAMMA_Q, per_fist=True)
    return (seed, row, T_q), (_frac(tot), {f: _frac(v) for f, v in pf.items()})


_TABS = {}


def _row_results(rows_T, extra_tables=None):
    """{row: T} -> the record's per-row block, evaluated on every evaluation seed."""
    jobs = [(s, r, grid_q(T)) for r, T in rows_T.items() for s in P.EVAL["seeds"]]
    with get_context("fork").Pool(WORKERS) as pool:
        got = dict(pool.imap_unordered(_eval, jobs))
    res = {}
    for r, T in rows_T.items():
        R = dict(per_eval_seed={}, per_fist={})
        for s in P.EVAL["seeds"]:
            o, pf = got[(s, r, grid_q(T))]
            R["per_eval_seed"][str(s)] = dict(cer=o["cer"] * 100, wer=o["wer"] * 100, exact=o["exact"] * 100)
            for f, v in pf.items():
                R["per_fist"].setdefault(f, {"cer": [], "wer": []})
                R["per_fist"][f]["cer"].append(v["cer"] * 100)
                R["per_fist"][f]["wer"].append(v["wer"] * 100)
        R["mean"] = {m: float(np.mean([v[m] for v in R["per_eval_seed"].values()])) for m in ("cer", "wer", "exact")}
        R["per_fist"] = {f: {m: float(np.mean(R["per_fist"][f][m])) for m in ("cer", "wer")} for f in OPS if f in R["per_fist"]}
        res[r] = R
    return res


def family_stats(res, names):
    out = {}
    for m in ("cer", "wer"):
        v = np.array([res[n]["mean"][m] for n in names])
        out[m] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()), max=float(v.max()),
                      spread=float(v.max() - v.min()), per_seed=v.tolist())
        ev_means = [np.mean([res[n]["per_eval_seed"][str(es)][m] for n in names]) for es in P.EVAL["seeds"]]
        out[m]["eval_spread"] = float(max(ev_means) - min(ev_means))
    Ts = [res[n]["T"] for n in names]
    out["T"] = dict(min=min(Ts), max=max(Ts), values=Ts)
    out["per_fist"] = {f: {m: dict(mean=float(np.mean([res[n]["per_fist"][f][m] for n in names])),
                                  per_seed=[res[n]["per_fist"][f][m] for n in names])
                          for m in ("cer", "wer")} for f in OPS}
    out["n"] = len(names)
    out["rows"] = list(names)
    return out


def compare(res, fam, cand, base):
    out = {}
    for m in ("cer", "wer"):
        x = [res[n]["mean"][m] for n in fam[cand]["rows"]]
        y = [res[n]["mean"][m] for n in fam[base]["rows"]]
        w = welch(x, y)
        w["wins"] = int(sum(1 for xi in x if xi < np.mean(y)))
        out[m] = w
    out["wins_both"] = int(sum(1 for n in fam[cand]["rows"] if res[n]["mean"]["cer"] < fam[base]["cer"]["mean"]
                               and res[n]["mean"]["wer"] < fam[base]["wer"]["mean"]))
    out["n"] = fam[cand]["n"]
    out["per_fist"] = {f: {m: welch(fam[cand]["per_fist"][f][m]["per_seed"], fam[base]["per_fist"][f][m]["per_seed"])
                           for m in ("cer", "wer")} for f in OPS}
    return out


def summarise(res):
    fam = {f: family_stats(res, FAMILY_ROWS[f]) for f in ("lif", "mlp", "bigru")}
    comp = {f"{a}_vs_{b}": compare(res, fam, a, b) for a, b in (("lif", "bigru"), ("lif", "mlp"), ("mlp", "bigru"))}
    return fam, comp


def anchor_cache_vs_checkpoints(txs):
    """The deployed ensemble from the quantized weights file must equal the same five seeds quantized
    from their float checkpoints, on every transmission of the first evaluation corpus (needs torch)."""
    a = bigru.producer([bigru.load_checkpoint(s) for s in bigru.DEPLOYED_SEEDS])
    b = bigru.producer(bigru.deployed())
    mism = sum(1 for tx in txs if a(tx) != b(tx))
    return dict(transmissions=len(txs), mismatching=mism, passed=mism == 0)


def evaluate(C, sources):
    global _TABS
    _TABS = {s: tables.load("rand", s) for s in P.EVAL["seeds"]}
    rows = list(C["rows"]) + ["perfect_gaps"]
    cal = _row_results({r: (C["rows"][r]["T"] if r in C["rows"] else 1) for r in rows})
    t1 = _row_results({r: 1 for r in rows})
    for block, Tof in ((cal, lambda r: C["rows"][r]["T"] if r in C["rows"] else 1), (t1, lambda r: 1)):
        for r, R in block.items():
            R["family"] = _family(r)
            R["source"] = str(sources[r].relative_to(ROOT)) if sources[r] else None
            R["source_sha256"] = sha256_file(sources[r]) if sources[r] else None
            R["T"], R["T_q"] = Tof(r), grid_q(Tof(r))
    fam, comp = summarise(cal)
    fam1, comp1 = summarise(t1)
    from ..corpus import corpus
    det = RECORDS / "training" / "determinism.json"
    return dict(experiment="exp207_temperature", protocol_hash=P.RANDOM_LETTERS_HASH, protocol=P.RANDOM_LETTERS,
                rule=P.RANDOM_LETTERS["rule"],
                calibration=dict(corpus=C["corpus"], grid=C["grid"], criterion=C["criterion"],
                                 T={n: C["rows"][n]["T"] for n in C["rows"]}),
                rows=cal, families=fam, comparisons=comp, t1=dict(rows=t1, families=fam1, comparisons=comp1),
                anchor=anchor_cache_vs_checkpoints(corpus(P.EVAL["per_fist"], P.EVAL["seeds"][0], "rand")),
                determinism=json.loads(det.read_text()) if det.exists() else None, finished=now())


def oracle_conf8():
    """The oracle at confidence 8 on the evaluation corpora (the ceiling the T = 1 record quoted)."""
    from ..corpus import corpus
    prod = oracle.producer(oracle.CONF_PRE_CALIBRATION)
    for s in P.EVAL["seeds"]:
        _TABS[s] = dict(_TABS[s], rows={"oracle_conf8": [prod(tx) for tx in corpus(P.EVAL["per_fist"], s, "rand")]})
    R = _row_results({"oracle_conf8": 1})["oracle_conf8"]
    return dict(experiment="random_letters_oracle_conf8", protocol_hash=P.RANDOM_LETTERS_HASH,
                conf=oracle.CONF_PRE_CALIBRATION, **R)


# ---- the joint (T, gamma) diagnostic, committed LIF on seed 4242 ----
GAMMAS = (0.0, 1.0, 2.0, 2.5, 3.0, 4.0)


def _joint(args):
    gamma, T = args
    tot = edit_sums(_TAB, "lif_committed", grid_q(T), grid_q(gamma))[0]
    return (gamma, T), dict(cer=100 * tot[0] / tot[1], wer=100 * tot[2] / tot[3])


def diag_joint_beta():
    global _TAB
    _TAB = tables.load("rand", P.CALIB["seed"])
    jobs = [(g, T) for g in GAMMAS for T in P.CALIB["grid"]]
    with get_context("fork").Pool(WORKERS) as pool:
        got = dict(pool.imap_unordered(_joint, jobs))
    grid = {f"{g}|{T}": dict(beta=g, T=T, **got[(g, T)]) for g, T in jobs}
    best = min(grid.values(), key=lambda r: r["cer"])
    pinned = min((r for r in grid.values() if r["beta"] == P.GAMMA_PINNED), key=lambda r: r["cer"])
    src = model_path("lif", 0).parent / "committed.pkl"
    return dict(experiment="exp207_diag_joint_beta", row="lif_committed", source=str(src.relative_to(ROOT)),
                corpus=dict(seed=P.CALIB["seed"], per_fist=P.CALIB["per_fist"]), betas=list(GAMMAS), grid=P.CALIB["grid"],
                results=grid, best_joint=best, best_pinned=pinned,
                gain_cer_pp=pinned["cer"] - best["cer"], gain_wer_pp=pinned["wer"] - best["wer"], finished=now())


def _dump(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2))


def main():
    t0 = time.time()
    tables.build()
    prods, sources = all_producers(fixed_point=False)
    rows = [r for r in prods if r != "perfect_gaps"]
    C = calibrate(rows, sources)
    _dump(C, "calibration.json")
    _dump({n: r["curve"] for n, r in C["rows"].items()}, "calibration_curves.json")
    print(f"  calibration ({time.time() - t0:.0f}s)", flush=True)
    R = evaluate(C, sources)
    _dump(R, "results.json")
    _dump(oracle_conf8(), "oracle_conf8.json")
    _dump(diag_joint_beta(), "diag_joint_beta.json")
    metas = {f"{fam}_seed{s}.meta.json": json.loads(model_path(fam, s).with_suffix(".meta.json").read_text())
             for fam in ("lif", "mlp") for s in range(10)}
    M = dict(experiment="exp207_temperature", protocol_hash=P.RANDOM_LETTERS_HASH, created=now(), git=git_info(),
             env=env_info(), files={f: sha256_file(OUT / f) for f in ("calibration.json", "calibration_curves.json",
                                                                       "results.json", "oracle_conf8.json",
                                                                       "diag_joint_beta.json")},
             feature_matrix_sha256=sorted({m["feature_sha256"] for m in metas.values()}),
             artifacts={n: R["rows"][n]["source_sha256"] for n in R["rows"] if R["rows"][n]["source_sha256"]},
             model_sidecars=metas, determinism=R["determinism"])
    _dump(M, "manifest.json")
    print(f"records -> {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
