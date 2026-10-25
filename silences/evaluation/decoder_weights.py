"""The evaluation on English words and random letters, decoder weights fitted per model and per text
(protocol/decoder_weights.json; Section 8 and Tables 2 and 3).

    python -m silences.evaluation.decoder_weights [fit|eval|stats|edge|all]

The decoder weights are the posterior scale T and the insertion bonus gamma. For each model and each
text they are chosen on that text's calibration corpus (random letters 4242, English 4243) as the
grid pair minimising decoded CER (ties to the smaller T, then the smaller gamma). Every model is then
evaluated on the text's four evaluation corpora at its own weights and at the default weights
(T = 1, gamma = 2.5), and the families are compared by Welch's interval on their seed means.

Records written to records/decoder_weights/:
    fit.json             the fitted (T, gamma) per model and text, with the whole grid
    eval.json            every model on every evaluation corpus, at its own and the default weights
    results.json         seed means, family statistics, the Welch comparisons, the anchors
    diag_beta_edge.json  the English refit with gamma extended to 8 (the fitted bonus sat on the grid's
                         edge for the feedforward models; reported, never the record)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json          # noqa: E402
import sys           # noqa: E402
import time          # noqa: E402
from multiprocessing import get_context   # noqa: E402

import numpy as np   # noqa: E402

from .. import RECORDS   # noqa: E402
from ..corpus import english_vocabulary   # noqa: E402
from ..fixedpoint.q16 import grid_q   # noqa: E402
from . import protocol as P, tables   # noqa: E402
from .metrics import edit_sums, welch   # noqa: E402
from .producers import FAMILY_ROWS   # noqa: E402
from .provenance import env_info, git_info, now, sha256_file   # noqa: E402

OUT = RECORDS / "decoder_weights"
PAIRS = [("lif", "bigru"), ("mlp", "bigru"), ("mlp", "lif")]
OPS = P.OPERATORS
WORKERS = tables.WORKERS
EDGE_GAMMAS = P.GAMMA_GRID + [5.0, 6.0, 8.0]


def _pack(x):
    CE, NC, WE, NW, EX, N = x
    return dict(cer=100 * CE / NC, wer=100 * WE / NW, exact=100 * EX / N, n=N)


# ---- fit ----
_TAB = None


def _fit_row(args):
    row, gammas = args
    grid = {}
    for gamma in gammas:
        for T in P.T_GRID:
            grid[f"{T}|{gamma}"] = dict(T=T, beta=gamma, **_pack(edit_sums(_TAB, row, grid_q(T), grid_q(gamma))[0]))
    best = min(grid.values(), key=lambda r: (round(r["cer"], 10), r["T"], r["beta"]))
    return row, dict(T=best["T"], beta=best["beta"], cer=best["cer"], wer=best["wer"],
                     T_edge=best["T"] in (P.T_GRID[0], P.T_GRID[-1]),
                     beta_edge=best["beta"] in (gammas[0], gammas[-1]), grid=grid)


def fit_text(text, gammas=P.GAMMA_GRID):
    global _TAB
    _TAB = tables.load(text, P.CALIB_SEED[text])
    rows = [r for r in _TAB["rows"] if r != "perfect_gaps"]
    with get_context("fork").Pool(WORKERS) as pool:
        got = dict(pool.imap_unordered(_fit_row, [(r, gammas) for r in rows]))
    return {r: got[r] for r in rows}


# ---- evaluate ----
def _eval_row(args):
    row, settings = args
    out = {}
    for cond, (T, gamma) in settings.items():
        tot, pf = edit_sums(_TAB, row, grid_q(T), grid_q(gamma), per_fist=True)
        out[cond] = dict(overall=_pack(tot), per_fist={f: _pack(v) for f, v in pf.items()})
    return row, out


def eval_text(text, fits, conds=("deployed", "own_fit")):
    global _TAB
    dep = (P.DEFAULT_WEIGHTS["T"], P.DEFAULT_WEIGHTS["beta"])
    res = {}
    for seed in P.EVAL["seeds"]:
        _TAB = tables.load(text, seed)
        jobs = []
        for row in _TAB["rows"]:
            s = {"deployed": dep} if "deployed" in conds else {}
            if row in fits and "own_fit" in conds:
                s["own_fit"] = (fits[row]["T"], fits[row]["beta"])
            if s:
                jobs.append((row, s))
        with get_context("fork").Pool(WORKERS) as pool:
            got = dict(pool.imap_unordered(_eval_row, jobs))
        res[str(seed)] = {r: got[r] for r, _ in jobs}
    return res


# ---- statistics ----
def row_means(res_text, cond):
    rows = {}
    for seed, R in res_text.items():
        for row, conds in R.items():
            if cond in conds:
                rows.setdefault(row, []).append(conds[cond])
    return {row: dict(cer=float(np.mean([x["overall"]["cer"] for x in L])),
                      wer=float(np.mean([x["overall"]["wer"] for x in L])),
                      exact=float(np.mean([x["overall"]["exact"] for x in L])),
                      per_fist={f: {m: float(np.mean([x["per_fist"][f][m] for x in L])) for m in ("cer", "wer")}
                                for f in OPS})
            for row, L in rows.items()}


def outcome(cmp):
    c, w = cmp["cer"], cmp["wer"]
    if c["win"] and w["win"]:
        return "HOLDS"
    if c["loss"] or w["loss"]:
        return "REVERSES"
    return "UNRESOLVED"


def comparisons(M, per_fist=True):
    cmps = {}
    for cand, base in PAIRS:
        c = {m: welch([M[r][m] for r in FAMILY_ROWS[cand]], [M[r][m] for r in FAMILY_ROWS[base]]) for m in ("cer", "wer")}
        c["outcome"] = outcome(c)
        if per_fist:      # reported alongside, never the claim
            c["per_fist"] = {}
            for f in OPS:
                pc = {m: welch([M[r]["per_fist"][f][m] for r in FAMILY_ROWS[cand]],
                               [M[r]["per_fist"][f][m] for r in FAMILY_ROWS[base]]) for m in ("cer", "wer")}
                pc["outcome"] = outcome(pc)
                c["per_fist"][f] = pc
        cmps[f"{cand}_vs_{base}"] = c
    return cmps


def stats(fits, ev):
    members = sum(FAMILY_ROWS.values(), [])
    out = dict(experiment="exp209_corpus_decoder_weights", protocol_hash=P.DECODER_WEIGHTS_HASH,
               protocol=P.DECODER_WEIGHTS, texts={})
    for text in P.TEXTS:
        T = out["texts"][text] = {}
        for cond in ("own_fit", "deployed"):
            M = row_means(ev[text], cond)
            fam = {f: dict(rows=rs, n=len(rs), cer=float(np.mean([M[r]["cer"] for r in rs])),
                           wer=float(np.mean([M[r]["wer"] for r in rs])),
                           cer_sd=float(np.std([M[r]["cer"] for r in rs], ddof=1)),
                           wer_sd=float(np.std([M[r]["wer"] for r in rs], ddof=1)),
                           per_fist={f2: {m: float(np.mean([M[r]["per_fist"][f2][m] for r in rs])) for m in ("cer", "wer")}
                                     for f2 in OPS})
                   for f, rs in FAMILY_ROWS.items()}
            T[cond] = dict(rows=M, families=fam, comparisons=comparisons(M),
                           singles={r: {k: M[r][k] for k in ("cer", "wer", "exact")} for r in M if r not in members})
        F = fits[text]
        T["fits"] = {f: [dict(row=r, T=F[r]["T"], beta=F[r]["beta"]) for r in rs] for f, rs in FAMILY_ROWS.items()}
        T["fits_singles"] = {r: dict(T=F[r]["T"], beta=F[r]["beta"]) for r in F if r not in members}
    # anchor: the default weights on random letters must reproduce the random-letter record's T = 1 block
    t1 = json.loads((RECORDS / "random_letters" / "results.json").read_text())["t1"]["rows"]
    dep = out["texts"]["rand"]["deployed"]["rows"]
    diffs = {r: (dep[r]["cer"] - t1[r]["mean"]["cer"], dep[r]["wer"] - t1[r]["mean"]["wer"]) for r in t1 if r in dep}
    worst = max(max(abs(a), abs(b)) for a, b in diffs.values())
    out["anchor_rand_deployed_vs_exp207_t1"] = dict(rows=len(diffs), max_abs_pp=worst, pass_2dp=worst < 0.005)
    lb = out["texts"]["rand"]["own_fit"]["comparisons"]["lif_vs_bigru"]
    out["valid_rand_direction"] = bool(lb["cer"]["win"] and lb["wer"]["win"])
    out["escalate"] = out["texts"]["eng"]["own_fit"]["comparisons"]["lif_vs_bigru"]["outcome"] == "REVERSES"
    out["eng_vocab_size"] = {str(L): len(v) for L, v in english_vocabulary().items()}
    out["git"], out["env"], out["finished"] = git_info(), env_info(), now()
    return out


def edge():
    """English refit with gamma extended to 8, re-evaluated and re-compared."""
    fits = {r: {k: v for k, v in f.items() if k in ("T", "beta", "cer", "wer")}
            for r, f in fit_text("eng", EDGE_GAMMAS).items()}
    M = row_means(eval_text("eng", fits, conds=("own_fit",)), "own_fit")
    out = dict(betas=EDGE_GAMMAS, fits=fits, families={}, comparisons=comparisons(M, per_fist=False))
    for f, rs in FAMILY_ROWS.items():
        out["families"][f] = dict(cer=sum(M[r]["cer"] for r in rs) / len(rs), wer=sum(M[r]["wer"] for r in rs) / len(rs),
                                  fits=[(fits[r]["T"], fits[r]["beta"]) for r in rs])
    members = sum(FAMILY_ROWS.values(), [])
    out["singles"] = {r: dict(cer=M[r]["cer"], wer=M[r]["wer"], T=fits[r]["T"], beta=fits[r]["beta"])
                      for r in M if r not in members}
    return out


def _dump(obj, name, indent=1):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=indent))


def _load(name):
    return json.loads((OUT / name).read_text())


def main(step="all"):
    tables.build()
    t0 = time.time()
    if step in ("fit", "all"):
        _dump(dict(protocol_hash=P.DECODER_WEIGHTS_HASH, protocol=P.DECODER_WEIGHTS,
                   fits={t: fit_text(t) for t in P.TEXTS}), "fit.json")
        print(f"  fit ({time.time() - t0:.0f}s)", flush=True)
    if step in ("eval", "all"):
        F = _load("fit.json")["fits"]
        _dump(dict(protocol_hash=P.DECODER_WEIGHTS_HASH, eval={t: eval_text(t, F[t]) for t in P.TEXTS}), "eval.json", None)
        print(f"  eval ({time.time() - t0:.0f}s)", flush=True)
    if step in ("stats", "all"):
        _dump(stats(_load("fit.json")["fits"], _load("eval.json")["eval"]), "results.json")
    if step in ("edge", "all"):
        _dump(edge(), "diag_beta_edge.json")
        print(f"  edge ({time.time() - t0:.0f}s)", flush=True)
    if step == "all":
        _dump(dict(experiment="exp209_corpus_decoder_weights", protocol_hash=P.DECODER_WEIGHTS_HASH,
                   base_protocol_hash=P.RANDOM_LETTERS_HASH, created=now(), git=git_info(), env=env_info(),
                   files={f: sha256_file(OUT / f) for f in ("fit.json", "eval.json", "results.json", "diag_beta_edge.json")}),
              "manifest.json")
    print(f"records -> {OUT}")


if __name__ == "__main__":
    main((sys.argv[1:] or ["all"])[0])
