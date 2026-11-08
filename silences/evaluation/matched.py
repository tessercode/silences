"""The record of the paper since v1.1.0: every segmenter at matched training data (protocol/matched.json;
Sections 5.4 and 8, Tables 2 to 5).

    python -m silences.evaluation.matched [tables|fit|eval|crit1a|stats|manifest|all]

The families of record are the feedforward networks trained on 12,000 operator transmissions per seed
(lif_op12, mlp_op12) and the BiGRU recipe with its final phase on the same kind of corpus (bigru_fist);
the deployed single networks are seed 0 of each feedforward family. The grid (every architecture under
each of three training corpora) is scored alongside, on the record evaluation corpora and on four fresh
ones. The scoring is the decoder-weight protocol's, unchanged: (T, gamma) fitted per row and text on the
calibration corpora, evaluation at the fitted and at the default weights, Welch intervals on seed means.

Records written to records/matched/:
    fit.json            (T, gamma) per row and text, with the whole grid, for every row
    eval.json           every row on every record and fresh evaluation corpus, at its own and the default
                        weights (the Q16.16 rows also at their float twin's weights)
    fidelity.json       the deployed networks' Q16.16 decisions against their float twins, every gap
    criterion_1a.json   the random-letter reference means at the fitted posterior scale, for the proposal
    results.json        the record: seed means, comparisons and singles for the families of record, on the
                        record and fresh seeds; the grid block; the deployed networks; anchors and ruling
The original research records of the two grid experiments ship verbatim under records/matched/research/.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json          # noqa: E402
import sys           # noqa: E402
import time          # noqa: E402
from multiprocessing import get_context   # noqa: E402

import numpy as np   # noqa: E402

from .. import RECORDS, ROOT   # noqa: E402
from ..fixedpoint.q16 import grid_q   # noqa: E402
from . import protocol as P, tables   # noqa: E402
from .decoder_weights import _fit_row, _eval_row, _pack, outcome, row_means   # noqa: E402
from .metrics import edit_sums, welch   # noqa: E402
from .producers import DEPLOYED, FAMILY_ROWS, grid_family_rows   # noqa: E402
from .provenance import env_info, git_info, now, sha256_file   # noqa: E402

OUT = RECORDS / "matched"
OPS = P.OPERATORS
WORKERS = tables.WORKERS
PAIRS = [("lif", "bigru"), ("mlp", "bigru"), ("mlp", "lif")]
# the grid's comparisons (protocol/grid_window.json's experiment), candidate first
GRID_PAIRS = [("mlp_op12", "mlp"), ("lif_op12", "lif"),
              ("lif_op12", "bigru_fist"), ("mlp_op12", "bigru_fist"),
              ("lif_gen", "bigru"), ("mlp_gen", "bigru"),
              ("lif", "bigru_shared"), ("mlp", "bigru_shared"),
              ("mlp_op12", "mlp_gen"), ("lif_op12", "lif_gen"), ("bigru_fist", "bigru"),
              ("mlp_op12", "lif_op12"), ("mlp_gen", "lif_gen"), ("mlp", "lif"),
              ("lif", "bigru"), ("mlp", "bigru"), ("lif", "bigru_fist"), ("mlp", "bigru_fist")]
FX_ROWS = {"lif_op12_seed0_fx": "lif", "mlp_op12_seed0_fx": "mlp"}


def all_families():
    """{family: rows}: the original three, then the grid's (the records' family names)."""
    return {**FAMILY_ROWS, **grid_family_rows()}


def record_families():
    F = all_families()
    return {f: F[src] for f, src in P.MATCHED_FAMILIES.items()}


def deployed_meta():
    """The deployed networks: seed 0 of each feedforward family (anchor C1: the file the firmware was generated
    from is the record's seed-0 model)."""
    out = {}
    for fam in ("lif", "mlp"):
        p = DEPLOYED[fam]
        meta = json.loads(p.with_suffix(".meta.json").read_text())
        sha = sha256_file(p)
        out[fam] = dict(source=str(p.relative_to(ROOT)), installed=str(p.relative_to(ROOT)), pickle_sha256=meta["pickle_sha256"],
                        installed_sha256=sha, c1_pass=sha == meta["pickle_sha256"], corpus_seed=meta["corpus_seed"],
                        train_seed=meta["seed"])
    return out


_TAB = None


def _set_tab(text, seed):
    global _TAB
    _TAB = tables.load(text, seed, matched=True)
    import silences.evaluation.decoder_weights as DW
    DW._TAB = _TAB                      # _fit_row and _eval_row read the module-level table
    return _TAB


# ---- fit: every row, both texts, on the calibration corpora ----
def fit_text(text):
    T = _set_tab(text, P.CALIB_SEED[text])
    rows = [r for r in T["rows"] if r != "perfect_gaps"]
    with get_context("fork").Pool(WORKERS) as pool:
        got = dict(pool.imap_unordered(_fit_row, [(r, P.GAMMA_GRID) for r in rows]))
    return {r: got[r] for r in rows}


# ---- evaluate: record and fresh seeds ----
def eval_text(text, fits):
    dep = (P.DEFAULT_WEIGHTS["T"], P.DEFAULT_WEIGHTS["beta"])
    res = dict(record={}, fresh={})
    for kind, seeds in (("record", P.EVAL["seeds"]), ("fresh", P.FRESH_SEEDS)):
        for seed in seeds:
            T = _set_tab(text, seed)
            jobs = []
            for row in T["rows"]:
                s = {"deployed": dep}
                if row in fits:
                    s["own_fit"] = (fits[row]["T"], fits[row]["beta"])
                if row in FX_ROWS:          # the same-weights fidelity measure: at the float twin's fit
                    tw = fits[row[:-3]]
                    s["float_fit"] = (tw["T"], tw["beta"])
                jobs.append((row, s))
            with get_context("fork").Pool(WORKERS) as pool:
                got = dict(pool.imap_unordered(_eval_row, jobs))
            res[kind][str(seed)] = {r: got[r] for r, _ in jobs}
    return res


# ---- fidelity: the deployed networks' Q16.16 decisions against their float twins ----
def fidelity():
    out = {}
    for fx_row, fam in FX_ROWS.items():
        float_row = fx_row[:-3]
        agree = total = 0
        per_corpus = {}
        for text, kind, seed in tables.corpora(matched=True):
            D = tables.load(text, seed, matched=True)
            a = t = 0
            for lf, lq in zip(D["rows"][float_row], D["rows"][fx_row]):
                af = np.argmax(np.asarray(lf, dtype=np.float64), axis=1)
                aq = np.argmax(np.asarray(lq, dtype=np.int64), axis=1)
                assert len(af) == len(aq)
                a += int((af == aq).sum()); t += len(af)
            per_corpus[f"{text}_{seed}"] = dict(agree=a, gaps=t)
            agree += a; total += t
        out[fam] = dict(float_row=float_row, fx_row=fx_row, agree=agree, gaps=total, agreement=agree / total,
                        per_corpus=per_corpus)
    return out


# ---- statistics ----
def _cmp(M, FAM, cand, base):
    c = {}
    for m in ("cer", "wer"):
        x = [M[r][m] for r in FAM[cand]]; y = [M[r][m] for r in FAM[base]]
        w = welch(x, y); w["wins"] = int(sum(1 for xi in x if xi < np.mean(y))); c[m] = w
    c["outcome"] = outcome(c)
    c["per_fist"] = {}
    for f in OPS:
        pc = {m: welch([M[r]["per_fist"][f][m] for r in FAM[cand]], [M[r]["per_fist"][f][m] for r in FAM[base]])
              for m in ("cer", "wer")}
        pc["outcome"] = outcome(pc)
        c["per_fist"][f] = pc
    return c


def _family(M, rs, with_rows):
    d = dict(rows=rs) if with_rows else {}
    d.update(n=len(rs), cer=float(np.mean([M[r]["cer"] for r in rs])), wer=float(np.mean([M[r]["wer"] for r in rs])),
             cer_sd=float(np.std([M[r]["cer"] for r in rs], ddof=1)), wer_sd=float(np.std([M[r]["wer"] for r in rs], ddof=1)),
             per_fist={f: {m: float(np.mean([M[r]["per_fist"][f][m] for r in rs])) for m in ("cer", "wer")} for f in OPS})
    return d


def record_block(res, FAM):
    """The families of record: rows, families, the three comparisons, the named singles."""
    out = {}
    for cond in ("own_fit", "deployed"):
        M = row_means(res, cond)
        out[cond] = dict(rows=M, families={f: _family(M, rs, True) for f, rs in FAM.items()},
                         comparisons={f"{a}_vs_{b}": _cmp(M, FAM, a, b) for a, b in PAIRS},
                         singles={name: {k: M[row][k] for k in ("cer", "wer", "exact")}
                                  for name, row in P.MATCHED_SINGLES.items() if row in M})
    return out


def grid_block(res, FAM):
    """The grid: every family, the grid's comparisons, rows as overall numbers."""
    out = {}
    for cond in ("own_fit", "deployed"):
        M = row_means(res, cond)
        out[cond] = dict(rows={r: {k: M[r][k] for k in ("cer", "wer", "exact")} for r in M if r not in FX_ROWS},
                         families={f: _family(M, rs, False) for f, rs in FAM.items()},
                         comparisons={f"{a}_vs_{b}": _cmp(M, FAM, a, b) for a, b in GRID_PAIRS if a in FAM and b in FAM})
    return out


def _label_r1(c):
    w, l = (c["cer"]["win"] or c["wer"]["win"]), (c["cer"]["loss"] or c["wer"]["loss"])
    return "HELPS" if (w and not l) else "NO EFFECT" if not (w or l) else "HURTS"


def _label_lead(c):
    if c["cer"]["loss"] or c["wer"]["loss"]:
        return "TRAILS"
    if c["cer"]["win"] and c["wer"]["win"]:
        return "LEADS"
    return "LEADS ON ONE" if (c["cer"]["win"] or c["wer"]["win"]) else "TIE"


def grid_labels(grid_texts):
    """The grid experiment's pre-registered outcome names, and its one-line pattern."""
    lab = {}
    for text in ("eng", "rand"):
        for kind in ("record", "fresh"):
            C = grid_texts[text][kind]["own_fit"]["comparisons"]
            L = {}
            for k in ("mlp_op12_vs_mlp", "lif_op12_vs_lif"):
                L[f"R1 {k}"] = _label_r1(C[k])
            for tag, ks in (("R2", ("lif_op12_vs_bigru_fist", "mlp_op12_vs_bigru_fist")), ("R3", ("lif_gen_vs_bigru", "mlp_gen_vs_bigru")),
                            ("5k", ("lif_vs_bigru_shared", "mlp_vs_bigru_shared"))):
                for k in ks:
                    L[f"{tag} {k}"] = _label_lead(C[k])
            for k in ("mlp_op12_vs_lif_op12", "mlp_gen_vs_lif_gen"):
                L[f"R6 {k}"] = C[k]["outcome"]
            lab[f"{text}_{kind}"] = L
    E = lab["eng_record"]
    r2 = [E["R2 lif_op12_vs_bigru_fist"], E["R2 mlp_op12_vs_bigru_fist"]]
    r3 = [E["R3 lif_gen_vs_bigru"], E["R3 mlp_gen_vs_bigru"]]
    pattern = ("BiGRU LEADS AT MATCHED DATA" if all(x == "TRAILS" for x in r2 + r3) else
               "BUDGET ARTIFACT" if all(x != "TRAILS" for x in r2) else "MIXED")
    return lab, pattern


def stats(fits, ev, fid):
    FAMALL = all_families()
    FAM = record_families()
    out = dict(experiment="exp212_matched_record", protocol_hash=P.MATCHED_HASH, protocol=P.MATCHED, deployed=deployed_meta(),
               texts={}, fresh={})
    grid = {}
    for text in P.TEXTS:
        T = out["texts"][text] = record_block(ev[text]["record"], FAM)
        out["fresh"][text] = record_block(ev[text]["fresh"], FAM)
        F = fits[text]
        T["fits"] = {f: [dict(row=r, T=F[r]["T"], beta=F[r]["beta"]) for r in rs] for f, rs in FAM.items()}
        T["fits_singles"] = {name: dict(T=F[row]["T"], beta=F[row]["beta"]) for name, row in P.MATCHED_SINGLES.items() if row in F}
        grid[text] = dict(record=grid_block(ev[text]["record"], FAMALL), fresh=grid_block(ev[text]["fresh"], FAMALL),
                          fits={f: [[F[r]["T"], F[r]["beta"]] for r in rs] for f, rs in FAMALL.items()})
    labels, pattern = grid_labels(grid)
    out["grid"] = dict(texts=grid, labels=labels, pattern=pattern, exp211_protocol_hash=P.GRID_WINDOW_HASH)
    # anchors: C1 the deployed files; C4 the record families equal the grid's; H1 the fixed-point fidelity
    c1 = all(m["c1_pass"] for m in out["deployed"].values())
    c4 = 0.0
    for text in P.TEXTS:
        for f, src in P.MATCHED_FAMILIES.items():
            a = out["texts"][text]["own_fit"]["families"][f]
            b = grid[text]["record"]["own_fit"]["families"][src]
            c4 = max(c4, abs(a["cer"] - b["cer"]), abs(a["wer"] - b["wer"]))
    fx_ok = all(v["agreement"] >= P.MATCHED["fidelity"]["min_agreement"] for v in fid.values())
    same = {}
    S = P.MATCHED_SINGLES
    for t in P.TEXTS:
        Mf = row_means(ev[t]["record"], "float_fit")
        for fam in ("lif", "mlp"):
            fl = out["texts"][t]["own_fit"]["rows"][S[f"{fam}_committed"]]
            fq = Mf[S[f"{fam}_committed_fx"]]
            dp = out["texts"][t]["deployed"]["singles"]
            same[f"{t}_{fam}"] = dict(own_fit_float_vs_fx_at_float_fit={m: fq[m] - fl[m] for m in ("cer", "wer")},
                                      deployed={m: dp[f"{fam}_committed_fx"][m] - dp[f"{fam}_committed"][m] for m in ("cer", "wer")})
    fx_pp = max(abs(v) for s_ in same.values() for d in s_.values() for v in d.values())
    fx_pp_ownfit = max(abs(out["texts"][t]["own_fit"]["singles"][f"{fam}_committed_fx"][m] - out["texts"][t]["own_fit"]["singles"][f"{fam}_committed"][m])
                       for t in P.TEXTS for fam in ("lif", "mlp") for m in ("cer", "wer"))
    out["anchors"] = dict(c1_model_identity=c1, c4_family_means_vs_exp211_max_abs_pp=c4, c4_pass=c4 < 1e-9,
                          h1_fx_agreement={f: v["agreement"] for f, v in fid.items()}, h1_fx_agreement_pass=fx_ok,
                          h1_fx_same_weights=same, h1_fx_max_abs_pp=fx_pp, h1_fx_pp_pass=fx_pp <= P.MATCHED["fidelity"]["max_abs_pp"],
                          h1_fx_max_abs_pp_each_own_fit=fx_pp_ownfit)
    out["anchors_pass_preregistered"] = bool(c1 and out["anchors"]["c4_pass"] and fx_ok and out["anchors"]["h1_fx_pp_pass"])
    out["ruling"] = P.MATCHED_RULING
    out["anchors_pass"] = bool(all(out["anchors"][k] for k in P.MATCHED_RULING["gating"]))
    out["escalate"] = dict(
        beats_mlp_control_eng=bool(out["texts"]["eng"]["own_fit"]["comparisons"]["mlp_vs_lif"]["outcome"] == "REVERSES"),
        lif_beats_bigru_eng=bool(out["texts"]["eng"]["own_fit"]["comparisons"]["lif_vs_bigru"]["outcome"] == "HOLDS"))
    out["git"], out["env"], out["finished"] = git_info(), env_info(), now()
    return out


# ---- criterion 1A: the random-letter protocol's procedure on the record families ----
def _crit1a_family(rows, kind_tables, calib_doc):
    """T per row from the random-letter protocol's grid, minimising CER on its calibration corpus with gamma pinned
    (ties to the smaller T); the record evaluation seeds at that T."""
    out = {}
    g = grid_q(P.GAMMA_PINNED)
    for row in rows:
        best = None
        for T in P.CALIB["grid"]:
            r = _pack(edit_sums(calib_doc, row, grid_q(T), g)[0])
            if best is None or r["cer"] < best[1]["cer"]:
                best = (T, r)
        T = best[0]
        per = {str(seed): _pack(edit_sums(doc, row, grid_q(T), g)[0]) for seed, doc in kind_tables.items()}
        out[row] = dict(T=T, at_grid_edge=T == P.CALIB["grid"][-1], calib=best[1],
                        cer=float(np.mean([v["cer"] for v in per.values()])), wer=float(np.mean([v["wer"] for v in per.values()])),
                        per_eval_seed=per)
    return out


def _crit1a_job(args):
    fam, rows = args
    calib = tables.load("rand", P.CALIB_SEED["rand"], matched=True)
    ev = {s: tables.load("rand", s, matched=True) for s in P.EVAL["seeds"]}
    return fam, _crit1a_family(rows, ev, calib)


def criterion_1a():
    F = all_families()
    jobs = [("lif", F["lif_op12"]), ("mlp", F["mlp_op12"]), ("bigru", F["bigru_fist"]),
            ("lif_old", F["lif"]), ("mlp_old", F["mlp"]), ("bigru_old", F["bigru"])]
    with get_context("fork").Pool(len(jobs)) as pool:
        res = dict(pool.imap_unordered(_crit1a_job, jobs))
    out = dict(protocol_hash=P.MATCHED_HASH,
               procedure=f"Exp 207: T per row from P.CALIB grid minimising CER on rand {P.CALIB_SEED['rand']} (per_fist {P.CALIB['per_fist']}), "
                         f"gamma pinned {P.GAMMA_PINNED}; record seeds {P.EVAL['seeds']} at that T",
               rows=res, families={}, anchor={})
    for fam, R in res.items():
        c = [v["cer"] for v in R.values()]; w = [v["wer"] for v in R.values()]
        out["families"][fam] = dict(n=len(c), cer=float(np.mean(c)), cer_sd=float(np.std(c, ddof=1)), wer=float(np.mean(w)),
                                    wer_sd=float(np.std(w, ddof=1)), T=[v["T"] for v in R.values()])
    rec = json.loads((RECORDS / "random_letters" / "results.json").read_text())
    worst = max(abs(out["families"][f"{fam}_old"][m] - rec["families"][fam][m]["mean"])
                for fam in ("lif", "mlp", "bigru") for m in ("cer", "wer"))
    out["anchor"] = dict(old_families_vs_exp207_record_max_abs_pp=worst, pass_2dp=worst < 0.005)
    for a, b in PAIRS:
        out[f"{a}_vs_{b}"] = {m: welch([v[m] for v in res[a].values()], [v[m] for v in res[b].values()]) for m in ("cer", "wer")}
    out["finished"] = now()
    return out


def _dump(obj, name, indent=1):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=indent))


def _load(name):
    return json.loads((OUT / name).read_text())


def main(step="all"):
    t0 = time.time()
    if step in ("tables", "all"):
        tables.build(matched=True)
        print(f"  tables ({time.time() - t0:.0f}s)", flush=True)
    if step in ("fit", "all"):
        _dump(dict(protocol_hash=P.MATCHED_HASH, fits={t: fit_text(t) for t in P.TEXTS}), "fit.json")
        print(f"  fit ({time.time() - t0:.0f}s)", flush=True)
    if step in ("eval", "all"):
        F = _load("fit.json")["fits"]
        _dump(dict(protocol_hash=P.MATCHED_HASH, eval={t: eval_text(t, F[t]) for t in P.TEXTS}), "eval.json", None)
        _dump(dict(protocol_hash=P.MATCHED_HASH, fidelity=fidelity()), "fidelity.json")
        print(f"  eval ({time.time() - t0:.0f}s)", flush=True)
    if step in ("crit1a", "all"):
        _dump(criterion_1a(), "criterion_1a.json")
        print(f"  criterion 1A ({time.time() - t0:.0f}s)", flush=True)
    if step in ("stats", "all"):
        R = stats(_load("fit.json")["fits"], _load("eval.json")["eval"], _load("fidelity.json")["fidelity"])
        _dump(R, "results.json")
        assert R["anchors_pass"], "the matched record's anchors failed"
        print(f"  anchors: C1 {R['anchors']['c1_model_identity']}, C4 {R['anchors']['c4_pass']}, "
              f"fidelity {R['anchors']['h1_fx_agreement']} (pp bound {'met' if R['anchors']['h1_fx_pp_pass'] else 'missed, reported'})")
    if step in ("manifest", "all"):
        _dump(dict(experiment="exp212_matched_record", protocol_hash=P.MATCHED_HASH, base_protocol_hash=P.GRID_WINDOW_HASH,
                   exp210_protocol_hash=P.GRID_BIGRU_HASH, exp209_protocol_hash=P.DECODER_WEIGHTS_HASH, created=now(),
                   git=git_info(), env=env_info(),
                   files={f: sha256_file(OUT / f) for f in ("fit.json", "eval.json", "fidelity.json", "criterion_1a.json", "results.json")}),
              "manifest.json")
    print(f"records -> {OUT}")


if __name__ == "__main__":
    main((sys.argv[1:] or ["all"])[0])
