"""The spike-fed network (v1.2.0): the LIF segmenter fed the window as spike trains, scored against the
matched-data record (protocol/spike_fed.json; Sections 8.1, 8.5 and 8.6, Table 6).

    python -m silences.evaluation.spike_fed [tables|check|check-train|fit|eval|crit1a|fidelity|stats|manifest|all]

Three families of ten training seeds each, trained on the matched record's op12 corpora with the LIF recipe
unchanged except for the model and the presentation length: spk40_op12 (fed spikes, T = 40; the family of
record, its seed 0 the deployed network), spk80_op12 (fed spikes, T = 80) and lif80_op12 (fed currents at
T = 80, the equal-T control). They are scored beside the matched record's lif_op12, mlp_op12 and bigru_fist
with the same decoder-weight protocol, on the record and the fresh evaluation corpora, and the comparisons
the protocol names are made with Welch intervals on seed means. Rows of the matched record are read from its
tables, never recomputed.

Records written to records/spike_fed/:
    check.json          C3: the harness reproduces the matched record's lif_op12_seed0 row; C4: no training
                        transmission is in any evaluation corpus
    check_train.json    C1: the op12 feature matrices hash to the record's; C2 (reported): the inference forward
                        against the training-side snnTorch forward (needs torch)
    fit.json            (T, gamma) per new row and text, with the grid
    eval.json           every new row on every record and fresh corpus, at its own and the default weights
    criterion_1a.json   the random-letter reference means at the fitted posterior scale, spike-fed and
                        current-fed families, anchored on the matched record's
    fidelity.json       the deployed spike-fed network's Q16.16 decisions against its float twin, every gap
                        of the eighteen corpora, and CER/WER of both at the same decoder weights
    results.json        the record: seed means, the protocol's comparisons, the pre-registered labels, anchors
The ruling that adopted the network, the research report and the spike-character diagnostic ship under
records/spike_fed/research/.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib       # noqa: E402
import json          # noqa: E402
import sys           # noqa: E402
import time          # noqa: E402
from multiprocessing import get_context   # noqa: E402

import numpy as np   # noqa: E402

from .. import RECORDS, ROOT   # noqa: E402
from ..corpus import corpus   # noqa: E402
from ..fixedpoint.q16 import grid_q   # noqa: E402
from . import protocol as P, tables   # noqa: E402
from .decoder_weights import _fit_row, _eval_row, _pack, row_means   # noqa: E402
from .matched import _cmp, _crit1a_family, _label_lead, _label_r1   # noqa: E402
from .metrics import edit_sums, welch   # noqa: E402
from .producers import DEPLOYED, SEEDS, SPIKE_FED_CORPUS, SPIKE_FED_FAMILIES, grid_family_rows, grid_model_path, spike_fed_family_rows   # noqa: E402
from .provenance import env_info, git_info, now, sha256_file   # noqa: E402

OUT = RECORDS / "spike_fed"
MATCHED = RECORDS / "matched"
OPS = P.OPERATORS
WORKERS = tables.WORKERS
NEW = [f"{a}_{SPIKE_FED_CORPUS}" for a in SPIKE_FED_FAMILIES]            # spk40_op12, spk80_op12, lif80_op12
REF = ("lif_op12", "mlp_op12", "bigru_fist")
PAIRS = [("spk40_op12", "lif_op12"), ("spk80_op12", "lif_op12"),             # S1 what spike-encoded input costs
         ("spk80_op12", "spk40_op12"),                                       # S2 quantization
         ("lif80_op12", "lif_op12"), ("spk80_op12", "lif80_op12"),           # S3 equal T
         ("spk40_op12", "bigru_fist"), ("spk80_op12", "bigru_fist"), ("lif80_op12", "bigru_fist"),   # S4 the baseline
         ("spk40_op12", "mlp_op12"), ("spk80_op12", "mlp_op12"), ("lif80_op12", "mlp_op12")]
FREE_MARGIN = 0.5
DEPLOYED_ROW = "spk40_op12_seed0"
FX_ROW = DEPLOYED_ROW + "_fx"
FIDELITY_MIN_AGREEMENT = 0.999
FIDELITY_MAX_ABS_PP = 0.1


def new_rows():
    """{row: (family, model file)} for every spike-fed model on disk."""
    out = {}
    for fam in SPIKE_FED_FAMILIES:
        for s in SEEDS:
            p = grid_model_path(fam, SPIKE_FED_CORPUS, s)
            if p.exists() and p.with_suffix(".meta.json").exists():
                out[f"{fam}_{SPIKE_FED_CORPUS}_seed{s}"] = (fam, p)
    return out


def families():
    """{family: rows}: the matched record's families and the spike-fed ones present."""
    F = dict(grid_family_rows())
    for r in new_rows():
        F.setdefault(r.rsplit("_seed", 1)[0], []).append(r)
    return {f: rs for f, rs in F.items() if len(rs) >= 2}


def _meta(row):
    return json.loads(new_rows()[row][1].with_suffix(".meta.json").read_text())


_TAB = None


def _set_tab(text, seed):
    global _TAB
    _TAB = tables.load(text, seed, matched="spike_fed")
    import silences.evaluation.decoder_weights as DW
    DW._TAB = _TAB
    return _TAB


# ---- check: C3 the harness reproduces the matched record's row; C4 disjointness ----
def _fingerprint(ticks):
    return hashlib.sha1(np.asarray(ticks[:12], dtype=np.int64).tobytes()).hexdigest()[:16]


def _train_fingerprints(seed):
    n = P.GRID_N_TX // len(P.OPERATORS)
    return seed, {_fingerprint(tx["ticks"]) for tx in corpus(n, P.GRID_CORPUS_SEED0[SPIKE_FED_CORPUS] + seed, "rand")}


def check():
    from ..segmenters.lif import LIF
    prod = LIF.load(DEPLOYED["lif"]).lp3
    c3 = {}
    for text in P.TEXTS:
        seed = P.EVAL["seeds"][0]
        D = tables.load(text, seed, matched=True)
        mine = [prod(dict(ticks=t, ch=np.asarray(c, dtype=np.int64))) for t, c in zip(D["ticks"], D["ch"])]
        c3[f"{text}_{seed}"] = dict(transmissions=len(mine), equal=mine == D["rows"]["lif_op12_seed0"])
    with get_context("fork").Pool(min(WORKERS, len(SEEDS))) as pool:
        train = set().union(*[fp for _, fp in pool.imap_unordered(_train_fingerprints, SEEDS)])
    bad = {}
    for text, kind, seed in tables.corpora(matched=True):
        D = tables.load(text, seed, matched=True)
        n = len(train & {_fingerprint(t) for t in D["ticks"]})
        if n:
            bad[f"{text}_{kind}_{seed}"] = n
    out = dict(c3=c3, c3_pass=all(v["equal"] for v in c3.values()),
               c4=dict(train_fingerprints=len(train), overlaps=bad, c4_disjoint=not bad))
    assert out["c3_pass"] and not bad
    return out


# ---- check-train: C1 the feature matrices, C2 the training-side forward (needs torch) ----
def check_train():
    from ..training.window_nets import grid_matrix, check_forward
    out = dict(c1={}, c2={})
    rows = new_rows()
    for s in SEEDS:
        F, _, sha = grid_matrix(SPIKE_FED_CORPUS, s)
        want = {r: _meta(r)["feature_sha256"] for r in rows if r.endswith(f"_seed{s}")}
        out["c1"][str(s)] = dict(corpus_seed=P.GRID_CORPUS_SEED0[SPIKE_FED_CORPUS] + s, feature_sha256=sha,
                                 equals_models=all(v == sha for v in want.values()))
        for r, (fam, p) in rows.items():
            if r.endswith(f"_seed{s}"):
                from .producers import _LOAD
                out["c2"][r] = check_forward(_LOAD[fam](p), F, np.random.default_rng(213 + s))
    out["c1_pass"] = all(v["equals_models"] for v in out["c1"].values())
    out["c2_min_argmax_agreement"] = min([v["argmax_agreement"] for v in out["c2"].values()] or [None])
    return out


# ---- fit and evaluate the new rows ----
def fit_text(text):
    _set_tab(text, P.CALIB_SEED[text])
    rows = [r for r in new_rows() if r in _TAB["rows"]]
    with get_context("fork").Pool(min(WORKERS, len(rows))) as pool:
        got = dict(pool.imap_unordered(_fit_row, [(r, P.GAMMA_GRID) for r in rows]))
    return {r: got[r] for r in rows}


def eval_text(text, fits):
    dep = (P.DEFAULT_WEIGHTS["T"], P.DEFAULT_WEIGHTS["beta"])
    res = dict(record={}, fresh={})
    for kind, seeds in (("record", P.EVAL["seeds"]), ("fresh", P.FRESH_SEEDS)):
        for seed in seeds:
            _set_tab(text, seed)
            jobs = [(row, {"deployed": dep, "own_fit": (fits[row]["T"], fits[row]["beta"])})
                    for row in new_rows() if row in _TAB["rows"] and row in fits]
            with get_context("fork").Pool(min(WORKERS, len(jobs))) as pool:
                got = dict(pool.imap_unordered(_eval_row, jobs))
            res[kind][str(seed)] = {r: got[r] for r, _ in jobs}
    return res


# ---- criterion 1A: the random-letter procedure on the spike-fed and current-fed families ----
def _crit1a_job(args):
    fam, rows = args
    calib = tables.load("rand", P.CALIB_SEED["rand"], matched="spike_fed")
    ev = {s: tables.load("rand", s, matched="spike_fed") for s in P.EVAL["seeds"]}
    return fam, _crit1a_family(rows, ev, calib)


def criterion_1a():
    """The matched record's procedure on the spike-fed families; its current-fed LIF and MLP families are
    recomputed from the same tables and asserted equal to records/matched/criterion_1a.json (the anchor)."""
    c12 = json.loads((MATCHED / "criterion_1a.json").read_text())
    FAM = families()
    jobs = [("spk", FAM["spk40_op12"]), ("spk80", FAM["spk80_op12"]), ("lif", FAM["lif_op12"]), ("mlp", FAM["mlp_op12"])]
    with get_context("fork").Pool(len(jobs)) as pool:
        res = dict(pool.imap_unordered(_crit1a_job, jobs))
    out = dict(protocol_hash=P.SPIKE_FED_HASH, exp212_protocol_hash=c12["protocol_hash"], procedure=c12["procedure"],
               rows=res, families={}, anchor={})
    for fam, R in res.items():
        c = [v["cer"] for v in R.values()]; w = [v["wer"] for v in R.values()]
        out["families"][fam] = dict(n=len(c), cer=float(np.mean(c)), cer_sd=float(np.std(c, ddof=1)), wer=float(np.mean(w)),
                                    wer_sd=float(np.std(w, ddof=1)), T=[v["T"] for v in R.values()])
    worst = max(abs(out["families"][f][m] - c12["families"][f][m]) for f in ("lif", "mlp") for m in ("cer", "wer"))
    out["anchor"] = dict(lif_mlp_vs_exp212_criterion_1a_max_abs_pp=worst, pass_2dp=worst < 0.005,
                         exp212_rows_identical=all(abs(res[f][r][m] - c12["rows"][f][r][m]) < 1e-9
                                                   for f in ("lif", "mlp") for r in res[f] for m in ("cer", "wer")))
    for a, b in (("spk", "lif"), ("spk", "mlp"), ("spk80", "spk"), ("spk", "bigru")):
        if b == "bigru":
            rb = c12["rows"]["bigru"]; out["families"]["bigru"] = c12["families"]["bigru"]
            out[f"{a}_vs_{b}"] = {m: welch([v[m] for v in res[a].values()], [v[m] for v in rb.values()]) for m in ("cer", "wer")}
        else:
            out[f"{a}_vs_{b}"] = {m: welch([v[m] for v in res[a].values()], [v[m] for v in res[b].values()]) for m in ("cer", "wer")}
    out["git"] = git_info(); out["finished"] = now()
    return out


# ---- fidelity: the deployed spike-fed network in Q16.16 against its float twin ----
def _identity():
    dec = json.loads((OUT / "research" / "decision.json").read_text())
    t = dec["target"]
    p = DEPLOYED["spk"]
    meta = json.loads(p.with_suffix(".meta.json").read_text())
    installed = sha256_file(p)
    return dict(installed=str(p.relative_to(ROOT)), installed_sha256=installed, target=t, meta_sha256=meta["pickle_sha256"],
                decided=dec["decided"], decided_by=dec["decided_by"],
                g1_pass=(installed == t["sha256"] == meta["pickle_sha256"] and t["row"] == DEPLOYED_ROW))


def _fidelity_corpus(job):
    text, seed = job
    t0 = time.time()
    D = tables.load(text, seed, matched="spike_fed")
    a = n = 0
    for lf, lq in zip(D["rows"][DEPLOYED_ROW], D["rows"][FX_ROW]):
        af = np.argmax(np.asarray(lf, dtype=np.float64), axis=1); aq = np.argmax(np.asarray(lq, dtype=np.int64), axis=1)
        assert len(af) == len(aq)
        a += int((af == aq).sum()); n += len(af)
    fits = json.loads((OUT / "fit.json").read_text())
    assert fits["protocol_hash"] == P.SPIKE_FED_HASH
    own = (fits["fits"][text][DEPLOYED_ROW]["T"], fits["fits"][text][DEPLOYED_ROW]["beta"])
    dep = (P.DEFAULT_WEIGHTS["T"], P.DEFAULT_WEIGHTS["beta"])
    sums = {}
    d2 = dict(ch=D["ch"], truth=D["truth"], fist=D["fist"], rows={"float": D["rows"][DEPLOYED_ROW], "fx": D["rows"][FX_ROW]})
    for cond, (T, b) in (("deployed", dep), ("float_fit", own)):
        for r in ("float", "fx"):
            sums[f"{cond}/{r}"] = edit_sums(d2, r, grid_q(T), grid_q(b))[0]
    return f"{text}_{seed}", dict(agree=a, gaps=n, own_fit=own, sums=sums, seconds=round(time.time() - t0))


def fidelity():
    g1 = _identity()
    assert g1["g1_pass"], "the deployed spike-fed model is not the one the ruling adopted"
    jobs = [(text, seed) for text, kind, seed in tables.corpora(matched=True)]
    with get_context("fork").Pool(min(WORKERS, len(jobs))) as pool:
        per = dict(pool.imap_unordered(_fidelity_corpus, jobs))
    agree = sum(v["agree"] for v in per.values()); gaps = sum(v["gaps"] for v in per.values())
    texts = {}
    for text in P.TEXTS:
        names = [f"{text}_{s}" for s in P.EVAL["seeds"]]
        tt = {}
        for key in per[names[0]]["sums"]:
            tt[key] = _pack([sum(per[nm]["sums"][key][i] for nm in names) for i in range(6)])
        texts[text] = dict(rows=tt, diff={c: {m: tt[f"{c}/fx"][m] - tt[f"{c}/float"][m] for m in ("cer", "wer")}
                                           for c in ("deployed", "float_fit")})
    max_pp = max(abs(v) for t in texts.values() for d in t["diff"].values() for v in d.values())
    return dict(experiment="exp213_port_gate", step="fidelity", row=DEPLOYED_ROW, exp213_protocol_hash=P.SPIKE_FED_HASH, g1=g1,
                agree=agree, gaps=gaps, agreement=agree / gaps, g2_agreement_pass=agree / gaps >= FIDELITY_MIN_AGREEMENT,
                max_abs_pp_same_weights=max_pp, g2_pp_within=max_pp <= FIDELITY_MAX_ABS_PP, per_corpus=per, texts=texts,
                rule=dict(min_agreement=FIDELITY_MIN_AGREEMENT, max_abs_pp=FIDELITY_MAX_ABS_PP, pp_is_reported_not_gating=True),
                git=git_info(), env=env_info(), finished=now())


# ---- statistics ----
def _s1(c):
    """S1: COSTS / FREE / GAINS / UNRESOLVED for a candidate against the record's current-fed LIF."""
    if c["cer"]["loss"] or c["wer"]["loss"]:
        return "COSTS"
    if c["cer"]["win"] and c["wer"]["win"]:
        return "GAINS"
    if abs(c["cer"]["lo"]) < FREE_MARGIN and abs(c["cer"]["hi"]) < FREE_MARGIN and not (c["cer"]["win"] or c["wer"]["win"]):
        return "FREE"
    return "UNRESOLVED"


def stats(fits, ev, chk, tr):
    m_eval = json.loads((MATCHED / "eval.json").read_text()); m_fit = json.loads((MATCHED / "fit.json").read_text())
    m_res = json.loads((MATCHED / "results.json").read_text())
    assert m_eval["protocol_hash"] == m_fit["protocol_hash"] == m_res["protocol_hash"] == P.MATCHED_HASH
    FAM = families()
    pairs = [(a, b) for a, b in PAIRS if a in FAM and b in FAM]
    out = dict(experiment="exp213_spike_input", protocol_hash=P.SPIKE_FED_HASH, protocol=P.SPIKE_FED,
               families={f: FAM[f] for f in FAM if f in NEW + list(REF)}, texts={})
    c5 = 0.0
    for text in P.TEXTS:
        out["texts"][text] = {}
        for kind in ("record", "fresh"):
            merged = {s: {**m_eval["eval"][text][kind][s], **ev[text][kind].get(s, {})} for s in m_eval["eval"][text][kind]}
            B = {}
            for cond in ("own_fit", "deployed"):
                M = row_means(merged, cond)
                fam = {f: dict(n=len(rs), cer=float(np.mean([M[r]["cer"] for r in rs])), wer=float(np.mean([M[r]["wer"] for r in rs])),
                               cer_sd=float(np.std([M[r]["cer"] for r in rs], ddof=1)), wer_sd=float(np.std([M[r]["wer"] for r in rs], ddof=1)),
                               per_fist={f2: {m: float(np.mean([M[r]["per_fist"][f2][m] for r in rs])) for m in ("cer", "wer")} for f2 in OPS})
                       for f, rs in FAM.items() if f in NEW + list(REF)}
                cmp = {f"{a}_vs_{b}": _cmp(M, FAM, a, b) for a, b in pairs}
                B[cond] = dict(rows={r: {k: M[r][k] for k in ("cer", "wer", "exact")} for r in M if r.rsplit("_seed", 1)[0] in NEW},
                               families=fam, comparisons=cmp)
                for f in REF:                                        # C5: the reference families equal the matched record's
                    v = m_res["grid"]["texts"][text][kind][cond]["families"][f]
                    c5 = max(c5, abs(fam[f]["cer"] - v["cer"]), abs(fam[f]["wer"] - v["wer"]))
            out["texts"][text][kind] = B
        allf = {**{r: (v["T"], v["beta"]) for r, v in m_fit["fits"][text].items()},
                **{r: (v["T"], v["beta"]) for r, v in fits[text].items()}}
        out["texts"][text]["fits"] = {f: [list(allf[r]) for r in FAM[f]] for f in FAM if f in NEW + list(REF)}
    out["anchors"] = dict(c1_pass=tr.get("c1_pass"), c2_min_argmax_agreement=tr.get("c2_min_argmax_agreement"),
                          c3_pass=chk["c3_pass"], c4=chk["c4"], c5=dict(max_abs=c5, passed=c5 < 1e-9))
    out["anchors_pass"] = bool(out["anchors"]["c1_pass"] and chk["c3_pass"] and chk["c4"]["c4_disjoint"] and c5 < 1e-9)
    lab = {}
    for text in ("eng", "rand"):
        for kind in ("record", "fresh"):
            C = out["texts"][text][kind]["own_fit"]["comparisons"]; L = {}
            for k in ("spk40_op12_vs_lif_op12", "spk80_op12_vs_lif_op12", "spk80_op12_vs_lif80_op12"):
                if k in C:
                    L[f"S1/S3 {k}"] = _s1(C[k])
            for k in ("spk80_op12_vs_spk40_op12", "lif80_op12_vs_lif_op12"):
                if k in C:
                    L[f"S2/S3 {k}"] = _label_r1(C[k])
            for a in NEW:
                for b in ("bigru_fist", "mlp_op12"):
                    k = f"{a}_vs_{b}"
                    if k in C:
                        L[f"S4 {k}"] = _label_lead(C[k])
            lab[f"{text}_{kind}"] = L
    out["labels"] = lab
    E = lab["eng_record"]
    out["escalate"] = any(v in ("TIE", "LEADS", "LEADS ON ONE") for k, v in E.items() if k.startswith("S4 ") and k.endswith("_vs_bigru_fist")) \
        or any(v in ("LEADS", "LEADS ON ONE") for k, v in E.items() if k.startswith("S4 ") and k.endswith("_vs_mlp_op12"))
    walls, c2 = {}, {}
    for r, (fam, p) in new_rows().items():
        m = _meta(r)
        walls.setdefault(f"{fam}_{SPIKE_FED_CORPUS}", []).append(m["wall_s"]); c2.setdefault(f"{fam}_{SPIKE_FED_CORPUS}", []).append(m["c2_forward"]["argmax_agreement"])
    out["train_wall_s"] = {k: dict(n=len(v), min=min(v), max=max(v)) for k, v in walls.items()}
    out["c2_argmax_agreement"] = {k: dict(n=len(v), min=min(v), mean=float(np.mean(v))) for k, v in c2.items()}
    out["git"] = git_info(); out["env"] = env_info(); out["finished"] = now()
    return out


def _dump(obj, name, indent=1):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=indent))


def _load(name):
    return json.loads((OUT / name).read_text())


def main(step="all"):
    t0 = time.time()
    if step in ("tables", "all"):
        tables.build(matched="spike_fed")
        print(f"  tables ({time.time() - t0:.0f}s)", flush=True)
    if step in ("check", "all"):
        _dump(check(), "check.json")
        print(f"  check: C3 and C4 pass ({time.time() - t0:.0f}s)", flush=True)
    if step == "check-train":
        _dump(check_train(), "check_train.json")
        print(f"  check-train ({time.time() - t0:.0f}s)", flush=True)
    if step in ("fit", "all"):
        _dump(dict(protocol_hash=P.SPIKE_FED_HASH, fits={t: fit_text(t) for t in P.TEXTS}), "fit.json")
        print(f"  fit ({time.time() - t0:.0f}s)", flush=True)
    if step in ("eval", "all"):
        F = _load("fit.json")["fits"]
        _dump(dict(protocol_hash=P.SPIKE_FED_HASH, eval={t: eval_text(t, F[t]) for t in P.TEXTS}), "eval.json", None)
        print(f"  eval ({time.time() - t0:.0f}s)", flush=True)
    if step in ("crit1a", "all"):
        _dump(criterion_1a(), "criterion_1a.json")
        print(f"  criterion 1A ({time.time() - t0:.0f}s)", flush=True)
    if step in ("fidelity", "all"):
        R = fidelity()
        _dump(R, "fidelity.json")
        print(f"  fidelity: {R['agree']}/{R['gaps']} ({100 * R['agreement']:.3f}%), same-weights max {R['max_abs_pp_same_weights']:.2f} pp "
              f"({time.time() - t0:.0f}s)", flush=True)
    if step in ("stats", "all"):
        tr = _load("check_train.json") if (OUT / "check_train.json").exists() else {}
        R = stats(_load("fit.json")["fits"], _load("eval.json")["eval"], _load("check.json"), tr)
        _dump(R, "results.json")
        assert R["anchors"]["c3_pass"] and R["anchors"]["c4"]["c4_disjoint"] and R["anchors"]["c5"]["passed"], "the spike-fed record's anchors failed"
        print(f"  anchors: {R['anchors']}; escalate {R['escalate']}", flush=True)
    if step in ("manifest", "all"):
        files = ["check.json", "fit.json", "eval.json", "criterion_1a.json", "fidelity.json", "results.json"]
        if (OUT / "check_train.json").exists():
            files.insert(1, "check_train.json")
        _dump(dict(experiment="exp213_spike_input", protocol_hash=P.SPIKE_FED_HASH, base_protocol_hash=P.GRID_WINDOW_HASH,
                   matched_protocol_hash=P.MATCHED_HASH, created=now(), git=git_info(), env=env_info(),
                   files={f: sha256_file(OUT / f) for f in files}), "manifest.json")
    print(f"records -> {OUT}")


if __name__ == "__main__":
    main((sys.argv[1:] or ["all"])[0])
