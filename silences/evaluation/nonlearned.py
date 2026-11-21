"""Non-learned baselines, and two checks on the decoder-weight fits (protocol/nonlearned.json; Sections 6.4
and 8.4, Table 3).

    python -m silences.evaluation.nonlearned [fitrule|tables|audio|check|fit|eval|gapacc|werfit|extgrid|rulegrid|stats|manifest|all]

Part A scores the threshold readout (segmenters/readout.py) under the paper's protocol exactly as a network:
its own decoder weights fitted per text, the record and the fresh evaluation corpora, Welch intervals against
the learned families. Four forms: the fixed thresholds of the earlier rule-based reader at the front end's
time-unit estimate (rule_fixed) and at the estimate they were designed for (rule_fixed_raf); the five
parameters fitted by per-gap cross-entropy on each training seed's op12 corpus (rule_fit_op12, ten rows); and
the adaptive variant (rule_adapt). Because every readout's fitted insertion bonus sits at the bottom of the
grid, the readouts are also refitted over negative bonuses (rulegrid), reported beside the protocol fit.

Part B runs multimon-ng on audio rendered from the record corpora (evaluation/audio.py; needs the program).

The two checks re-score the learned families: fitted to word error instead of character error (werfit), and on
a grid extended to lambda = 32 and gamma = 8 (extgrid). Per-gap accuracy carries the inter-word confusion.

Records written to records/nonlearned/:
    readout_fits.json   the fitted readout's parameters per training seed
    check.json          R1: the tables' corpora and rows are the width record's; R2: the fixed readout is the
                        reader the per-word oracle record re-ran; R3: readout rows are valid log-posteriors;
                        R4: the audio's marks reproduce the corpora
    fit.json, eval.json the readouts' fits; every scored family at its own fit and at the default weights
    gapacc.json         per-gap accuracy and the inter-word confusion of every scored row
    audio.json          multimon-ng, both settings, per corpus and per operator
    werfit_*.json, extgrid_*.json, rulegrid_*.json   the alternative fits and their evaluations
    results.json        the record
The research repository's records of the same run ship under records/nonlearned/research/.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib       # noqa: E402
import json          # noqa: E402
import pickle        # noqa: E402
import sys           # noqa: E402
import time          # noqa: E402
from multiprocessing import get_context   # noqa: E402

import numpy as np   # noqa: E402

from .. import OUTPUTS, RECORDS, timing   # noqa: E402
from ..features import fe_tu_us   # noqa: E402
from ..fixedpoint.q16 import grid_q   # noqa: E402
from ..segmenters import readout   # noqa: E402
from . import protocol as P, tables, width, width_lif   # noqa: E402
from .decoder_weights import _eval_row, _fit_row, _pack   # noqa: E402
from .matched import _label_lead, _label_r1   # noqa: E402
from .metrics import edit_sums   # noqa: E402
from .producers import SEEDS   # noqa: E402
from .provenance import env_info, git_info, now, sha256_file   # noqa: E402

OUT = RECORDS / "nonlearned"
TABLES = OUTPUTS / "tables_nonlearned"
WORKERS = width.WORKERS
OPS = P.OPERATORS
READOUTS = P.NONLEARNED["readouts"]
W_FIT = READOUTS["rule_fit_op12"]["W"]                  # 5: the matrix whose centre gap the thresholds are fitted on
BIGRU = "bigru_fist_fp"       # the baseline through the float path, which the width record shows equals its fixed-point rows
RECORD = [BIGRU, "mlp_op12", "lif_op12", "spk40_op12", "mlp_w5_op12", "spk40_w5_op12", "lif_w5_op12"]
RULES = ["rule_fixed", "rule_fixed_raf", "rule_adapt", "rule_fit_op12"]
LEARNED = (BIGRU, "mlp_op12", "spk40_op12")
PAIRS = ([(r, l) for r in RULES for l in LEARNED] +
         [("rule_fit_op12", "rule_fixed"), ("rule_adapt", "rule_fixed"), ("rule_fit_op12", "rule_adapt"), ("rule_fixed_raf", "rule_fixed")])
RECORD_PAIRS = [("spk40_op12", BIGRU), ("mlp_op12", BIGRU), ("mlp_op12", "spk40_op12"), ("spk40_op12", "lif_op12"),
                ("mlp_w5_op12", "mlp_op12"), ("spk40_w5_op12", "spk40_op12"), ("lif_w5_op12", "lif_op12")]
FIT_SOURCES = [RECORDS / "matched", RECORDS / "spike_fed", RECORDS / "width", RECORDS / "width_lif"]
NEG_GAMMAS = [-8.0, -4.0, -2.0, -1.0, -0.5]
NEG_LAMBDAS = P.T_GRID + [24, 32]
KINDS = (("record", P.EVAL["seeds"]), ("fresh", P.FRESH_SEEDS))


# ---- the readouts ----
def new_rows():
    """{row: (kind, parameters)}; the fitted rows exist once fitrule has run."""
    out = {"rule_fixed_seed0": ("fixed", READOUTS["rule_fixed"]), "rule_fixed_raf_seed0": ("fixed_raf", READOUTS["rule_fixed_raf"]),
           "rule_adapt_seed0": ("adaptive", READOUTS["rule_adapt"])}
    if (OUT / "readout_fits.json").exists():
        R = _load("readout_fits.json")
        assert R["protocol_hash"] == P.NONLEARNED_HASH
        for s, prm in R["fits"].items():
            out[f"rule_fit_op12_seed{s}"] = ("fitted", prm)
    return out


def producer(kind, prm):
    if kind in ("fixed", "fixed_raf", "fitted"):
        p = readout.Params(tuple(prm["theta_dot"]), tuple(prm["theta_dash"]), prm["k"], 1.0)
        tu_of = timing.estimate_tu_us if kind == "fixed_raf" else fe_tu_us

        def f(tx):
            ticks = np.asarray(tx["ticks"], dtype=np.float64)
            if len(ticks) < 2:
                return []
            return readout.lp3(readout.normalized_intervals(ticks, tu_of(ticks)), np.asarray(tx["ch"])[1:], p)
        return f

    def g(tx):
        ticks = np.asarray(tx["ticks"], dtype=np.float64)
        if len(ticks) < 2:
            return []
        return readout.lp3_adaptive(readout.normalized_intervals(ticks, fe_tu_us(ticks)), np.asarray(tx["ch"])[1:],
                                    prm["k"], prm["t_lo"], prm["t_hi"], 1.0, prm["plow"])
    return g


def _fit_one(seed):
    t0 = time.time()
    F, y, sha, _ = width.width_matrix(W_FIT, seed)
    x = np.exp(F[:, W_FIT]); bit = F[:, 2 * W_FIT + 1 + W_FIT]      # the centre gap: its interval in time units and its channel bit
    assert set(np.unique(bit)) <= {0.0, 1.0}
    med = lambda b_, c: float(np.median(x[(bit == b_) & (y == c)]))                                   # noqa: E731
    mid = [((med(b_, 0) + med(b_, 1)) / 2, (med(b_, 1) + med(b_, 2)) / 2) for b_ in (0.0, 1.0)]       # per bank: between the class medians
    starts = [readout.Params(READOUTS["rule_fixed"]["theta_dot"], READOUTS["rule_fixed"]["theta_dash"], k) for k in (1.5, 3.0, 6.0)]
    starts += [readout.Params(mid[0], mid[1], k) for k in (1.5, 3.0, 6.0)]
    tried = []
    for p0 in starts:
        p = readout.fit(x, bit, y, p0)
        tried.append((readout.cross_entropy(x, bit, y, p), p, p0))
    ce, p, p0 = min(tried, key=lambda t: t[0])
    acc = lambda q: float((readout.class_probs(x, bit, q).argmax(1) == y).mean())                     # noqa: E731
    return str(seed), dict(theta_dot=list(p.theta_dot), theta_dash=list(p.theta_dash), k=p.k, gaps=int(len(y)), feature_sha256=sha,
                           corpus_seed=P.GRID_CORPUS_SEED0[width.ARM] + seed, train_gap_accuracy=acc(p), cross_entropy=ce,
                           start=dict(theta_dot=list(p0.theta_dot), theta_dash=list(p0.theta_dash), k=p0.k),
                           starts=[dict(ce=c, k0=q0.k, theta_dot0=list(q0.theta_dot), acc=acc(q)) for c, q, q0 in tried],
                           wall_s=round(time.time() - t0, 1))


def fitrule():
    with get_context("fork").Pool(min(WORKERS, len(SEEDS))) as pool:
        fits = dict(pool.imap_unordered(_fit_one, SEEDS))
    feat = json.loads((width.OUT / "features.json").read_text())
    assert all(f["feature_sha256"] == feat["corpora"][s]["widths"][str(W_FIT)]["feature_sha256"] for s, f in fits.items())
    return dict(protocol_hash=P.NONLEARNED_HASH, readout=READOUTS["rule_fit_op12"], fits={s: fits[s] for s in sorted(fits, key=int)})


# ---- families and tables ----
def families():
    """{family: rows}: the feedforward families of the earlier records and the readouts (single-row families kept)."""
    F = dict(width_lif._gapacc_families())
    for r in new_rows():
        F.setdefault(r.rsplit("_seed", 1)[0], []).append(r)
    return F


def table_path(text, seed):
    return TABLES / f"{text}_{seed}.pkl"


def _make_table(job):
    text, kind, seed = job
    t0 = time.time()
    added, total = width.extend_table(table_path(text, seed), lambda: width_lif.load(text, seed), P.NONLEARNED_HASH, new_rows, producer,
                                      lambda k, prm: hashlib.sha256(json.dumps(prm, sort_keys=True).encode()).hexdigest())
    return f"{text}_{seed}", added, total, round(time.time() - t0)


def build_tables():
    TABLES.mkdir(parents=True, exist_ok=True)
    jobs = tables.corpora(matched=True)
    with get_context("fork").Pool(min(WORKERS, len(jobs))) as pool:
        for name, added, total, sec in pool.imap_unordered(_make_table, jobs):
            print(f"  {name}: +{added} rows, {total} in all ({sec}s)", flush=True)


def load(text, seed):
    with open(table_path(text, seed), "rb") as fh:
        doc = pickle.load(fh)
    assert doc["protocol_hash"] == P.NONLEARNED_HASH
    return doc


# ---- check ----
def check():
    out = {}
    text, seed = "eng", P.EVAL["seeds"][0]
    doc, ref = load(text, seed), width_lif.load(text, seed)
    out["r1_corpus_identity"] = bool(doc["truth"] == ref["truth"] and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(doc["ticks"], ref["ticks"]))
                                     and all(np.array_equal(doc["rows"][r][0], ref["rows"][r][0]) for r in ref["rows"]))
    leg = json.loads((RECORDS / "tempo" / "legacy_readout.json").read_text())["readout"]
    out["r2_fixed_equals_legacy"] = bool(list(leg["dur"]) == READOUTS["rule_fixed"]["theta_dot"] and list(leg["dash"]) == READOUTS["rule_fixed"]["theta_dash"]
                                         and leg["k"] == READOUTS["rule_fixed"]["k"])
    fmt = True
    for r in new_rows():
        for tx_rows, net_rows, ticks in zip(doc["rows"][r], doc["rows"]["spk40_op12_seed0"], doc["ticks"]):
            a = np.asarray(tx_rows, dtype=np.int64).reshape(-1, 3)
            fmt &= len(a) == len(net_rows) == len(ticks) - 1 and bool(np.all((a >= -(2 ** 31)) & (a <= -1)))
    out["r3_lp3_format"] = bool(fmt)
    A = _load("audio.json")["r4_parity"]
    out["r4_audio_parity"] = all(v["equal"] == v["transmissions"] for v in A.values())
    out["rows"] = sorted(new_rows())
    out["families"] = {f: len(rs) for f, rs in families().items()}
    out["pass"] = all(v for k, v in out.items() if k.startswith("r"))
    assert out["pass"], out
    return dict(protocol_hash=P.NONLEARNED_HASH, **out)


# ---- fits and evaluation ----
_FITS = {}


def own_fit_of(row, text):
    """The fit (with its grid) a learned row received in its own record."""
    for d in FIT_SOURCES:
        if d not in _FITS:
            _FITS[d] = json.loads((d / "fit.json").read_text())["fits"]
        if row in _FITS[d][text]:
            return _FITS[d][text][row]
    raise KeyError(row)


def all_fits(text):
    """{row: (T, gamma)} at own fit for every scored row: the readouts from this record, the rest from theirs."""
    mine = _load("fit.json")["fits"][text]
    out = {}
    for rows in families().values():
        for r in rows:
            src = mine[r] if r in mine else own_fit_of(r, text)
            out[r] = (src["T"], src["beta"])
    return out


def _eval_all(settings_of, kinds=KINDS):
    """Evaluate every scored row under settings_of(text, row) -> {condition: (T, gamma)} on the listed corpora."""
    res = {}
    for text in P.TEXTS:
        res[text] = {}
        for kind, seeds in kinds:
            res[text][kind] = {}
            for seed in seeds:
                tab = width._set_tab(text, seed, load)
                jobs = [(row, settings_of(text, row)) for rows in families().values() for row in rows
                        if row in tab["rows"] and settings_of(text, row)]
                with get_context("fork").Pool(min(WORKERS, len(jobs))) as pool:
                    res[text][kind][str(seed)] = dict(pool.imap_unordered(_eval_row, jobs))
    return res


def evaluate():
    dep = (P.DEFAULT_WEIGHTS["T"], P.DEFAULT_WEIGHTS["beta"])
    F = {t: all_fits(t) for t in P.TEXTS}
    return _eval_all(lambda text, row: ({"deployed": dep, "own_fit": F[text][row]} if row in F[text] else None))


# ---- per-gap accuracy and the inter-word confusion ----
def _gapacc_corpus(job):
    text, seed = job
    doc = load(text, seed)
    txs = width.corpus_with_labels(text, seed, doc)
    labels = np.concatenate([np.asarray(tx["labels"], dtype=np.int64) for tx in txs])
    fist_of_gap = np.concatenate([[tx["fist"]] * len(tx["labels"]) for tx in txs])
    out = {}
    for rows in families().values():
        for r in rows:
            if r not in doc["rows"]:
                continue
            lp = np.concatenate([np.asarray(x, dtype=np.int64).reshape(-1, 3) for x in doc["rows"][r] if len(x)])
            assert len(lp) == len(labels), (r, len(lp), len(labels))
            pred = lp.argmax(1)
            conf = np.zeros((3, 3), dtype=np.int64)
            np.add.at(conf, (labels, pred), 1)
            out[r] = dict(overall=float((pred == labels).mean()), per_fist={o: float((pred == labels)[fist_of_gap == o].mean()) for o in OPS},
                          confusion=conf.tolist(), gaps=int(len(labels)))
    return f"{text}_{seed}", out


def gapacc():
    jobs = [(t, s) for t in P.TEXTS for s in P.EVAL["seeds"]]
    with get_context("fork").Pool(len(jobs)) as pool:
        per = dict(pool.imap_unordered(_gapacc_corpus, jobs))
    out = dict(protocol_hash=P.NONLEARNED_HASH, corpora=per, texts={})
    for text in P.TEXTS:
        keys = [f"{text}_{s}" for s in P.EVAL["seeds"]]
        fam = {}
        for f, rs in families().items():
            rs = [r for r in rs if r in per[keys[0]]]
            if not rs:
                continue
            seed_means = [np.mean([per[k][r]["overall"] for k in keys]) for r in rs]
            conf = np.sum([np.asarray(per[k][r]["confusion"]) for k in keys for r in rs], axis=0)
            gaps = int(conf.sum())
            iw_fp, iw_fn = int(conf[:2, 2].sum()), int(conf[2, :2].sum())
            fam[f] = dict(n=len(rs), mean=100 * float(np.mean(seed_means)), sd=100 * float(np.std(seed_means, ddof=1)) if len(rs) > 1 else 0.0,
                          per_fist={o: 100 * float(np.mean([per[k][r]["per_fist"][o] for k in keys for r in rs])) for o in OPS},
                          confusion=(conf / len(rs)).tolist(), gaps_per_row=gaps // len(rs),
                          iw=dict(false_pos_per_1000=1000 * iw_fp / gaps, false_neg_per_1000=1000 * iw_fn / gaps,
                                  fp_over_fn=(iw_fp / iw_fn) if iw_fn else None,
                                  recall=float(conf[2, 2] / conf[2].sum()), precision=float(conf[2, 2] / conf[:, 2].sum()),
                                  share_of_errors=float((iw_fp + iw_fn) / (gaps - np.trace(conf)))))
        out["texts"][text] = dict(families=fam)
    return out


# ---- the alternative fits: to word error, on the extended grid, and the readouts over negative bonuses ----
def _stored(text, row):
    mine = _load("fit.json")["fits"][text]
    return mine[row] if row in mine else own_fit_of(row, text)


def werfit():
    chosen = {}
    for text in P.TEXTS:
        chosen[text] = {}
        for f in RECORD + ["rule_fit_op12"]:
            for row in families()[f]:
                src = _stored(text, row)
                best = min(src["grid"].values(), key=lambda r: (round(r["wer"], 10), r["T"], r["beta"]))
                chosen[text][row] = dict(T=best["T"], beta=best["beta"], cer=best["cer"], wer=best["wer"], own=(src["T"], src["beta"]),
                                         moved=(best["T"], best["beta"]) != (src["T"], src["beta"]))
    ev = _eval_all(lambda text, row: ({"wer_fit": (chosen[text][row]["T"], chosen[text][row]["beta"])} if row in chosen[text] else None),
                   kinds=KINDS[:1])
    return dict(protocol_hash=P.NONLEARNED_HASH, fits=chosen), dict(protocol_hash=P.NONLEARNED_HASH, eval=ev)


def _cells(args):
    row, lambdas, gammas, skip_stored = args
    import silences.evaluation.decoder_weights as DW
    grid = {}
    for gamma in gammas:
        for T in lambdas:
            if skip_stored and gamma in P.GAMMA_GRID and T in P.T_GRID:
                continue
            grid[f"{T}|{gamma}"] = dict(T=T, beta=gamma, **_pack(edit_sums(DW._TAB, row, grid_q(T), grid_q(gamma))[0]))
    return row, grid


def _regrid(fams, lambdas, gammas, skip_stored, cells_key, cond):
    chosen = {}
    for text in P.TEXTS:
        chosen[text] = {}
        width._set_tab(text, P.CALIB_SEED[text], load)
        rows = [r for f in fams for r in families()[f]]
        with get_context("fork").Pool(min(WORKERS, len(rows))) as pool:
            extra = dict(pool.imap_unordered(_cells, [(r, lambdas, gammas, skip_stored) for r in rows]))
        for row in rows:
            src = _stored(text, row)
            grid = dict(src["grid"]); grid.update(extra[row])
            best = min(grid.values(), key=lambda r: (round(r["cer"], 10), r["T"], r["beta"]))
            chosen[text][row] = {**dict(T=best["T"], beta=best["beta"], cer=best["cer"], wer=best["wer"], own=(src["T"], src["beta"]),
                                        moved=(best["T"], best["beta"]) != (src["T"], src["beta"])), cells_key: extra[row]}
    ev = _eval_all(lambda text, row: ({cond: (chosen[text][row]["T"], chosen[text][row]["beta"])} if row in chosen[text] else None),
                   kinds=KINDS[:1])
    return chosen, dict(protocol_hash=P.NONLEARNED_HASH, eval=ev)


def extgrid():
    G = P.NONLEARNED["ext_grid"]
    chosen, ev = _regrid(RECORD + ["rule_fit_op12"], G["T_grid"], G["beta_grid"], True, "ext_cells", "ext_fit")
    return dict(protocol_hash=P.NONLEARNED_HASH, grid=G, fits=chosen), ev


def rulegrid():
    """Every readout's protocol fit sat at gamma = 0, the bottom of the grid, so the readouts are refitted over
    negative bonuses (and the extended lambdas); the protocol fit is reported first, this one beside it."""
    chosen, ev = _regrid(RULES, NEG_LAMBDAS, NEG_GAMMAS, False, "neg_cells", "neg_fit")
    return dict(protocol_hash=P.NONLEARNED_HASH, gammas=NEG_GAMMAS, lambdas=NEG_LAMBDAS, fits=chosen), ev


# ---- statistics ----
def welch(cand, base, ci=0.95):
    """Difference of seed means in error (pp), positive = candidate better; a single-row side contributes no variance."""
    from scipy import stats as st
    a, b = np.asarray(cand, float), np.asarray(base, float)
    na, nb = len(a), len(b)
    D = float(b.mean() - a.mean())
    va = a.var(ddof=1) / na if na > 1 else 0.0
    vb = b.var(ddof=1) / nb if nb > 1 else 0.0
    se = float(np.sqrt(va + vb))
    if na > 1 and nb > 1:
        df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    else:
        df = (nb - 1) if nb > 1 else (na - 1)
    tc = float(st.t.ppf(0.5 + ci / 2, df)) if se > 0 else float("inf")
    lo, hi = (D - tc * se, D + tc * se) if se > 0 else (D, D)
    t = D / se if se > 0 else float("inf")
    p = float(2 * st.t.sf(abs(t), df)) if se > 0 else 0.0
    return dict(d=D, lo=lo, hi=hi, se=se, df=float(df), t=t, p=p, n_candidate=na, n_baseline=nb, ci=ci,
                win=bool(lo > 0), loss=bool(hi < 0), wins=int(np.sum(a < b.mean())))


def _fam_stats(res_text, cond):
    out = {}
    for f, rows in families().items():
        rs = [r for r in rows if r in res_text and cond in res_text[r]]
        if not rs:
            continue
        cer = [res_text[r][cond]["overall"]["cer"] for r in rs]
        wer = [res_text[r][cond]["overall"]["wer"] for r in rs]
        out[f] = dict(n=len(rs), cer=float(np.mean(cer)), wer=float(np.mean(wer)),
                      cer_sd=float(np.std(cer, ddof=1)) if len(rs) > 1 else 0.0, wer_sd=float(np.std(wer, ddof=1)) if len(rs) > 1 else 0.0,
                      cer_rows=cer, wer_rows=wer,
                      per_fist={o: dict(cer=float(np.mean([res_text[r][cond]["per_fist"][o]["cer"] for r in rs])),
                                        wer=float(np.mean([res_text[r][cond]["per_fist"][o]["wer"] for r in rs]))) for o in OPS})
    return out


def _seed_mean_rows(ev, text, kind, cond):
    """{row: {cond: {overall, per_fist}}} averaged over the corpora of one kind."""
    seeds = list(ev[text][kind])
    rows = set.intersection(*[set(ev[text][kind][s]) for s in seeds])
    out = {}
    for r in rows:
        if not all(cond in ev[text][kind][s][r] for s in seeds):
            continue
        o = dict(cer=float(np.mean([ev[text][kind][s][r][cond]["overall"]["cer"] for s in seeds])),
                 wer=float(np.mean([ev[text][kind][s][r][cond]["overall"]["wer"] for s in seeds])))
        pf = {f: dict(cer=float(np.mean([ev[text][kind][s][r][cond]["per_fist"][f]["cer"] for s in seeds])),
                      wer=float(np.mean([ev[text][kind][s][r][cond]["per_fist"][f]["wer"] for s in seeds]))) for f in OPS}
        out[r] = {cond: dict(overall=o, per_fist=pf)}
    return out


def _compare(FS, pairs):
    return {f"{a}_vs_{b}": dict(cer=welch(FS[a]["cer_rows"], FS[b]["cer_rows"]), wer=welch(FS[a]["wer_rows"], FS[b]["wer_rows"]))
            for a, b in pairs if a in FS and b in FS}


def _public(FS):
    return {f: {k: v for k, v in s.items() if k not in ("cer_rows", "wer_rows")} for f, s in FS.items()}


def stats():
    ev = _load("eval.json"); assert ev["protocol_hash"] == P.NONLEARNED_HASH
    ev = ev["eval"]
    R16 = width._load("results.json", width_lif.OUT); R15 = width._load("results.json")
    R = dict(experiment="exp217_nonlearned_baselines", protocol_hash=P.NONLEARNED_HASH, protocol=P.NONLEARNED, texts={}, labels={}, anchors={})
    # R5: the learned families re-decoded here equal the half-width-5 record's (the baseline through the float path
    # against its fixed-point record row, within what the width record itself shows; every other family exactly)
    r5, r5_bigru = 0.0, 0.0
    for text in P.TEXTS:
        R["texts"][text] = {}
        for kind in ("record", "fresh"):
            R["texts"][text][kind] = {}
            for cond in ("own_fit", "deployed"):
                FS = _fam_stats(_seed_mean_rows(ev, text, kind, cond), cond)
                R["texts"][text][kind][cond] = dict(families=_public(FS), comparisons=_compare(FS, PAIRS + RECORD_PAIRS))
                for f in RECORD:
                    ref = R16["texts"][text][kind][cond]["families"].get("bigru_fist" if f == BIGRU else f)
                    if ref and f in FS:
                        d = max(abs(ref["cer"] - FS[f]["cer"]), abs(ref["wer"] - FS[f]["wer"]))
                        if f == BIGRU:
                            r5_bigru = max(r5_bigru, d)
                        else:
                            r5 = max(r5, d)
    Ex = _load("rulegrid_eval.json")["eval"]; Fx = _load("rulegrid_fits.json")["fits"]
    for text in P.TEXTS:
        merged = _seed_mean_rows(ev, text, "record", "own_fit")
        for r, v in _seed_mean_rows(Ex, text, "record", "neg_fit").items():
            merged[r] = {"own_fit": v["neg_fit"]}
        FS = _fam_stats(merged, "own_fit")
        R["texts"][text]["record"]["neg_fit"] = dict(
            families={f: s for f, s in _public(FS).items() if f in RULES}, comparisons=_compare(FS, PAIRS),
            fits={f: sorted({(Fx[text][r]["T"], Fx[text][r]["beta"]) for r in families()[f] if r in Fx[text]}) for f in RULES},
            note="readouts at their negative-bonus fit, learned families at their own fit")
        C = R["texts"][text]["record"]["neg_fit"]["comparisons"]
        R["labels"][f"{text}_record_neg"] = {f"A {r} vs {l} (neg)": _label_lead(C[f"{r}_vs_{l}"]) for r in RULES for l in LEARNED if f"{r}_vs_{l}" in C}
    b1 = max(abs(R15["texts"][t][k][c]["families"]["bigru_fist_fp"][m] - R15["texts"][t][k][c]["families"]["bigru_fist"][m])
             for t in P.TEXTS for k in ("record", "fresh") for c in ("own_fit", "deployed") for m in ("cer", "wer"))
    R["anchors"].update(r5_max_abs_diff_pp=r5, r5_bigru_float_vs_fixed_pp=r5_bigru, r5_bigru_bound_from_exp215_pp=b1,
                        r5_pass=bool(r5 < 1e-9 and r5_bigru <= b1 + 1e-9))
    R["anchors"].update({k: v for k, v in _load("check.json").items() if k.startswith("r")})
    for text in P.TEXTS:
        for kind in ("record", "fresh"):
            C = R["texts"][text][kind]["own_fit"]["comparisons"]
            L = {}
            for r in RULES:
                for l in LEARNED:
                    if f"{r}_vs_{l}" in C:
                        L[f"A {r} vs {l}"] = _label_lead(C[f"{r}_vs_{l}"])
            for k, nm in (("rule_fit_op12_vs_rule_fixed", "A fit vs fixed"), ("rule_adapt_vs_rule_fixed", "A adapt vs fixed"),
                          ("rule_fit_op12_vs_rule_adapt", "A fit vs adapt"), ("rule_fixed_raf_vs_rule_fixed", "A fixed at its own unit vs at the front end's")):
                if k in C:
                    L[nm] = _label_r1(C[k])
            R["labels"][f"{text}_{kind}"] = L
    # the collapse under drift: the high-jitter sideswiper against the fast regular operator, per family
    R["drift"] = {}
    for text in P.TEXTS:
        FS = dict(R["texts"][text]["record"]["own_fit"]["families"])
        for f, v in R["texts"][text]["record"]["neg_fit"]["families"].items():
            FS[f + "_neg"] = v
        R["drift"][text] = {f: dict(fast=FS[f]["per_fist"]["fast_normalr"]["cer"], cootie=FS[f]["per_fist"]["cootie_high_jitter"]["cer"],
                                    gap_pp=FS[f]["per_fist"]["cootie_high_jitter"]["cer"] - FS[f]["per_fist"]["fast_normalr"]["cer"],
                                    ratio=FS[f]["per_fist"]["cootie_high_jitter"]["cer"] / FS[f]["per_fist"]["fast_normalr"]["cer"])
                            for f in RULES + [r + "_neg" for r in RULES] + list(LEARNED) if f in FS}
    G = _load("gapacc.json"); assert G["protocol_hash"] == P.NONLEARNED_HASH
    R["gapacc"] = G["texts"]
    A = _load("audio.json")
    R["audio"] = dict(decoder_version=A["decoder_version"], texts=A["texts"], r4_parity=A["r4_parity"],
                      samples={k: v["samples"][:3] for k, v in A["corpora"].items()})
    for name, stem, cond in (("q1_wer_fit", "werfit", "wer_fit"), ("q2_ext_grid", "extgrid", "ext_fit")):
        Fx = _load(f"{stem}_fits.json")["fits"]; Ex = _load(f"{stem}_eval.json")["eval"]
        Q = {}
        for text in P.TEXTS:
            FS = _fam_stats(_seed_mean_rows(Ex, text, "record", cond), cond)
            own = R["texts"][text]["record"]["own_fit"]
            comp = _compare(FS, RECORD_PAIRS + [("rule_fit_op12", l) for l in LEARNED])
            changes = []
            for k, c in comp.items():
                if k in own["comparisons"]:
                    for m in ("cer", "wer"):
                        o, n = own["comparisons"][k][m], c[m]
                        if (o["win"], o["loss"]) != (n["win"], n["loss"]):
                            changes.append(dict(pair=k, metric=m, own=dict(d=o["d"], lo=o["lo"], hi=o["hi"]), alt=dict(d=n["d"], lo=n["lo"], hi=n["hi"])))
            fam = {}
            for f, s in FS.items():
                if f in own["families"]:
                    fam[f] = dict(cer=s["cer"], wer=s["wer"], d_cer=s["cer"] - own["families"][f]["cer"], d_wer=s["wer"] - own["families"][f]["wer"],
                                  rows_moved=sum(Fx[text][r]["moved"] for r in families()[f] if r in Fx[text]), n=s["n"],
                                  fits=sorted({(Fx[text][r]["T"], Fx[text][r]["beta"]) for r in families()[f] if r in Fx[text]}))
            Q[text] = dict(families=fam, comparisons=comp, label_changes=changes,
                           max_abs_shift_cer=max(abs(v["d_cer"]) for v in fam.values()), max_abs_shift_wer=max(abs(v["d_wer"]) for v in fam.values()))
        R[name] = Q
    R.update(git=git_info(), env=env_info(), finished=now())
    return R


def _dump(obj, name, indent=1):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=indent))


def _load(name):
    return json.loads((OUT / name).read_text())


STEPS = ("fitrule", "tables", "audio", "check", "fit", "eval", "gapacc", "werfit", "extgrid", "rulegrid", "stats", "manifest")


def main(step="all"):
    assert step == "all" or step in STEPS, step
    t0 = time.time()
    done = lambda s: print(f"  {s} ({time.time() - t0:.0f}s)", flush=True)      # noqa: E731
    if step in ("fitrule", "all"):
        _dump(fitrule(), "readout_fits.json"); done("readout fits")
    if step in ("tables", "all"):
        build_tables(); done("tables")
    if step in ("audio", "all"):
        from . import audio
        _dump(audio.run(), "audio.json"); done("audio")
    if step in ("check", "all"):
        _dump(check(), "check.json"); done("check: R1 to R4 pass")
    if step in ("fit", "all"):
        fits = {}
        for t in P.TEXTS:
            tab = width._set_tab(t, P.CALIB_SEED[t], load)
            rows = [r for r in new_rows() if r in tab["rows"]]
            with get_context("fork").Pool(min(WORKERS, len(rows))) as pool:
                got = dict(pool.imap_unordered(_fit_row, [(r, P.GAMMA_GRID) for r in rows]))
            fits[t] = {r: got[r] for r in rows}
        _dump(dict(protocol_hash=P.NONLEARNED_HASH, fits=fits), "fit.json"); done("fit")
    if step in ("eval", "all"):
        _dump(dict(protocol_hash=P.NONLEARNED_HASH, eval=evaluate()), "eval.json", None); done("eval")
    if step in ("gapacc", "all"):
        _dump(gapacc(), "gapacc.json"); done("per-gap accuracy")
    if step in ("werfit", "all"):
        F, E = werfit(); _dump(F, "werfit_fits.json"); _dump(E, "werfit_eval.json", None); done("fit to word error")
    if step in ("extgrid", "all"):
        F, E = extgrid(); _dump(F, "extgrid_fits.json"); _dump(E, "extgrid_eval.json", None); done("extended grid")
    if step in ("rulegrid", "all"):
        F, E = rulegrid(); _dump(F, "rulegrid_fits.json"); _dump(E, "rulegrid_eval.json", None); done("readouts over negative bonuses")
    if step in ("stats", "all"):
        R = stats()
        _dump(R, "results.json")
        assert R["anchors"]["r5_pass"], R["anchors"]
        print(f"  anchors: {R['anchors']}", flush=True)
    if step in ("manifest", "all"):
        files = ["readout_fits.json", "check.json", "fit.json", "eval.json", "gapacc.json", "audio.json", "werfit_fits.json", "werfit_eval.json",
                 "extgrid_fits.json", "extgrid_eval.json", "rulegrid_fits.json", "rulegrid_eval.json", "results.json"]
        _dump(dict(experiment="exp217_nonlearned_baselines", protocol_hash=P.NONLEARNED_HASH, base_protocol_hash=P.WIDTH_LIF_HASH,
                   created=now(), git=git_info(), env=env_info(), files={f: sha256_file(OUT / f) for f in files}), "manifest.json")
    print(f"records -> {OUT}")


if __name__ == "__main__":
    main((sys.argv[1:] or ["all"])[0])
