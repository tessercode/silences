"""The window width and the input representation (protocol/width.json; Section 7.2, Table 4).

    python -m silences.evaluation.width [tables|check|features|fit|eval|gapacc|stats|manifest|all]

The control is retrained at four other half-widths of the gap window (2, 5, 15 and 20 gaps either side; 10 is
the paper's, read from the matched record), ten training seeds each on the matched op12 corpora, with the
MLP recipe unchanged. Beside it, the baseline's recurrent recipe is trained on the window's two raw values
per gap in place of its nine features (bigru_raw, eight seeds), and the baseline of record is re-scored
through the same floating-point path (bigru_fist_fp), the anchor that the float path stands in for the
fixed-point one. Every new row is scored by the matched record's protocol: (T, gamma) fitted per row and text
on the calibration corpora, evaluation at the fitted and at the default weights on the record and the fresh
evaluation corpora, Welch intervals on seed means. Rows of the matched record are read from its tables and
records, never recomputed.

Records written to records/width/:
    check.json      C3: the harness reproduces the matched record's mlp_op12_seed0 row; C4: no training
                    transmission is in any evaluation corpus; B2: the float nine-feature encoder equals the
                    training code's; the raw-feature network's final-phase corpora are the op12 corpora
    features.json   W1: the op12 feature matrices at each width hash to the ones the shipped models were
                    trained on, and at the paper's width to the matched record's (needs no torch)
    fit.json        (T, gamma) per new row and text, with the grid
    eval.json       every new row on every record and fresh corpus, at its own and the default weights
    gapacc.json     per-gap accuracy of every feedforward row on the record corpora; the inter-spike interval
                    and the window's lookahead in ms and in time units; the renderer's channel error
    results.json    the record: seed means, the comparisons, the pre-registered labels, anchors
The research repository's records of the same run ship under records/width/research/.
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

from .. import MODELS, OUTPUTS, RECORDS   # noqa: E402
from ..corpus import corpus   # noqa: E402
from ..features import fe_tu_us, window_features   # noqa: E402
from ..morse import MORSE_TABLE   # noqa: E402
from ..segmenters import bigru   # noqa: E402
from ..segmenters.mlp import MLP   # noqa: E402
from . import protocol as P, tables   # noqa: E402
from .decoder_weights import _fit_row, _eval_row, row_means   # noqa: E402
from .matched import _cmp, _label_lead, _label_r1   # noqa: E402
from .producers import DEPLOYED, SEEDS, grid_family_rows, grid_model_path   # noqa: E402
from .provenance import env_info, git_info, now, sha256_file   # noqa: E402

OUT = RECORDS / "width"
MATCHED = RECORDS / "matched"
TABLES = OUTPUTS / "tables_width"
OPS = P.OPERATORS
WORKERS = min(tables.WORKERS, 12)
WIDTHS = tuple(P.WIDTH["widths"])                 # 2, 5, 15, 20
W_RECORD = P.WIDTH["record_width"]                # 10
ARM = P.WIDTH["corpus_arm"]                       # op12
RAW_SEEDS = P.WIDTH["bigru_raw"]["seeds"]         # 42..49
ALL_W = sorted(set(WIDTHS) | {W_RECORD})
REF = ("mlp_op12", "lif_op12", "bigru_fist")
FEEDFORWARD = ("mlp_op12", "mlp_gen", "lif_op12", "lif_gen")     # the matched families whose per-gap accuracy is reported
RAW_TRAINING = OUT / "research" / "exp215_train_raw.json"       # the raw-feature network's training record


def family(kind, W):
    return f"{kind}_w{W}_{ARM}"


def mlp_family(W):
    return "mlp_op12" if W == W_RECORD else family("mlp", W)


def width_model_path(kind, W, seed):
    return grid_model_path(f"{kind}_w{W}", ARM, seed)


def raw_checkpoint(seed):
    return MODELS / "bigru" / "raw" / f"bigru3_seed{seed}.pt"


def new_rows():
    """{row: (kind, model file)} for every new model on disk; kind is window, raw or fp."""
    out = {}
    for W in WIDTHS:
        for s in SEEDS:
            p = width_model_path("mlp", W, s)
            if p.exists() and p.with_suffix(".meta.json").exists():
                out[f"{family('mlp', W)}_seed{s}"] = ("window", p)
    R = json.loads(RAW_TRAINING.read_text())
    for s in RAW_SEEDS:
        if R["rows"][str(s)]["status"] == "COMPLETED" and raw_checkpoint(s).exists():
            out[f"bigru_raw_seed{s}"] = ("raw", raw_checkpoint(s))
    for s in RAW_SEEDS:
        p = bigru.arm_dir("fist") / f"bigru3_seed{s}.pt"
        if p.exists():
            out[f"bigru_fist_fp_seed{s}"] = ("fp", p)
    return out


def families():
    """{family: rows}: the matched record's grid families and the new ones present."""
    F = dict(grid_family_rows())
    for r in new_rows():
        F.setdefault(r.rsplit("_seed", 1)[0], []).append(r)
    return {f: rs for f, rs in F.items() if len(rs) >= 2}


def _producer(kind, path):
    if kind == "window":
        return MLP.load(path).lp3
    from ..segmenters import bigru_float
    return bigru_float.producer(kind, path)


def _row_sha(kind, path):
    return json.loads(path.with_suffix(".meta.json").read_text())["pickle_sha256"] if kind == "window" else sha256_file(path)


# ---- tables: the matched set's eighteen corpora with the new rows added ----
def table_path(text, seed):
    return TABLES / f"{text}_{seed}.pkl"


def extend_table(path, base_doc, protocol_hash, rows_fn, producer_fn, sha_fn):
    """Add the rows of rows_fn() that the table at path lacks (creating it from base_doc); returns (added, total)."""
    if path.exists():
        with open(path, "rb") as fh:
            doc = pickle.load(fh)
        assert doc["protocol_hash"] == protocol_hash
    else:
        doc = dict(base_doc(), protocol_hash=protocol_hash, row_sha256={})
        doc["rows"] = dict(doc["rows"])
    todo = {n: v for n, v in rows_fn().items() if n not in doc["rows"]}
    txs = [dict(ticks=t, ch=np.asarray(c, dtype=np.int64)) for t, c in zip(doc["ticks"], doc["ch"])]
    for n, (k, p) in todo.items():
        prod = producer_fn(k, p)
        doc["rows"][n] = [prod(tx) for tx in txs]
        doc["row_sha256"][n] = sha_fn(k, p)
    if todo:
        with open(str(path) + ".tmp", "wb") as fh:
            pickle.dump(doc, fh)
        os.replace(str(path) + ".tmp", path)
    return len(todo), len(doc["rows"])


def _make_table(job):
    text, kind, seed = job
    t0 = time.time()
    added, total = extend_table(table_path(text, seed), lambda: tables.load(text, seed, matched=True), P.WIDTH_HASH,
                                new_rows, _producer, _row_sha)
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
    assert doc["protocol_hash"] == P.WIDTH_HASH
    return doc


def _set_tab(text, seed, loader=None):
    import silences.evaluation.decoder_weights as DW
    DW._TAB = (loader or load)(text, seed)
    return DW._TAB


# ---- check: the anchors that need no training ----
def _fingerprint(ticks):
    return hashlib.sha1(np.asarray(ticks[:12], dtype=np.int64).tobytes()).hexdigest()[:16]


def _train_fingerprints(seed):
    n = P.GRID_N_TX // len(OPS)
    return seed, {_fingerprint(tx["ticks"]) for tx in corpus(n, P.GRID_CORPUS_SEED0[ARM] + seed, "rand")}


def training_fingerprints():
    with get_context("fork").Pool(min(WORKERS, len(SEEDS))) as pool:
        return set().union(*[fp for _, fp in pool.imap_unordered(_train_fingerprints, SEEDS)])


def disjointness(train):
    bad = {}
    for text, kind, seed in tables.corpora(matched=True):
        D = tables.load(text, seed, matched=True)
        n = len(train & {_fingerprint(t) for t in D["ticks"]})
        if n:
            bad[f"{text}_{kind}_{seed}"] = n
    return dict(train_fingerprints=len(train), overlaps=bad, c4_disjoint=not bad)


def _b2(n=200):
    """The float nine-feature encoder of the evaluation against the training code's, on n transmissions."""
    from ..segmenters import bigru_float
    from ..training import bigru as TB
    worst = 0.0
    txs = corpus(n // len(OPS), P.BIGRU_FIST["seed_base"] + RAW_SEEDS[0], "rand")
    for tx in txs:
        spikes = [(float(t) * 1e-6, int(c)) for t, c in zip(tx["ticks"], tx["ch"])]
        tu = TB.TU_SCALAR_C * TB.estimate_tu(spikes)
        ref = TB.nine_features(spikes, tu)
        mine = bigru_float.nine_features(tx["ticks"], tx["ch"], tu * 1e6)
        assert ref.shape == mine.shape
        worst = max(worst, float(np.max(np.abs(ref.astype(np.float64) - mine.astype(np.float64)))) if len(ref) else 0.0)
    return dict(transmissions=len(txs), max_abs_diff=worst, equal=worst < 1e-5)


def check():
    prod = MLP.load(DEPLOYED["mlp"]).lp3
    c3 = {}
    for text in P.TEXTS:
        seed = P.EVAL["seeds"][0]
        D = tables.load(text, seed, matched=True)
        mine = [prod(dict(ticks=t, ch=np.asarray(c, dtype=np.int64))) for t, c in zip(D["ticks"], D["ch"])]
        c3[f"{text}_{seed}"] = dict(transmissions=len(mine), equal=mine == D["rows"]["mlp_op12_seed0"])
    c4 = disjointness(training_fingerprints())
    feat = json.loads((OUT / "features.json").read_text())
    b2 = _b2()
    # the raw-feature network's final-phase corpus of seed 42 + s is the op12 corpus of training seed s
    ident = all(P.GRID_CORPUS_SEED0[ARM] + s == P.BIGRU_FIST["seed_base"] + RAW_SEEDS[s] for s in range(len(RAW_SEEDS)))
    out = dict(protocol_hash=P.WIDTH_HASH, c3=c3, c3_pass=all(v["equal"] for v in c3.values()), c4=c4,
               w1_pass=bool(feat["w1_pass"]), b2=b2, b2_pass=bool(b2["equal"]), corpus_identity=ident)
    assert out["c3_pass"] and c4["c4_disjoint"] and out["w1_pass"] and ident and out["b2_pass"]
    return out


# ---- features: W1, the feature matrices at every width ----
def width_matrix(W, seed):
    """(F, y, sha256) of the op12 corpus of one training seed at half-width W."""
    txs = corpus(P.GRID_N_TX // len(OPS), P.GRID_CORPUS_SEED0[ARM] + seed, "rand")
    tus = [fe_tu_us(tx["ticks"]) for tx in txs]
    F = np.concatenate([window_features(tx["ticks"], tx["ch"], tu, W) for tx, tu in zip(txs, tus)]).astype(np.float32)
    y = np.concatenate([np.asarray(tx["labels"], dtype=np.int64) for tx in txs])
    return F, y, hashlib.sha256(F.tobytes() + y.tobytes()).hexdigest(), len(txs)


def _features_one(seed):
    t0 = time.time()
    want = json.loads((RECORDS / "training" / "window_nets_grid.json").read_text())["corpora"][f"{ARM}_{seed}"]
    info = dict(train_seed=seed, corpus_seed=P.GRID_CORPUS_SEED0[ARM] + seed, widths={})
    for W in (W_RECORD,) + WIDTHS:
        F, y, sha, n = width_matrix(W, seed)
        info.update(transmissions=n, gaps=int(len(y)))
        if W == W_RECORD:
            info["w10_equals_exp211"] = sha == want["feature_sha256"]
        else:
            info["widths"][str(W)] = dict(feature_sha256=sha, n_features=int(F.shape[1]))
    info["seconds"] = round(time.time() - t0, 1)
    return info


def features():
    with get_context("fork").Pool(len(SEEDS)) as pool:
        infos = pool.map(_features_one, SEEDS)
    models = {}
    for i in infos:                     # each shipped model was trained on the matrix this code rebuilds
        for W in WIDTHS:
            for kind in ("mlp", "lif", "spk40"):
                p = width_model_path(kind, W, i["train_seed"]).with_suffix(".meta.json")
                if p.exists():
                    models[f"{kind}_w{W}_{ARM}_seed{i['train_seed']}"] = json.loads(p.read_text())["feature_sha256"] == i["widths"][str(W)]["feature_sha256"]
    return dict(experiment="exp215_window_width", widths=list(WIDTHS), record_width=W_RECORD,
                corpora={str(i["train_seed"]): i for i in infos}, w1_pass=all(i["w10_equals_exp211"] for i in infos),
                models_trained_on_these=models, models_pass=all(models.values()), finished=now())


# ---- fit and evaluate the new rows ----
def fit_text(text, rows_fn=new_rows, loader=None):
    tab = _set_tab(text, P.CALIB_SEED[text], loader)
    rows = [r for r in rows_fn() if r in tab["rows"]]
    with get_context("fork").Pool(min(WORKERS, len(rows))) as pool:
        got = dict(pool.imap_unordered(_fit_row, [(r, P.GAMMA_GRID) for r in rows]))
    return {r: got[r] for r in rows}


def eval_text(text, fits, rows_fn=new_rows, loader=None):
    dep = (P.DEFAULT_WEIGHTS["T"], P.DEFAULT_WEIGHTS["beta"])
    res = dict(record={}, fresh={})
    for kind, seeds in (("record", P.EVAL["seeds"]), ("fresh", P.FRESH_SEEDS)):
        for seed in seeds:
            tab = _set_tab(text, seed, loader)
            jobs = [(row, {"deployed": dep, "own_fit": (fits[row]["T"], fits[row]["beta"])})
                    for row in rows_fn() if row in tab["rows"] and row in fits]
            with get_context("fork").Pool(min(WORKERS, len(jobs))) as pool:
                got = dict(pool.imap_unordered(_eval_row, jobs))
            res[kind][str(seed)] = {r: got[r] for r, _ in jobs}
    return res


# ---- per-gap accuracy, the interval, the lookahead and the channel error (record corpora) ----
def corpus_with_labels(text, seed, ref):
    """The evaluation corpus regenerated with its gap labels, checked against the table it was scored from."""
    txs = corpus(P.EVAL["per_fist"], seed, text)
    same = (len(txs) == len(ref["ticks"])
            and all(np.array_equal(np.asarray(t["ticks"]), np.asarray(r)) for t, r in zip(txs, ref["ticks"]))
            and all(" ".join(t["words"]) == r for t, r in zip(txs, ref["truth"]))
            and all(t["fist"] == r for t, r in zip(txs, ref["fist"])))
    assert same, f"the regenerated corpus {text} {seed} differs from the table's"
    return txs


def row_accuracy(doc, rows, labels, fist_of_gap):
    acc = {}
    for r in rows:
        lp = np.concatenate([np.asarray(x, dtype=np.int64).reshape(-1, 3) for x in doc["rows"][r] if len(x)])
        assert len(lp) == len(labels), (r, len(lp), len(labels))
        ok = lp.argmax(1) == labels
        acc[r] = dict(overall=float(ok.mean()), per_fist={f: float(ok[fist_of_gap == f].mean()) for f in OPS})
    return acc


def _gapacc_families():
    FAM = families()
    new = {r.rsplit("_seed", 1)[0] for r in new_rows()}
    return {f: rs for f, rs in FAM.items() if f in FEEDFORWARD or f in new}


def _gapacc_corpus(job):
    text, seed = job
    doc = load(text, seed)
    txs = corpus_with_labels(text, seed, doc)
    labels = np.concatenate([np.asarray(tx["labels"], dtype=np.int64) for tx in txs])
    fist_of_gap = np.concatenate([[tx["fist"]] * len(tx["labels"]) for tx in txs])
    FAM = _gapacc_families()
    acc = row_accuracy(doc, [r for f in FAM for r in FAM[f] if r in doc["rows"]], labels, fist_of_gap)
    # lookahead: from the spike that closes gap i to the spike that closes gap i + W, per width and operator, in ms
    # and in units of the transmission's true mean time unit; the channel error: each spike's bit against the
    # ideal element of its letter
    look = {str(W): {f: [] for f in OPS} for W in ALL_W}
    look_tu = {str(W): {f: [] for f in OPS} for W in ALL_W}
    interval, interval_tu, cherr = {f: [] for f in OPS}, {f: [] for f in OPS}, {f: [0, 0] for f in OPS}
    for tx in txs:
        dt = np.diff(np.asarray(tx["ticks"], dtype=np.float64)) / 1000.0
        tu_ms = float(tx["true_tu_us"]) / 1000.0
        interval[tx["fist"]].append(dt); interval_tu[tx["fist"]].append(dt / tu_ms)
        cs = np.concatenate([[0.0], np.cumsum(dt)])
        for W in ALL_W:
            if len(dt) > W:
                la = cs[1 + W:len(dt) + 1] - cs[1:len(dt) + 1 - W]
                look[str(W)][tx["fist"]].append(la); look_tu[str(W)][tx["fist"]].append(la / tu_ms)
        ideal = np.array([1 if el == "-" else 0 for w in tx["words"] for c in w for el in MORSE_TABLE[c]], dtype=np.int64)
        assert len(ideal) == len(tx["ch"])
        cherr[tx["fist"]][0] += int(np.sum(ideal != np.asarray(tx["ch"]))); cherr[tx["fist"]][1] += int(len(ideal))

    def stat(v):
        return dict(mean=float(np.mean(v)), p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)), n=int(len(v)))
    return f"{text}_{seed}", dict(gaps=int(len(labels)), acc=acc,
                                  interval_ms={f: stat(np.concatenate(interval[f])) for f in OPS},
                                  interval_tu={f: stat(np.concatenate(interval_tu[f])) for f in OPS},
                                  lookahead_ms={W: {f: stat(np.concatenate(v[f])) for f in OPS} for W, v in look.items()},
                                  lookahead_tu={W: {f: stat(np.concatenate(v[f])) for f in OPS} for W, v in look_tu.items()},
                                  channel_error={f: dict(errors=e, elements=n, rate=e / n) for f, (e, n) in cherr.items()})


def family_accuracy(per, keys, FAM):
    fam = {}
    for f, rs in FAM.items():
        if not all(r in per[keys[0]]["acc"] for r in rs):
            continue
        seed_means = [np.mean([per[k]["acc"][r]["overall"] for k in keys]) for r in rs]
        fam[f] = dict(n=len(rs), mean=100 * float(np.mean(seed_means)), sd=100 * float(np.std(seed_means, ddof=1)),
                      per_fist={o: 100 * float(np.mean([per[k]["acc"][r]["per_fist"][o] for k in keys for r in rs])) for o in OPS})
    return fam


def gapacc():
    jobs = [(t, s) for t in P.TEXTS for s in P.EVAL["seeds"]]
    with get_context("fork").Pool(len(jobs)) as pool:
        per = dict(pool.imap_unordered(_gapacc_corpus, jobs))
    FAM = _gapacc_families()
    out = dict(protocol_hash=P.WIDTH_HASH, corpora=per, texts={})
    for text in P.TEXTS:
        keys = [f"{text}_{s}" for s in P.EVAL["seeds"]]
        allint = {o: float(np.mean([per[k]["interval_ms"][o]["mean"] for k in keys])) for o in OPS}

        def agg(field):
            lk = {W: {o: {q: float(np.mean([per[k][field][W][o][q] for k in keys])) for q in ("mean", "p5", "p95")} for o in OPS}
                  for W in map(str, ALL_W)}
            for W in lk:
                lk[W]["overall"] = dict(mean=float(np.mean([v["mean"] for o, v in lk[W].items() if o in OPS])),
                                        p5=float(np.min([v["p5"] for o, v in lk[W].items() if o in OPS])),
                                        p95=float(np.max([v["p95"] for o, v in lk[W].items() if o in OPS])))
            return lk
        int_tu = {o: {q: float(np.mean([per[k]["interval_tu"][o][q] for k in keys])) for q in ("mean", "p5", "p95")} for o in OPS}
        int_tu["overall"] = dict(mean=float(np.mean([v["mean"] for v in int_tu.values()])), p5=float(np.min([v["p5"] for v in int_tu.values()])),
                                 p95=float(np.max([v["p95"] for v in int_tu.values()])))
        ce = {o: dict(errors=sum(per[k]["channel_error"][o]["errors"] for k in keys),
                      elements=sum(per[k]["channel_error"][o]["elements"] for k in keys)) for o in OPS}
        for o in ce:
            ce[o]["rate"] = ce[o]["errors"] / ce[o]["elements"]
        ce["overall"] = dict(errors=sum(v["errors"] for v in ce.values()), elements=sum(v["elements"] for v in ce.values()))
        ce["overall"]["rate"] = ce["overall"]["errors"] / ce["overall"]["elements"]
        out["texts"][text] = dict(families=family_accuracy(per, keys, FAM), interval_ms=allint, interval_tu=int_tu,
                                  lookahead_ms=agg("lookahead_ms"), lookahead_tu=agg("lookahead_tu"), channel_error=ce)
    return out


# ---- statistics ----
def _pairs(FAM):
    p = [(mlp_family(W), "mlp_op12") for W in WIDTHS]                                     # Q1 the width
    p += [(family("lif", W), "lif_op12") for W in WIDTHS]                                  # Q3 (scored by width_lif)
    p += [("bigru_fist_fp", "bigru_fist"), ("bigru_raw", "bigru_fist_fp"), ("bigru_raw", "bigru_fist"),   # B1, Q2
          ("bigru_raw", "mlp_op12"), ("mlp_op12", "bigru_raw")]
    p += [(mlp_family(W), "bigru_fist") for W in WIDTHS] + [(mlp_family(W), "lif_op12") for W in WIDTHS]
    return [(a, b) for a, b in p if a in FAM and b in FAM and a != b]


def family_block(M, FAM, show):
    return {f: dict(n=len(rs), cer=float(np.mean([M[r]["cer"] for r in rs])), wer=float(np.mean([M[r]["wer"] for r in rs])),
                    cer_sd=float(np.std([M[r]["cer"] for r in rs], ddof=1)), wer_sd=float(np.std([M[r]["wer"] for r in rs], ddof=1)),
                    per_fist={o: {m: float(np.mean([M[r]["per_fist"][o][m] for r in rs])) for m in ("cer", "wer")} for o in OPS})
            for f, rs in FAM.items() if f in show}


def matched_record():
    m_eval = json.loads((MATCHED / "eval.json").read_text()); m_fit = json.loads((MATCHED / "fit.json").read_text())
    m_res = json.loads((MATCHED / "results.json").read_text())
    assert m_eval["protocol_hash"] == m_fit["protocol_hash"] == m_res["protocol_hash"] == P.MATCHED_HASH
    return m_eval, m_fit, m_res


def stats(fits, ev, chk, gap):
    m_eval, m_fit, m_res = matched_record()
    FAM = families()
    NEW = sorted({r.rsplit("_seed", 1)[0] for r in new_rows()} & set(FAM))
    SHOW = [f for f in FAM if f in NEW + list(REF)]
    pairs = _pairs(FAM)
    out = dict(experiment="exp215_window_width", protocol_hash=P.WIDTH_HASH, protocol=P.WIDTH, families={f: FAM[f] for f in SHOW}, texts={})
    c5 = 0.0
    for text in P.TEXTS:
        out["texts"][text] = {}
        for kind in ("record", "fresh"):
            merged = {s: {**m_eval["eval"][text][kind][s], **ev[text][kind].get(s, {})} for s in m_eval["eval"][text][kind]}
            B = {}
            for cond in ("own_fit", "deployed"):
                M = row_means(merged, cond)
                fam = family_block(M, FAM, SHOW)
                cmp = {f"{a}_vs_{b}": _cmp(M, FAM, a, b) for a, b in pairs}
                B[cond] = dict(rows={r: {k: M[r][k] for k in ("cer", "wer", "exact")} for r in M if r.rsplit("_seed", 1)[0] in NEW},
                               families=fam, comparisons=cmp)
                for f in REF:                                        # C5: the reference families equal the matched record's
                    v = m_res["grid"]["texts"][text][kind][cond]["families"][f]
                    c5 = max(c5, abs(fam[f]["cer"] - v["cer"]), abs(fam[f]["wer"] - v["wer"]))
            out["texts"][text][kind] = B
        allf = {**{r: (v["T"], v["beta"]) for r, v in m_fit["fits"][text].items()},
                **{r: (v["T"], v["beta"]) for r, v in fits[text].items()}}
        out["texts"][text]["fits"] = {f: [list(allf[r]) for r in FAM[f]] for f in SHOW}
    X = out["texts"]["eng"]["record"]["own_fit"]["families"]
    b1 = dict(cer_pp=X["bigru_fist_fp"]["cer"] - X["bigru_fist"]["cer"], wer_pp=X["bigru_fist_fp"]["wer"] - X["bigru_fist"]["wer"])
    b1["within_bound"] = abs(b1["cer_pp"]) <= P.WIDTH["anchors"]["b1_float_vs_fixed_pp"]
    out["anchors"] = dict(c3_pass=chk["c3_pass"], c4=chk["c4"], w1_pass=chk["w1_pass"], b2_pass=chk["b2_pass"],
                          corpus_identity=chk["corpus_identity"], c5=dict(max_abs=c5, passed=c5 < 1e-9), b1_float_vs_fixed=b1)
    out["anchors_pass"] = bool(chk["c3_pass"] and chk["c4"]["c4_disjoint"] and chk["w1_pass"] and chk["corpus_identity"]
                               and chk["b2_pass"] and c5 < 1e-9)
    out["gapacc"] = gap["texts"]
    lab = {}
    for text in ("eng", "rand"):
        for kind in ("record", "fresh"):
            C = out["texts"][text][kind]["own_fit"]["comparisons"]; L = {}
            for W in WIDTHS:
                k = f"{mlp_family(W)}_vs_mlp_op12"
                if k in C:
                    L[f"Q1 W={W}"] = _label_r1(C[k])
                k = f"{family('lif', W)}_vs_lif_op12"
                if k in C:
                    L[f"Q3 W={W}"] = _label_r1(C[k])
            for q, k in (("Q2 raw vs fp", "bigru_raw_vs_bigru_fist_fp"), ("B1 fp vs fixed", "bigru_fist_fp_vs_bigru_fist")):
                if k in C:
                    L[q] = _label_r1(C[k])
            if "bigru_raw_vs_mlp_op12" in C:
                L["Q2 raw vs MLP"] = _label_lead(C["bigru_raw_vs_mlp_op12"])
            for W in WIDTHS:
                k = f"{mlp_family(W)}_vs_bigru_fist"
                if k in C:
                    L[f"E W={W} vs BiGRU"] = _label_lead(C[k])
            lab[f"{text}_{kind}"] = L
    out["labels"] = lab
    # the narrowest width that is NO EFFECT against the paper's on both texts (own fit, record seeds)
    ok = {W: all(lab[f"{t}_record"].get(f"Q1 W={W}", "NO EFFECT" if W == W_RECORD else "") == "NO EFFECT" for t in ("eng", "rand"))
          for W in ALL_W if W == W_RECORD or f"Q1 W={W}" in lab["eng_record"]}
    out["w_star"] = dict(no_effect_widths=[W for W, v in ok.items() if v], w_star=min([W for W, v in ok.items() if v] or [W_RECORD]),
                         widths_scored=sorted(ok), wider_helps=[W for W in WIDTHS if W > W_RECORD and lab["eng_record"].get(f"Q1 W={W}") == "HELPS"],
                         conditional_lif_arm=min([W for W, v in ok.items() if v] or [W_RECORD]) < W_RECORD)
    E = lab["eng_record"]
    out["escalate"] = dict(mlp_beats_control=any(v == "HELPS" for k, v in E.items() if k.startswith("Q1 ")),
                           new_family_leads_bigru=any(v in ("LEADS", "LEADS ON ONE") for k, v in E.items() if k.startswith("E ")),
                           raw_bigru_vs_record_bigru=E.get("Q2 raw vs fp"))
    out["escalate"]["any"] = bool(out["escalate"]["mlp_beats_control"] or out["escalate"]["new_family_leads_bigru"]
                                  or out["escalate"]["raw_bigru_vs_record_bigru"] == "HELPS")
    walls, params = {}, {}
    for r, (k, p) in new_rows().items():
        if k == "window":
            m = json.loads(p.with_suffix(".meta.json").read_text())
            walls.setdefault(r.rsplit("_seed", 1)[0], []).append(m["wall_s"]); params[r.rsplit("_seed", 1)[0]] = m["n_params"]
    out["train_wall_s"] = {k: dict(n=len(v), min=min(v), max=max(v)) for k, v in walls.items()}
    out["n_params"] = params
    R = json.loads(RAW_TRAINING.read_text())
    out["bigru_raw_train"] = dict(alive=R["alive"], select_balacc=R["select_balacc"], n_params=R["recipe"]["n_params"],
                                  n_params_gap9=R["recipe"]["n_params_gap9"], wall_s=R["wall_s"])
    out["git"] = git_info(); out["env"] = env_info(); out["finished"] = now()
    return out


def _dump(obj, name, indent=1, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(obj, indent=indent))


def _load(name, out=OUT):
    return json.loads((out / name).read_text())


def main(step="all"):
    t0 = time.time()
    if step in ("tables", "all"):
        build_tables()
        print(f"  tables ({time.time() - t0:.0f}s)", flush=True)
    if step in ("features", "all"):
        R = features()
        _dump(R, "features.json")
        assert R["w1_pass"] and R["models_pass"], "a feature matrix is not the one a shipped model was trained on"
        print(f"  features: W1 and the models' matrices pass ({time.time() - t0:.0f}s)", flush=True)
    if step in ("check", "all"):
        _dump(check(), "check.json")
        print(f"  check: C3, C4, B2 and the corpus identity pass ({time.time() - t0:.0f}s)", flush=True)
    if step in ("fit", "all"):
        _dump(dict(protocol_hash=P.WIDTH_HASH, fits={t: fit_text(t) for t in P.TEXTS}), "fit.json")
        print(f"  fit ({time.time() - t0:.0f}s)", flush=True)
    if step in ("eval", "all"):
        F = _load("fit.json")["fits"]
        _dump(dict(protocol_hash=P.WIDTH_HASH, eval={t: eval_text(t, F[t]) for t in P.TEXTS}), "eval.json", None)
        print(f"  eval ({time.time() - t0:.0f}s)", flush=True)
    if step in ("gapacc", "all"):
        _dump(gapacc(), "gapacc.json")
        print(f"  per-gap accuracy ({time.time() - t0:.0f}s)", flush=True)
    if step in ("stats", "all"):
        R = stats(_load("fit.json")["fits"], _load("eval.json")["eval"], _load("check.json"), _load("gapacc.json"))
        _dump(R, "results.json")
        assert R["anchors_pass"] and R["anchors"]["b1_float_vs_fixed"]["within_bound"], "the width record's anchors failed"
        print(f"  anchors: {R['anchors']}", flush=True)
    if step in ("manifest", "all"):
        files = ["check.json", "features.json", "fit.json", "eval.json", "gapacc.json", "results.json"]
        _dump(dict(experiment="exp215_window_width", protocol_hash=P.WIDTH_HASH, base_protocol_hash=P.GRID_WINDOW_HASH,
                   matched_protocol_hash=P.MATCHED_HASH, created=now(), git=git_info(), env=env_info(),
                   files={f: sha256_file(OUT / f) for f in files}), "manifest.json")
    print(f"records -> {OUT}")


if __name__ == "__main__":
    main((sys.argv[1:] or ["all"])[0])
