"""The per-word time-unit oracle on the trained segmenters (protocol/tempo.json; Sections 5.2 and 10.3).

    python -m silences.evaluation.tempo [check|tables|eval|legacy|parity|sensitivity|stats|manifest|all]

Every segmenter normalises its intervals by one time-unit estimate per transmission. Here each family of the
record (the spike-fed LIF, the current-fed LIF, the control, the baseline) is re-run on the record evaluation
corpora with the unit supplied four ways, its weights and its fitted decoder weights unchanged:

    single   the front end's one estimate per transmission (the record itself; reproduced exactly)
    local    one estimate per gap from the marks within three of it (timing.local_tu_us)
    stream   the same from the marks before it only
    oracle   the renderer's true unit of the word that opens the interval: the upper bound of any tracker

No model is trained. Records written to records/tempo/:
    check.json            A1: the regenerated corpora equal the tables'; A2: the per-gap path under "single"
                          reproduces the record's stored rows
    eval.json             per-gap accuracy and CER / WER of every (row, mode) on every record corpus
    legacy_readout.json   the earlier rule-based reader under the single, local and oracle unit (the 38% to 85%
                          of Section 5.2)
    tu_parity.json        the resonator estimate's spread against the front end's
    sensitivity.json      the deployed spike-fed network refitted under the oracle unit (English)
    results.json          the record: seed statistics per family and mode, paired differences, outcomes
The research repository's records of the same run ship under records/tempo/research/.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json          # noqa: E402
import pickle        # noqa: E402
import sys           # noqa: E402
import time          # noqa: E402
from multiprocessing import get_context   # noqa: E402

import numpy as np   # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view   # noqa: E402

from .. import OUTPUTS, RECORDS, timing   # noqa: E402
from ..corpus import corpus   # noqa: E402
from ..features import PAD_LOGX, fe_tu_us   # noqa: E402
from ..fixedpoint import q16   # noqa: E402
from ..fixedpoint.q16 import ONE, fxmul, grid_q, quantize_lp3   # noqa: E402
from ..segmenters import bigru, readout   # noqa: E402
from ..segmenters.lif import LIF, SpikeFedLIF   # noqa: E402
from ..segmenters.mlp import MLP   # noqa: E402
from . import protocol as P, tables   # noqa: E402
from .decoder_weights import _pack   # noqa: E402
from .metrics import edit_sums   # noqa: E402
from .producers import SEEDS, grid_model_path   # noqa: E402
from .provenance import env_info, git_info, now, sha256_file   # noqa: E402

OUT = RECORDS / "tempo"
TABLES = OUTPUTS / "tables_tempo"
WORKERS = min(tables.WORKERS, 12)
OPS = P.OPERATORS
MODES = tuple(P.TEMPO["modes"])                     # single, local, stream, oracle
HALF_WINDOW = P.TEMPO["half_window"]                # 3
FAMILIES = dict(spk40_op12=[f"spk40_op12_seed{s}" for s in SEEDS], lif_op12=[f"lif_op12_seed{s}" for s in SEEDS],
                mlp_op12=[f"mlp_op12_seed{s}" for s in SEEDS], bigru_fist=[f"bigru_fist_seed{s}" for s in bigru.POOL_SEEDS])
assert {f: len(rs) for f, rs in FAMILIES.items()} == P.TEMPO["families"]
ROWS = [r for rs in FAMILIES.values() for r in rs]
REGULAR = ("fast_normalr", "slow_normalr")
PRIMARY = P.TEMPO["primary"]
CRITERIA = P.TEMPO["criteria"]
_LOAD = dict(spk40_op12=("spk40", SpikeFedLIF.load), lif_op12=("lif", LIF.load), mlp_op12=("mlp", MLP.load))


def corpora():
    return [(t, s) for t in P.TEMPO["texts"] for s in P.TEMPO["corpora"]["seeds"]]


def family_of(row):
    return row.rsplit("_seed", 1)[0]


# ---- the models and their per-gap feature paths ----
def load_row(row):
    fam, s = row.rsplit("_seed", 1)
    if fam == "bigru_fist":
        return "bigru", [bigru.load_q16(int(s), bigru.arm_q16("fist"))]
    name, load = _LOAD[fam]
    return "window", load(grid_model_path(name, "op12", int(s)))


def window_features(ticks, ch, W, tu):
    """features.window_features with one time unit per gap (a constant array reproduces the scalar path exactly)."""
    ticks = np.asarray(ticks, dtype=np.float64)
    ch = np.asarray(ch, dtype=np.int64)
    x = np.diff(ticks) / np.asarray(tu, dtype=np.float64)
    logx = np.log(np.clip(x, 1e-3, None))
    bit = ch[1:].astype(np.float64)
    if len(x) == 0:
        return np.zeros((0, 2 * (2 * W + 1)))
    xp = np.concatenate([np.full(W, PAD_LOGX), logx, np.full(W, PAD_LOGX)])
    bp = np.concatenate([np.zeros(W), bit, np.zeros(W)])
    return np.concatenate([sliding_window_view(xp, 2 * W + 1), sliding_window_view(bp, 2 * W + 1)], axis=1)


def lp3_window(model, ticks, ch, tu):
    F = window_features(ticks, ch, model.W, tu)
    if len(F) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(quantize_lp3(model.log_posteriors(F)), dtype=np.int64)


def lp3_bigru(seeds, ticks, ch, tu):
    """segmenters.bigru.lp3_ticks with one integer time unit per gap."""
    ticks, ch = np.asarray(ticks, np.int64), np.asarray(ch, np.int64)
    dt = np.diff(ticks)
    if len(dt) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    isi_q = (dt * ONE) // np.asarray(tu, dtype=np.int64)
    chn = ch[1:]
    cols = [L.vec(isi_q) for L in bigru.POP]
    cols.append(np.where(chn != 0, ONE, 0).astype(np.int64))
    cols.append(q16.SIGMOID.vec(fxmul(isi_q - bigru.TAU_DUR_Q, np.int64(bigru.INV_GAP_S_Q))))
    tau = np.where(chn != 0, bigru.TAU_DASH_Q, bigru.TAU_DOT_Q).astype(np.int64)
    cols.append(q16.SIGMOID.vec(fxmul(isi_q - tau, np.int64(bigru.INV_GAP_S_Q))))
    feats = np.stack(cols, axis=1)
    psum = sum(S.posteriors_q(feats) for S in seeds)
    return np.asarray(q16.fxlog_vec(psum // len(seeds)), dtype=np.int64)


def tu_arrays(tx):
    """One time unit per gap under each mode (floats; the baseline's path rounds them)."""
    ticks = np.asarray(tx["ticks"])
    n = max(len(ticks) - 1, 0)
    return dict(single=np.full(n, float(fe_tu_us(ticks))),
                local=np.asarray(timing.local_tu_us(ticks, half_window=HALF_WINDOW), dtype=np.float64),
                stream=np.asarray(timing.local_tu_us(ticks, half_window=HALF_WINDOW, causal=True), dtype=np.float64),
                oracle=np.asarray(tx["tu_words_us"], dtype=np.float64)[np.asarray(tx["widx"])[:-1]])


def row_lp3(kind, model, tx, tu):
    if kind == "window":
        return lp3_window(model, tx["ticks"], tx["ch"], tu)
    return lp3_bigru(model, tx["ticks"], tx["ch"], np.rint(tu).astype(np.int64))


def record_row(text, seed, row):
    """The record's stored rows of one model on one record corpus, as one (gaps, 3) array."""
    doc = tables.load(text, seed, matched="spike_fed")
    return np.concatenate([np.asarray(r, dtype=np.int64).reshape(-1, 3) for r in doc["rows"][row]], axis=0)


def make_corpus(text, seed):
    txs = corpus(P.TEMPO["corpora"]["per_fist"], seed, text)
    ref = tables.load(text, seed, matched="spike_fed")
    same = (len(txs) == len(ref["ticks"])
            and all(np.array_equal(np.asarray(t["ticks"]), np.asarray(r)) for t, r in zip(txs, ref["ticks"]))
            and all(" ".join(t["words"]) == r for t, r in zip(txs, ref["truth"]))
            and all(t["fist"] == r for t, r in zip(txs, ref["fist"])))
    assert same, f"A1: the regenerated corpus {text} {seed} differs from the spike-fed table's"
    return txs


# ---- check: A1 and A2 on one seed of each family ----
def check():
    out = dict(protocol_hash=P.TEMPO_HASH, a1={}, a2={}, timing_s={})
    for text, seed in corpora():
        t0 = time.time()
        txs = make_corpus(text, seed)
        out["a1"][f"{text}_{seed}"] = dict(transmissions=len(txs), equal=True, seconds=round(time.time() - t0, 1))
    text, seed = "eng", P.TEMPO["corpora"]["seeds"][0]
    txs = make_corpus(text, seed)
    t0 = time.time()
    tus = [tu_arrays(tx) for tx in txs[:100]]
    out["timing_s"]["time_units_per_100_tx"] = round(time.time() - t0, 2)
    for fam, rows in FAMILIES.items():
        row = rows[0]
        kind, model = load_row(row)
        t0 = time.time()
        mine = np.concatenate([row_lp3(kind, model, tx, tu["single"]) for tx, tu in zip(txs[:100], tus)], axis=0)
        ref = record_row(text, seed, row)[:len(mine)]
        out["a2"][row] = dict(gaps=int(len(mine)), equal=bool(np.array_equal(mine, ref)), seconds=round(time.time() - t0, 1))
    out["a1_pass"] = all(v["equal"] for v in out["a1"].values())
    out["a2_pass"] = all(v["equal"] for v in out["a2"].values())
    assert out["a1_pass"] and out["a2_pass"], out["a2"]
    return out


# ---- tables: every record row under every mode, on the eight record corpora ----
def table_path(text, seed):
    return TABLES / f"{text}_{seed}.pkl"


def _make_table(job):
    text, seed = job
    t0 = time.time()
    path = table_path(text, seed)
    if path.exists():
        with open(path, "rb") as fh:
            doc = pickle.load(fh)
        assert doc["protocol_hash"] == P.TEMPO_HASH
    else:
        txs = make_corpus(text, seed)
        tus = [tu_arrays(tx) for tx in txs]
        doc = dict(protocol_hash=P.TEMPO_HASH, text=text, seed=seed, ticks=[tx["ticks"] for tx in txs], ch=[tx["ch"] for tx in txs],
                   truth=[" ".join(tx["words"]) for tx in txs], fist=[tx["fist"] for tx in txs],
                   labels=np.concatenate([np.asarray(tx["labels"], dtype=np.int64) for tx in txs]),
                   n_gaps=np.array([len(tx["labels"]) for tx in txs], dtype=np.int64),
                   tu={m: [t[m] for t in tus] for m in MODES}, rows={m: {} for m in MODES}, a2={})
    todo = [r for r in ROWS if r not in doc["rows"]["oracle"]]
    txs = [dict(ticks=t, ch=c) for t, c in zip(doc["ticks"], doc["ch"])]
    for row in todo:
        kind, model = load_row(row)
        for m in MODES:
            doc["rows"][m][row] = np.concatenate([row_lp3(kind, model, tx, tu) for tx, tu in zip(txs, doc["tu"][m])], axis=0)
        assert len(doc["rows"]["single"][row]) == int(doc["n_gaps"].sum())
        doc["a2"][row] = bool(np.array_equal(doc["rows"]["single"][row], record_row(text, seed, row)))
        assert doc["a2"][row], f"A2: {row} on {text} {seed} does not reproduce the record's row"
    if todo:
        with open(str(path) + ".tmp", "wb") as fh:
            pickle.dump(doc, fh)
        os.replace(str(path) + ".tmp", path)
    return f"{text}_{seed}", len(todo), round(time.time() - t0)


def build_tables():
    TABLES.mkdir(parents=True, exist_ok=True)
    jobs = corpora()
    with get_context("fork").Pool(min(WORKERS, len(jobs))) as pool:
        for name, added, sec in pool.imap_unordered(_make_table, jobs):
            print(f"  {name}: +{added} rows, A2 held ({sec}s)", flush=True)


def load(text, seed):
    with open(table_path(text, seed), "rb") as fh:
        doc = pickle.load(fh)
    assert doc["protocol_hash"] == P.TEMPO_HASH
    return doc


# ---- evaluate ----
def own_fits():
    f13 = json.loads((RECORDS / "spike_fed" / "fit.json").read_text()); f12 = json.loads((RECORDS / "matched" / "fit.json").read_text())
    assert f13["protocol_hash"] == P.SPIKE_FED_HASH and f12["protocol_hash"] == P.MATCHED_HASH
    out = {}
    for text in P.TEMPO["texts"]:
        out[text] = {}
        for row in ROWS:
            f = (f13 if family_of(row) == "spk40_op12" else f12)["fits"][text][row]
            out[text][row] = (f["T"], f["beta"])
    return out


_DOC = None


def decode_sums(doc, lp, T, gamma):
    """Edit sums overall and per operator for one (gaps, 3) array under the record's decoding."""
    rows, off = [], 0
    for n in doc["n_gaps"]:
        rows.append([[int(v) for v in r] for r in lp[off:off + int(n)]])
        off += int(n)
    return edit_sums(dict(ch=doc["ch"], truth=doc["truth"], fist=doc["fist"], rows={"x": rows}), "x", grid_q(T), grid_q(gamma), per_fist=True)


def gap_accuracy(doc, lp):
    ok = lp.argmax(axis=1) == doc["labels"]
    fist_of_gap = np.repeat(np.asarray(doc["fist"]), doc["n_gaps"])
    out = dict(overall=float(ok.mean()), n=int(len(ok)), per_fist={})
    for f in OPS:
        m = fist_of_gap == f
        out["per_fist"][f] = dict(acc=float(ok[m].mean()), n=int(m.sum()))
    return out


def _eval_job(args):
    row, mode, (T, gamma) = args
    lp = _DOC["rows"][mode][row]
    tot, pf = decode_sums(_DOC, lp, T, gamma)
    return (row, mode), dict(gap=gap_accuracy(_DOC, lp), decode=dict(overall=_pack(tot), per_fist={f: _pack(v) for f, v in pf.items()}),
                             weights=dict(T=T, beta=gamma))


def evaluate():
    global _DOC
    fits = own_fits()
    res = {}
    for text, seed in corpora():
        t0 = time.time()
        _DOC = load(text, seed)
        jobs = [(row, m, fits[text][row]) for row in ROWS for m in MODES]
        done = res.setdefault(text, {}).setdefault(str(seed), {})
        with get_context("fork").Pool(WORKERS) as pool:
            for (row, m), r in pool.imap_unordered(_eval_job, jobs, chunksize=4):
                done[f"{row}|{m}"] = r
        print(f"  eval {text} {seed}: {len(jobs)} (row, mode) cells ({time.time() - t0:.0f}s)", flush=True)
        _DOC = None
    return res


# ---- the earlier rule-based reader (the figures Section 5.2 cites) ----
def legacy():
    p = readout.Params((3.0, 6.0), (5.1, 8.2), k=6.0)
    out = dict(source="the earlier rule-based reader: fixed thresholds, 120 transmissions per operator, seed 7",
               readout=dict(dur=(3.0, 6.0), dash=(5.1, 8.2), k=6.0), per_fist={})
    for f in OPS:
        txs = corpus(120, 7, "rand", operators=[f])
        a, n = {"single": 0, "local": 0, "oracle": 0}, 0
        for tx in txs:
            isi = np.diff(tx["ticks"].astype(float)); bit = tx["ch"][1:].astype(float); y = tx["labels"]
            for key, tu in (("single", timing.estimate_tu_us(tx["ticks"])), ("local", timing.local_tu_us(tx["ticks"], half_window=3)),
                            ("oracle", tx["tu_words_us"][tx["widx"][:-1]])):
                a[key] += int((readout.class_probs(isi / tu, bit, p).argmax(1) == y).sum())
            n += len(y)
        out["per_fist"][f] = {k: 100 * v / n for k, v in a.items()}
    c, h = out["per_fist"]["cootie_high_jitter"], out["per_fist"]["high_r_bug"]
    out["a3"] = dict(claimed=dict(cootie_single=38, cootie_oracle=85, high_r_oracle=68.5),
                     measured=dict(cootie_single=c["single"], cootie_oracle=c["oracle"], high_r_oracle=h["oracle"]))
    out["a3_pass"] = bool(abs(c["single"] - 38) < 1.5 and abs(c["oracle"] - 85) < 1.5 and abs(h["oracle"] - 68.5) < 1.5)
    return out


# ---- the resonator estimate's spread against the front end's ----
def parity(per_fist=150, seed=20260720, Q=4.0):
    def robust(v):
        v = np.asarray(v, float)
        m = np.median(v)
        return float(m), float(np.median(np.abs(v - m)) * 1.4826)
    byf = {}
    for tx in corpus(per_fist, seed, "rand"):
        byf.setdefault(tx["fist"], []).append(tx)
    res = {}
    for f, txs in byf.items():
        fe = [fe_tu_us(tx["ticks"]) / tx["true_tu_us"] for tx in txs]
        rf = [timing.estimate_tu_us(tx["ticks"], Q=Q) / tx["true_tu_us"] for tx in txs]
        (fem, fes), (rfm, rfs) = robust(fe), robust(rf)
        res[f] = dict(fe_med=fem, fe_spread=fes, raf_med=rfm, raf_spread=rfs, n=len(txs),
                      spread_ratio_fe_over_raf=(fes / rfs) if rfs else None)
    return dict(source="per_fist 150, seed 20260720, Q 4.0: robust spread (MAD x 1.4826) of estimate / true unit", per_fist=res)


# ---- sensitivity: the deployed spike-fed network refitted under the oracle unit (English) ----
def sensitivity():
    row, text = "spk40_op12_seed0", "eng"
    kind, model = load_row(row)
    seed = P.CALIB_SEED[text]
    txs = corpus(P.CALIB["per_fist"], seed, text)
    doc = dict(ch=[tx["ch"] for tx in txs], truth=[" ".join(tx["words"]) for tx in txs], fist=[tx["fist"] for tx in txs],
               n_gaps=np.array([len(tx["labels"]) for tx in txs], dtype=np.int64))
    lp = np.concatenate([row_lp3(kind, model, tx, tu_arrays(tx)["oracle"]) for tx in txs], axis=0)
    grid = {}
    for gamma in P.GAMMA_GRID:
        for T in P.T_GRID:
            grid[f"{T}|{gamma}"] = dict(T=T, beta=gamma, **_pack(decode_sums(doc, lp, T, gamma)[0]))
    best = min(grid.values(), key=lambda r: (round(r["cer"], 10), r["T"], r["beta"]))
    rec = {}
    for s in P.TEMPO["corpora"]["seeds"]:
        d = load(text, s)
        tot, pf = decode_sums(d, d["rows"]["oracle"][row], best["T"], best["beta"])
        rec[str(s)] = dict(overall=_pack(tot), per_fist={f: _pack(v) for f, v in pf.items()})
    own = own_fits()[text][row]
    return dict(row=row, text=text, calib_seed=seed, own_fit=dict(T=own[0], beta=own[1]), oracle_fit=dict(T=best["T"], beta=best["beta"]),
                grid=grid, record_corpora_at_oracle_fit=rec,
                mean_cer_at_oracle_fit=float(np.mean([v["overall"]["cer"] for v in rec.values()])),
                mean_wer_at_oracle_fit=float(np.mean([v["overall"]["wer"] for v in rec.values()])))


# ---- statistics ----
def _paired(x, y):
    """x - y over seeds: mean, sd, 95% t interval, the number positive."""
    from scipy import stats as st
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = len(d)
    h = st.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return dict(mean=float(d.mean()), sd=float(d.std(ddof=1)) if n > 1 else 0.0, lo=float(d.mean() - h), hi=float(d.mean() + h), n=int(n),
                positive=int((d > 0).sum()))


def _msd(v):
    v = np.asarray(v, dtype=float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)))


def stats(E, records):
    seeds = [str(s) for s in P.TEMPO["corpora"]["seeds"]]
    out = dict(experiment=P.TEMPO["experiment"], protocol_hash=P.TEMPO_HASH, protocol=P.TEMPO, texts={})
    for text in P.TEMPO["texts"]:
        T = out["texts"][text] = {}
        for fam, rows in FAMILIES.items():
            F = T[fam] = dict(modes={}, vs_single={}, recovered={})
            per_seed = {}                 # mode -> metric -> operator or overall -> one value per training seed (mean over corpora)
            for m in MODES:
                per_seed[m] = {}
                for row in rows:
                    cells = [E[text][s][f"{row}|{m}"] for s in seeds]
                    vals = dict(gap_acc={"overall": 100 * np.mean([c["gap"]["overall"] for c in cells])},
                                cer={"overall": np.mean([c["decode"]["overall"]["cer"] for c in cells])},
                                wer={"overall": np.mean([c["decode"]["overall"]["wer"] for c in cells])})
                    for f in OPS:
                        vals["gap_acc"][f] = 100 * np.mean([c["gap"]["per_fist"][f]["acc"] for c in cells])
                        vals["cer"][f] = np.mean([c["decode"]["per_fist"][f]["cer"] for c in cells])
                        vals["wer"][f] = np.mean([c["decode"]["per_fist"][f]["wer"] for c in cells])
                    for met, d in vals.items():
                        for k, v in d.items():
                            per_seed[m].setdefault(met, {}).setdefault(k, []).append(float(v))
                F["modes"][m] = {met: {k: _msd(v) for k, v in d.items()} for met, d in per_seed[m].items()}
            for m in MODES[1:]:
                F["vs_single"][m] = {met: {k: _paired(per_seed[m][met][k], per_seed["single"][met][k]) for k in per_seed[m][met]}
                                     for met in per_seed[m]}
            for m in ("local", "stream"):
                F["recovered"][m] = {}
                for met in ("gap_acc", "cer"):
                    F["recovered"][m][met] = {}
                    for k in per_seed[m][met]:
                        g = np.asarray(per_seed["oracle"][met][k]) - np.asarray(per_seed["single"][met][k])
                        r = np.asarray(per_seed[m][met][k]) - np.asarray(per_seed["single"][met][k])
                        with np.errstate(divide="ignore", invalid="ignore"):
                            frac = np.where(np.abs(g) > 1e-9, r / g, np.nan)
                        frac = frac[np.isfinite(frac)]
                        F["recovered"][m][met][k] = _paired(frac, np.zeros_like(frac)) if len(frac) > 1 else dict(mean=float("nan"), n=int(len(frac)))
            F["seeds"] = rows
    p, c = PRIMARY, CRITERIA
    prim = out["texts"][p["text"]][p["family"]]
    h1 = prim["vs_single"]["oracle"]["gap_acc"][p["fist"]]
    h1_label = ("LARGE" if h1["mean"] >= c["h1"]["large_pp"] and h1["lo"] > 0 else
                "MODERATE" if h1["mean"] >= c["h1"]["moderate_pp"] and h1["lo"] > 0 else "SMALL")
    h2 = {m: prim["recovered"][m]["gap_acc"][p["fist"]] for m in ("local", "stream")}
    h2_label = {m: ("DROP" if h1_label == "SMALL" or not (v.get("lo", 0) > 0) else "REPORT") for m, v in h2.items()}
    orc = prim["modes"]["oracle"]["gap_acc"]
    h3_gap = float(np.mean([orc[f]["mean"] for f in REGULAR]) - orc["high_r_bug"]["mean"])
    par = records["tu_parity.json"]["per_fist"]
    h4_ratios = {f: par[f]["spread_ratio_fe_over_raf"] for f in REGULAR if f in par}
    h4_label = "KEEP" if h4_ratios and all(r is not None and r >= c["h4_min_ratio"] for r in h4_ratios.values()) else ("DROP" if h4_ratios else "NOT RUN")
    h5 = prim["vs_single"]["oracle"]["cer"]["overall"]
    h5_label = "KEEP" if (-h5["mean"]) > c["h5_between_family_cer_pp"] and (-h5["hi"]) > c["h5_between_family_cer_pp"] else "DROP"
    out["outcomes"] = dict(
        h1=dict(label=h1_label, gain_pp=h1, single=prim["modes"]["single"]["gap_acc"][p["fist"]], oracle=orc[p["fist"]]),
        h2=dict(label=h2_label, fraction_recovered=h2, gain_pp={m: prim["vs_single"][m]["gap_acc"][p["fist"]] for m in ("local", "stream")}),
        h3=dict(label="REPORT" if h3_gap >= c["h3_min_gap_pp"] else "DROP", regular_minus_high_r_pp=h3_gap, oracle_gap_acc={f: orc[f] for f in OPS}),
        h4=dict(label=h4_label, spread_ratio_fe_over_raf=h4_ratios),
        h5=dict(label=h5_label, oracle_cer_gain_pp=-h5["mean"], interval=[-h5["hi"], -h5["lo"]], threshold_pp=c["h5_between_family_cer_pp"]))
    out["records"] = records
    out.update(git=git_info(), env=env_info(), finished=now())
    return out


def _dump(obj, name, indent=1):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=indent))


def _load(name):
    return json.loads((OUT / name).read_text())


def main(step="all"):
    t0 = time.time()
    if step in ("check", "all"):
        _dump(check(), "check.json")
        print(f"  check: A1 and A2 pass ({time.time() - t0:.0f}s)", flush=True)
    if step in ("tables", "all"):
        build_tables()
        print(f"  tables ({time.time() - t0:.0f}s)", flush=True)
    if step in ("eval", "all"):
        _dump(dict(protocol_hash=P.TEMPO_HASH, eval=evaluate()), "eval.json", None)
        print(f"  eval ({time.time() - t0:.0f}s)", flush=True)
    if step in ("legacy", "all"):
        R = legacy()
        _dump(R, "legacy_readout.json")
        assert R["a3_pass"], R["a3"]
        print(f"  legacy readout: {R['a3']['measured']} ({time.time() - t0:.0f}s)", flush=True)
    if step in ("parity", "all"):
        _dump(parity(), "tu_parity.json")
    if step in ("sensitivity", "all"):
        _dump(sensitivity(), "sensitivity.json")
        print(f"  sensitivity ({time.time() - t0:.0f}s)", flush=True)
    if step in ("stats", "all"):
        ev = _load("eval.json")
        assert ev["protocol_hash"] == P.TEMPO_HASH
        R = stats(ev["eval"], {n: _load(n) for n in ("check.json", "legacy_readout.json", "tu_parity.json", "sensitivity.json")})
        _dump(R, "results.json")
        print("  outcomes: " + ", ".join(f"{k} {v['label']}" for k, v in R["outcomes"].items()), flush=True)
    if step in ("manifest", "all"):
        files = ["check.json", "eval.json", "legacy_readout.json", "tu_parity.json", "sensitivity.json", "results.json"]
        _dump(dict(experiment=P.TEMPO["experiment"], protocol_hash=P.TEMPO_HASH, base_protocol_hash=P.SPIKE_FED_HASH,
                   created=now(), git=git_info(), env=env_info(), files={f: sha256_file(OUT / f) for f in files}), "manifest.json")
    print(f"records -> {OUT}")


if __name__ == "__main__":
    main((sys.argv[1:] or ["all"])[0])
