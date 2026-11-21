"""The spiking networks at the width sweep's best width (protocol/width_lif.json; Section 7.2, Table 4).

    python -m silences.evaluation.width_lif [tables|check|fit|eval|gapacc|stats|manifest|all]

The paper's spike-fed network and its current-fed form are retrained with the window halved to 5 gaps either
side (spk40_w5_op12, lif_w5_op12; ten seeds each on the matched op12 corpora, the LIF recipe unchanged) and
scored by the same protocol beside their own forms at the paper's width, the control at both widths and the
baseline. Rows of the matched, spike-fed and width records are read from their tables and records.

Records written to records/width_lif/:
    check.json      F1: every model was trained on the width record's half-width-5 matrix of its seed;
                    C3: the harness reproduces the spike-fed record's spk40_op12_seed0 row; C4: no training
                    transmission is in any evaluation corpus; C2 (reported): each model's inference forward
                    against its training-side forward
    fit.json, eval.json   as in the width record, for the twenty new rows
    gapacc.json     per-gap accuracy of every feedforward row on the record corpora
    results.json    the record: seed means, the comparisons, the pre-registered labels, anchors
The research repository's records of the same run ship under records/width_lif/research/.
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

from .. import OUTPUTS, RECORDS   # noqa: E402
from ..segmenters.lif import LIF, SpikeFedLIF   # noqa: E402
from . import protocol as P, tables, width   # noqa: E402
from .decoder_weights import row_means   # noqa: E402
from .matched import _cmp, _label_lead, _label_r1   # noqa: E402
from .producers import DEPLOYED, SEEDS, spike_fed_family_rows   # noqa: E402
from .provenance import env_info, git_info, now, sha256_file   # noqa: E402
from .spike_fed import _s1   # noqa: E402

OUT = RECORDS / "width_lif"
TABLES = OUTPUTS / "tables_width_lif"
OPS = P.OPERATORS
WORKERS = width.WORKERS
W = P.WIDTH_LIF["W"]                                        # 5
FAMILY = P.WIDTH_LIF["families"]                            # spk -> spk40_w5_op12, lif -> lif_w5_op12
_LOAD = dict(spk=SpikeFedLIF.load, lif=LIF.load)
_KIND = dict(spk="spk40", lif="lif")
SPK, LIF5, MLP5 = FAMILY["spk"], FAMILY["lif"], width.mlp_family(W)
NEW = [SPK, LIF5]
REF = ("spk40_op12", "lif_op12", "mlp_op12", "bigru_fist", MLP5)
PAIRS = [(SPK, "spk40_op12"), (LIF5, "lif_op12"),                        # Q1, Q3: the width's transfer
         (SPK, MLP5), (LIF5, MLP5), (SPK, LIF5),                         # Q2: the ladder at half-width 5
         (SPK, "bigru_fist"), (LIF5, "bigru_fist"),
         (SPK, "mlp_op12"), (SPK, "lif_op12"), (MLP5, "spk40_op12")]


def new_rows():
    """{row: (kind, model file)} for every half-width-5 spiking model on disk."""
    out = {}
    for kind in ("spk", "lif"):
        for s in SEEDS:
            p = width.width_model_path(_KIND[kind], W, s)
            if p.exists() and p.with_suffix(".meta.json").exists():
                out[f"{FAMILY[kind]}_seed{s}"] = (kind, p)
    return out


def families():
    F = dict(width.families())
    F.update(spike_fed_family_rows())
    for r in new_rows():
        F.setdefault(r.rsplit("_seed", 1)[0], []).append(r)
    return {f: rs for f, rs in F.items() if len(rs) >= 2}


# ---- tables: the width set with the spike-fed record's rows and the new rows ----
def table_path(text, seed):
    return TABLES / f"{text}_{seed}.pkl"


def _base(text, seed):
    T15 = width.load(text, seed); T13 = tables.load(text, seed, matched="spike_fed")
    assert T15["truth"] == T13["truth"] and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(T15["ticks"], T13["ticks"]))
    rows = dict(T15["rows"])
    rows.update({r: T13["rows"][r] for rs in spike_fed_family_rows().values() for r in rs if r in T13["rows"]})
    return dict(T15, rows=rows)


def _make_table(job):
    text, kind, seed = job
    t0 = time.time()
    added, total = width.extend_table(table_path(text, seed), lambda: _base(text, seed), P.WIDTH_LIF_HASH, new_rows,
                                      lambda k, p: _LOAD[k](p).lp3,
                                      lambda k, p: json.loads(p.with_suffix(".meta.json").read_text())["pickle_sha256"])
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
    assert doc["protocol_hash"] == P.WIDTH_LIF_HASH
    return doc


# ---- check ----
def check():
    feat = json.loads((width.OUT / "features.json").read_text())
    f1, c2 = {}, {}
    for r, (k, p) in new_rows().items():
        m = json.loads(p.with_suffix(".meta.json").read_text())
        f1[r] = m["feature_sha256"] == feat["corpora"][str(m["seed"])]["widths"][str(W)]["feature_sha256"]
        c2[r] = m["c2_forward"]["argmax_agreement"]
    prod = SpikeFedLIF.load(DEPLOYED["spk"]).lp3
    c3 = {}
    for text in P.TEXTS:
        seed = P.EVAL["seeds"][0]
        D = tables.load(text, seed, matched="spike_fed")
        mine = [prod(dict(ticks=t, ch=np.asarray(c, dtype=np.int64))) for t, c in zip(D["ticks"], D["ch"])]
        c3[f"{text}_{seed}"] = dict(transmissions=len(mine), equal=mine == D["rows"]["spk40_op12_seed0"])
    c4 = width.disjointness(width.training_fingerprints())
    out = dict(protocol_hash=P.WIDTH_LIF_HASH, f1=f1, f1_pass=all(f1.values()) and bool(f1), c3=c3,
               c3_pass=all(v["equal"] for v in c3.values()), c4=c4,
               c2_min_argmax_agreement=min(c2.values()) if c2 else None, c2=c2)
    assert out["f1_pass"] and out["c3_pass"] and c4["c4_disjoint"]
    return out


# ---- per-gap accuracy (record corpora) ----
def _gapacc_families():
    FAM = families()
    keep = set(width._gapacc_families()) | set(spike_fed_family_rows()) | set(NEW)
    return {f: rs for f, rs in FAM.items() if f in keep}


def _gapacc_corpus(job):
    text, seed = job
    doc = load(text, seed)
    txs = width.corpus_with_labels(text, seed, doc)
    labels = np.concatenate([np.asarray(tx["labels"], dtype=np.int64) for tx in txs])
    fist_of_gap = np.concatenate([[tx["fist"]] * len(tx["labels"]) for tx in txs])
    FAM = _gapacc_families()
    acc = width.row_accuracy(doc, [r for f in FAM for r in FAM[f] if r in doc["rows"]], labels, fist_of_gap)
    return f"{text}_{seed}", dict(gaps=int(len(labels)), acc=acc)


def gapacc():
    jobs = [(t, s) for t in P.TEXTS for s in P.EVAL["seeds"]]
    with get_context("fork").Pool(len(jobs)) as pool:
        per = dict(pool.imap_unordered(_gapacc_corpus, jobs))
    FAM = _gapacc_families()
    out = dict(protocol_hash=P.WIDTH_LIF_HASH, corpora=per, texts={})
    for text in P.TEXTS:
        out["texts"][text] = dict(families=width.family_accuracy(per, [f"{text}_{s}" for s in P.EVAL["seeds"]], FAM))
    return out


# ---- statistics ----
def _records():
    m_eval, m_fit, _ = width.matched_record()
    E = dict(matched=m_eval, spike_fed=json.loads((RECORDS / "spike_fed" / "eval.json").read_text()), width=width._load("eval.json"))
    F = dict(matched=m_fit, spike_fed=json.loads((RECORDS / "spike_fed" / "fit.json").read_text()), width=width._load("fit.json"))
    r13 = json.loads((RECORDS / "spike_fed" / "results.json").read_text()); r15 = width._load("results.json")
    assert E["spike_fed"]["protocol_hash"] == F["spike_fed"]["protocol_hash"] == r13["protocol_hash"] == P.SPIKE_FED_HASH
    assert E["width"]["protocol_hash"] == F["width"]["protocol_hash"] == r15["protocol_hash"] == P.WIDTH_HASH
    return E, F, r13, r15


def stats(fits, ev, chk, gap):
    E, F, r13, r15 = _records()
    FAM = families()
    SHOW = [f for f in FAM if f in NEW + list(REF)]
    pairs = [(a, b) for a, b in PAIRS if a in FAM and b in FAM]
    out = dict(experiment="exp216_lif_width", protocol_hash=P.WIDTH_LIF_HASH, protocol=P.WIDTH_LIF, families={f: FAM[f] for f in SHOW}, texts={})
    c5 = 0.0
    for text in P.TEXTS:
        out["texts"][text] = {}
        for kind in ("record", "fresh"):
            merged = {s: {**E["matched"]["eval"][text][kind][s], **E["spike_fed"]["eval"][text][kind].get(s, {}),
                          **E["width"]["eval"][text][kind].get(s, {}), **ev[text][kind].get(s, {})} for s in E["matched"]["eval"][text][kind]}
            B = {}
            for cond in ("own_fit", "deployed"):
                M = row_means(merged, cond)
                fam = width.family_block(M, FAM, SHOW)
                cmp = {f"{a}_vs_{b}": _cmp(M, FAM, a, b) for a, b in pairs}
                B[cond] = dict(rows={r: {k: M[r][k] for k in ("cer", "wer", "exact")} for r in M if r.rsplit("_seed", 1)[0] in NEW},
                               families=fam, comparisons=cmp)
                for f, src in (("spk40_op12", r13), ("lif_op12", r13), ("mlp_op12", r13), ("bigru_fist", r13), (MLP5, r15)):
                    v = src["texts"][text][kind][cond]["families"][f]      # C5: the reference families equal their records'
                    c5 = max(c5, abs(fam[f]["cer"] - v["cer"]), abs(fam[f]["wer"] - v["wer"]))
            out["texts"][text][kind] = B
        allf = {}
        for name in ("matched", "spike_fed", "width"):
            allf.update({r: (v["T"], v["beta"]) for r, v in F[name]["fits"][text].items()})
        allf.update({r: (v["T"], v["beta"]) for r, v in fits[text].items()})
        out["texts"][text]["fits"] = {f: [list(allf[r]) for r in FAM[f]] for f in SHOW}
    out["anchors"] = dict(f1_pass=chk["f1_pass"], c3_pass=chk["c3_pass"], c4=chk["c4"], c2_min_argmax_agreement=chk["c2_min_argmax_agreement"],
                          c5=dict(max_abs=c5, passed=c5 < 1e-9))
    out["anchors_pass"] = bool(chk["f1_pass"] and chk["c3_pass"] and chk["c4"]["c4_disjoint"] and c5 < 1e-9)
    out["gapacc"] = gap["texts"]
    lab = {}
    for text in ("eng", "rand"):
        for kind in ("record", "fresh"):
            C = out["texts"][text][kind]["own_fit"]["comparisons"]; L = {}
            if f"{SPK}_vs_spk40_op12" in C:
                L["Q1 spk w5 vs w10"] = _label_r1(C[f"{SPK}_vs_spk40_op12"])
            if f"{LIF5}_vs_lif_op12" in C:
                L["Q3 lif w5 vs w10"] = _label_r1(C[f"{LIF5}_vs_lif_op12"])
            for q, k in (("Q2 spk w5 vs mlp w5", f"{SPK}_vs_{MLP5}"), ("Q2 lif w5 vs mlp w5", f"{LIF5}_vs_{MLP5}"), ("Q2 spk w5 vs lif w5", f"{SPK}_vs_{LIF5}")):
                if k in C:
                    L[q] = _s1(C[k])
            for q, k in (("E spk w5 vs BiGRU", f"{SPK}_vs_bigru_fist"), ("E lif w5 vs BiGRU", f"{LIF5}_vs_bigru_fist")):
                if k in C:
                    L[q] = _label_lead(C[k])
            lab[f"{text}_{kind}"] = L
    out["labels"] = lab
    Eng = lab["eng_record"]
    own = out["texts"]["eng"]["record"]["own_fit"]["comparisons"]
    mlp_gain = r15["texts"]["eng"]["record"]["own_fit"]["comparisons"][f"{MLP5}_vs_mlp_op12"]["cer"]["d"]
    spk_gain = own.get(f"{SPK}_vs_spk40_op12", {}).get("cer", {}).get("d")
    out["transfer"] = dict(mlp_gain_cer_pp=mlp_gain, spk_gain_cer_pp=spk_gain,
                           lif_gain_cer_pp=own.get(f"{LIF5}_vs_lif_op12", {}).get("cer", {}).get("d"))
    out["escalate"] = dict(spk_leads_bigru=Eng.get("E spk w5 vs BiGRU") in ("LEADS", "LEADS ON ONE"),
                           spk_gain_exceeds_mlp=bool(spk_gain is not None and spk_gain > mlp_gain and Eng.get("Q1 spk w5 vs w10") == "HELPS"))
    out["escalate"]["any"] = bool(out["escalate"]["spk_leads_bigru"] or out["escalate"]["spk_gain_exceeds_mlp"])
    walls = {}
    for r, (k, p) in new_rows().items():
        m = json.loads(p.with_suffix(".meta.json").read_text())
        walls.setdefault(FAMILY[k], []).append(m["wall_s"]); out.setdefault("n_params", {})[FAMILY[k]] = m["n_params"]
    out["train_wall_s"] = {k: dict(n=len(v), min=min(v), max=max(v)) for k, v in walls.items()}
    out["git"] = git_info(); out["env"] = env_info(); out["finished"] = now()
    return out


def main(step="all"):
    t0 = time.time()
    dump = lambda obj, name, indent=1: width._dump(obj, name, indent, OUT)      # noqa: E731
    get = lambda name: width._load(name, OUT)                                    # noqa: E731
    if step in ("tables", "all"):
        build_tables()
        print(f"  tables ({time.time() - t0:.0f}s)", flush=True)
    if step in ("check", "all"):
        dump(check(), "check.json")
        print(f"  check: F1, C3 and C4 pass ({time.time() - t0:.0f}s)", flush=True)
    if step in ("fit", "all"):
        dump(dict(protocol_hash=P.WIDTH_LIF_HASH, fits={t: width.fit_text(t, new_rows, load) for t in P.TEXTS}), "fit.json")
        print(f"  fit ({time.time() - t0:.0f}s)", flush=True)
    if step in ("eval", "all"):
        Fi = get("fit.json")["fits"]
        dump(dict(protocol_hash=P.WIDTH_LIF_HASH, eval={t: width.eval_text(t, Fi[t], new_rows, load) for t in P.TEXTS}), "eval.json", None)
        print(f"  eval ({time.time() - t0:.0f}s)", flush=True)
    if step in ("gapacc", "all"):
        dump(gapacc(), "gapacc.json")
        print(f"  per-gap accuracy ({time.time() - t0:.0f}s)", flush=True)
    if step in ("stats", "all"):
        R = stats(get("fit.json")["fits"], get("eval.json")["eval"], get("check.json"), get("gapacc.json"))
        dump(R, "results.json")
        assert R["anchors_pass"], "the half-width-5 record's anchors failed"
        print(f"  anchors: {R['anchors']['c5']}, labels {R['labels']['eng_record']}", flush=True)
    if step in ("manifest", "all"):
        files = ["check.json", "fit.json", "eval.json", "gapacc.json", "results.json"]
        dump(dict(experiment="exp216_lif_width", protocol_hash=P.WIDTH_LIF_HASH, base_protocol_hash=P.WIDTH_HASH,
                  spike_fed_protocol_hash=P.SPIKE_FED_HASH, created=now(), git=git_info(), env=env_info(),
                  files={f: sha256_file(OUT / f) for f in files}), "manifest.json")
    print(f"records -> {OUT}")


if __name__ == "__main__":
    main((sys.argv[1:] or ["all"])[0])
