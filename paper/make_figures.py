#!/usr/bin/env python3
"""Every table, figure and cited number in "Playing the Silences", from the records.

    python paper/make_figures.py

Writes paper/figures/{fig_collision,fig_coverage,fig_decoder_weights,fig_cerwer_eng}.pdf, the tables
paper/{fists_table,training_table,cerwer_table_eng,grid_table,seed_table_dw}.tex, and paper/numbers.tex,
the LaTeX macros the prose cites (the paper types no number by hand; consumed_macros.txt lists them). Before a record is used, its manifest hashes and its
protocol hash are checked, and so is the record's claim to come from a clean commit.

Sources:
    firmware/encoder/fists.h                    the five operators (Table 1)
    data/operator_style_fits.csv                the per-segment fits of the real recordings (Figure 2)
    records/random_letters/                     random letters, one posterior scale per model (protocol facts)
    records/decoder_weights/                    the superseded 5,000-transmission record (the FiveK values)
    records/matched/                            the record of the paper: every family at matched training data,
                                                the grid, the deployed networks' fixed-point fidelity, criterion 1A
    protocol/, silences/training/, records/training/   the training recipes (Table 2)
    records/collision.json                      the 4.0 time-unit collision study (Figure 1's numbers)
    models/{lif,mlp}/op12/seed0.pkl             the deployed networks' hyperparameters and parameter count
    records/deployment/                         the RP2040 chain (2026-11-06), the bench test (2026-11-08), footprint
"""
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from silences.operators import parse_fists_h  # noqa: E402
from silences.segmenters import load_dict  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
REC = ROOT / "records"
FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": False, "figure.dpi": 150})
# PDFs without a creation date, so a rebuild is byte-identical
SAVE = dict(bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})

MACROS = {}


def macro(name, value):
    MACROS[name] = value


def _tfmt(T):
    return f"{T:g}"


def _sd1(x):
    """Signed one-decimal percentage points; a difference that rounds to zero prints as 0.0."""
    t = f"{x:+.1f}"
    return "0.0" if t in ("+0.0", "-0.0") else t


def load_record(d, files):
    """A record directory's files, after checking them against its manifest."""
    M = json.loads((d / "manifest.json").read_text())
    out = {}
    for fn in files:
        raw = (d / fn).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == M["files"][fn], f"{d.name}/{fn} does not match its manifest"
        out[fn] = json.loads(raw)
    if not os.environ.get("SILENCES_ALLOW_DIRTY"):
        assert M["git"]["commit"] and M["git"]["clean_for_record"], f"{d.name} was not produced from a clean commit"
    return M, out


# ---- the five operators (Table 1) and the segment fits they summarise (Figure 2) ----
FISTS = parse_fists_h()
FBY = dict(FISTS)
macro("NFists", len(FISTS))
rows = list(csv.DictReader(open(ROOT / "data" / "operator_style_fits.csv")))


def _col(name):
    return np.array([float(r[name]) if r[name] not in ("", "nan") else np.nan for r in rows])


wpm, rr, se, om = _col("wpm"), _col("r"), _col("sigma_elem"), _col("omega")
sg = se * 1.15                                      # sigma_gap = 1.15 sigma_elem (fists.h)
WPM_TRIM, R_TRIM = (6.0, 60.0), (2.0, 7.0)          # the fits the operators were derived from
keep = (wpm >= WPM_TRIM[0]) & (wpm <= WPM_TRIM[1]) & (rr >= R_TRIM[0]) & (rr <= R_TRIM[1]) & np.isfinite(se) & np.isfinite(om)
wk, rk, sek, omk, sgk = wpm[keep], rr[keep], se[keep], om[keep], sg[keep]
macro("NSegAll", len(rows))
macro("NSegTrim", int(keep.sum()))
macro("NSegDropped", len(rows) - int(keep.sum()))
macro("NRecordings", len({r["recording"] for r in rows}))
HIGH_R, FAST_WPM = 4.0, 26.0                        # the stratum rule (fists.h)
macro("StratHighR", f"{HIGH_R:.1f}")
macro("StratFastWpm", f"{FAST_WPM:.0f}")


def stratum(w, r):
    return "high_r_bug" if r >= HIGH_R else ("fast_normalr" if w >= FAST_WPM else "slow_normalr")


strat = np.array([stratum(w, r) for w, r in zip(wk, rk)])
STRATA = ["fast_normalr", "slow_normalr", "high_r_bug"]
for s in STRATA:
    macro("NStrat" + s.replace("_", ""), int((strat == s).sum()))
STY = {"fast_normalr": dict(marker="o", facecolors="none", edgecolors="#0072B2"),
       "slow_normalr": dict(marker="s", facecolors="none", edgecolors="#E69F00"),
       "high_r_bug": dict(marker="^", facecolors="none", edgecolors="#009E73")}
FIST_MARK = {"fast_normalr": ("*", 220), "slow_normalr": ("p", 150), "high_r_bug": ("h", 150),
             "cootie_high_jitter": ("D", 80), "cootie_corrected": ("P", 120)}
FIST_COLS = ("fast_normalr", "slow_normalr", "high_r_bug", "cootie_high_jitter", "cootie_corrected")
FIST_SHORT = {"fast_normalr": "Fast", "slow_normalr": "Slow", "high_r_bug": "HighR",
              "cootie_high_jitter": "CootieHj", "cootie_corrected": "CootieCorr"}

# ---- random letters, one posterior scale per model ----
RL_M, RL = load_record(REC / "random_letters", ["results.json"])
RL = RL["results.json"]
assert RL["anchor"]["passed"], "the deployed ensemble's weights file does not reproduce its checkpoints"
for fam, short in (("lif", "Lif"), ("bigru", "Bigru")):
    macro(f"NSeeds{short}", RL["families"][fam]["n"])
macro("CiLevel", int(round(RL["protocol"]["ci"] * 100)))
macro("EvalPerFist", RL["protocol"]["eval"]["per_fist"])
macro("NEvalSeeds", len(RL["protocol"]["eval"]["seeds"]))

# ---- the decoder-weight record of the 5,000-transmission networks (the FiveK values; Section 8.4's grid-edge check) ----
DW_M, DW = load_record(REC / "decoder_weights", ["results.json", "fit.json", "diag_beta_edge.json"])
E209, E209FIT, E209EDGE = DW["results.json"], DW["fit.json"], DW["diag_beta_edge.json"]
assert E209["protocol_hash"] == DW_M["protocol_hash"] == E209FIT["protocol_hash"]
assert DW_M["base_protocol_hash"] == RL["protocol_hash"], "the decoder-weight protocol is not chained to the random-letter one"
assert E209["anchor_rand_deployed_vs_exp207_t1"]["pass_2dp"] and E209["valid_rand_direction"]

# ---- the record of the paper: every family at matched training data (Sections 5.4 and 8, Tables 2 to 5) ----
MA_M, MA = load_record(REC / "matched", ["results.json", "fit.json", "fidelity.json", "criterion_1a.json"])
E212, E212FIT, E212FID, E212CRIT = MA["results.json"], MA["fit.json"], MA["fidelity.json"]["fidelity"], MA["criterion_1a.json"]
assert E212["protocol_hash"] == MA_M["protocol_hash"] == E212FIT["protocol_hash"] == E212CRIT["protocol_hash"]
assert MA_M["exp209_protocol_hash"] == E209["protocol_hash"], "the matched record is not chained to the decoder-weight one"
assert E212["anchors_pass"] and E212CRIT["anchor"]["pass_2dp"]
assert E212["texts"]["eng"]["own_fit"]["families"]["lif"]["rows"] == [f"lif_op12_seed{s}" for s in range(10)]
assert E212["texts"]["eng"]["own_fit"]["families"]["mlp"]["rows"] == [f"mlp_op12_seed{s}" for s in range(10)]
assert E212["texts"]["eng"]["own_fit"]["families"]["bigru"]["rows"] == [f"bigru_fist_seed{s}" for s in range(42, 50)]
FAM_SHORT = {"lif": "Lif", "mlp": "Mlp", "bigru": "Bigru"}
PAIR_SHORT = {"lif_vs_bigru": "LifVsBigru", "mlp_vs_bigru": "MlpVsBigru", "mlp_vs_lif": "MlpVsLif"}
TEXT_SHORT = {"eng": "Eng", "rand": "Rand"}
SINGLES = E212["protocol"]["singles"]                    # record alias -> row (lif_committed -> lif_op12_seed0, ...)


def emit_text_record(E, tag):
    """The per-text comparison macros of a record in the decoder-weight schema; tag "" for the record of the
    paper, "FiveK" for the superseded 5,000-transmission record."""
    for text, ts in TEXT_SHORT.items():
        for cond, cs in (("own_fit", ""), ("deployed", "Dep")):
            X = E["texts"][text][cond]
            sfx = ts + cs + tag
            for fam, short in FAM_SHORT.items():
                macro(f"Cer{short}SeedMean{sfx}", f"{X['families'][fam]['cer']:.1f}")
                macro(f"Wer{short}SeedMean{sfx}", f"{X['families'][fam]['wer']:.1f}")
            for pair, ps in PAIR_SHORT.items():
                C = X["comparisons"][pair]
                for m, M_ in (("cer", "Cer"), ("wer", "Wer")):
                    c = C[m]
                    macro(f"{M_}{ps}{sfx}", _sd1(c["d"]))
                    macro(f"{M_}{ps}{sfx}Abs", f"{abs(c['d']):.1f}")
                    macro(f"{M_}{ps}{sfx}Lo", _sd1(c["lo"]))
                    macro(f"{M_}{ps}{sfx}Hi", _sd1(c["hi"]))
                    macro(f"{M_}{ps}{sfx}HiAbs", f"{abs(c['hi']):.1f}")
                    if cond == "own_fit":
                        for f in FIST_COLS:
                            pc = C["per_fist"][f][m]
                            macro(f"{M_}{ps}{sfx}{FIST_SHORT[f]}", _sd1(pc["d"]))
                            macro(f"{M_}{ps}{sfx}{FIST_SHORT[f]}Abs", f"{abs(pc['d']):.1f}")
                            macro(f"{M_}{ps}{sfx}{FIST_SHORT[f]}Lo", _sd1(pc["lo"]))
                            macro(f"{M_}{ps}{sfx}{FIST_SHORT[f]}Hi", _sd1(pc["hi"]))
            for row, rs in (("lif_committed", "Lif"), ("mlp_committed", "Mlp"), ("bigru_deployed", "Gapnine"), ("bigru_matched_ens5", "BigruEns")):
                if row in X["singles"]:
                    macro(f"Cer{rs}{sfx}", f"{X['singles'][row]['cer']:.1f}")
                    macro(f"Wer{rs}{sfx}", f"{X['singles'][row]['wer']:.1f}")
            if "perfect_gaps" in X["singles"]:
                macro(f"CerCeiling{sfx}", f"{X['singles']['perfect_gaps']['cer']:.1f}")
                macro(f"WerCeiling{sfx}", f"{X['singles']['perfect_gaps']['wer']:.1f}")
        for fam, short in FAM_SHORT.items():
            fits = E["texts"][text]["fits"][fam]
            macro(f"T{short}Median{ts}{tag}", _tfmt(float(np.median([x["T"] for x in fits]))))
            macro(f"Gamma{short}Median{ts}{tag}", _tfmt(float(np.median([x["beta"] for x in fits]))))
    for fam, short in FAM_SHORT.items():
        for m, M_ in (("cer", "Cer"), ("wer", "Wer")):
            g = E["texts"]["eng"]["deployed"]["families"][fam][m] - E["texts"]["eng"]["own_fit"]["families"][fam][m]
            macro(f"{M_}{short}OwnFitGainEng{tag}", f"{g:.1f}")


emit_text_record(E212, "")
emit_text_record(E209, "FiveK")
eng_own, eng_dep = E212["texts"]["eng"]["own_fit"], E212["texts"]["eng"]["deployed"]
macro("EngWordLenMin", E209["protocol"]["eng_vocab"]["lengths"][0])
macro("EngWordLenMax", E209["protocol"]["eng_vocab"]["lengths"][1])
macro("GammaDefault", _tfmt(E209["protocol"]["deployed"]["beta"]))
# the deployed networks in Q16.16 against their float twins, at the same decoder weights
_same = E212["anchors"]["h1_fx_same_weights"]
for m, M_ in (("cer", "Cer"), ("wer", "Wer")):
    macro(f"FxMaxDiff{M_}", f"{max(abs(d[m]) for s_ in _same.values() for d in s_.values()):.2f}")
for fam, short in (("lif", "Lif"), ("mlp", "Mlp")):
    macro(f"FxAgree{short}", f"{100 * E212['anchors']['h1_fx_agreement'][fam]:.3f}")
assert len({v["gaps"] for v in E212FID.values()}) == 1
macro("FxGaps", f"{E212FID['lif']['gaps']:,}".replace(",", "{,}"))
# the grid-edge check of the superseded record (Section 8.4 cites it as such)
_own = E209["texts"]["eng"]["own_fit"]["comparisons"]
macro("EdgeMaxShiftFiveK", f"{max(abs(E209EDGE['comparisons'][k][m]['d'] - _own[k][m]['d']) for k in _own for m in ('cer', 'wer')):.2f}")

# ---- the model x training-corpus grid (Table 4) ----
GRID = E212["grid"]["texts"]
GRID_FAM = {("bigru", "Gen"): "bigru", ("bigru", "Op"): "bigru_fist", ("bigru", "OpFiveK"): "bigru_shared",
            ("lif", "Gen"): "lif_gen", ("lif", "Op"): "lif_op12", ("lif", "OpFiveK"): "lif",
            ("mlp", "Gen"): "mlp_gen", ("mlp", "Op"): "mlp_op12", ("mlp", "OpFiveK"): "mlp"}
GRID_PAIRS = {"BigruOpVsGen": "bigru_fist_vs_bigru", "LifOpVsOpFiveK": "lif_op12_vs_lif", "MlpOpVsOpFiveK": "mlp_op12_vs_mlp"}
for text, ts in TEXT_SHORT.items():
    G = GRID[text]["record"]["own_fit"]
    for name, key in GRID_PAIRS.items():
        for m, M_ in (("cer", "Cer"), ("wer", "Wer")):
            macro(f"Grid{M_}{name}{ts}Abs", f"{abs(G['comparisons'][key][m]['d']):.1f}")
for fam in FAM_SHORT:
    for m in ("cer", "wer"):
        assert abs(GRID["eng"]["record"]["own_fit"]["families"][GRID_FAM[(fam, "Op")]][m] - eng_own["families"][fam][m]) < 1e-9
PROT = json.loads((ROOT / "protocol" / "grid_bigru.json").read_text())
PROTW = json.loads((ROOT / "protocol" / "grid_window.json").read_text())
assert PROT["part_a"]["fist"]["per_fist"] * len(FIST_COLS) == PROTW["n_tx"]
macro("NTxRecord", f"{PROTW['n_tx']:,}".replace(",", "{,}"))
macro("NTxPerFistRecord", f"{PROT['part_a']['fist']['per_fist']:,}".replace(",", "{,}"))
macro("NTxFiveK", f"{RL['protocol']['train']['per_fist'] * len(FIST_COLS):,}".replace(",", "{,}"))
macro("NTxBigruPhase", f"{PROTW['n_tx']:,}".replace(",", "{,}"))

# ---- criterion 1A's software reference (the proposal): the random-letter procedure on the record families ----
for fam, short in FAM_SHORT.items():
    F_ = E212CRIT["families"][fam]
    assert F_["n"] == RL["families"][fam]["n"]
    macro(f"Cer{short}SeedMean", f"{F_['cer']:.1f}")
    macro(f"Wer{short}SeedMean", f"{F_['wer']:.1f}")

# ---- how the segmenters were trained (Table 2): the recipes, read from the training code and the records ----
from silences.evaluation import protocol as _P   # noqa: E402
from silences.training import bigru as _B        # noqa: E402


def _sci(x):
    e = int(np.floor(np.log10(x)))
    m = x / 10 ** e
    assert abs(m - round(m)) < 1e-9
    m = int(round(m))
    return f"10^{{{e}}}" if m == 1 else f"{m}\\times10^{{{e}}}"


for fam, short in (("lif", "Lif"), ("mlp", "Mlp")):
    R_ = _P.RANDOM_LETTERS[fam]
    macro(f"Epochs{short}", R_["epochs"])
    macro(f"Batch{short}", f"{R_['batch']:,}".replace(",", "{,}"))
assert _P.RANDOM_LETTERS["lif"]["lr"] == _P.RANDOM_LETTERS["mlp"]["lr"] and _P.RANDOM_LETTERS["lif"]["wd"] == _P.RANDOM_LETTERS["mlp"]["wd"]
macro("LrFf", _sci(_P.RANDOM_LETTERS["lif"]["lr"]))
macro("WdFf", _sci(_P.RANDOM_LETTERS["lif"]["wd"]))
macro("ValFracFf", int(round(100 * _P.RANDOM_LETTERS["lif"]["val"])))
macro("NSeedsFf", 10)
assert E212["texts"]["eng"]["own_fit"]["families"]["lif"]["n"] == 10
macro("BigruHidden", _B.N_HIDDEN)
_NP_BIGRU = 2 * 3 * (_B.N_HIDDEN * _B.N_INPUT + _B.N_HIDDEN * _B.N_HIDDEN + 2 * _B.N_HIDDEN) + (2 * _B.N_HIDDEN * _B.N_OUTPUT + _B.N_OUTPUT)
assert _NP_BIGRU == json.loads((REC / "training" / "bigru_pool.json").read_text())["recipe"]["n_params"]
macro("NParamsBigru", f"{_NP_BIGRU:,}".replace(",", "{,}"))
macro("EpochsBigru", _B.N_EPOCHS_TOTAL)
macro("EpochsBigruFinal", _B.N_EPOCHS_PHASE2B)
macro("BatchBigru", _B.BATCH_SIZE)
macro("LrBigru", _sci(_B.LR_INIT))
macro("KIwBigru", f"{_B.K_IW:g}")
macro("BwBigruFirst", f"{_B.BW_PHASE1A:g}")
macro("BwBigruLast", f"{_B.BW_PHASE2B:g}")
macro("NPhasesBigru", 4)
macro("SurrogateSlope", 25)                                     # snnTorch's fast_sigmoid default, as training/window_nets.py calls it
assert "surrogate.fast_sigmoid()" in (ROOT / "silences" / "training" / "window_nets.py").read_text()

# ---- the deployed LIF's hyperparameters and parameter count, from the model itself ----
SNN = load_dict(ROOT / "models" / "lif" / "op12" / "seed0.pkl")
assert SINGLES["lif_committed"] == "lif_op12_seed0"
H1, H2 = SNN["hidden"]
macro("NNeuronsLif", H1 + H2)
macro("HiddenOne", H1)
macro("HiddenTwo", H2)
macro("WinHalf", SNN["W"])
macro("WinGaps", 2 * SNN["W"] + 1)
macro("NFeat", 2 * (2 * SNN["W"] + 1))
macro("LifBeta", f"{SNN['beta']:.2f}")
macro("LifTheta", f"{SNN['threshold']:.0f}")
macro("LifT", SNN["T"])
_NP = sum(int(np.asarray(a).size) for layer in SNN["fcs"] for a in layer)
_MLP = load_dict(ROOT / "models" / "mlp" / "op12" / "seed0.pkl")
assert _NP == sum(int(np.asarray(a).size) for layer in _MLP["layers"] for a in layer)
macro("NParamsFf", f"{_NP:,}".replace(",", "{,}"))
macro("ParamRatioFfBigru", f"{_NP / _NP_BIGRU:.1f}")

# ---- the collision study (Section 3) ----
COL = json.loads((REC / "collision.json").read_text())["prod_cell"]
macro("CollisionIndicator", f"{100 * COL['subset_indicator_acc']:.1f}")
macro("CollisionMajority", f"{100 * COL['subset_majority_base']:.1f}")
macro("CollisionContext", f"{100 * COL['subset_context_acc']:.1f}")

# ---- the RP2040 deployment (Section 9): the chain run and the bench test of the deployed networks ----
DEP = REC / "deployment"
CHAIN_RUN, BENCH_RUN = "_20261106", "_20261108"           # the runs of the deployed (12,000-transmission) networks
chain = {}
for tag in ("lif", "mlp"):
    data = json.loads((DEP / f"chain_{tag}{CHAIN_RUN}.json").read_text())
    hops = ("h06", "h03", "h05")
    match = sum(1 for v in data if v["h03"]["present"] and v["h03"]["verdict"] in ("drift", "exact"))
    struct = sum(1 for v in data for h in hops if v[h]["present"] and v[h]["verdict"] not in ("drift", "exact"))
    chain[tag] = dict(n=len(data), match=match, struct=struct, repro=sum(v["dec"]["repro_ok"] for v in data),
                      deaths=sum(not v["dec"]["died_ok"] for v in data))
assert chain["lif"]["n"] == chain["mlp"]["n"]
macro("ChainSents", chain["lif"]["n"])
for tag, short in (("lif", "Lif"), ("mlp", "Mlp")):
    macro(f"ChainRepro{short}", chain[tag]["repro"])        # sentences reproducing the chain golden's decode
    macro(f"ChainStructDiff{short}", chain[tag]["struct"])  # structural differences on any hop
    macro(f"ChainClfHopMatch{short}", chain[tag]["match"])  # the segmenter's own 0x03 frames reconciled
macro("ChainBeamDeathsLif", chain["lif"]["deaths"])
for tag, short in (("lif", "Lif"), ("mlp", "Mlp")):
    lat = json.loads((DEP / f"latency_{tag}{BENCH_RUN}.json").read_text())
    golden = (ROOT / "firmware" / "golden" / f"segmenter_{tag}.json").read_bytes()
    assert lat["reference_sha256"] == hashlib.sha256(golden).hexdigest(), f"the {tag} bench test scored a different golden"
    assert lat["compute_ok"] == lat["nvec"] == len(json.loads(golden)["vectors"])
    macro(f"MsPerGap{short}", f"{lat['slope_ms_per_gap']:.0f}" if lat["slope_ms_per_gap"] >= 100 else f"{lat['slope_ms_per_gap']:.1f}")
    macro(f"MaxLatS{short}", f"{lat['max_gap_ms'] / 1000:.2f}")
    macro(f"NClfVectors{short}", lat["nvec"])
    macro(f"NClfValues{short}", f"{lat['n_values']:,}".replace(",", "{,}"))
assert MACROS["NClfVectorsLif"] == MACROS["NClfVectorsMlp"] and MACROS["NClfValuesLif"] == MACROS["NClfValuesMlp"]
macro("NClfVectors", MACROS.pop("NClfVectorsLif")); MACROS.pop("NClfVectorsMlp")
macro("NClfValues", MACROS.pop("NClfValuesLif")); MACROS.pop("NClfValuesMlp")
macro("LatRatio", f"{float(MACROS['MsPerGapLif']) / float(MACROS['MsPerGapMlp']):.1f}")
size = re.search(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+[0-9a-f]+\s+clf_hw\.elf", (DEP / "footprint_lif.txt").read_text(), re.M)
text_b, data_b, bss_b = (int(x) for x in size.groups())
macro("FlashKBLif", (text_b + data_b + 500) // 1000)
macro("SramKBLif", (data_b + bss_b + 500) // 1000)
macro("StackBLif", json.loads((DEP / "silicon.json").read_text())["lif"]["stack_bytes"])
macro("RpSramKB", 264)                                         # the RP2040's SRAM

# the macros the paper (and the proposal) cite: the set is pinned so a macro cannot vanish unnoticed
CITED = set((HERE / "consumed_macros.txt").read_text().split())


# ---- Figure 1: the 4.0 time-unit collision ----
def fig_collision():
    """Two compositions with the same spike interval: intra-letter gap + dash, inter-letter gap + dot."""
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 2.9), sharex=True)
    seqs = [("intra-letter gap $+$ dash", [("mark", 0, 1, "dot"), ("gap", 1, 2, "intra"), ("mark", 2, 5, "dash")]),
            ("inter-letter gap $+$ dot", [("mark", 0, 1, "dash"), ("gap", 1, 4, "inter"), ("mark", 4, 5, "dot")])]
    for ax, (title, elems) in zip(axes, seqs):
        for kind, t0, t1, lab in elems:
            if kind == "mark":
                fc = "#0072B2" if lab == "dash" else "#E69F00"
                txt = "white" if lab == "dash" else "black"
                ax.add_patch(plt.Rectangle((t0, 0.25), t1 - t0, 0.5, fc=fc, ec="black", lw=0.8))
                ax.text((t0 + t1) / 2, 0.5, lab, ha="center", va="center", color=txt, fontsize=8, fontweight="bold")
            else:
                ax.plot([t0, t1], [0.5, 0.5], ls=":", color="black", lw=1.0)
                ax.text((t0 + t1) / 2, 0.80, f"{lab}\n({t1 - t0:.0f} $T_u$)", ha="center", va="center", fontsize=7.5)
        spikes = [t1 for kind, _, t1, _ in elems if kind == "mark"]
        for s in spikes:
            ax.annotate("", xy=(s, 1.15), xytext=(s, 0.75), arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
        a, b = spikes[0], spikes[-1]
        ax.annotate("", xy=(a, -0.18), xytext=(b, -0.18), arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.0))
        ax.text((a + b) / 2, -0.42, r"ISI $= 4.0\,T_u$", ha="center", va="center", fontsize=8.5)
        ax.set_title(title, fontsize=9, loc="left", pad=2)
        ax.set_ylim(-0.62, 1.35)
        ax.set_xlim(-0.25, 5.6)
        ax.axis("off")
    axes[1].set_xlabel("time ($T_u$)")
    fig.tight_layout(h_pad=0.9)
    fig.savefig(FIGS / "fig_collision.pdf", **SAVE)
    plt.close(fig)


# ---- Figure 2: operator-style coverage ----
def fig_coverage():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.6, 3.4))
    for s in STRATA:
        m = strat == s
        axl.scatter(wk[m], rk[m], s=9, lw=0.6, alpha=0.55, label=s, **STY[s])
        axr.scatter(sgk[m], omk[m], s=9, lw=0.6, alpha=0.55, **STY[s])
    for name, (w, r, s_e, s_g, o) in FISTS:
        mk, sz = FIST_MARK[name]
        axl.scatter([w], [r], marker=mk, s=sz, c="black", edgecolors="white", lw=0.8, zorder=6)
        axr.scatter([s_g], [o], marker=mk, s=sz, c="black", edgecolors="white", lw=0.8, zorder=6)
    hj, cc = FBY["cootie_high_jitter"], FBY["cootie_corrected"]          # the drift correction, as an arrow
    axr.annotate("", xy=(cc[3], cc[4]), xytext=(hj[3], hj[4]), arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1, ls="--"))
    axl.set_xlabel("keying speed (wpm)")
    axl.set_ylabel(r"dash/dot ratio $r$")
    axl.axhline(HIGH_R, color="0.7", lw=0.7, ls="--")
    axl.axvline(FAST_WPM, color="0.7", lw=0.7, ls="--")
    axl.set_title("(a) style plane", fontsize=9, loc="left")
    axr.set_xlabel(r"$\sigma_\mathrm{gap}$ ($T_u$)")
    axr.set_ylabel(r"per-word drift $\omega$")
    axr.set_title("(b) jitter plane", fontsize=9, loc="left")
    axl.legend(title="segment fits (stratum)", fontsize=6.5, title_fontsize=7, frameon=False, loc="upper right")
    handles = [Line2D([0], [0], marker=FIST_MARK[n][0], color="none", markerfacecolor="black", markeredgecolor="white",
                      markeredgewidth=0.8, markersize=9, label=n) for n, _ in FISTS]
    fig.legend(handles=handles, title="synthetic operators (fists)", fontsize=6.5, title_fontsize=7, frameon=False,
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIGS / "fig_coverage.pdf", **SAVE)
    plt.close(fig)


# ---- Figure 8: decoded CER against the posterior scale ----
def fig_decoder_weights():
    """Each family's mean calibration CER at each T, every seed at its best insertion bonus for that T."""
    P_ = E209FIT["protocol"]
    Ts = P_["T_grid"]
    fams = [("bigru", "BiGRU (baseline)", "#E69F00", "^"), ("lif", "LIF (spiking)", "#0072B2", "o"),
            ("mlp", "MLP (control)", "#009E73", "s")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    for ax, (text, title) in zip(axes, (("eng", "English words"), ("rand", "Random letters"))):
        F_ = E212FIT["fits"][text]
        for fam, label, c, mk in fams:
            rs = E212["texts"][text]["own_fit"]["families"][fam]["rows"]
            assert all(len(F_[r]["grid"]) == len(Ts) * len(P_["beta_grid"]) for r in rs)
            y = [float(np.mean([min(F_[r]["grid"][f"{T}|{b}"]["cer"] for b in P_["beta_grid"]) for r in rs])) for T in Ts]
            ax.plot(Ts, y, marker=mk, ms=4, lw=1.2, color=c, label=label, mec="0.15", mew=0.6)
            i = int(np.argmin(y))
            ax.plot([Ts[i]], [y[i]], marker=mk, ms=9, mfc="none", mec="black", mew=1.1, ls="none")
        ax.axvline(1.0, color="0.6", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ts)
        ax.set_xticklabels([f"{t:g}" for t in Ts], fontsize=7)
        ax.minorticks_off()
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("posterior scale $T$")
    axes[0].set_ylabel("CER (%), calibration corpus")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGS / "fig_decoder_weights.pdf", **SAVE)
    plt.close(fig)


# ---- Figure 7: error by operator on English ----
def fig_cerwer_eng():
    X, orc = eng_own["families"], eng_dep["rows"]["perfect_gaps"]
    series = [("BiGRU seeds (baseline)", X["bigru"], "", "#E69F00"), ("MLP seeds (control)", X["mlp"], "///", "#009E73"),
              ("LIF seeds (spiking)", X["lif"], "", "#0072B2"), ("oracle gaps (ceiling)", orc, "xxx", "0.72")]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw={"width_ratios": [2.6, 1.0]})
    n, x = len(series), np.arange(len(FIST_COLS))
    w = 0.8 / n
    for i, (lab, d, h, c) in enumerate(series):
        axa.bar(x + i * w - 0.4 + w / 2, [d["per_fist"][f]["cer"] for f in FIST_COLS], w, label=lab, fc=c, ec="black", lw=0.7, hatch=h)
    tick = {"fast_normalr": "fast", "slow_normalr": "slow", "high_r_bug": "high_r",
            "cootie_high_jitter": "cootie_hj", "cootie_corrected": "cootie_corr"}
    axa.set_xticks(x)
    axa.set_xticklabels([tick[f] for f in FIST_COLS], fontsize=7.5)
    axa.set_ylabel("CER (%)")
    axa.set_title("(a) character error rate by operator fist", fontsize=9, loc="left")
    axa.set_ylim(0, max(d["per_fist"][f]["cer"] for _, d, _, _ in series for f in FIST_COLS) * 1.34)
    axa.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
    for i, (lab, d, h, c) in enumerate(series):
        axb.bar([i - 0.19], [d["cer"]], 0.36, fc=c, ec="black", lw=0.7, hatch=h)
        axb.bar([i + 0.19], [d["wer"]], 0.36, fc=c, ec="black", lw=0.7, hatch=h, alpha=0.55)
    axb.set_xticks(range(n))
    axb.set_xticklabels(["BiGRU", "MLP", "LIF", "oracle"], fontsize=7.5)
    axb.set_ylabel("error rate (%): solid CER, faded WER")
    axb.set_title("(b) overall CER / WER", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_cerwer_eng.pdf", **SAVE)
    plt.close(fig)


# ---- Table 3: training-seed distributions ----
def write_seed_table_dw():
    ci = int(round(RL["protocol"]["ci"] * 100))
    out = [r"% Generated by paper/make_figures.py from records/matched/results.json -- do not edit.",
           r"\begin{tabular}{lrccrrrr}", r"\toprule",
           r"& & \multicolumn{2}{c}{fitted} & \multicolumn{2}{c}{seed mean $\pm$ sd (\%)} & \multicolumn{2}{c}{vs BiGRU seeds, pp [" + str(ci) + r"\% CI]} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
           r"segmenter & $n$ & $T$ & $\gamma$ & CER & WER & CER & WER \\"]

    def dci(c):
        return f"{_sd1(c['d'])} [{_sd1(c['lo'])}, {_sd1(c['hi'])}]"

    def rng(v):
        lo, hi = min(v), max(v)
        return f"{lo:g}" if lo == hi else f"{lo:g}--{hi:g}"
    for text, head in (("eng", "English words"), ("rand", "Random letters")):
        X, fits = E212["texts"][text]["own_fit"], E212["texts"][text]["fits"]
        out += [r"\midrule", r"\multicolumn{8}{l}{\emph{" + head + r"}} \\"]
        for label, fam, pair in (("LIF (spiking)", "lif", "lif_vs_bigru"), ("MLP (control)", "mlp", "mlp_vs_bigru"),
                                 ("BiGRU (baseline)", "bigru", None)):
            F_ = X["families"][fam]
            d = [dci(X["comparisons"][pair]["cer"]), dci(X["comparisons"][pair]["wer"])] if pair else ["---", "---"]
            out.append(f"{label} & {F_['n']} & {rng([x['T'] for x in fits[fam]])} & {rng([x['beta'] for x in fits[fam]])} & "
                       f"{F_['cer']:.1f} $\\pm$ {F_['cer_sd']:.1f} & {F_['wer']:.1f} $\\pm$ {F_['wer_sd']:.1f} & {d[0]} & {d[1]} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "seed_table_dw.tex").write_text("\n".join(out) + "\n")


# ---- Table 2: English error rates of the committed models ----
def write_cerwer_table_eng():
    own, dep = eng_own["rows"], eng_dep["rows"]
    for alias in ("lif_committed", "mlp_committed", "bigru_matched_ens5"):
        assert abs(own[SINGLES[alias]]["cer"] - eng_own["singles"][alias]["cer"]) < 1e-9
    order = [(r"\textbf{LIF (spiking)}", own[SINGLES["lif_committed"]], True), ("MLP (control)", own[SINGLES["mlp_committed"]], False),
             ("BiGRU ensemble (baseline)", own[SINGLES["bigru_matched_ens5"]], False), ("oracle gaps (ceiling)", dep["perfect_gaps"], False)]
    hdr = " & ".join(f.replace("_", r"\_") for f in FIST_COLS)
    out = [r"% Generated by paper/make_figures.py from records/matched/results.json -- do not edit.",
           r"\setlength{\tabcolsep}{3.5pt}%", r"\begin{tabular}{l" + "r" * len(FIST_COLS) + "rr}", r"\toprule",
           r"& \multicolumn{%d}{c}{CER by operator (\%%)} & \multicolumn{2}{c}{overall (\%%)} \\" % len(FIST_COLS),
           r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (len(FIST_COLS) + 1, len(FIST_COLS) + 2, len(FIST_COLS) + 3),
           r"segmenter & " + hdr + r" & CER & WER \\", r"\midrule"]
    for label, r, bold in order:
        tot = [f"{r['cer']:.1f}", f"{r['wer']:.1f}"]
        if bold:
            tot = [r"\textbf{%s}" % t for t in tot]
        out.append(label + " & " + " & ".join([f"{r['per_fist'][f]['cer']:.1f}" for f in FIST_COLS] + tot) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "cerwer_table_eng.tex").write_text("\n".join(out) + "\n")


# ---- Table 1: the five operators ----
def write_fists_table():
    out = [r"% Generated by paper/make_figures.py from firmware/encoder/fists.h -- do not edit.",
           r"\begin{tabular}{lrrrrr}", r"\toprule", r"operator & wpm & $r$ & $\sigelem$ & $\siggap$ & $\omega$ \\", r"\midrule"]
    for name, (w, r, se_, sg_, om_) in FISTS:
        cells = [f"{w:g}"] + [f"{v:.4f}" for v in (r, se_, sg_, om_)]
        out.append(r"\texttt{" + name.replace("_", r"\_") + "} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "fists_table.tex").write_text("\n".join(out) + "\n")


# ---- Table 4: model x training corpus ----
def write_grid_table():
    G = GRID["eng"]["record"]["own_fit"]["families"]
    out = [r"% Generated by paper/make_figures.py from records/matched/results.json (grid) -- do not edit.",
           r"\begin{tabular}{lccc}", r"\toprule", r"training corpus & BiGRU & MLP & LIF \\", r"\midrule"]
    for label, corp in ((r"generic operators, \NTxBigruPhase{} per seed", "Gen"),
                        (r"the five operators, \NTxRecord{} per seed (record)", "Op"),
                        (r"the five operators, \NTxFiveK{} per seed", "OpFiveK")):
        cells = []
        for fam in ("bigru", "mlp", "lif"):
            F_ = G[GRID_FAM[(fam, corp)]]
            cell = f"{F_['cer']:.1f} / {F_['wer']:.1f}"
            cells.append(r"\textbf{%s}" % cell if corp == "Op" else cell)
        out.append(label + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "grid_table.tex").write_text("\n".join(out) + "\n")


# ---- Table 2: the training recipes ----
def write_training_table():
    out = [r"% Generated by paper/make_figures.py (protocol/, silences/training/, records/training/, the deployed models) -- do not edit.",
           r"\begin{tabular}{lccc}", r"\toprule",
           r" & BiGRU baseline & MLP control & LIF network \\", r"\midrule",
           r"hidden units & $\BigruHidden{}$ per direction & $\HiddenOne{}$, $\HiddenTwo{}$ & $\HiddenOne{}$, $\HiddenTwo{}$ \\",
           r"trainable parameters & \NParamsBigru{} & \NParamsFf{} & \NParamsFf{} \\",
           r"training transmissions per seed & \NTxBigruPhase{} per phase, \NPhasesBigru{} phases & \NTxRecord{} & \NTxRecord{} \\",
           r"operator corpus & final phase: the five, \NTxPerFistRecord{} each & the five, \NTxPerFistRecord{} each & the five, \NTxPerFistRecord{} each \\",
           r"loss & weighted cross-entropy & cross-entropy & cross-entropy \\",
           r"optimizer, learning rate & Adam, $\LrBigru$, cosine & Adam, $\LrFf$, constant & Adam, $\LrFf$, constant \\",
           r"weight decay & none & $\WdFf$ & $\WdFf$ \\",
           r"batch & \BatchBigru{} transmissions & \BatchMlp{} gaps & \BatchLif{} gaps \\",
           r"epochs & \EpochsBigru{} & \EpochsMlp{} & \EpochsLif{} \\",
           r"checkpoint & final epoch & lowest validation loss & lowest validation loss \\",
           r"training seeds & \NSeedsBigru{} & \NSeedsFf{} & \NSeedsFf{} \\",
           r"\bottomrule", r"\end{tabular}"]
    (HERE / "training_table.tex").write_text("\n".join(out) + "\n")


def write_numbers():
    """numbers.tex: one \\newcommand per cited number (the set in consumed_macros.txt). A leading minus is
    written \\textminus."""
    def tex(v):
        v = str(v)
        return "\\textminus" + v[1:] if v.startswith("-") and v[1:2].isdigit() else v
    missing = CITED - set(MACROS)
    assert not missing, f"the paper cites numbers this script no longer computes: {sorted(missing)}"
    lines = ["% Generated by paper/make_figures.py -- do not edit.", ""]
    lines += [f"\\newcommand{{\\{k}}}{{{tex(MACROS[k])}}}" for k in sorted(MACROS) if k in CITED]
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    fig_collision()
    fig_coverage()
    fig_decoder_weights()
    fig_cerwer_eng()
    write_seed_table_dw()
    write_cerwer_table_eng()
    write_grid_table()
    write_training_table()
    write_fists_table()
    write_numbers()
    print(f"{len(CITED)} numbers, 5 tables and 4 figures -> {HERE}")
