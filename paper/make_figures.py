#!/usr/bin/env python3
"""Every table, figure and cited number in "Playing the Silences", from the records.

    python paper/make_figures.py

Writes paper/figures/{fig_collision,fig_coverage,fig_decoder_weights,fig_cerwer_eng}.pdf, the tables
paper/{fists_table,cerwer_table_eng,seed_table_dw}.tex, and paper/numbers.tex, the LaTeX macros the
prose cites (the paper types no number by hand). Before a record is used, its manifest hashes and its
protocol hash are checked, and so is the record's claim to come from a clean commit.

Sources:
    firmware/encoder/fists.h                    the five operators (Table 1)
    data/operator_style_fits.csv                the per-segment fits of the real recordings (Figure 2)
    records/random_letters/                     random letters, one posterior scale per model
    records/decoder_weights/                    English and random letters, decoder weights per model and text
    records/collision.json                      the 4.0 time-unit collision study (Figure 1's numbers)
    models/lif/committed.pkl                    the deployed LIF's hyperparameters
    records/deployment/                         the RP2040 chain, latency, transport and footprint
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
macro("CerLifSeedMean", f"{RL['families']['lif']['cer']['mean']:.1f}")
macro("WerLifSeedMean", f"{RL['families']['lif']['wer']['mean']:.1f}")
macro("CiLevel", int(round(RL["protocol"]["ci"] * 100)))
macro("EvalPerFist", RL["protocol"]["eval"]["per_fist"])
macro("NEvalSeeds", len(RL["protocol"]["eval"]["seeds"]))

# ---- English and random letters, decoder weights per model and text (Tables 2 and 3, Figures 7 and 8) ----
DW_M, DW = load_record(REC / "decoder_weights", ["results.json", "fit.json"])
E209, E209FIT = DW["results.json"], DW["fit.json"]
assert E209["protocol_hash"] == DW_M["protocol_hash"] == E209FIT["protocol_hash"]
assert DW_M["base_protocol_hash"] == RL["protocol_hash"], "the decoder-weight protocol is not chained to the random-letter one"
assert E209["anchor_rand_deployed_vs_exp207_t1"]["pass_2dp"] and E209["valid_rand_direction"]
eng_own, eng_dep = E209["texts"]["eng"]["own_fit"], E209["texts"]["eng"]["deployed"]
for fam, short in (("lif", "Lif"), ("bigru", "Bigru")):
    macro(f"Cer{short}SeedMeanEng", f"{eng_own['families'][fam]['cer']:.1f}")
    macro(f"Wer{short}SeedMeanEng", f"{eng_own['families'][fam]['wer']:.1f}")
macro("CerLifEng", f"{eng_own['singles']['lif_committed']['cer']:.1f}")
macro("CerMlpEng", f"{eng_own['singles']['mlp_committed']['cer']:.1f}")
macro("CerGapnineEng", f"{eng_own['singles']['bigru_deployed']['cer']:.1f}")
macro("WerGapnineEng", f"{eng_own['singles']['bigru_deployed']['wer']:.1f}")
macro("CerCeilingEngDep", f"{eng_dep['singles']['perfect_gaps']['cer']:.1f}")
macro("WerCeilingEngDep", f"{eng_dep['singles']['perfect_gaps']['wer']:.1f}")
macro("CerMlpVsBigruEngAbs", f"{abs(eng_own['comparisons']['mlp_vs_bigru']['cer']['d']):.1f}")
macro("CerLifVsBigruEngCootieHjAbs", f"{abs(eng_own['comparisons']['lif_vs_bigru']['per_fist']['cootie_high_jitter']['cer']['d']):.1f}")
macro("CerLifVsBigruRandAbs", f"{abs(E209['texts']['rand']['own_fit']['comparisons']['lif_vs_bigru']['cer']['d']):.1f}")
macro("EngWordLenMin", E209["protocol"]["eng_vocab"]["lengths"][0])
macro("EngWordLenMax", E209["protocol"]["eng_vocab"]["lengths"][1])
macro("GammaDefault", _tfmt(E209["protocol"]["deployed"]["beta"]))

# ---- the deployed LIF's hyperparameters, from the model itself ----
SNN = load_dict(ROOT / "models" / "lif" / "committed.pkl")
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

# ---- the collision study (Section 3) ----
COL = json.loads((REC / "collision.json").read_text())["prod_cell"]
macro("CollisionIndicator", f"{100 * COL['subset_indicator_acc']:.1f}")
macro("CollisionMajority", f"{100 * COL['subset_majority_base']:.1f}")
macro("CollisionContext", f"{100 * COL['subset_context_acc']:.1f}")

# ---- the RP2040 deployment (Section 9) ----
DEP = REC / "deployment"
chain = {}
for tag in ("lif", "mlp"):
    data = json.loads((DEP / f"chain_{tag}.json").read_text())
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
golden = json.loads((ROOT / "firmware" / "golden" / "segmenter_lif.json").read_text())["vectors"]
for tag, short in (("lif", "Lif"), ("mlp", "Mlp")):
    lat = json.loads((DEP / f"latency_{tag}.json").read_text())
    tra = json.loads((DEP / f"transport_{tag}.json").read_text())
    assert lat["compute_ok"] == lat["nvec"] == tra["npass"] == tra["nvec"] == len(golden) and tra["correct"]
    macro(f"MsPerGap{short}", f"{lat['slope_ms_per_gap']:.3g}")           # least-squares slope over the 50 vectors
    macro(f"MaxLatS{short}", f"{lat['max_gap_ms'] / 1000:.3g}")           # the longest vector's compute time
macro("NClfVectors", len(golden))
macro("NClfValues", sum(3 * v["n_gaps"] for v in golden))
macro("LatRatio", f"{float(MACROS['MsPerGapLif']) / float(MACROS['MsPerGapMlp']):.1f}")
size = re.search(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+[0-9a-f]+\s+clf_hw\.elf", (DEP / "footprint_lif.txt").read_text(), re.M)
text_b, data_b, bss_b = (int(x) for x in size.groups())
macro("FlashKBLif", (text_b + data_b + 500) // 1000)
macro("SramKBLif", (data_b + bss_b + 500) // 1000)
macro("StackBLif", json.loads((DEP / "silicon.json").read_text())["lif"]["stack_bytes"])
macro("RpSramKB", 264)                                         # the RP2040's SRAM


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
        F_ = E209FIT["fits"][text]
        for fam, label, c, mk in fams:
            rs = E209["texts"][text]["own_fit"]["families"][fam]["rows"]
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
    out = [r"% Generated by paper/make_figures.py from records/decoder_weights/results.json -- do not edit.",
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
        X, fits = E209["texts"][text]["own_fit"], E209["texts"][text]["fits"]
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
    order = [(r"\textbf{LIF (spiking)}", own["lif_committed"], True), ("MLP (control)", own["mlp_committed"], False),
             ("BiGRU (baseline)", own["bigru_deployed"], False), ("oracle gaps (ceiling)", dep["perfect_gaps"], False)]
    hdr = " & ".join(f.replace("_", r"\_") for f in FIST_COLS)
    out = [r"% Generated by paper/make_figures.py from records/decoder_weights/results.json -- do not edit.",
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


def write_numbers():
    """numbers.tex: one \\newcommand per cited number. A leading minus is written \\textminus."""
    def tex(v):
        v = str(v)
        return "\\textminus" + v[1:] if v.startswith("-") and v[1:2].isdigit() else v
    lines = ["% Generated by paper/make_figures.py -- do not edit.", ""]
    lines += [f"\\newcommand{{\\{k}}}{{{tex(MACROS[k])}}}" for k in sorted(MACROS)]
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    fig_collision()
    fig_coverage()
    fig_decoder_weights()
    fig_cerwer_eng()
    write_seed_table_dw()
    write_cerwer_table_eng()
    write_fists_table()
    write_numbers()
    print(f"{len(MACROS)} numbers, 3 tables and 4 figures -> {HERE}")
