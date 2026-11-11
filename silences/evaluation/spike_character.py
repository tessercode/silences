"""What the spikes carry (Section 8.6): the LIF network's spike statistics and timing-destroying controls.

    python -m silences.evaluation.spike_character [spk40|lif] [forward|decode|torch|stats|all]

For the ten training seeds of a family (spk40_op12, the spike-fed network of record, seed 0 deployed; or
lif_op12, the current-fed rung), every gap of the record evaluation corpora is run through the network with
within-presentation spike timing destroyed at one layer while the spike counts are kept:
    shuffle1    layer-1 spike times permuted within the presentation per (gap, neuron); layer 2 and readout re-run
    collapse1   layer-1 spike trains replaced by their mean rate (count/T) as a constant current into layer 2
    shuffle2    layer 1 intact; layer-2 spike times permuted into the readout
    collapse2   layer 1 intact; layer-2 spike trains replaced by their mean rate into the readout
and, for the spike-fed network, at the input:
    shuffle0    the regular input spike trains have their spike times permuted per (gap, channel), counts kept
    collapse0   each input train replaced by its mean rate as a constant current into layer 1
Each control is scored by per-gap decision agreement with the intact network and by decoded CER/WER at the
network's own fitted decoder weights. The intact path is run per transmission exactly as the producer of
record runs it, and asserted equal, value for value, to the stored rows (anchor B0) before any control is read.
The spike counts per neuron per presentation, the operation counts per decision, and (step torch, needs torch)
the training-side snnTorch forward against the inference forward are recorded with them.

Writes records/spike_fed/spikes_<family>.json; the per-corpus rows are cached under outputs/spike_character/.
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
from ..features import fe_tu_us, window_features   # noqa: E402
from ..fixedpoint.q16 import grid_q, quantize_lp3   # noqa: E402
from . import protocol as P, tables   # noqa: E402
from .decoder_weights import _pack   # noqa: E402
from .metrics import edit_sums   # noqa: E402
from .producers import DEPLOYED, SEEDS, _LOAD, grid_model_path   # noqa: E402
from .provenance import env_info, git_info, now, sha256_file   # noqa: E402

OUT = RECORDS / "spike_fed"
ROWS = OUTPUTS / "spike_character"
TEXTS = P.TEXTS
EVAL_SEEDS = P.EVAL["seeds"]
WORKERS = 8
RNG_BASE = 213
FAMILIES = {"spk40": ("spk40_op12", "spike_fed"), "lif": ("lif_op12", True)}      # family -> (rows' family, table set)


class Family:
    def __init__(self, name):
        self.name = name
        self.fam, self.tables = FAMILIES[name]
        self.spk = name.startswith("spk")
        self.variants = (("intact", "shuffle0", "collapse0") if self.spk else ("intact",)) + ("shuffle1", "collapse1", "shuffle2", "collapse2")
        self.hash = P.SPIKE_FED_HASH if self.spk else P.MATCHED_HASH
        self.protocol = dict(exp211_protocol_hash=P.GRID_WINDOW_HASH, exp213_protocol_hash=P.SPIKE_FED_HASH if self.spk else None,
                             family=self.fam, seeds=SEEDS, texts=TEXTS, eval_seeds=EVAL_SEEDS, variants=self.variants,
                             rng="numpy default_rng(213 + eval seed + 1000 * train seed), one permutation per gap and neuron",
                             decoder="own fit of each seed (Exp 211 fit.json), per text",
                             reading=dict(rate_code="every control agrees on >= 99% of decisions and moves CER by < 0.3 pp",
                                          timing_matters="any control agrees on < 95% or moves CER by >= 1 pp", mixed="otherwise"))

    def model_path(self, s):
        return grid_model_path(self.name, "op12", s)

    def load_model(self, s):
        return _LOAD[self.name](self.model_path(s))

    def load_doc(self, text, es):
        return tables.load(text, es, matched=self.tables)

    def fits(self):
        src = OUT / "fit.json" if self.spk else RECORDS / "matched" / "fit.json"
        F = json.loads(src.read_text())
        assert F["protocol_hash"] == self.hash
        return F["fits"]

    def row_path(self, s, text, es):
        return ROWS / f"{self.fam}_seed{s}_{text}_{es}.pkl"


_F = None


def step_layer(Wt, b, cur_seq, beta, theta, T, spiking):
    """One LIF layer over a (T, n, in) input sequence, exactly as the segmenters' logits() run it."""
    n, out_dim = cur_seq.shape[1], Wt.shape[1]
    mem = np.zeros((n, out_dim))
    if spiking:
        S = np.zeros((T, n, out_dim))
        for t in range(T):
            mem = beta * mem + (cur_seq[t] @ Wt + b)
            spk = (mem >= theta).astype(np.float64)
            mem = mem - spk * theta
            S[t] = spk
        return S
    acc = np.zeros((n, 3))
    for t in range(T):
        mem = beta * mem + (cur_seq[t] @ Wt + b)
        acc += mem
    return acc / T


def forward_tx(m, F, rng, spk):
    T, beta, theta = m.T, m.beta, m.threshold
    (W1, b1), (W2, b2), (W3, b3) = m.fcs
    if spk:
        X = m.encode(F)
    else:
        Z = (np.asarray(F, np.float64) - m.mu) / m.sd
        X = np.broadcast_to(Z, (T,) + Z.shape)
    S1 = step_layer(W1, b1, X, beta, theta, T, True)
    S2 = step_layer(W2, b2, S1, beta, theta, T, True)
    out = {"intact": step_layer(W3, b3, S2, beta, theta, T, False)}
    c0 = None
    if spk:
        perm0 = rng.random(X.shape).argsort(axis=0)
        X0 = np.take_along_axis(X, perm0, axis=0)
        out["shuffle0"] = step_layer(W3, b3, step_layer(W2, b2, step_layer(W1, b1, X0, beta, theta, T, True), beta, theta, T, True), beta, theta, T, False)
        R0 = np.broadcast_to(X.mean(0), X.shape)
        out["collapse0"] = step_layer(W3, b3, step_layer(W2, b2, step_layer(W1, b1, R0, beta, theta, T, True), beta, theta, T, True), beta, theta, T, False)
        c0 = X.sum(0).astype(np.int64)
    perm = rng.random(S1.shape).argsort(axis=0)
    out["shuffle1"] = step_layer(W3, b3, step_layer(W2, b2, np.take_along_axis(S1, perm, axis=0), beta, theta, T, True), beta, theta, T, False)
    out["collapse1"] = step_layer(W3, b3, step_layer(W2, b2, np.broadcast_to(S1.mean(0), S1.shape), beta, theta, T, True), beta, theta, T, False)
    perm2 = rng.random(S2.shape).argsort(axis=0)
    out["shuffle2"] = step_layer(W3, b3, np.take_along_axis(S2, perm2, axis=0), beta, theta, T, False)
    out["collapse2"] = step_layer(W3, b3, np.broadcast_to(S2.mean(0), S2.shape), beta, theta, T, False)
    return out, S1.sum(0).astype(np.int64), S2.sum(0).astype(np.int64), c0


def to_lp3(L):
    L = L - L.max(1, keepdims=True)
    return quantize_lp3(L - np.log(np.exp(L).sum(1, keepdims=True)))


def _job_forward(job):
    s, text, es = job
    path = _F.row_path(s, text, es)
    if path.exists():
        return path.name, "cached", 0.0
    t0 = time.time()
    m = _F.load_model(s)
    doc = _F.load_doc(text, es)
    stored = doc["rows"][f"{_F.fam}_seed{s}"]
    rng = np.random.default_rng(RNG_BASE + es + 1000 * s)
    rows = {v: [] for v in _F.variants}
    h1 = np.zeros(m.T + 1, dtype=np.int64); h2 = np.zeros(m.T + 1, dtype=np.int64)
    n_in = m.fcs[0][0].shape[0]
    h0 = np.zeros(n_in * m.T + 1, dtype=np.int64); hact = np.zeros(n_in + 1, dtype=np.int64)
    anchor_equal, n_gaps = True, 0
    for i, (ticks, ch) in enumerate(zip(doc["ticks"], doc["ch"])):
        F = window_features(ticks, np.asarray(ch, dtype=np.int64), fe_tu_us(ticks), m.W)
        if len(F) == 0:
            for v in _F.variants:
                rows[v].append([])
            assert stored[i] == []
            continue
        out, c1, c2, c0 = forward_tx(m, F, rng, _F.spk)
        for v in _F.variants:
            rows[v].append(to_lp3(out[v]))
        anchor_equal = anchor_equal and rows["intact"][-1] == stored[i]
        n_gaps += len(F)
        h1 += np.bincount(c1.ravel(), minlength=m.T + 1); h2 += np.bincount(c2.ravel(), minlength=m.T + 1)
        if c0 is not None:
            h0 += np.bincount(c0.sum(1), minlength=len(h0)); hact += np.bincount((c0 > 0).sum(1), minlength=len(hact))
    agree = {v: float(np.mean(np.concatenate([np.argmax(np.asarray(a), 1) == np.argmax(np.asarray(b), 1)
                                              for a, b in zip(rows[v], rows["intact"]) if a]))) for v in _F.variants}
    ROWS.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".tmp", "wb") as fh:
        pickle.dump(dict(seed=s, text=text, eval_seed=es, rows=rows, anchor_equal=anchor_equal, gaps=n_gaps,
                         hist1=h1.tolist(), hist2=h2.tolist(), hist0=h0.tolist(), hist_active=hact.tolist(), n_in=n_in,
                         T=m.T, hidden=list(m.hidden), agreement=agree), fh)
    os.replace(str(path) + ".tmp", path)
    return path.name, f"anchor {'OK' if anchor_equal else 'FAIL'} " + " ".join(f"{v} {100 * a:.2f}" for v, a in agree.items()), round(time.time() - t0, 1)


def step_forward():
    jobs = [(s, t, es) for s in SEEDS for t in TEXTS for es in EVAL_SEEDS]
    with get_context("fork").Pool(WORKERS) as pool:
        for name, how, sec in pool.imap_unordered(_job_forward, jobs):
            print(f"  {name}: {how} ({sec}s)", flush=True)


def _job_decode(job):
    s, text, es, v = job
    f = _F.fits()[text][f"{_F.fam}_seed{s}"]
    with open(_F.row_path(s, text, es), "rb") as fh:
        R = pickle.load(fh)
    doc = _F.load_doc(text, es)
    d = dict(ch=doc["ch"], truth=doc["truth"], fist=doc["fist"], rows={v: R["rows"][v]})
    tot, pf = edit_sums(d, v, grid_q(f["T"]), grid_q(f["beta"]), per_fist=True)
    return (s, text, es, v), dict(overall=_pack(tot), per_fist={k: _pack(x) for k, x in pf.items()}, T=f["T"], beta=f["beta"])


def step_decode():
    path = ROWS / f"decode_{_F.name}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    jobs = [(s, t, es, v) for s in SEEDS for t in TEXTS for es in EVAL_SEEDS for v in _F.variants if f"{s}|{t}|{es}|{v}" not in done]
    t0 = time.time()
    with get_context("fork").Pool(WORKERS) as pool:
        for (s, t, es, v), r in pool.imap_unordered(_job_decode, jobs):
            done[f"{s}|{t}|{es}|{v}"] = r
    path.write_text(json.dumps(done))
    print(f"  decode done: {len(done)} cells ({time.time() - t0:.0f}s)", flush=True)


def step_torch():
    """The training-side snnTorch forward of seed 0 against the inference forward on every evaluation gap."""
    import torch
    from ..training.window_nets import torch_logits
    torch.set_num_threads(4)
    m = _F.load_model(0)
    fits = _F.fits()
    out = {}
    for text in TEXTS:
        agree_n, n, cells = 0, 0, {}
        for es in EVAL_SEEDS:
            doc = _F.load_doc(text, es)
            rows_t, rows_n = [], []
            for ticks, ch in zip(doc["ticks"], doc["ch"]):
                F = window_features(ticks, np.asarray(ch, dtype=np.int64), fe_tu_us(ticks), m.W)
                if len(F) == 0:
                    rows_t.append([]); rows_n.append([]); continue
                Lt = torch_logits(m, F); Ln = m.logits(F)
                agree_n += int(np.sum(Lt.argmax(1) == Ln.argmax(1))); n += len(F)
                rows_t.append(to_lp3(Lt)); rows_n.append(to_lp3(Ln))
            f = fits[text][f"{_F.fam}_seed0"]
            d = dict(ch=doc["ch"], truth=doc["truth"], fist=doc["fist"], rows=dict(t=rows_t, n=rows_n))
            cells[str(es)] = {k: _pack(edit_sums(d, k, grid_q(f["T"]), grid_q(f["beta"]))[0]) for k in ("t", "n")}
            print(f"  torch {text} {es}: agreement so far {100 * agree_n / n:.3f}%", flush=True)
        out[text] = dict(gaps=n, argmax_agreement=agree_n / n, cells=cells,
                         cer_numpy=float(np.mean([c["n"]["cer"] for c in cells.values()])), cer_torch=float(np.mean([c["t"]["cer"] for c in cells.values()])),
                         wer_numpy=float(np.mean([c["n"]["wer"] for c in cells.values()])), wer_torch=float(np.mean([c["t"]["wer"] for c in cells.values()])))
    (ROWS / f"torch_vs_numpy_{_F.name}.json").write_text(json.dumps(dict(
        seed=0, note="snnTorch Leaky resets one step after the spike; numpy inference resets in the same step", texts=out), indent=1))


def _hist_stats(h, T):
    h = np.asarray(h, dtype=np.float64); n = h.sum(); k = np.arange(T + 1)
    cdf = np.cumsum(h) / n
    q = lambda p: float(k[np.searchsorted(cdf, p)])  # noqa: E731
    return dict(mean=float((h * k).sum() / n), median=q(0.5), p95=q(0.95), frac_zero=float(h[0] / n), frac_max=float(h[T] / n), pairs=int(n))


def step_stats():
    V = _F.variants
    if _F.spk:
        meta = json.loads(_F.model_path(0).with_suffix(".meta.json").read_text())
        assert sha256_file(_F.model_path(0)) == meta["pickle_sha256"] == sha256_file(DEPLOYED["spk"])
        R13 = json.loads((OUT / "results.json").read_text())
        assert R13["protocol_hash"] == P.SPIKE_FED_HASH and f"{_F.fam}_seed0" in R13["families"][_F.fam]
        sha0, target = meta["pickle_sha256"], f"{_F.fam}_seed0 (decision 2026-11-09, port pending)"
    else:
        rec = json.loads((RECORDS / "matched" / "results.json").read_text())
        dep = rec["deployed"]["lif"]
        assert sha256_file(_F.model_path(0)) == dep["pickle_sha256"] == sha256_file(DEPLOYED["lif"])
        sha0, target = dep["pickle_sha256"], "lif_op12_seed0 (deployed, Exp 212 record)"
    dec = json.loads((ROWS / f"decode_{_F.name}.json").read_text())
    out = dict(experiment="exp213_spike_character", protocol=_F.protocol, family=_F.fam, target=target, deployed_pickle_sha256=sha0,
               anchors=dict(b0_intact_equals_stored_rows=True, gaps=0), per_seed={}, texts={})
    H1 = {t: None for t in TEXTS}; H2 = {t: None for t in TEXTS}; H0 = {t: None for t in TEXTS}; HA = {t: None for t in TEXTS}
    T = hidden = n_in = None
    agree = {t: {v: [] for v in V} for t in TEXTS}; cer = {t: {v: [] for v in V} for t in TEXTS}; wer = {t: {v: [] for v in V} for t in TEXTS}
    for s in SEEDS:
        ps = out["per_seed"][str(s)] = {}
        for t in TEXTS:
            a = {v: [] for v in V}
            for es in EVAL_SEEDS:
                with open(_F.row_path(s, t, es), "rb") as fh:
                    R = pickle.load(fh)
                out["anchors"]["b0_intact_equals_stored_rows"] &= R["anchor_equal"]; out["anchors"]["gaps"] += R["gaps"]
                T, hidden, n_in = R["T"], R["hidden"], R["n_in"]
                H1[t] = np.asarray(R["hist1"]) if H1[t] is None else H1[t] + np.asarray(R["hist1"])
                H2[t] = np.asarray(R["hist2"]) if H2[t] is None else H2[t] + np.asarray(R["hist2"])
                if _F.spk:
                    H0[t] = np.asarray(R["hist0"]) if H0[t] is None else H0[t] + np.asarray(R["hist0"])
                    HA[t] = np.asarray(R["hist_active"]) if HA[t] is None else HA[t] + np.asarray(R["hist_active"])
                for v in V:
                    a[v].append(R["agreement"][v])
            ps[t] = dict(agreement={v: float(np.mean(a[v])) for v in V},
                         cer={v: float(np.mean([dec[f"{s}|{t}|{es}|{v}"]["overall"]["cer"] for es in EVAL_SEEDS])) for v in V},
                         wer={v: float(np.mean([dec[f"{s}|{t}|{es}|{v}"]["overall"]["wer"] for es in EVAL_SEEDS])) for v in V},
                         fit=dict(T=dec[f"{s}|{t}|{EVAL_SEEDS[0]}|intact"]["T"], beta=dec[f"{s}|{t}|{EVAL_SEEDS[0]}|intact"]["beta"]))
            for v in V:
                agree[t][v].append(ps[t]["agreement"][v]); cer[t][v].append(ps[t]["cer"][v]); wer[t][v].append(ps[t]["wer"][v])
    nfeat = 2 * (2 * P.TRAIN["W"] + 1)
    for t in TEXTS:
        st1, st2 = _hist_stats(H1[t], T), _hist_stats(H2[t], T)
        fam = {v: dict(agreement=float(np.mean(agree[t][v])), agreement_min=float(np.min(agree[t][v])),
                       cer=float(np.mean(cer[t][v])), wer=float(np.mean(wer[t][v])),
                       cer_delta=float(np.mean(cer[t][v]) - np.mean(cer[t]["intact"])), wer_delta=float(np.mean(wer[t][v]) - np.mean(wer[t]["intact"]))) for v in V}
        d0 = out["per_seed"]["0"][t]
        dep0 = {v: dict(agreement=d0["agreement"][v], cer=d0["cer"][v], wer=d0["wer"][v],
                        cer_delta=d0["cer"][v] - d0["cer"]["intact"], wer_delta=d0["wer"][v] - d0["wer"]["intact"]) for v in V}
        ctrl = [v for v in V if v != "intact"]
        worst_agree = min(dep0[v]["agreement"] for v in ctrl); worst_cer = max(abs(dep0[v]["cer_delta"]) for v in ctrl)
        reading = ("RATE CODE" if worst_agree >= 0.99 and worst_cer < 0.3 else "TIMING MATTERS" if worst_agree < 0.95 or worst_cer >= 1.0 else "MIXED")
        inp = None
        if H0[t] is not None:
            st0 = _hist_stats(H0[t], len(H0[t]) - 1); sta = _hist_stats(HA[t], len(HA[t]) - 1)
            inp = dict(spikes_per_presentation=st0, active_channels=sta, channels=n_in, possible=n_in * T)
        out["texts"][t] = dict(spikes=dict(layer1=st1, layer2=st2, per_decision=st1["mean"] * hidden[0] + st2["mean"] * hidden[1],
                                           per_neuron=(st1["mean"] * hidden[0] + st2["mean"] * hidden[1]) / sum(hidden)),
                               input=inp, controls_seed_mean=fam, controls_deployed=dep0, reading_deployed=reading,
                               ops=dict(lif_input_macs=nfeat * hidden[0], lif_input_macs_per_step=nfeat * hidden[0] * T,
                                        input_events=(inp["spikes_per_presentation"]["mean"] * hidden[0]) if inp else None,
                                        lif_synaptic_events=st1["mean"] * hidden[0] * hidden[1] + st2["mean"] * hidden[1] * 3,
                                        lif_leak_updates=sum(hidden) * T + 3 * T,
                                        mlp_macs=nfeat * hidden[0] + hidden[0] * hidden[1] + hidden[1] * 3, T=T, hidden=hidden))
    tv = ROWS / f"torch_vs_numpy_{_F.name}.json"
    out["torch_vs_numpy"] = json.loads(tv.read_text()) if tv.exists() else None
    out["git"] = git_info(); out["env"] = env_info(); out["finished"] = now()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"spikes_{_F.name}.json").write_text(json.dumps(out, indent=1))
    A = out["anchors"]
    print(f"B0 intact == stored rows: {'PASS' if A['b0_intact_equals_stored_rows'] else 'FAIL'} over {A['gaps']} gaps")
    for t in TEXTS:
        X = out["texts"][t]
        print(f"  [{t}] L1 {X['spikes']['layer1']['mean']:.2f} L2 {X['spikes']['layer2']['mean']:.2f} spikes/neuron; reading {X['reading_deployed']}; "
              + " ".join(f"{v} {100 * X['controls_deployed'][v]['agreement']:.2f}%" for v in V if v != "intact"))


def main(argv):
    global _F
    _F = Family(argv[0] if argv else "spk40")
    step = argv[1] if len(argv) > 1 else "all"
    steps = dict(forward=step_forward, decode=step_decode, torch=step_torch, stats=step_stats)
    for name in (steps if step == "all" else [step]):
        t0 = time.time(); print(f"== {_F.name} {name}", flush=True); steps[name](); print(f"== {name} done ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
