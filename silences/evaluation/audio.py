"""An off-the-shelf software decoder on audio rendered from the evaluation corpora (Section 6.4, Table 3).

The corpora hold completion spikes only, so each transmission is rendered again with its marks' on and off
times (corpus(..., with_marks=True); anchor R4: the spikes equal the corpus's on every transmission) and
keyed as a sine tone with no noise. multimon-ng decodes the raw 16-bit audio in its MORSE_CW mode, and the
decoded text is scored by the paper's CER / WER against the keyed words. Two settings: the decoder's own
automatic timing, and its dit and gap lengths set to the transmission's true mean time unit, which no
receiver could know.

Needs the multimon-ng program on the PATH (the record was made with version 1.3.0); everything else in the
package runs without it. Called by silences.evaluation.nonlearned (step "audio").
"""
import re
import subprocess
import time
from multiprocessing import get_context

import numpy as np

from ..corpus import corpus
from . import protocol as P
from .metrics import cer_wer

PROTO = P.NONLEARNED["audio"]
WORKERS = 12
OPS = P.OPERATORS


def corpus_with_marks(text, seed):
    """The record corpus of (text, seed), each transmission with its marks; the number whose spikes equal the corpus's."""
    txs = corpus(P.EVAL["per_fist"], seed, text, with_marks=True)
    ref = corpus(P.EVAL["per_fist"], seed, text)
    n_equal = 0
    for tx, r in zip(txs, ref):
        t_us = np.array([round(m[1] * 1e6) for m in tx["marks"]], dtype=np.int64)
        assert np.array_equal(t_us, np.asarray(r["ticks"])) and np.array_equal(tx["ticks"], r["ticks"]) and tx["words"] == r["words"], \
            "R4: the mark-recording renderer diverged from the corpus"
        n_equal += 1
    return txs, n_equal


def render_audio(marks, sr=PROTO["sample_rate"], f0=PROTO["tone_hz"], amp=PROTO["amplitude"]):
    """A keyed sine tone: each mark with raised-cosine edges, silence before and after."""
    lead, tail, edge = PROTO["lead_s"], PROTO["tail_s"], PROTO["edge_ms"] / 1000.0
    n = int((lead + marks[-1][1] + tail) * sr) + 1
    env = np.zeros(n)
    ne = max(int(edge * sr), 1)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ne)))
    for a, b in marks:
        i, j = int((lead + a) * sr), int((lead + b) * sr)
        if j - i < 2 * ne:
            env[i:j] = 1.0
        else:
            env[i:i + ne] = np.maximum(env[i:i + ne], ramp); env[i + ne:j - ne] = 1.0
            env[j - ne:j] = np.maximum(env[j - ne:j], ramp[::-1])
    t = np.arange(n) / sr
    return (amp * env * np.sin(2 * np.pi * f0 * t)).astype("<i2")


def normalise(s):
    s = re.sub(r"[^A-Z ]", " ", s.upper())
    return re.sub(r" +", " ", s).strip()


def decode(audio, extra_args):
    p = subprocess.run(["multimon-ng", "-q", "-c", "-a", "MORSE_CW", "-t", "raw", "-"] + list(extra_args),
                       input=audio.tobytes(), capture_output=True)
    return normalise(p.stdout.decode(errors="replace")), p.returncode


def _one(args):
    i, fist, words, marks, tu_us = args
    audio = render_audio(marks)
    truth = " ".join(words)
    out = {}
    for name, tmpl in PROTO["settings"].items():
        extra = [a.format(tu_ms=max(int(round(tu_us / 1000.0)), 1)) for a in tmpl]
        dec, rc = decode(audio, extra)
        ce, nc, we, nw = cer_wer(truth, dec)
        out[name] = dict(dec=dec, rc=rc, ce=ce, nc=nc, we=we, nw=nw, nonempty=int(bool(dec)))
    return i, fist, truth, out


def run():
    t0 = time.time()
    h = subprocess.run(["multimon-ng", "-h"], capture_output=True)
    ver = (h.stdout + h.stderr).decode(errors="replace").splitlines()[0].strip()
    res = dict(protocol=PROTO, decoder_version=ver, corpora={}, r4_parity={})
    for text in P.TEXTS:
        for seed in P.EVAL["seeds"]:
            txs, n_eq = corpus_with_marks(text, seed)
            res["r4_parity"][f"{text}_{seed}"] = dict(transmissions=len(txs), equal=n_eq)
            jobs = [(i, tx["fist"], tx["words"], tx["marks"], tx["true_tu_us"]) for i, tx in enumerate(txs)]
            with get_context("fork").Pool(WORKERS) as pool:
                rows = sorted(pool.imap_unordered(_one, jobs), key=lambda r: r[0])
            agg = {}
            for _, fist, truth, out in rows:
                for name, o in out.items():
                    a = agg.setdefault(name, {}).setdefault(fist, [0, 0, 0, 0, 0, 0])
                    for k, v in enumerate((o["ce"], o["nc"], o["we"], o["nw"], o["nonempty"], 1)):
                        a[k] += v
            samples = [dict(fist=r[1], truth=r[2], **{n: o["dec"] for n, o in r[3].items()}) for r in rows[:5]]
            res["corpora"][f"{text}_{seed}"] = dict(sums=agg, samples=samples, returncodes=sorted({o["rc"] for r in rows for o in r[3].values()}))
            print(f"  audio {text} {seed}: {len(txs)} transmissions, R4 {n_eq}/{len(txs)} " +
                  ", ".join(f"{n} CER {100 * sum(v[0] for v in a.values()) / sum(v[1] for v in a.values()):.1f}" for n, a in agg.items()) +
                  f" ({time.time() - t0:.0f}s)", flush=True)
    res["texts"] = {}
    for text in P.TEXTS:                      # per text: the mean over the four record corpora, overall and per operator
        keys = [f"{text}_{s}" for s in P.EVAL["seeds"]]
        T = {}
        for name in PROTO["settings"]:
            def pack(sel):
                vals = []
                for k in keys:
                    S = res["corpora"][k]["sums"][name]
                    v = [sum(S[f][i] for f in S if sel(f)) for i in range(6)]
                    vals.append(dict(cer=100 * v[0] / v[1], wer=100 * v[2] / v[3], nonempty=100 * v[4] / v[5]))
                return dict(cer=float(np.mean([x["cer"] for x in vals])), wer=float(np.mean([x["wer"] for x in vals])),
                            cer_sd=float(np.std([x["cer"] for x in vals], ddof=1)), wer_sd=float(np.std([x["wer"] for x in vals], ddof=1)),
                            nonempty=float(np.mean([x["nonempty"] for x in vals])), n=len(vals))
            T[name] = dict(overall=pack(lambda f: True), per_fist={f: pack(lambda g, f=f: g == f) for f in OPS})
        res["texts"][text] = T
    res["wall_s"] = round(time.time() - t0, 1)
    return res
