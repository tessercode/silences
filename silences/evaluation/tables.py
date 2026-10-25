"""Posterior tables: every segmenter's per-gap log-posteriors on every evaluation corpus, computed once.

    python -m silences.evaluation.tables

Ten corpora: for each text (random letters, English), its calibration corpus (seeds 4242, 4243) and
the four evaluation corpora (seeds 999, 777, 555, 321), 250 transmissions per operator each. Every
later step (decoder-weight fits, evaluation, diagnostics) decodes from these tables, so the segmenters
run once. Tables are cached under outputs/tables/ and are regenerated, never shipped.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per worker
    os.environ.setdefault(_v, "1")

import pickle            # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402
from multiprocessing import get_context   # noqa: E402

from .. import OUTPUTS, ROOT   # noqa: E402
from ..corpus import corpus   # noqa: E402
from . import protocol as P   # noqa: E402
from .producers import all_producers   # noqa: E402

TABLES = OUTPUTS / "tables"
WORKERS = int(os.environ.get("SILENCES_WORKERS", "20"))


def corpora():
    """[(text, kind, seed)]: one calibration and four evaluation corpora per text."""
    out = []
    for t in P.TEXTS:
        out.append((t, "calib", P.CALIB_SEED[t]))
        out += [(t, "eval", s) for s in P.EVAL["seeds"]]
    return out


def table_path(text, seed):
    return TABLES / f"{text}_{seed}.pkl"


def _job(args):
    text, seed, row = args
    prod = all_producers()[0][row]
    return text, seed, row, [prod(tx) for tx in corpus(P.EVAL["per_fist"], seed, text)]


def build(force=False):
    TABLES.mkdir(parents=True, exist_ok=True)
    todo = [(t, s) for t, _, s in corpora() if force or not table_path(t, s).exists()]
    if not todo:
        return
    rows, sources = all_producers()
    jobs = [(t, s, r) for t, s in todo for r in rows]
    got = {}
    t0 = time.time()
    with get_context("fork").Pool(WORKERS) as pool:
        for i, (t, s, r, lp) in enumerate(pool.imap_unordered(_job, jobs), 1):
            got[(t, s, r)] = lp
            if i % 33 == 0:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    for t, s in todo:
        txs = corpus(P.EVAL["per_fist"], s, t)
        doc = dict(text=t, seed=s, per_fist=P.EVAL["per_fist"], protocol_hash=P.DECODER_WEIGHTS_HASH,
                   truth=[" ".join(tx["words"]) for tx in txs], fist=[tx["fist"] for tx in txs],
                   ch=[[int(c) for c in tx["ch"]] for tx in txs], rows={r: got[(t, s, r)] for r in rows},
                   sources={r: (str(p.relative_to(ROOT)) if p else None) for r, p in sources.items()})
        with open(table_path(t, s), "wb") as fh:
            pickle.dump(doc, fh)


def load(text, seed):
    with open(table_path(text, seed), "rb") as fh:
        doc = pickle.load(fh)
    assert doc["protocol_hash"] == P.DECODER_WEIGHTS_HASH
    return doc


if __name__ == "__main__":
    build(force="--force" in sys.argv[1:])
    print(f"tables -> {TABLES}")
