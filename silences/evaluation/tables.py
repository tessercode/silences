"""Posterior tables: every segmenter's per-gap log-posteriors on every evaluation corpus, computed once.

    python -m silences.evaluation.tables

Ten corpora: for each text (random letters, English), its calibration corpus (seeds 4242, 4243) and
the four evaluation corpora (seeds 999, 777, 555, 321), 250 transmissions per operator each. Every
later step (decoder-weight fits, evaluation, diagnostics) decodes from these tables, so the segmenters
run once. Tables are cached under outputs/tables/ and are regenerated, never shipped.

The matched-data record (silences.evaluation.matched) uses a second set, outputs/tables_matched/: the
same ten corpora plus four fresh evaluation corpora per text (seeds 1106, 2212, 3318, 4424), with every
row of the grid added (producers.grid_producers). Rows already in outputs/tables/ are copied, not recomputed.
The spike-fed record (silences.evaluation.spike_fed) uses a third, outputs/tables_spike_fed/: the matched
set's eighteen corpora with the spike-fed rows added (producers.spike_fed_producers); the matched rows are
copied from outputs/tables_matched/, which must exist.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per worker
    os.environ.setdefault(_v, "1")

import pickle            # noqa: E402
import sys               # noqa: E402

import numpy as np       # noqa: E402
import time              # noqa: E402
from multiprocessing import get_context   # noqa: E402

from .. import OUTPUTS, ROOT   # noqa: E402
from ..corpus import corpus   # noqa: E402
from . import protocol as P   # noqa: E402
from .producers import all_producers, grid_producers, spike_fed_producers   # noqa: E402

TABLES = OUTPUTS / "tables"
TABLES_MATCHED = OUTPUTS / "tables_matched"
TABLES_SPIKE_FED = OUTPUTS / "tables_spike_fed"
_DIRS = {False: TABLES, True: TABLES_MATCHED, "spike_fed": TABLES_SPIKE_FED}
_HASHES = {False: P.DECODER_WEIGHTS_HASH, True: P.MATCHED_HASH, "spike_fed": P.SPIKE_FED_HASH}
WORKERS = int(os.environ.get("SILENCES_WORKERS", "20"))


def corpora(matched=False):
    """[(text, kind, seed)]: one calibration and four evaluation corpora per text; with matched=True also
    the four fresh evaluation corpora per text (kind "fresh"; the record ones are then kind "record")."""
    out = []
    for t in P.TEXTS:
        out.append((t, "calib", P.CALIB_SEED[t]))
        out += [(t, "record" if matched else "eval", s) for s in P.EVAL["seeds"]]
        if matched:
            out += [(t, "fresh", s) for s in P.FRESH_SEEDS]
    return out


def table_path(text, seed, matched=False):
    """matched is False (the ten corpora), True (the matched set) or "spike_fed" (the spike-fed set)."""
    return _DIRS[matched] / f"{text}_{seed}.pkl"


_MATCHED = False


def _producers(matched):
    if matched == "spike_fed":
        return spike_fed_producers()
    prods, sources = all_producers()
    if matched:
        g, gs = grid_producers()
        prods.update(g); sources.update(gs)
    return prods, sources


def _job(args):
    text, seed, row = args
    prod = _producers(_MATCHED)[0][row]
    return text, seed, row, [prod(tx) for tx in corpus(P.EVAL["per_fist"], seed, text)]


def build(force=False, matched=False):
    global _MATCHED
    _MATCHED = matched
    d = _DIRS[matched]
    d.mkdir(parents=True, exist_ok=True)
    todo = [(t, s) for t, _, s in corpora(bool(matched)) if force or not table_path(t, s, matched).exists()]
    if not todo:
        return
    rows, sources = _producers(matched)
    jobs, reused, prev_sources = [], {}, {}
    for t, s in todo:
        have = {}
        if matched == "spike_fed":                            # the matched set's rows: copied, never recomputed
            prev = load(t, s, matched=True)
            have, prev_sources[(t, s)] = prev["rows"], prev["sources"]
        elif matched and table_path(t, s).exists():            # the record rows on the ten corpora: copy, do not recompute
            prev = load(t, s)
            have = prev["rows"]
        reused[(t, s)] = have
        jobs += [(t, s, r) for r in rows if r not in have]
    got = {}
    t0 = time.time()
    with get_context("fork").Pool(WORKERS) as pool:
        for i, (t, s, r, lp) in enumerate(pool.imap_unordered(_job, jobs), 1):
            got[(t, s, r)] = lp
            if i % 33 == 0:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    for t, s in todo:
        txs = corpus(P.EVAL["per_fist"], s, t)
        truth = [" ".join(tx["words"]) for tx in txs]
        ch = [[int(c) for c in tx["ch"]] for tx in txs]
        if reused[(t, s)]:
            prev = load(t, s, matched=True if matched == "spike_fed" else False)
            assert prev["truth"] == truth and prev["ch"] == ch, f"{t}_{s}: the corpus does not reproduce the cached table's"
        all_rows = {**reused[(t, s)], **{r: got[(t, s, r)] for r in rows}} if matched == "spike_fed" else \
            {r: (reused[(t, s)][r] if r in reused[(t, s)] else got[(t, s, r)]) for r in rows}
        doc = dict(text=t, seed=s, per_fist=P.EVAL["per_fist"], protocol_hash=_HASHES[matched],
                   truth=truth, fist=[tx["fist"] for tx in txs], ch=ch, rows=all_rows,
                   sources={**prev_sources.get((t, s), {}), **{r: (str(p.relative_to(ROOT)) if p else None) for r, p in sources.items()}})
        if matched:
            doc["ticks"] = [np.asarray(tx["ticks"], dtype=np.int64) for tx in txs]
        with open(table_path(t, s, matched), "wb") as fh:
            pickle.dump(doc, fh)


def load(text, seed, matched=False):
    with open(table_path(text, seed, matched), "rb") as fh:
        doc = pickle.load(fh)
    assert doc["protocol_hash"] == _HASHES[matched]
    return doc


if __name__ == "__main__":
    matched = "spike_fed" if "--spike-fed" in sys.argv[1:] else "--matched" in sys.argv[1:]
    build(force="--force" in sys.argv[1:], matched=matched)
    print(f"tables -> {_DIRS[matched]}")
