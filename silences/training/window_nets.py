"""Training the LIF segmenter, in both forms, and the MLP control (Sections 7 and 8).

    python -m silences.training.window_nets spk40 0 --corpus op12  # the network of record: the LIF fed spikes, T = 40
    python -m silences.training.window_nets lif 0                  # the original 5,000-transmission corpus
    python -m silences.training.window_nets mlp 0 --corpus op12    # the matched corpus: 12,000 operator transmissions
    python -m silences.training.window_nets lif 0 --corpus gen     # the BiGRU recipe's generic corpus
    python -m silences.training.window_nets spk80 0 --corpus op12  # the spike-fed network at T = 80
    python -m silences.training.window_nets lif80 0 --corpus op12  # the current-fed network at T = 80 (equal-T control)

The recipe is the same for every corpus: the 42 window features of every gap in the training corpus with
their gap labels, standardised by the features' own mean and spread; ten percent held out for model
selection (the weights kept are those of the epoch with the lowest validation loss); Adam, cross-entropy,
weight decay 1e-5. The corpora differ (Section 5.4): the original one is shared by every seed (random
letters, 1,000 transmissions per operator, seed 13); op12 gives each training seed its own 12,000
operator transmissions (2,400 per operator, corpus seed 210442 + training seed), the families of record
since v1.1.0; gen gives each seed the 12,000 generic transmissions of the BiGRU recipe's final phase
(generator seed 197442 + training seed, silences.training.bigru.generic_transmissions).

The spike-fed families (protocol/spike_fed.json: spk40, spk80, and the current-fed lif80) use the LIF recipe
unchanged on the op12 corpora; the spike-fed network encodes each batch into its ON/OFF spike trains on the
fly (silences.segmenters.lif.SpikeFedLIF.encode) and the first layer receives spikes.

Training runs on the CPU with six threads and torch's deterministic algorithms, and is bit for bit
reproducible: a retrained seed must produce a pickle identical to models/<family>/seed<N>.pkl (or
models/<family>/<corpus>/seed<N>.pkl). The initialisation and the data order consume random numbers in a
fixed order; do not reorder the steps. (models/*/deployed_5k.pkl, the first deployed models, were trained
before the LIF's seed-before-build fix and are artifacts of record; they are not reproduced by this script.)

snnTorch's Leaky applies the subtract reset one step after the spike; the numpy inference resets in the
same step. Training and inference are therefore slightly different dynamical systems for every LIF model
here; check_forward() reports their agreement and the records carry it (c2_forward), while the paper's
numbers are the inference path's, which the evaluation anchors by reproducing stored rows exactly.
"""
import os

OMP_THREADS = 6
os.environ.setdefault("OMP_NUM_THREADS", str(OMP_THREADS))   # before torch is imported

import hashlib   # noqa: E402
import json      # noqa: E402
import sys       # noqa: E402
import time      # noqa: E402

import numpy as np   # noqa: E402

from .. import MODELS, OUTPUTS, RECORDS   # noqa: E402
from ..corpus import corpus   # noqa: E402
from ..evaluation import protocol as P   # noqa: E402
from ..evaluation.provenance import env_info, git_info, now, sha256_file   # noqa: E402
from ..features import transmission_features   # noqa: E402
from ..segmenters.lif import LIF, SpikeFedLIF   # noqa: E402
from ..segmenters.mlp import MLP   # noqa: E402

RETRAIN = OUTPUTS / "retrain"
FEATURES_SHA256 = "119dbc647d8d8a7e1415787baa21e6ccd39e924d5e6abf78b8944dc937a74085"
# The same matrix rounded to 9 decimals: equal on any CPU, where the exact bits depend on which SIMD
# kernels numpy uses for log (AVX2 and AVX-512 can differ in the last bit).
FEATURES_SHA256_9DP = "0cb308eac5078c7e169a4b7acdfca093ac821f0539308d06a6cdf2e1a765c126"


def training_matrix():
    """(F, y, sha256 of F and y) for the protocol's training corpus, cached under outputs/."""
    cache = OUTPUTS / "training_matrix.npz"
    if cache.exists():
        z = np.load(cache)
        F, y = z["F"], z["y"]
    else:
        txs = corpus(P.TRAIN["per_fist"], P.TRAIN["corpus_seed"], "rand")
        F = np.concatenate([transmission_features(tx, P.TRAIN["W"]) for tx in txs])
        y = np.concatenate([tx["labels"] for tx in txs])
        OUTPUTS.mkdir(exist_ok=True)
        np.savez(cache, F=F, y=y)
    sha = hashlib.sha256(F.tobytes() + y.tobytes()).hexdigest()
    if sha != FEATURES_SHA256:
        rounded = hashlib.sha256(np.round(F, 9).tobytes() + y.tobytes()).hexdigest()
        assert rounded == FEATURES_SHA256_9DP, "the training matrix is not the one the models were trained on"
        print("note: this CPU's floating-point kernels give the training matrix in different last bits; it is the\n"
              "      same matrix to 9 decimals, but retrained models will not be bit-identical to the shipped ones", flush=True)
    return F, y, sha


def grid_matrix(arm, seed):
    """(F, y, sha256) for one grid corpus; the sha must equal the one recorded when the shipped model was trained."""
    from ..features import fe_tu_us
    if arm == "op12":
        n = P.GRID_N_TX // len(P.OPERATORS)
        txs = corpus(n, P.GRID_CORPUS_SEED0["op12"] + seed, "rand")
    else:
        from .bigru import generic_transmissions
        txs = generic_transmissions(P.GRID_N_TX, P.GRID_CORPUS_SEED0["gen"] + seed)
    assert all(fe_tu_us(tx["ticks"]) > 0 for tx in txs)
    F = np.concatenate([transmission_features(tx, P.TRAIN["W"]) for tx in txs]).astype(np.float32)
    y = np.concatenate([np.asarray(tx["labels"], dtype=np.int64) for tx in txs])
    sha = hashlib.sha256(F.tobytes() + y.tobytes()).hexdigest()
    rec = json.loads((RECORDS / "training" / "window_nets_grid.json").read_text())["corpora"][f"{arm}_{seed}"]
    assert len(txs) == rec["transmissions"] and len(y) == rec["gaps"], f"{arm} corpus {seed}: size differs from the record"
    if sha != rec["feature_sha256"]:
        print(f"note: the {arm} feature matrix for seed {seed} differs in its last bits from the recorded one "
              f"(this CPU's kernels); the retrained model will not be bit-identical", flush=True)
    return F, y, sha


def fit_mlp(m, F, y, epochs, lr, batch, wd, val, seed):
    import torch
    import torch.nn as nn
    F = np.asarray(F, np.float32)
    y = np.asarray(y, np.int64)
    m.mu = F.mean(0)
    m.sd = F.std(0) + 1e-6
    Z = (F - m.mu) / m.sd
    g = torch.Generator().manual_seed(seed)
    n = len(Z)
    nv = int(n * val)
    perm = torch.randperm(n, generator=g).numpy()
    tr, va = perm[nv:], perm[:nv]
    Zt, yt = torch.tensor(Z), torch.tensor(y)
    dims = [Z.shape[1]] + list(m.hidden) + [3]
    mods = []
    for a, b in zip(dims[:-1], dims[1:]):
        mods += [nn.Linear(a, b), nn.Tanh()]
    net = nn.Sequential(*mods[:-1])                  # no tanh after the output layer
    torch.manual_seed(seed)
    for mod in net:
        if isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            nn.init.zeros_(mod.bias)
    lossf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best = (1e9, None)
    for _ in range(epochs):
        net.train()
        idx = tr[torch.randperm(len(tr), generator=g).numpy()]
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            opt.zero_grad()
            loss = lossf(net(Zt[b]), yt[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Zt[va]), yt[va]).item()
        if vl < best[0]:
            best = (vl, [(mod.weight.detach().numpy().T.copy(), mod.bias.detach().numpy().copy())
                         for mod in net if isinstance(mod, nn.Linear)])
    m.layers = best[1]
    return m


def fit_lif(m, F, y, epochs, lr, batch, wd, val, seed):
    import torch
    import torch.nn as nn
    from snntorch import Leaky, surrogate
    F = np.asarray(F, np.float32)
    y = np.asarray(y, np.int64)
    m.mu = F.mean(0)
    m.sd = F.std(0) + 1e-6
    Z = torch.tensor((F - m.mu) / m.sd)
    yt = torch.tensor(y)
    torch.manual_seed(seed)                          # before the layers are built: the seed sets the initialisation
    sg = surrogate.fast_sigmoid()
    dims = [2 * (2 * m.W + 1)] + list(m.hidden) + [3]
    fcs = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
    lifs = nn.ModuleList([Leaky(beta=m.beta, threshold=m.threshold, spike_grad=sg, reset_mechanism="subtract")
                          for _ in m.hidden] +
                         [Leaky(beta=m.beta, threshold=1e9, spike_grad=sg, reset_mechanism="subtract")])   # readout

    def forward(x):
        mems = [lif.init_leaky() for lif in lifs]
        out = 0.0
        for _ in range(m.T):
            cur = x
            for i, (fc, lif) in enumerate(zip(fcs, lifs)):
                spk, mems[i] = lif(fc(cur), mems[i])
                cur = spk
            out = out + mems[-1]
        return out / m.T

    g = torch.Generator().manual_seed(seed)
    n = len(Z)
    nv = int(n * val)
    perm = torch.randperm(n, generator=g).numpy()
    tr, va = perm[nv:], perm[:nv]
    opt = torch.optim.Adam(list(fcs.parameters()), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    best = (1e9, None)
    for _ in range(epochs):
        fcs.train()
        idx = tr[torch.randperm(len(tr), generator=g).numpy()]
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            opt.zero_grad()
            loss = lossf(forward(Z[b]), yt[b])
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = lossf(forward(Z[va]), yt[va]).item()
        if vl < best[0]:
            best = (vl, [(fc.weight.detach().numpy().T.copy(), fc.bias.detach().numpy().copy()) for fc in fcs])
    m.fcs = best[1]
    return m


def fit_spk(m, F, y, epochs, lr, batch, wd, val, seed):
    """The spike-fed network: the LIF recipe with each batch encoded into its spike trains on the fly."""
    import torch
    import torch.nn as nn
    from snntorch import Leaky, surrogate
    F = np.asarray(F, np.float32)
    y = np.asarray(y, np.int64)
    m.mu = F.mean(0)
    m.sd = F.std(0) + 1e-6
    yt = torch.tensor(y)

    def enc(rows):
        return torch.tensor(m.encode(F[rows]), dtype=torch.float32)        # (T, b, 84)
    sg = surrogate.fast_sigmoid()
    dims = [2 * F.shape[1]] + list(m.hidden) + [3]
    torch.manual_seed(seed)
    fc = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
    lif = nn.ModuleList([Leaky(beta=m.beta, threshold=m.threshold, spike_grad=sg, reset_mechanism="subtract") for _ in m.hidden] +
                        [Leaky(beta=m.beta, threshold=1e9, spike_grad=sg, reset_mechanism="subtract")])

    def fwd(spk):
        mems = [l.init_leaky() for l in lif]
        out = 0.0
        for t in range(m.T):
            cur = spk[t]
            for i, (f, l) in enumerate(zip(fc, lif)):
                cur = f(cur)
                s, mems[i] = l(cur, mems[i])
                cur = s
            out = out + mems[-1]
        return out / m.T

    g = torch.Generator().manual_seed(seed)
    n = F.shape[0]
    nv = int(n * val)
    perm = torch.randperm(n, generator=g).numpy()
    tr, va = perm[nv:], perm[:nv]
    opt = torch.optim.Adam(fc.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    best = (1e9, None)
    for _ in range(epochs):
        idx = tr[torch.randperm(len(tr), generator=g).numpy()]
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            opt.zero_grad()
            loss = lossf(fwd(enc(b)), yt[b])
            loss.backward()
            opt.step()
        with torch.no_grad():
            vrows = va if nv else np.arange(batch)
            vl = lossf(fwd(enc(vrows)), yt[vrows]).item()
        if vl < best[0]:
            best = (vl, [(f.weight.detach().numpy().T.copy(), f.bias.detach().numpy().copy()) for f in fc])
    m.fcs = best[1]
    return m


def torch_logits(m, F):
    """The training-side forward of a trained LIF (either form), rebuilt in snnTorch from the saved weights."""
    import torch
    import torch.nn as nn
    from snntorch import Leaky
    Z = torch.tensor(((np.asarray(F, np.float64) - m.mu) / m.sd).astype(np.float32))
    fcs = []
    for Wt, b in m.fcs:
        lin = nn.Linear(Wt.shape[0], Wt.shape[1])
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(Wt.T)); lin.bias.copy_(torch.tensor(b))
        fcs.append(lin)
    lifs = [Leaky(beta=m.beta, threshold=m.threshold, reset_mechanism="subtract") for _ in m.hidden] + \
           [Leaky(beta=m.beta, threshold=1e9, reset_mechanism="subtract")]
    mems = [l.init_leaky() for l in lifs]
    out = 0.0
    spk_in = torch.tensor(m.encode(F), dtype=torch.float32) if hasattr(m, "encode") else None
    with torch.no_grad():
        for t in range(m.T):
            cur = spk_in[t] if spk_in is not None else Z
            for i, (f, l) in enumerate(zip(fcs, lifs)):
                cur = f(cur)
                s, mems[i] = l(cur, mems[i])
                cur = s
            out = out + mems[-1]
    return (out / m.T).numpy()


C2_BATCH = 2048


def check_forward(m, F, rng):
    """Reported, not gating: the inference forward against the training-side forward on a held batch."""
    rows = rng.choice(len(F), size=min(C2_BATCH, len(F)), replace=False)
    a = m.logits(F[rows]); b = torch_logits(m, F[rows])
    return dict(batch=int(len(rows)), max_abs_logit_diff=float(np.max(np.abs(a - b))),
                argmax_agreement=float(np.mean(a.argmax(1) == b.argmax(1))),
                note="snnTorch resets one step after the spike; numpy inference resets in the same step")


SPIKE_FED = ("spk40", "spk80", "lif80")


def train(family, seed, outdir=RETRAIN, arm=None):
    import torch
    torch.set_num_threads(OMP_THREADS)
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    F, y, f_sha = training_matrix() if arm is None else grid_matrix(arm, seed)
    if family in SPIKE_FED:
        assert arm == P.SPIKE_FED_CORPUS, f"{family} trains on the {P.SPIKE_FED_CORPUS} corpora"
        c = P.SPIKE_FED_ARMS[family]
        if c["model"] == "SpikeInputSNN":
            m = fit_spk(SpikeFedLIF(W=P.TRAIN["W"], hidden=tuple(c["hidden"]), beta=c["beta"], threshold=c["threshold"], T=c["T"], scale=c["scale"]),
                        F, y, c["epochs"], c["lr"], c["batch"], c["wd"], c["val"], seed)
        else:
            m = fit_lif(LIF(W=P.TRAIN["W"], hidden=tuple(c["hidden"]), beta=c["beta"], threshold=c["threshold"], T=c["T"]),
                        F, y, c["epochs"], c["lr"], c["batch"], c["wd"], c["val"], seed)
    else:
        c = P.RANDOM_LETTERS[family]
        if family == "lif":
            m = fit_lif(LIF(W=P.TRAIN["W"], hidden=tuple(c["hidden"]), beta=c["beta"], threshold=c["threshold"], T=c["T"]),
                        F, y, c["epochs"], c["lr"], c["batch"], c["wd"], c["val"], seed)
        else:
            m = fit_mlp(MLP(W=P.TRAIN["W"], hidden=tuple(c["hidden"])),
                        F, y, c["epochs"], c["lr"], c["batch"], c["wd"], c["val"], seed)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / (f"{family}_seed{seed}.pkl" if arm is None else f"{family}_{arm}_seed{seed}.pkl")
    m.save(path)
    if family in SPIKE_FED:
        c2 = check_forward(m, F, np.random.default_rng(213 + seed))
        meta = dict(family=family, seed=seed, corpus_seed=P.GRID_CORPUS_SEED0[arm] + seed, protocol=c,
                    base_protocol_hash=P.RANDOM_LETTERS_HASH, exp211_protocol_hash=P.GRID_WINDOW_HASH, feature_sha256=f_sha,
                    gaps=int(len(y)), pickle_sha256=sha256_file(path), c2_forward=c2, wall_s=round(time.time() - t0, 1),
                    finished=now(), git=git_info(), env=env_info(), corpus=arm)
    else:
        meta = dict(family=family, seed=seed, corpus=arm or "record", protocol_hash=P.RANDOM_LETTERS_HASH, protocol=c, train=P.TRAIN,
                    feature_sha256=f_sha, pickle_sha256=sha256_file(path), wall_s=round(time.time() - t0, 1),
                    finished=now(), git=git_info(), env=env_info())
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return path, meta


def main(argv):
    family, seed = argv[0], int(argv[1])
    arm = argv[argv.index("--corpus") + 1] if "--corpus" in argv else None
    assert family in ("lif", "mlp") + SPIKE_FED, family
    assert arm in (None, "op12", "gen"), arm
    path, meta = train(family, seed, arm=arm)
    shipped = MODELS / family / f"seed{seed}.pkl" if arm is None else MODELS / family / arm / f"seed{seed}.pkl"
    same = meta["pickle_sha256"] == sha256_file(shipped)
    print(f"{family} seed {seed}: retrained in {meta['wall_s']:.0f}s, sha256 {meta['pickle_sha256'][:16]} "
          f"{'IDENTICAL to' if same else 'DIFFERS from'} {shipped.relative_to(MODELS.parent)}")
    if family == "lif" and seed == 0 and arm is None:          # the determinism record the evaluation quotes
        (RECORDS / "training").mkdir(parents=True, exist_ok=True)
        (RECORDS / "training" / "determinism.json").write_text(json.dumps(dict(
            deterministic=bool(same), max_abs_weight_diff=0.0 if same else None, checked=now(),
            protocol_hash=P.RANDOM_LETTERS_HASH, omp_threads=OMP_THREADS,
            method="LIF seed 0 retrained in a fresh process and compared byte for byte with the shipped model"), indent=2))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
