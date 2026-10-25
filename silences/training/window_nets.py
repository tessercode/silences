"""Training the LIF segmenter and the MLP control (Sections 7 and 8).

    python -m silences.training.window_nets lif 0        # retrain one seed and compare with the shipped model
    python -m silences.training.window_nets mlp 0

Both train on the same matrix: the 42 window features of every gap in the training corpus (random
letters, 1000 transmissions per operator, seed 13) with their gap labels, standardised by the
features' own mean and spread. Ten percent is held out for model selection: the weights kept are
those of the epoch with the lowest validation loss. Adam, cross-entropy, weight decay 1e-5.

Training runs on the CPU with six threads and torch's deterministic algorithms, and is bit for bit
reproducible: a retrained seed must produce a pickle identical to models/<family>/seed<N>.pkl. The
initialisation and the data order consume random numbers in a fixed order; do not reorder the steps.
(models/*/committed.pkl, the deployed models, were trained before the LIF's seed-before-build fix
and are artifacts of record; they are not reproduced by this script.)
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
from ..segmenters.lif import LIF   # noqa: E402
from ..segmenters.mlp import MLP   # noqa: E402

RETRAIN = OUTPUTS / "retrain"
FEATURES_SHA256 = "119dbc647d8d8a7e1415787baa21e6ccd39e924d5e6abf78b8944dc937a74085"


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
    assert sha == FEATURES_SHA256, "the training matrix is not the one the models were trained on"
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


def train(family, seed, outdir=RETRAIN):
    import torch
    torch.set_num_threads(OMP_THREADS)
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    F, y, f_sha = training_matrix()
    c = P.RANDOM_LETTERS[family]
    if family == "lif":
        m = fit_lif(LIF(W=P.TRAIN["W"], hidden=tuple(c["hidden"]), beta=c["beta"], threshold=c["threshold"], T=c["T"]),
                    F, y, c["epochs"], c["lr"], c["batch"], c["wd"], c["val"], seed)
    else:
        m = fit_mlp(MLP(W=P.TRAIN["W"], hidden=tuple(c["hidden"])),
                    F, y, c["epochs"], c["lr"], c["batch"], c["wd"], c["val"], seed)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{family}_seed{seed}.pkl"
    m.save(path)
    meta = dict(family=family, seed=seed, protocol_hash=P.RANDOM_LETTERS_HASH, protocol=c, train=P.TRAIN,
                feature_sha256=f_sha, pickle_sha256=sha256_file(path), wall_s=round(time.time() - t0, 1),
                finished=now(), git=git_info(), env=env_info())
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return path, meta


def main(argv):
    family, seed = argv[0], int(argv[1])
    path, meta = train(family, seed)
    shipped = MODELS / family / f"seed{seed}.pkl"
    same = meta["pickle_sha256"] == sha256_file(shipped)
    print(f"{family} seed {seed}: retrained in {meta['wall_s']:.0f}s, sha256 {meta['pickle_sha256'][:16]} "
          f"{'IDENTICAL to' if same else 'DIFFERS from'} {shipped.relative_to(MODELS.parent)}")
    if family == "lif" and seed == 0:          # the determinism record the evaluation quotes
        (RECORDS / "training").mkdir(parents=True, exist_ok=True)
        (RECORDS / "training" / "determinism.json").write_text(json.dumps(dict(
            deterministic=bool(same), max_abs_weight_diff=0.0 if same else None, checked=now(),
            protocol_hash=P.RANDOM_LETTERS_HASH, omp_threads=OMP_THREADS,
            method="LIF seed 0 retrained in a fresh process and compared byte for byte with the shipped model"), indent=2))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
