"""Training the bidirectional-GRU baseline (Section 6): an eight-seed pool, the five best kept.

    python -m silences.training.bigru            # needs a CUDA GPU; about five minutes on one
    python -m silences.training.bigru --smoke    # a few epochs on small corpora, to check the pipeline

Each seed trains for 180 epochs through four phases of growing difficulty: two-letter, two-word
transmissions with no jitter, then jitter up to 0.2 and 0.3 time units, then two to four words of
two to six letters with jitter up to 0.45. Class weights favour the rare classes (inter-word 7.5
times the inter-letter weight) and are relaxed phase by phase. After the first phase a gate aborts a
seed whose validation recall collapsed to zero for any class. The surviving seeds are ranked by
balanced accuracy on a selection corpus rendered by the five-operator style model; the top five form
the deployed ensemble (models/bigru/pool/, DEPLOYED_SEEDS), checked once on a disjoint test corpus.

The training corpus has its own renderer (generate_sentence below), the one the shipped checkpoints
were trained on: per-word speed uniform on [0.5, 1.5] and dash ratio uniform on [2.5, 4.5] rather
than the five operators, letters and digits drawn uniformly, and gap jitter 0.8924 times the element
jitter. The front end's estimate here is the float one (tenth-percentile interval / 2, times 0.9).

On the GPU and software the checkpoints were trained with (an RTX 5080, torch 2.11.0 with CUDA
12.8) the retrain is bit-identical: every weight equals the shipped checkpoint's. bfloat16 kernels
differ across GPUs and library versions, so elsewhere the retrain is checked statistically against
records/training/bigru_pool.json. Checkpoints go to outputs/retrain/bigru/, never over the shipped ones.
"""
import contextlib
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import OUTPUTS, RECORDS
from ..morse import MORSE_TABLE
from ..operators import OPERATORS

SMOKE = "--smoke" in sys.argv[1:]
OUT = OUTPUTS / "retrain" / "bigru"

# ---- the recipe ----
POOL_SEEDS = list(range(42, 50))
N_SELECTED = 5
N_TX_TRAIN, N_TX_GATEVAL = 12000, 400                  # transmissions per phase per seed; gate validation
N_TX_SELECT_FIST, N_TX_TEST_FIST, N_TX_TEST_RAND = 300, 400, 1500
N_EPOCHS_TOTAL = 180
EP_1B, EP_2A, EP_2B = 30, 60, 90                       # epochs at which phases 1B, 2A, 2B start
N_EPOCHS_PHASE2B = 90
K_IW = 7.5                                             # inter-word weight, relative to inter-letter
GATE_FLOOR = 0.05                                      # after phase 1A every class recall must exceed this
BW_PHASE1A, BW_PHASE1B, BW_PHASE2A, BW_PHASE2B = 16.0, 8.0, 6.0, 4.0   # inter-letter class weight per phase
BALACC_FLOOR = 0.939
CONSISTENCY_BAR = 0.005
BATCH_SIZE, LR_INIT = 64, 5e-4

T_U = 60e-3
T_THRESH = 1.92
SIGMA_STAR = 0.450
OMEGA_LO, OMEGA_HI = 0.5, 1.5
R_LO, R_HI = 2.5, 4.5
SIGMA_GAP_RATIO = 0.8924
WORD_GAP_TU = 7.0
BETA_ISI = np.array([0.9885, 0.9942, 0.9962, 0.9974, 0.9981, 0.9987])
V_KICK = 1.0
GAP_TAU_DUR, GAP_TAU_DOT, GAP_TAU_DASH, GAP_S = 3.5, 3.0, 5.0, 0.6
TU_PCTILE, TU_DIVISOR, TU_SCALAR_C = 10.0, 2.0, 0.90
N_INPUT, N_HIDDEN, N_OUTPUT = 9, 32, 3

SIGMA_P1A, SIGMA_P1B, SIGMA_P2A, SIGMA_P2B = (0.0, 0.0), (0.0, 0.2), (0.0, 0.3), (0.0, 0.45)
NW_P1, WL_P1 = [2], [2]
NW_P2B, WL_P2B = [2, 3, 4], [2, 3, 4, 5, 6]
LETTERS = sorted(MORSE_TABLE.keys())                   # letters and digits
MAX_T = max(NW_P2B) * max(WL_P2B) * max(len(v) for v in MORSE_TABLE.values()) + 2

# corpus seeds: independent of the model seeds; the training phases offset by the model seed
RNG_TRAIN_P1A, RNG_TRAIN_P1B, RNG_TRAIN_P2A, RNG_TRAIN_P2B = 197_100, 197_200, 197_300, 197_400
RNG_GATE, RNG_SELECT, RNG_TEST_FIST, RNG_TEST_RAND = 197_444, 197_555, 197_888, 197_777
SELECTION_OPERATORS = ["fast_normalr", "slow_normalr", "high_r_bug", "cootie_high_jitter"]

if SMOKE:
    N_EPOCHS_TOTAL, EP_1B, EP_2A, EP_2B, N_EPOCHS_PHASE2B = 6, 2, 4, 5, 1
    N_TX_TRAIN, N_TX_GATEVAL = 400, 100
    N_TX_SELECT_FIST, N_TX_TEST_FIST, N_TX_TEST_RAND = 30, 40, 100
    POOL_SEEDS, N_SELECTED = [42, 43, 44], 2
    GATE_FLOOR = -1.0          # two epochs cannot leave the all-inter-word start; let the pipeline run


# ---- the training renderer and its features ----
def generate_sentence(n_words_list, wlen_list, t_u, sigma_frac, t_thresh_frac, rng):
    n_words = int(rng.choice(n_words_list))
    spikes, spike_letter, spike_word, tu_words = [], [], [], []
    t_cursor, gletter = 0.0, -1
    for wi in range(n_words):
        omega = rng.uniform(OMEGA_LO, OMEGA_HI)
        r_val = rng.uniform(R_LO, R_HI)
        t_u_w = t_u * omega
        tu_words.append(t_u_w)
        t_thresh = t_thresh_frac * t_u_w
        sigma_elem = sigma_frac * t_u_w
        sigma_gap = sigma_frac * SIGMA_GAP_RATIO * t_u_w
        wlen = int(rng.choice(wlen_list))
        for ci in range(wlen):
            char = str(rng.choice(LETTERS))
            gletter += 1
            elements = MORSE_TABLE[char]
            for eidx, element in enumerate(elements):
                t_ideal = (1.0 if element == '.' else r_val) * t_u
                noise = rng.normal(0.0, sigma_elem) if sigma_elem > 0 else 0.0
                t_dur = max(t_ideal * omega + noise, 0.1 * t_u_w)
                t_spike = t_cursor + t_dur
                spikes.append((t_spike, 0 if t_dur < t_thresh else 1))
                spike_letter.append(gletter)
                spike_word.append(wi)
                if eidx < len(elements) - 1:
                    g = rng.normal(0.0, sigma_gap) if sigma_gap > 0 else 0.0
                    t_cursor = t_spike + max(1.0 * t_u_w + g, 0.1 * t_u_w)
                else:
                    t_cursor = t_spike
            g = rng.normal(0.0, sigma_gap) if sigma_gap > 0 else 0.0
            gap_nom = WORD_GAP_TU if (ci == wlen - 1 and wi < n_words - 1) else 3.0
            t_cursor += max(gap_nom * t_u_w + g, 0.1 * t_u_w)
    if len(spikes) < 2:
        return spikes, [], tu_words
    labels = [0 if spike_letter[i + 1] == spike_letter[i] else (1 if spike_word[i + 1] == spike_word[i] else 2)
              for i in range(len(spikes) - 1)]
    return spikes, labels, tu_words


def features(spikes, t_u_norm_s):
    """The nine float features per gap (the fixed-point versions are segmenters.bigru.features_q)."""
    rows = []
    for i in range(len(spikes) - 1):
        isi_tu = (spikes[i + 1][0] - spikes[i][0]) / t_u_norm_s
        ch_next = float(spikes[i + 1][1])
        pop = V_KICK * (BETA_ISI ** (isi_tu * 60.0))
        dur_gap = 1.0 / (1.0 + np.exp(-(isi_tu - GAP_TAU_DUR) / GAP_S))
        tau_bit = GAP_TAU_DASH if ch_next >= 0.5 else GAP_TAU_DOT
        durbit = 1.0 / (1.0 + np.exp(-(isi_tu - tau_bit) / GAP_S))
        rows.append(np.array(list(pop) + [ch_next, dur_gap, durbit], dtype=np.float32))
    return np.stack(rows) if rows else np.zeros((0, N_INPUT), dtype=np.float32)


def estimate_tu(spikes):
    if len(spikes) < 2:
        return None
    isis = np.diff(np.array([s[0] for s in spikes], dtype=float))
    isis = isis[isis > 0]
    return float(np.percentile(isis, TU_PCTILE) / TU_DIVISOR) if isis.size else None


def build_dataset(n_tx, sigma_min, sigma_max, n_words_list, wlen_list, rng_seed):
    rng, samples, tu_ratios = np.random.default_rng(rng_seed), [], []
    for _ in range(n_tx):
        sigma_frac = rng.uniform(sigma_min, sigma_max) if sigma_max > sigma_min else sigma_min
        spikes, labels, tu_words = generate_sentence(n_words_list, wlen_list, T_U, sigma_frac, T_THRESH, rng)
        if len(spikes) < 2 or not labels:
            continue
        est = estimate_tu(spikes)
        if est is None or est <= 0:
            continue
        x = features(spikes, TU_SCALAR_C * est)
        if len(x) == 0:
            continue
        tu_ratios.append(TU_SCALAR_C * est / float(np.mean(tu_words)))
        samples.append((x, np.array(labels, dtype=np.int64)))
    return samples, np.array(tu_ratios)


# ---- the selection and test corpora (the operator style model) ----
def render_selection(words, op, rng):
    """The style renderer the selection corpus was made with (no inter-word gap after the last word)."""
    t_u_base, r_val = 1.2 / op["wpm"], op["r"]
    spikes, lidx, widx = [], [], []
    t_cursor, gletter = 0.0, -1
    for wi, word in enumerate(words):
        omega_word = max(0.2, rng.normal(1.0, op["omega"]))
        t_u = t_u_base * omega_word
        t_thresh = T_THRESH * t_u
        se, sg = op["sigma_elem"] * t_u, op["sigma_gap"] * t_u
        for ci, ch in enumerate(word):
            if ch not in MORSE_TABLE:
                continue
            gletter += 1
            for eidx, el in enumerate(MORSE_TABLE[ch]):
                t_ideal = (1.0 if el == "." else r_val) * t_u
                t_dur = max(t_ideal + (rng.normal(0.0, se) if se > 0 else 0.0), 0.1 * t_u)
                t_spike = t_cursor + t_dur
                spikes.append((t_spike, 0 if t_dur < t_thresh else 1))
                lidx.append(gletter)
                widx.append(wi)
                if eidx < len(MORSE_TABLE[ch]) - 1:
                    g = rng.normal(0.0, sg) if sg > 0 else 0.0
                    t_cursor = t_spike + max(1.0 * t_u + g, 0.1 * t_u)
                else:
                    t_cursor = t_spike
            g = rng.normal(0.0, sg) if sg > 0 else 0.0
            gap_nom = WORD_GAP_TU if (ci == len(word) - 1 and wi < len(words) - 1) else 3.0
            t_cursor += max(gap_nom * t_u + g, 0.1 * t_u)
    labels = [0 if lidx[i + 1] == lidx[i] else (1 if widx[i + 1] == widx[i] else 2) for i in range(len(spikes) - 1)]
    return spikes, np.array(labels, dtype=np.int64)


def operator_corpus(n_per_fist, rng_seed):
    from nltk.corpus import words as nltk_words
    vocab = sorted({w.upper() for w in nltk_words.words() if 2 <= len(w) <= 8 and w.isalpha()})
    rng = np.random.default_rng(rng_seed)
    out = {name: [] for name in SELECTION_OPERATORS}
    for name in SELECTION_OPERATORS:
        for _ in range(n_per_fist):
            nw = int(rng.integers(2, 7))
            words = [str(vocab[rng.integers(0, len(vocab))]) for _ in range(nw)]
            spikes, labels = render_selection(words, OPERATORS[name], rng)
            if len(spikes) >= 2 and len(labels):
                out[name].append((spikes, labels))
    return out


def random_corpus(n_tx, rng_seed):
    rng, out = np.random.default_rng(rng_seed), []
    for _ in range(n_tx):
        spikes, labels, _ = generate_sentence(NW_P2B, WL_P2B, T_U, SIGMA_STAR, T_THRESH, rng)
        if len(spikes) >= 2 and labels:
            out.append((spikes, np.array(labels, dtype=np.int64)))
    return out


def encode(spike_sets):
    enc = []
    for spikes, labels in spike_sets:
        est = estimate_tu(spikes)
        if est is not None and est > 0:
            x = features(spikes, TU_SCALAR_C * est)
            if len(x):
                enc.append((x, labels))
    return enc


# ---- the model and its training ----
class BidirectionalSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=N_INPUT, hidden_size=N_HIDDEN, num_layers=1, batch_first=False, bidirectional=True)
        self.fc = nn.Linear(N_HIDDEN * 2, N_OUTPUT)

    def forward(self, x_seq):
        return self.fc(self.gru(x_seq)[0])


def training_tensors(samples):
    X = torch.zeros(len(samples), MAX_T, N_INPUT, dtype=torch.float32)
    y = torch.full((len(samples), MAX_T), -1, dtype=torch.long)
    for i, (x, lbl) in enumerate(samples):
        t = min(x.shape[0], MAX_T)
        X[i, :t, :] = torch.from_numpy(x[:t])
        y[i, :t] = torch.from_numpy(lbl[:t])
    return X, y


def cuda_graph_step(model, device, initial_cw):
    """One captured training step (forward, weighted cross-entropy, backward) replayed per batch."""
    amp = torch.autocast('cuda', dtype=torch.bfloat16)
    s = torch.cuda.Stream(device=device)
    static_x = torch.zeros(MAX_T, BATCH_SIZE, N_INPUT, device=device)
    static_y = torch.full((MAX_T, BATCH_SIZE), -1, dtype=torch.long, device=device)
    static_loss = torch.zeros((), device=device)
    static_cw = torch.tensor(initial_cw, dtype=torch.float32, device=device)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        for _ in range(3):
            model.zero_grad(set_to_none=False)
            with amp:
                loss = F.cross_entropy(model(static_x).reshape(-1, N_OUTPUT).float(), static_y.reshape(-1),
                                       weight=static_cw, ignore_index=-1)
            loss.backward()
    torch.cuda.current_stream(device).wait_stream(s)
    model.zero_grad(set_to_none=False)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s, capture_error_mode="thread_local"):
        with amp:
            static_out = model(static_x)
            static_loss_val = F.cross_entropy(static_out.reshape(-1, N_OUTPUT).float(), static_y.reshape(-1),
                                              weight=static_cw, ignore_index=-1)
        static_loss_val.backward()
        static_loss.copy_(static_loss_val)
    return g, static_x, static_y, static_cw, static_loss, static_out


def train_epoch(model, X, y, optimizer, device, graph):
    model.train()
    g, static_x, static_y, _, _, _ = graph
    perm = torch.randperm(X.size(0)).to(device)
    for bi in range(0, X.size(0) - BATCH_SIZE + 1, BATCH_SIZE):
        idx = perm[bi:bi + BATCH_SIZE]
        static_x.copy_(X[idx].permute(1, 0, 2))
        static_y.copy_(y[idx].permute(1, 0))
        optimizer.zero_grad(set_to_none=False)
        g.replay()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def recalls(models, samples, device):
    """Per-class recall and balanced accuracy; several models average their posteriors."""
    for m in models:
        m.eval()
    conf = np.zeros((N_OUTPUT, N_OUTPUT), dtype=np.int64)
    for x, y in samples:
        xt = torch.from_numpy(x).unsqueeze(1).to(device)
        lps = torch.stack([F.log_softmax(m(xt).squeeze(1).float(), dim=-1) for m in models])
        mx = lps.max(dim=0, keepdim=True).values
        pred = ((lps - mx).exp().mean(dim=0).log() + mx.squeeze(0)).argmax(-1).cpu().numpy()
        np.add.at(conf, (y, pred), 1)
    rowsum = conf.sum(axis=1)
    rec = np.divide(np.diag(conf), rowsum, out=np.zeros(N_OUTPUT), where=rowsum > 0)
    return rec, float(np.mean(rec)), conf


def train_seed(seed, phases, device, gate_samples):
    torch.manual_seed(seed)
    model = BidirectionalSegmenter().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EP_2B)
    cw = lambda bw: [1.0, float(bw), float(K_IW * bw)]                          # noqa: E731
    graph = cuda_graph_step(model, device, cw(BW_PHASE1A))
    static_cw = graph[3]
    gate = {"checked": False, "passed": None, "epoch": EP_1B}
    for epoch in range(N_EPOCHS_TOTAL):
        X, y = phases[0 if epoch < EP_1B else 1 if epoch < EP_2A else 2 if epoch < EP_2B else 3]
        train_epoch(model, X, y, optimizer, device, graph)
        scheduler.step()
        if epoch == EP_1B - 1:
            vrec, _, _ = recalls([model], gate_samples, device)
            gate.update(checked=True, passed=bool(all(r > GATE_FLOOR for r in vrec)), val_recalls=[float(r) for r in vrec])
            if not gate["passed"]:
                return None, gate
            static_cw.copy_(torch.tensor(cw(BW_PHASE1B), device=device))
        elif epoch == EP_2A - 1:
            static_cw.copy_(torch.tensor(cw(BW_PHASE2A), device=device))
        elif epoch == EP_2B - 1:
            static_cw.copy_(torch.tensor(cw(BW_PHASE2B), device=device))
            for pg in optimizer.param_groups:
                pg['lr'] = LR_INIT
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_PHASE2B)
    return model, gate


def main():
    assert torch.cuda.is_available(), "the BiGRU pool trains on a CUDA GPU"
    device = torch.device("cuda:0")
    t0 = time.time()
    select = {k: encode(v) for k, v in operator_corpus(N_TX_SELECT_FIST, RNG_SELECT).items()}
    test = {k: encode(v) for k, v in operator_corpus(N_TX_TEST_FIST, RNG_TEST_FIST).items()}
    test_rand = encode(random_corpus(N_TX_TEST_RAND, RNG_TEST_RAND))
    gate_samples = build_dataset(N_TX_GATEVAL, *SIGMA_P1A, NW_P1, WL_P1, RNG_GATE)[0]
    (OUT / "pool").mkdir(parents=True, exist_ok=True)
    pool, models = {}, {}
    for seed in POOL_SEEDS:
        phases = []
        for (lo, hi), nw, wl, base in ((SIGMA_P1A, NW_P1, WL_P1, RNG_TRAIN_P1A), (SIGMA_P1B, NW_P1, WL_P1, RNG_TRAIN_P1B),
                                       (SIGMA_P2A, NW_P1, WL_P1, RNG_TRAIN_P2A), (SIGMA_P2B, NW_P2B, WL_P2B, RNG_TRAIN_P2B)):
            X, y = training_tensors(build_dataset(N_TX_TRAIN, lo, hi, nw, wl, base + seed)[0])
            phases.append((X.to(device), y.to(device)))
        model, gate = train_seed(seed, phases, device, gate_samples)
        del phases
        torch.cuda.empty_cache()
        if model is None:
            pool[str(seed)] = {"status": "ABORTED_PHASE1A", "gate": gate}
            continue
        torch.save(model.state_dict(), OUT / "pool" / f"bigru3_seed{seed}.pt")
        conf = sum(recalls([model], s, device)[2] for s in select.values())
        rec = np.divide(np.diag(conf), conf.sum(axis=1), out=np.zeros(3), where=conf.sum(axis=1) > 0)
        pool[str(seed)] = {"status": "COMPLETED", "gate": gate, "select_balacc": float(np.mean(rec)),
                           "select_recalls": rec.tolist()}
        models[seed] = model
        print(f"  seed {seed}: SELECT balanced accuracy {np.mean(rec) * 100:.2f}%", flush=True)
    alive = {int(s): p for s, p in pool.items() if p["status"] == "COMPLETED"}
    bals = np.array([p["select_balacc"] for p in alive.values()])
    selected = sorted(alive, key=lambda s: alive[s]["select_balacc"], reverse=True)[:N_SELECTED]
    conf = sum(recalls([models[s] for s in selected], t, device)[2] for t in test.values())
    ens_rec = np.diag(conf) / np.maximum(conf.sum(axis=1), 1)
    out = dict(smoke=SMOKE, pool=pool, selected_seeds=selected,
               consistency=dict(mean=float(bals.mean()), std=float(bals.std()), min=float(bals.min()), max=float(bals.max())),
               ensemble_test=dict(pooled_balacc=float(np.mean(ens_rec)), recalls=ens_rec.tolist(),
                                  random_letter_balacc=recalls([models[s] for s in selected], test_rand, device)[1]),
               wall_s=round(time.time() - t0, 1))
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    check(out)


def check(out):
    """A retrain passes if every seed's SELECT balanced accuracy lies within the shipped pool's range
    widened by its own spread, and the new ensemble's test balanced accuracy clears the floor."""
    ref = json.loads((RECORDS / "training" / "bigru_pool.json").read_text())
    lo, hi = ref["consistency"]["min"], ref["consistency"]["max"]
    band = (lo - (hi - lo), hi + (hi - lo))
    seeds_ok = all(band[0] <= p["select_balacc"] <= band[1] for p in out["pool"].values() if p["status"] == "COMPLETED")
    ens_ok = out["ensemble_test"]["pooled_balacc"] >= BALACC_FLOOR
    print(f"retrained pool: SELECT balanced accuracy {out['consistency']['min'] * 100:.2f}-{out['consistency']['max'] * 100:.2f}% "
          f"(shipped {lo * 100:.2f}-{hi * 100:.2f}%, band {band[0] * 100:.2f}-{band[1] * 100:.2f}%) -> {'PASS' if seeds_ok else 'FAIL'}; "
          f"ensemble test {out['ensemble_test']['pooled_balacc'] * 100:.2f}% (floor {BALACC_FLOOR * 100:.1f}%) -> "
          f"{'PASS' if ens_ok else 'FAIL'}; selected {out['selected_seeds']} (shipped {ref['selected_seeds']})")
    from ..segmenters.bigru import POOL_DIR
    same = [s for s, p in out["pool"].items() if p["status"] == "COMPLETED" and (OUT / "pool" / f"bigru3_seed{s}.pt").exists()
            and _weights_equal(POOL_DIR / f"bigru3_seed{s}.pt", OUT / "pool" / f"bigru3_seed{s}.pt")]
    print(f"weights identical to the shipped checkpoints for {len(same)} of {len(out['pool'])} seeds")
    if not SMOKE:
        sys.exit(0 if seeds_ok and ens_ok else 1)


def _weights_equal(a, b):
    sa, sb = torch.load(a, weights_only=True), torch.load(b, weights_only=True)
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


if __name__ == "__main__":
    main()
