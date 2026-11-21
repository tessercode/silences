"""The bidirectional GRU in floating point, for two rows of the window-width record (Section 7.2, Table 4).

The baseline is evaluated everywhere else in the Q16.16 arithmetic of its board (segmenters/bigru.py). Two
rows need the float network instead: the baseline's recipe trained on the window's two raw values per gap
(bigru_raw), for which no fixed-point lowering exists, and the baseline of record on its nine features
through the same float path (bigru_fist_fp), the anchor that shows the float path stands in for the
fixed-point one (they agree to well under 0.2 pp of character error).

Needs torch. The log-posteriors are rounded to Q16.16 so the rows enter the decoder like every other.
"""
import numpy as np

from ..features import fe_tu_us
from . import bigru

N_RAW = 2
ONE = 65536
INT32_MIN = -(2 ** 31)


def raw_features(ticks_us, ch, tu_us):
    """[log(interval / time unit), bit of the closing mark] per gap: the window networks' pair, unwindowed."""
    if len(ticks_us) < 2:
        return np.zeros((0, N_RAW), dtype=np.float32)
    t = np.array([float(t) * 1e-6 for t in ticks_us], dtype=np.float64)
    c = np.array([int(v) for v in ch], dtype=np.float64)
    x = np.diff(t) / (float(tu_us) * 1e-6)
    return np.stack([np.log(np.clip(x, 1e-3, None)), c[1:]], axis=1).astype(np.float32)


def nine_features(ticks_us, ch, tu_us):
    """The baseline's nine features in floating point (segmenters/bigru.py computes them in Q16.16)."""
    ticks = np.asarray(ticks_us, dtype=np.float64)
    ch = np.asarray(ch, dtype=np.int64)
    if len(ticks) < 2:
        return np.zeros((0, len(bigru.BETA_ISI) + 3), dtype=np.float32)
    isi = (np.diff(ticks) * 1e-6) / (float(tu_us) * 1e-6)
    nxt = ch[1:].astype(np.float64)
    pop = bigru.V_KICK * (np.asarray(bigru.BETA_ISI, dtype=np.float64)[None, :] ** (isi[:, None] * 60.0))
    dur = 1.0 / (1.0 + np.exp(-(isi - bigru.TAU_DUR) / bigru.GAP_S))
    tau = np.where(nxt >= 0.5, bigru.TAU_DASH, bigru.TAU_DOT)
    durbit = 1.0 / (1.0 + np.exp(-(isi - tau) / bigru.GAP_S))
    return np.concatenate([pop, nxt[:, None], dur[:, None], durbit[:, None]], axis=1).astype(np.float32)


def load(pt_path, n_input):
    import torch
    import torch.nn as nn

    class Net(nn.Module):                         # the training module's parameter names
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size=n_input, hidden_size=bigru.H, num_layers=1, batch_first=False, bidirectional=True)
            self.fc = nn.Linear(2 * bigru.H, 3)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out)
    torch.set_num_threads(1)
    m = Net()
    m.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    return m.eval()


def lp3(model, feats):
    import torch
    if len(feats) == 0:
        return []
    with torch.no_grad():
        L = model(torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)).unsqueeze(1)).squeeze(1).double()
        logp = torch.log_softmax(L, dim=-1).numpy()
    q = np.clip(np.rint(logp * ONE).astype(np.int64), INT32_MIN, -1)
    return [[int(v) for v in r] for r in q]


def producer(kind, pt_path):
    """kind "raw": the two-value network; "fp": a nine-feature checkpoint in floating point."""
    if kind == "raw":
        m = load(pt_path, N_RAW)
        return lambda tx: lp3(m, raw_features(tx["ticks"], tx["ch"], fe_tu_us(tx["ticks"])))
    m = load(pt_path, len(bigru.BETA_ISI) + 3)
    return lambda tx: lp3(m, nine_features(tx["ticks"], tx["ch"], fe_tu_us(tx["ticks"])))
