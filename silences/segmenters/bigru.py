"""The bidirectional-GRU baseline (Section 6), in the Q16.16 arithmetic of its RP2040 board.

Each gap contributes nine engineered features computed from the interval in time units (isi):
six exponential traces beta_j^(60 isi), the channel bit of the mark that ends the gap, and two
sigmoids, one at 3.5 time units and one at 3 or 5 time units depending on that bit. A one-layer
bidirectional GRU (32 units each way) reads the gap sequence and a linear head gives three logits.
The ensemble averages the per-seed posteriors (an integer mean) before the log.

The models were trained in floating point (silences.training.bigru) and are evaluated, here and on
the device, only in fixed point. models/bigru/pool/ holds the eight training seeds; the deployed
ensemble is five of them in DEPLOYED_SEEDS order (seed 42 alone is the one-seed board build).
"""
import math

import numpy as np

from .. import MODELS
from ..features import fe_tu_us
from ..fixedpoint import q16
from ..fixedpoint.q16 import FRAC, ONE, LUT, fxmul

POOL_SEEDS = list(range(42, 50))
DEPLOYED_SEEDS = [42, 47, 48, 49, 43]
POOL_DIR = MODELS / "bigru" / "pool"
POOL_Q16 = MODELS / "bigru" / "pool_q16.npz"
ARMS = ("rep", "fist", "shared")              # the retrained arms of the matched-data record (training/bigru.py --arm)


def arm_dir(arm):
    return MODELS / "bigru" / arm


def arm_q16(arm):
    return MODELS / "bigru" / arm / "q16.npz"
H = 32

# ---- the nine features ----
BETA_ISI = [0.9885, 0.9942, 0.9962, 0.9974, 0.9981, 0.9987]
V_KICK = 1.0
TAU_DUR, TAU_DOT, TAU_DASH, GAP_S = 3.5, 3.0, 5.0, 0.6
# Each trace's table covers the interval at which it rounds to zero (11.78 / its rate, in time units).
POP_HI = [16.0, 64.0, 64.0, 128.0, 128.0, 256.0]
POP = [LUT((lambda b: (lambda x: V_KICK * (b ** (x * 60.0))))(beta), 0.0, hi, 4096)
       for beta, hi in zip(BETA_ISI, POP_HI)]
TAU_DUR_Q, TAU_DOT_Q, TAU_DASH_Q = q16.to_q(TAU_DUR), q16.to_q(TAU_DOT), q16.to_q(TAU_DASH)
INV_GAP_S_Q = q16.to_q(1.0 / GAP_S)


def features_q(ticks, ch, tu_us):
    """(n_gaps, 9) Q16.16 feature rows from integer microsecond ticks and the time unit."""
    ticks, ch = np.asarray(ticks, np.int64), np.asarray(ch, np.int64)
    isi_q = (np.diff(ticks) * ONE) // int(tu_us)
    chn = ch[1:]
    cols = [P.vec(isi_q) for P in POP]
    cols.append(np.where(chn != 0, ONE, 0).astype(np.int64))
    cols.append(q16.SIGMOID.vec(fxmul(isi_q - TAU_DUR_Q, np.int64(INV_GAP_S_Q))))
    tau = np.where(chn != 0, TAU_DASH_Q, TAU_DOT_Q).astype(np.int64)
    cols.append(q16.SIGMOID.vec(fxmul(isi_q - tau, np.int64(INV_GAP_S_Q))))
    return np.stack(cols, axis=1)


class SeedQ16:
    """One trained seed in Q16.16: GRU gate rows stacked [r; z; n] as the firmware stores them."""

    def __init__(self, w_ih, w_hh, b_ih, b_hh, fc_w, fc_b):
        self.w_ih, self.w_hh, self.b_ih, self.b_hh = w_ih, w_hh, b_ih, b_hh     # {fwd, rev}: arrays
        self.fc_w, self.fc_b = fc_w, fc_b

    @classmethod
    def from_state_dict(cls, sd):
        """Round a float checkpoint to Q16.16 (half to even, as the board's weights were built)."""
        def Q(name):
            a = sd[name].detach().cpu().numpy().astype(np.float64)
            assert float(np.abs(a).max()) < 32767.0, f"Q16.16 saturation risk in {name}"
            return np.array([q16.to_q(v) for v in a.reshape(-1)], dtype=np.int64).reshape(a.shape)
        d = {"fwd": "", "rev": "_reverse"}
        return cls({k: Q(f"gru.weight_ih_l0{s}") for k, s in d.items()},
                   {k: Q(f"gru.weight_hh_l0{s}") for k, s in d.items()},
                   {k: Q(f"gru.bias_ih_l0{s}") for k, s in d.items()},
                   {k: Q(f"gru.bias_hh_l0{s}") for k, s in d.items()},
                   Q("fc.weight"), Q("fc.bias"))

    def _direction(self, key, feats):
        wih, whh, bih, bhh = self.w_ih[key], self.w_hh[key], self.b_ih[key], self.b_hh[key]
        h = np.zeros(H, dtype=np.int64)
        states = np.empty((len(feats), H), dtype=np.int64)
        for s, x in enumerate(feats if key == "fwd" else feats[::-1]):
            gi = (wih @ x) >> FRAC
            gh = (whh @ h) >> FRAC
            r = q16.SIGMOID.vec(gi[0:H] + bih[0:H] + gh[0:H] + bhh[0:H])
            z = q16.SIGMOID.vec(gi[H:2 * H] + bih[H:2 * H] + gh[H:2 * H] + bhh[H:2 * H])
            n = q16.TANH.vec(gi[2 * H:] + bih[2 * H:] + fxmul(r, gh[2 * H:] + bhh[2 * H:]))
            h = fxmul(ONE - z, n) + fxmul(z, h)
            states[s] = h
        return states[::-1] if key == "rev" else states

    def posteriors_q(self, feats):
        """(n_gaps, 3) Q16.16 posteriors: exp from the table (exp(0) exact), floor divide by the sum."""
        hcat = np.concatenate([self._direction("fwd", feats), self._direction("rev", feats)], axis=1)
        logits = ((hcat @ self.fc_w.T) >> FRAC) + self.fc_b
        d = logits - logits.max(axis=1, keepdims=True)
        e = np.where(d == 0, np.int64(ONE), q16.EXP.vec(d))
        return (e << FRAC) // e.sum(axis=1, keepdims=True)


def lp3_ticks(seeds, ticks, ch, tu_us):
    """Ensemble log-posteriors: integer mean of the seeds' posteriors, then one log per class."""
    feats = features_q(ticks, ch, tu_us)
    if len(feats) == 0:
        return []
    psum = sum(S.posteriors_q(feats) for S in seeds)
    return [[int(v) for v in row] for row in q16.fxlog_vec(psum // len(seeds))]


def producer(seeds):
    return lambda tx: lp3_ticks(seeds, tx["ticks"], tx["ch"], int(fe_tu_us(tx["ticks"])))


# ---- model files ----
_ARRAYS = ("w_ih", "w_hh", "b_ih", "b_hh")


def load_checkpoint(seed, directory=POOL_DIR):
    """One seed, quantized from its float checkpoint (needs torch)."""
    import torch
    return SeedQ16.from_state_dict(torch.load(directory / f"bigru3_seed{seed}.pt", map_location="cpu",
                                              weights_only=True))


def save_q16(path, seeds_by_id):
    d = {}
    for sid, S in seeds_by_id.items():
        for a in _ARRAYS:
            for k in ("fwd", "rev"):
                d[f"s{sid}_{a}_{k}"] = getattr(S, a)[k]
        d[f"s{sid}_fc_w"], d[f"s{sid}_fc_b"] = S.fc_w, S.fc_b
    np.savez(path, **d)


def load_q16(seed, path=POOL_Q16):
    """One pool seed from the quantized weights file (no torch needed)."""
    z = np.load(path)
    return SeedQ16(*({k: z[f"s{seed}_{a}_{k}"] for k in ("fwd", "rev")} for a in _ARRAYS),
                   z[f"s{seed}_fc_w"], z[f"s{seed}_fc_b"])


def deployed(n_seeds=5):
    """The deployed ensemble (n_seeds = 5), or its one-seed build (n_seeds = 1)."""
    return [load_q16(s) for s in DEPLOYED_SEEDS[:n_seeds]]


def quantize_arm(arm):
    """Write models/bigru/<arm>/q16.npz from the arm's float checkpoints (needs torch; run once per arm)."""
    import json
    alive = json.loads((MODELS.parent / "records" / "training" / "bigru_arms.json").read_text())["arms"][arm]["alive"]
    save_q16(arm_q16(arm), {s: load_checkpoint(s, arm_dir(arm)) for s in alive})
    return arm_q16(arm)


if __name__ == "__main__":
    import sys
    for a in (sys.argv[1:] or ARMS):
        print("wrote", quantize_arm(a))
