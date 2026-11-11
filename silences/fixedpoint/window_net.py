"""Q16.16 reference for the LIF and MLP segmenters: the specification their RP2040 firmware
(firmware/segmenter_spk, firmware/segmenter_lif, firmware/segmenter_mlp) reproduces bit for bit.

Weights, the feature mean and 1/sd are rounded to Q16.16 once, from the float model. At inference
every step is integer: log(interval / time unit) through the log table, standardisation as a
rounded multiply, matrix products accumulated in 64 bits, tanh from its table (MLP) or LIF
membranes with a Q16.16 decay and threshold (LIF), then the same exp/divide/log head as the
baseline's board. The spike-fed network ('spk') adds the input encoder: the clipped standardised
value zc = clip(+-Z_q, 0, scale_q) stays in Q16.16 and the train's cumulative count by step t is the
exact integer floor (t * zc) // scale_q, so the only float-vs-fixed difference at the input is the
Q16.16 rounding of Z itself.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..features import fe_tu_us
from . import q16
from .q16 import FRAC, ONE, fxmul_round

PAD_LOGX_Q = q16.fxlog(q16.to_q(12.0))      # the window's edge padding, through the same log the C uses


class FixedWindowNet:
    """kind is 'mlp', 'lif' (fed currents) or 'spk' (fed spikes)."""

    def __init__(self, kind, W, layers_q, mu_q, insd_q, beta_q=None, theta_q=None, T=None, scale_q=None):
        self.kind, self.W = kind, W
        self.layers_q = layers_q            # [(Wt int64[in, out], b int64[out]), ...]
        self.mu_q, self.insd_q = mu_q, insd_q
        self.beta_q, self.theta_q, self.T = beta_q, theta_q, T
        self.scale_q = scale_q              # 'spk' only: the encoder's saturation, in Q16.16 standard deviations

    @staticmethod
    def _q(a):
        return np.rint(np.asarray(a) * ONE).astype(np.int64)

    @classmethod
    def from_mlp(cls, m):
        return cls("mlp", m.W, [(cls._q(Wt), cls._q(b)) for Wt, b in m.layers], cls._q(m.mu), cls._q(1.0 / m.sd))

    @classmethod
    def from_lif(cls, m):
        return cls("lif", m.W, [(cls._q(Wt), cls._q(b)) for Wt, b in m.fcs], cls._q(m.mu), cls._q(1.0 / m.sd),
                   beta_q=q16.to_q(m.beta), theta_q=q16.to_q(m.threshold), T=m.T)

    @classmethod
    def from_spk(cls, m):
        return cls("spk", m.W, [(cls._q(Wt), cls._q(b)) for Wt, b in m.fcs], cls._q(m.mu), cls._q(1.0 / m.sd),
                   beta_q=q16.to_q(m.beta), theta_q=q16.to_q(m.threshold), T=m.T, scale_q=q16.to_q(m.scale))

    def encode_q(self, feat_q):
        """(G, 42) standardised Q16.16 -> (T, G, 84) regular ON/OFF spike trains in {0, 1} (SpikeFedLIF.encode)."""
        z = np.asarray(feat_q, dtype=np.int64)
        zc = np.concatenate([np.clip(z, 0, self.scale_q), np.clip(-z, 0, self.scale_q)], axis=1)
        t = np.arange(1, self.T + 1, dtype=np.int64)[:, None, None]
        cum = (t * zc[None]) // self.scale_q
        prev = np.concatenate([np.zeros((1,) + zc.shape, np.int64), cum[:-1]], axis=0)
        return cum - prev

    def features_q(self, ticks, ch, tu_us):
        ticks, ch, W = np.asarray(ticks, np.int64), np.asarray(ch, np.int64), self.W
        x_q = np.maximum((np.diff(ticks) << FRAC) // int(tu_us), 66)     # interval / time unit, >= ~1e-3
        logx = q16.fxlog_vec(x_q)
        bit = (ch[1:] * ONE).astype(np.int64)
        xp = np.concatenate([np.full(W, PAD_LOGX_Q, np.int64), logx, np.full(W, PAD_LOGX_Q, np.int64)])
        bp = np.concatenate([np.zeros(W, np.int64), bit, np.zeros(W, np.int64)])
        raw = np.concatenate([sliding_window_view(xp, 2 * W + 1), sliding_window_view(bp, 2 * W + 1)], axis=1)
        return fxmul_round(self.insd_q, raw - self.mu_q)

    @staticmethod
    def _matvec(x_q, Wt, b, shift):
        acc = x_q.astype(np.int64) @ Wt
        return (acc >> FRAC if shift else acc) + b      # spike inputs are 0/1, so they need no shift

    def logits(self, feat_q):
        if self.kind == "mlp":
            z = feat_q
            for i, (Wt, b) in enumerate(self.layers_q):
                z = self._matvec(z, Wt, b, shift=True)
                if i < len(self.layers_q) - 1:
                    z = q16.TANH.vec(z)
            return z
        G = len(feat_q)
        if self.kind == "spk":
            S_in = self.encode_q(feat_q)                                    # the input spike trains, (T, G, 84)
        else:
            I0 = self._matvec(feat_q, *self.layers_q[0], shift=True)        # the constant input current
        mem = [np.zeros((G, Wt.shape[1]), np.int64) for Wt, _ in self.layers_q]
        out = np.zeros((G, 3), np.int64)
        for t in range(self.T):
            cur = None
            for i, (Wt, b) in enumerate(self.layers_q):
                if i == 0:
                    I = self._matvec(S_in[t], Wt, b, shift=False) if self.kind == "spk" else I0
                else:
                    I = self._matvec(cur, Wt, b, shift=False)
                m = fxmul_round(self.beta_q, mem[i]) + I
                if i < len(self.layers_q) - 1:
                    spk = (m >= self.theta_q).astype(np.int64)
                    mem[i] = m - spk * self.theta_q
                    cur = spk
                else:
                    mem[i] = m
                    out += m
        return out // self.T

    def lp3_ticks(self, ticks, ch, tu_us):
        if len(ticks) < 3:
            return []
        lp = q16.softmax_lp3(self.logits(self.features_q(ticks, ch, tu_us)))
        return [[int(v) for v in row] for row in lp]

    def lp3(self, tx):
        return self.lp3_ticks(tx["ticks"], tx["ch"], int(fe_tu_us(tx["ticks"])))
