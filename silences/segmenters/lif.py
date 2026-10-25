"""The feedforward LIF segmenter (Section 8).

The 42 window features, standardised, drive the first layer as a constant current for T = 40 steps.
Two hidden layers of leaky integrate-and-fire neurons (decay beta = 0.95, threshold 1, subtractive
reset) pass binary spikes forward; a non-spiking readout integrates its membrane, and its mean over
the T steps gives the three logits. Training (silences.training.window_nets) uses snnTorch surrogate
gradients; inference here is a plain numpy simulation of the same dynamics.
"""
import pickle

import numpy as np

from ..features import fe_tu_us, window_features
from ..fixedpoint.q16 import quantize_lp3
from . import load_dict


class LIF:
    def __init__(self, W=10, hidden=(256, 128), beta=0.95, threshold=1.0, T=40):
        self.W = W
        self.hidden = tuple(hidden)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.T = int(T)
        self.mu = None
        self.sd = None
        self.fcs = None             # [(Wt, b), ...]: input -> hidden 1 -> hidden 2 -> readout

    def logits(self, F):
        Z = (np.asarray(F, np.float64) - self.mu) / self.sd
        mem = [np.zeros((len(Z), Wt.shape[1])) for Wt, _ in self.fcs]
        out = np.zeros((len(Z), 3))
        for _ in range(self.T):
            cur = Z
            for i, (Wt, b) in enumerate(self.fcs):
                mem[i] = self.beta * mem[i] + (cur @ Wt + b)
                if i < len(self.fcs) - 1:                  # hidden layer: spike, subtract the threshold
                    spk = (mem[i] >= self.threshold).astype(np.float64)
                    mem[i] = mem[i] - spk * self.threshold
                    cur = spk
                else:                                      # readout: integrate, never spike
                    out += mem[i]
        return out / self.T

    def log_posteriors(self, F):
        L = self.logits(F)
        L -= L.max(1, keepdims=True)
        return L - np.log(np.exp(L).sum(1, keepdims=True))

    def lp3(self, tx):
        F = window_features(tx["ticks"], tx["ch"], fe_tu_us(tx["ticks"]), self.W)
        return quantize_lp3(self.log_posteriors(F)) if len(F) else []

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(dict(W=self.W, hidden=self.hidden, beta=self.beta, threshold=self.threshold,
                             T=self.T, mu=self.mu, sd=self.sd, fcs=self.fcs), fh)

    @classmethod
    def load(cls, path):
        d = load_dict(path)
        m = cls(W=d["W"], hidden=d["hidden"], beta=d["beta"], threshold=d["threshold"], T=d["T"])
        m.mu, m.sd, m.fcs = d["mu"], d["sd"], d["fcs"]
        return m
