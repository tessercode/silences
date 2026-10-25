"""The tanh MLP control (Section 7): the LIF network's structure with tanh units in place of spiking
neurons. Inputs are the 42 window features, standardised; two hidden layers; three logits."""
import pickle

import numpy as np

from ..features import fe_tu_us, window_features
from ..fixedpoint.q16 import quantize_lp3
from . import load_dict


class MLP:
    def __init__(self, W=10, hidden=(256, 128)):
        self.W = W
        self.hidden = tuple(hidden)
        self.mu = None
        self.sd = None
        self.layers = None          # [(Wt, b), ...]; the last layer is linear

    def logits(self, F):
        Z = (np.asarray(F, np.float64) - self.mu) / self.sd
        for Wt, b in self.layers[:-1]:
            Z = np.tanh(Z @ Wt + b)
        return Z @ self.layers[-1][0] + self.layers[-1][1]

    def log_posteriors(self, F):
        L = self.logits(F)
        L -= L.max(1, keepdims=True)
        return L - np.log(np.exp(L).sum(1, keepdims=True))

    def lp3(self, tx):
        F = window_features(tx["ticks"], tx["ch"], fe_tu_us(tx["ticks"]), self.W)
        return quantize_lp3(self.log_posteriors(F)) if len(F) else []

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(dict(W=self.W, hidden=self.hidden, mu=self.mu, sd=self.sd, layers=self.layers), fh)

    @classmethod
    def load(cls, path):
        d = load_dict(path)
        m = cls(W=d["W"], hidden=d["hidden"])
        m.mu, m.sd, m.layers = d["mu"], d["sd"], d["layers"]
        return m
