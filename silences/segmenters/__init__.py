"""Gap segmenters: each maps a transmission to per-gap log-posteriors over (intra-letter,
inter-letter, inter-word) in Q16.16, the rows the beam decoder consumes.

    lif     the feedforward LIF network, fed spikes (SpikeFedLIF, the network of record) or fed
            currents (LIF, the ladder's middle rung) (Section 8)
    mlp     its tanh MLP control, same inputs and layer sizes (Section 7)
    bigru   the bidirectional-GRU baseline (Section 6)
    oracle  confident posteriors from the true labels (the decoder's ceiling)

A producer is a function tx -> lp3 (a list of [lp0, lp1, lp2] integer rows, one per gap).
"""
import pickle


class _NumpyOnly(pickle.Unpickler):
    """Model files are plain dicts of numpy arrays; refuse anything else."""
    ALLOWED = {("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy._core.multiarray", "scalar"),
               ("numpy.core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"model file references {module}.{name}")
        return super().find_class(module, name)


def load_dict(path):
    with open(path, "rb") as fh:
        return _NumpyOnly(fh).load()
