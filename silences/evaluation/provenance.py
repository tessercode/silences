"""Provenance stamped into every record: the commit, whether the tree was clean, the environment."""
import datetime
import hashlib
import os
import platform
import subprocess

from .. import ROOT

# A record is clean when nothing under these paths differs from the commit it names.
RELEVANT = ("silences/", "protocol/", "models/", "data/", "firmware/encoder/fists.h")


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def git_info():
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    # porcelain lines are "XY path" or "XY old -> new"; keep the leading status columns intact
    dirty = [l[3:].split(" -> ")[-1] for l in run("status", "--porcelain").splitlines()]
    relevant = [d for d in dirty if d.startswith(RELEVANT)]
    return dict(commit=run("rev-parse", "HEAD").strip(), dirty_paths=dirty, relevant_dirty_paths=relevant,
                clean_for_record=not relevant)


def env_info():
    import numpy
    info = dict(python=platform.python_version(), numpy=numpy.__version__,
                blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"))
    try:
        import torch
        import snntorch
        info.update(torch=torch.__version__, snntorch=snntorch.__version__)
    except ImportError:
        pass
    return info


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")
