"""Compare result records value by value, ignoring only provenance.

    python tools/compare_records.py NEW.json OLD.json
    python tools/compare_records.py --git records/random_letters records/decoder_weights

The second form compares every JSON file under the given directories with its committed version
(git HEAD), so after `make evaluate` it shows whether the regenerated records agree with the shipped
ones. Every number, string and list element must be equal; the provenance keys (timestamps, git and
environment details, file hashes, model file paths) are skipped. Exit status 0 means everything agrees.
"""
import json
import subprocess
import sys
from pathlib import Path

PROVENANCE = {"created", "finished", "started", "wall_s", "timing_s", "git", "env", "record_git", "results_git",
              "source", "sources", "results_sha256", "calibration_sha256", "files", "checked", "seconds"}


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in PROVENANCE:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'new' if k in a else 'old'}")
            else:
                walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def compare(new, old, name):
    out = []
    walk(new, old, "", out)
    for line in out[:20]:
        print(f"  {name}{line}")
    print(f"{name}: {'agrees' if not out else f'{len(out)} differences'}")
    return not out


def main(argv):
    if argv[0] != "--git":
        return 0 if compare(json.load(open(argv[0])), json.load(open(argv[1])), argv[0]) else 1
    ok = True
    for d in argv[1:]:
        for p in sorted(Path(d).glob("*.json")):
            committed = subprocess.run(["git", "show", f"HEAD:{p.as_posix()}"], capture_output=True, text=True)
            if committed.returncode:
                print(f"{p}: not committed, skipped")
                continue
            ok &= compare(json.loads(p.read_text()), json.loads(committed.stdout), str(p))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
