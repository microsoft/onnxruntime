# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Compare profile_decoder.py validation archives; optionally require byte-exact logits and live KV."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference, candidate):
    rows = []
    with np.load(reference, allow_pickle=False) as expected, np.load(candidate, allow_pickle=False) as actual:
        if set(expected.files) != set(actual.files) or "logits" not in expected.files:
            raise ValueError("Archives must have the same output names, including logits.")
        for name in sorted(expected.files):
            a, b = expected[name], actual[name]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"Shape/type mismatch for {name}: {a.shape}/{a.dtype} versus {b.shape}/{b.dtype}.")
            x, y = a.astype(np.float64), b.astype(np.float64)
            delta = x - y
            finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
            norm = np.linalg.norm(x)
            row = {
                "name": name,
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "finite": finite,
                "byte_exact": finite and a.tobytes() == b.tobytes(),
                "max_abs": float(np.max(np.abs(delta), initial=0)) if finite else None,
                "rms": float(np.sqrt(np.mean(delta * delta))) if finite and a.size else None,
                "relative_l2": float(np.linalg.norm(delta) / norm) if finite and norm else None,
            }
            if name == "logits":
                row["top1_equal"] = bool(np.array_equal(np.argmax(x, axis=-1), np.argmax(y, axis=-1)))
            rows.append(row)
    return {"byte_exact": all(row["byte_exact"] for row in rows), "outputs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    print(json.dumps(result, indent=2, allow_nan=False))
    if not all(row["finite"] for row in result["outputs"]) or (args.require_exact and not result["byte_exact"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
