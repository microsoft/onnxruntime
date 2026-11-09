# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate the 84-case INT4 CPU compute-request/shape matrix."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

if __package__:
    from .generate_matmul_nbits_cases import Case, generate_cases
else:
    from generate_matmul_nbits_cases import Case, generate_cases


def matrix_cases() -> list[Case]:
    requests = [("float32", level) for level in range(5)] + [("float16", 0), ("float16", 4)]
    return [
        Case((m, k), n, dtype=dtype, accuracy_level=level)
        for dtype, level in requests
        for m in (1, 8, 32, 128)
        for k, n in ((64, 32), (128, 64), (1024, 1024))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New case bundle directory")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    try:
        cases = matrix_cases()
        generate_cases(cases, args.output, args.seed)
        (args.output / "config.json").write_text(
            json.dumps({"cases": [asdict(case) for case in cases]}, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Matrix generation failed: {error}\n")
    print(f"Generated {len(cases)} cases: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
