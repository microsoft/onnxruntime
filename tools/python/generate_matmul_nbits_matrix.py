# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate INT4 CPU compute-request or focused bias/tail experiment matrices."""

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


def bias_tail_cases(block_sizes: tuple[int, ...] = (32,)) -> list[Case]:
    """Pair bias off/on for isolated boundaries and one combined-tail probe."""
    cases = {}
    for block_size in block_sizes:
        shapes = []
        for m in (1, 8):
            shapes.extend(
                (m, k, 32)
                for k in (
                    block_size - 1,
                    block_size,
                    block_size + 1,
                    2 * block_size - 1,
                    2 * block_size,
                    2 * block_size + 1,
                    3 * block_size,
                    4 * block_size,
                )
            )
            shapes.extend((m, 4 * block_size, n) for n in (3, 4, 5, 31, 32, 33))
        shapes.extend((m, 4 * block_size, 32) for m in (2, 3, 7, 9, 32))
        shapes.append((3, 2 * block_size + 1, 33))
        for accuracy in (0, 4):
            for m, k, n in shapes:
                for bias in (False, True):
                    case = Case((m, k), n, block_size=block_size, accuracy_level=accuracy, bias=bias)
                    cases[case.identity()] = case
    return list(cases.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New case bundle directory")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--preset", choices=("compute", "bias-tails"), default="compute")
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Bias/tail preset only: block sizes (default: 32); e.g. 32 64 128",
    )
    args = parser.parse_args()
    try:
        if args.block_sizes is not None and args.preset != "bias-tails":
            raise ValueError("--block-sizes requires --preset bias-tails")
        cases = bias_tail_cases(tuple(args.block_sizes or (32,))) if args.preset == "bias-tails" else matrix_cases()
        generate_cases(cases, args.output, args.seed)
        (args.output / "config.json").write_text(
            json.dumps({"cases": [asdict(case) for case in cases]}, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Matrix generation failed: {error}\n")
    print(f"Generated {len(cases)} cases: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
