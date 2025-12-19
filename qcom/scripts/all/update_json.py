#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import jsonc


def main(
    input_path: Path,
    output_path: Path | None,
    updates: Iterable[tuple[str, str]],
) -> None:
    data = jsonc.load(input_path.open("rt"))
    assert isinstance(data, dict), "Input JSON is not a dict."
    data.update(updates)

    kwargs: dict = {
        "indent": 2,
    }
    if output_path is not None:
        jsonc.dump(data, output_path.open("wt"), **kwargs)
    else:
        jsonc.dump(data, sys.stdout, **kwargs)


def parse_kvp(kvp: str) -> tuple[str, str]:
    parts = kvp.split("=")
    if len(parts) != 2:
        raise ValueError(f"Key/value pair {kvp} is not of format 'key=value'.")
    return parts[0], parts[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input jsonc file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=False,
        help="Path to output jsonc file.",
    )
    parser.add_argument(
        "updates", nargs="*", metavar="KEY=VALUE", help="Key value pair(s) to set in the json", type=parse_kvp
    )

    args = parser.parse_args()
    main(args.input, args.output, args.updates)
