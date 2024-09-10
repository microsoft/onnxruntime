#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import pathlib
import subprocess

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()


def generate_cpp(flatc: pathlib.Path, schema_path: pathlib.Path):
    # run flatc to generate C++ code
    cmd = [str(flatc), "--cpp", "--scoped-enums", "--filename-suffix", ".fbs", str(schema_path)]
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Generate language bindings for the ORT flatbuffers schema.",
        usage="Provide the path to the flatbuffers flatc executable. "
        "Script can be executed from anywhere but must be located in its original "
        "directory in the ONNX Runtime enlistment.",
    )

    parser.add_argument(
        "-f",
        "--flatc",
        required=True,
        type=pathlib.Path,
        help="Path to flatbuffers flatc executable. "
        "Can be found in the build directory under _deps/flatbuffers-build/<config>/",
    )

    all_languages = ["cpp"]
    parser.add_argument(
        "-l",
        "--language",
        action="append",
        dest="languages",
        choices=all_languages,
        help="Specify which language bindings to generate.",
    )

    args = parser.parse_args()
    languages = args.languages if args.languages is not None else all_languages
    flatc = args.flatc.resolve(strict=True)
    schema_path = SCRIPT_DIR / "adapter_schema.fbs"

    if "cpp" in languages:
        generate_cpp(flatc, schema_path)


if __name__ == "__main__":
    main()
