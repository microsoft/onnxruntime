#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import pathlib
import shutil
import subprocess
import tempfile

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()


def update_namespace(schema_path: pathlib.Path, updated_schema_path: pathlib.Path):
    # create a copy of the schema so we can replace the namespace so that the generated module name doesn't clash
    # with the 'onnxruntime' package.
    with open(schema_path) as input, open(updated_schema_path, "w") as output:
        for line in input:
            # convert any line with the namespace to use ort_flatbuffers_py instead of onnxruntime as the top level
            # namespace. this doesn't change how anything works - it just avoids a naming clash with the 'real'
            # onnxruntime python package
            output.write(line.replace("onnxruntime.fbs", "ort_flatbuffers_py.fbs"))


def generate_python(flatc: pathlib.Path, schema_path: pathlib.Path, output_dir: pathlib.Path):
    # run flatc to generate Python code
    cmd = [str(flatc), "--python", str(schema_path)]
    subprocess.run(cmd, check=True, cwd=output_dir)


def create_init_py(output_dir: pathlib.Path):
    # create an __init__.py that imports all the py files so we can just 'import ort_flatbuffers_py.fbs'
    # in a script that wants to process an ORT format model
    init_py_path = output_dir / "ort_flatbuffers_py/fbs/__init__.py"
    with open(init_py_path, "w") as init_py:
        init_py.write(
            """from os.path import dirname, basename, isfile, join, splitext
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [splitext(basename(f))[0] for f in modules if isfile(f) and not f.endswith('__init__.py')]

from . import *
"""
        )


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
        "Can be found in the build directory under external/flatbuffers/<config>/",
    )

    all_languages = ["python", "cpp"]
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
    schema_path = SCRIPT_DIR / "ort.fbs"

    if "python" in languages:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = pathlib.Path(temp_dir_name).resolve()
            updated_schema_path = temp_dir / "ort.py.fbs"
            update_namespace(schema_path, updated_schema_path)

            output_dir = temp_dir / "out"
            output_dir.mkdir()
            generate_python(flatc, updated_schema_path, output_dir)
            create_init_py(output_dir)

            # replace generated files in repo
            target_dir = SCRIPT_DIR.parent / "ort_flatbuffers_py"
            if target_dir.is_dir():
                shutil.rmtree(target_dir)
            shutil.move(str(output_dir / "ort_flatbuffers_py"), str(target_dir))

    if "cpp" in languages:
        generate_cpp(flatc, schema_path)


if __name__ == "__main__":
    main()
