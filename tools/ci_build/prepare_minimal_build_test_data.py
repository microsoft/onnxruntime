#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Prepare data for the Linux minimal-build E2E jobs using an installed inference wheel.

Build and install the Debug wheel and Python requirements before running this script.
The required-operator configs are written beneath <build-dir>/minimal_build_test_data.
Model conversions use a neutral working directory to import the installed wheel.
A missing custom-op library is reported as a warning; conversion failures are fatal.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from logger import get_logger

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "tools" / "python"))

from util import run  # noqa: E402

log = get_logger("prepare_minimal_build_test_data")


def prepare_test_data(build_dir: Path) -> None:
    build_dir = build_dir.resolve()
    test_data_dir = build_dir / "minimal_build_test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = build_dir / "Debug"
    test_data_root = REPO_DIR / "onnxruntime" / "test" / "testdata"
    e2e_models = test_data_root / "ort_minimal_e2e_test_data"
    convert_script = REPO_DIR / "tools" / "python" / "convert_onnx_models_to_ort.py"
    config_script = REPO_DIR / "tools" / "python" / "create_reduced_build_config.py"

    with tempfile.TemporaryDirectory(prefix="ort_minimal_conversion_") as conversion_cwd:
        run(sys.executable, str(convert_script), str(e2e_models), cwd=conversion_cwd)
        run(
            sys.executable,
            "-m",
            "onnxruntime.tools.convert_onnx_models_to_ort",
            str(e2e_models),
            cwd=conversion_cwd,
        )

        standalone_ops = (e2e_models / "required_ops.standalone_invoker.config").read_text(encoding="utf-8")
        for config_name, type_args in (
            ("required_ops.ort_models.config", []),
            ("required_ops_and_types.ort_models.config", ["--enable_type_reduction"]),
        ):
            config_path = test_data_dir / config_name
            run(
                sys.executable,
                str(config_script),
                "--format",
                "ORT",
                *type_args,
                str(test_data_root),
                str(config_path),
                cwd=conversion_cwd,
            )
            with config_path.open("a", encoding="utf-8") as config:
                config.write("\n" + standalone_ops)

        custom_op_library = debug_dir / "libcustom_op_library.so"
        if not custom_op_library.is_file():
            log.warning("Custom-op library %s not found; skipping custom-op conversion.", custom_op_library)
        else:
            with tempfile.TemporaryDirectory(prefix="custom_ops_model_", dir=test_data_dir) as custom_models:
                models = sorted((test_data_root / "custom_op_library").glob("*.onnx"))
                if not models:
                    raise FileNotFoundError(f"No custom-op ONNX models found in {test_data_root / 'custom_op_library'}")
                for model in models:
                    shutil.copy2(model, custom_models)
                run(
                    sys.executable,
                    str(convert_script),
                    "--custom_op_library",
                    str(custom_op_library),
                    custom_models,
                    cwd=conversion_cwd,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path, help="Build output directory (RUNNER_TEMP in CI).")
    args = parser.parse_args()
    prepare_test_data(args.build_dir)


if __name__ == "__main__":
    main()
