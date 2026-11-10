# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_pyright_public_imports(tmp_path: Path):
    source = Path(__file__).resolve().parents[2] / "onnxruntime"
    site_packages = tmp_path / "site-packages"
    package = site_packages / "onnxruntime"
    # CMake installs the Python inference classes under onnxruntime/capi.
    (package / "capi").mkdir(parents=True)
    shutil.copy(source / "__init__.py", package)
    shutil.copy(source / "py.typed", package)
    shutil.copy(source / "python" / "onnxruntime_inference_collection.py", package / "capi")

    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "# pyright: reportPrivateImportUsage=error, reportMissingTypeStubs=error\n"
        "from onnxruntime import InferenceSession, IOBinding, ModelCompiler, OrtValue\n",
        encoding="utf-8",
    )
    # Use Python's package search path so Pyright treats onnxruntime as a third-party library.
    subprocess.run(
        [sys.executable, "-m", "pyright", "--pythonpath", sys.executable, consumer.name],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(site_packages)},
        check=True,
    )
