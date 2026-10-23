# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Build the normal CMake diagnostic target in an already configured WebGPU build."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True, help="Configured ORT CMake build directory.")
    parser.add_argument("--config", default="Release")
    args = parser.parse_args()
    subprocess.run(
        [
            "cmake",
            "--build",
            str(args.build_dir.resolve()),
            "--config",
            args.config,
            "--target",
            "onnxruntime_webgpu_matmul_nbits_benchmark",
            "--parallel",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
