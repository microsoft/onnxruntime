#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Build a wheel for the onnxruntime-ep-webgpu package.

Combines pre-built plugin EP binaries with the Python package source to produce
a platform-specific wheel.

Usage:
    python build_wheel.py --binary_dir <path> --version <ver> --output_dir <path>
"""

import argparse
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MIN_ONNXRUNTIME_VERSION_FILE = SCRIPT_DIR.parent / "MIN_ONNXRUNTIME_VERSION"

# Import the shared template helper from _packaging_utils.py in the parent directory.
sys.path.insert(0, str(SCRIPT_DIR.parent))
from _packaging_utils import gen_file_from_template  # noqa: E402, I001  (path setup must precede import)


REQUIRED_BINARIES = {
    "Windows": ("onnxruntime_providers_webgpu.dll", "dxcompiler.dll"),
    "Linux": ("libonnxruntime_providers_webgpu.so",),
    "Darwin": ("libonnxruntime_providers_webgpu.dylib",),
}

OPTIONAL_BINARIES = {
    "Windows": ("webgpu_dawn.dll",),
    "Linux": ("libwebgpu_dawn.so",),
    "Darwin": ("libwebgpu_dawn.dylib",),
}

# Libraries to exclude from auditwheel bundling (user-provided drivers)
AUDITWHEEL_EXCLUDE = [
    "libvulkan.so.1",
]

WINDOWS_AGILITY_SDK_BINARIES = [
    Path("D3D12/D3D12Core.dll"),
    Path("D3D12/d3d12SDKLayers.dll"),
]


def prepare_staging_dir(staging_dir: Path, binary_dir: Path, version: str, *, require_agility_sdk: bool = False):
    """Copy the package source tree into staging_dir, copy binaries, and stamp the version."""
    target_platform = platform.system()
    if target_platform not in REQUIRED_BINARIES:
        raise ValueError(f"Unsupported wheel platform: {target_platform}")
    if require_agility_sdk and target_platform != "Windows":
        raise ValueError("--require-agility-sdk is only supported for Windows wheels")

    required_binaries = [Path(filename) for filename in REQUIRED_BINARIES[target_platform]]
    optional_binaries = [Path(filename) for filename in OPTIONAL_BINARIES[target_platform]]
    if target_platform == "Windows":
        if require_agility_sdk or any(
            (binary_dir / relative_path).is_file() for relative_path in WINDOWS_AGILITY_SDK_BINARIES
        ):
            required_binaries.extend(WINDOWS_AGILITY_SDK_BINARIES)

    missing_binaries = [
        str(relative_path) for relative_path in required_binaries if not (binary_dir / relative_path).is_file()
    ]
    if missing_binaries:
        raise FileNotFoundError(
            f"Missing required {target_platform} wheel binaries from {binary_dir}: {', '.join(missing_binaries)}"
        )

    staging_dir.mkdir(parents=True, exist_ok=True)

    # Copy only the files needed to build the wheel
    shutil.copy2(SCRIPT_DIR / "setup.py", staging_dir / "setup.py")
    shutil.copytree(SCRIPT_DIR / "onnxruntime_ep_webgpu", staging_dir / "onnxruntime_ep_webgpu")

    # Stage the repo-root LICENSE and ThirdPartyNotices.txt next to setup.py so setuptools
    # can bundle them via the `license-files` entry in pyproject.toml (PEP 639).
    repo_root = SCRIPT_DIR.parents[1]
    for license_filename in ("LICENSE", "ThirdPartyNotices.txt"):
        src = repo_root / license_filename
        if not src.is_file():
            raise FileNotFoundError(f"Expected license file not found: {src}")
        shutil.copy2(src, staging_dir / license_filename)

    package_dir = staging_dir / "onnxruntime_ep_webgpu"
    binaries = required_binaries + [
        relative_path for relative_path in optional_binaries if (binary_dir / relative_path).is_file()
    ]
    for relative_path in binaries:
        src = binary_dir / relative_path
        dst = package_dir / relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying {src} -> {dst}")
        shutil.copy2(src, dst)

    # Substitute the minimum ORT version into the staged README in place.
    min_ort_version = MIN_ONNXRUNTIME_VERSION_FILE.read_text(encoding="utf-8").strip()
    if not min_ort_version:
        raise ValueError(f"{MIN_ONNXRUNTIME_VERSION_FILE} is empty")

    staged_readme = package_dir / "README.md"
    gen_file_from_template(
        staged_readme,
        staged_readme,
        {"min_onnxruntime_version": min_ort_version},
    )

    # Render pyproject.toml from its template
    gen_file_from_template(
        SCRIPT_DIR / "pyproject.toml.in",
        staging_dir / "pyproject.toml",
        {"version": version},
    )


def build_wheel(source_dir: Path, wheel_dir: Path):
    """Build the wheel using pip."""
    wheel_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        str(source_dir),
        "--wheel-dir",
        str(wheel_dir),
        "--no-deps",
        "--no-build-isolation",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def auditwheel_repair(wheel_dir: Path):
    """Run auditwheel repair on Linux to produce a manylinux-compliant wheel."""
    if platform.system() != "Linux":
        return

    original_wheels = list(wheel_dir.glob("onnxruntime_ep_webgpu-*.whl"))
    if not original_wheels:
        raise RuntimeError(f"No wheel found in {wheel_dir} to repair with auditwheel")

    with tempfile.TemporaryDirectory() as repaired_dir_name:
        repaired_dir = Path(repaired_dir_name)

        for wheel in original_wheels:
            cmd = [sys.executable, "-m", "auditwheel", "repair", str(wheel), "--wheel-dir", str(repaired_dir)]
            for lib in AUDITWHEEL_EXCLUDE:
                cmd.extend(["--exclude", lib])
            print(f"Running: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            # Remove the original wheel so only the repaired one remains
            wheel.unlink()

        repaired_wheels = list(repaired_dir.glob("*.whl"))
        if not repaired_wheels:
            raise RuntimeError(f"auditwheel repair produced no wheels in {repaired_dir}")

        # Move repaired wheels into wheel_dir
        for repaired_wheel in repaired_wheels:
            repaired_wheel.replace(wheel_dir / repaired_wheel.name)


def collect_wheels(wheel_dir: Path, output_dir: Path):
    """Copy built wheels to the output directory and verify at least one was produced."""
    wheels = list(wheel_dir.glob("onnxruntime_ep_webgpu-*.whl"))
    if not wheels:
        raise RuntimeError("No wheel was produced")

    output_dir.mkdir(parents=True, exist_ok=True)

    for wheel in wheels:
        dest = output_dir / wheel.name
        shutil.copy2(wheel, dest)
        print(f"Built wheel: {dest}")


def main():
    parser = argparse.ArgumentParser(description="Build onnxruntime-ep-webgpu wheel")
    parser.add_argument(
        "--binary_dir", required=True, type=Path, help="Directory containing the built plugin EP binaries"
    )
    parser.add_argument("--version", required=True, help="Package version string (PEP 440 format)")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to place the built wheel")
    parser.add_argument(
        "--require-agility-sdk",
        action="store_true",
        help="Require both D3D12 Agility SDK DLLs in the pre-built input (used by Windows release pipelines).",
    )
    args = parser.parse_args()

    if not args.binary_dir.is_dir():
        raise FileNotFoundError(f"Binary directory does not exist: {args.binary_dir}")

    with tempfile.TemporaryDirectory(prefix="ort_webgpu_wheel_") as tmp:
        staging_dir = Path(tmp) / "package"
        wheel_dir = Path(tmp) / "wheels"

        prepare_staging_dir(staging_dir, args.binary_dir, args.version, require_agility_sdk=args.require_agility_sdk)
        build_wheel(staging_dir, wheel_dir)
        auditwheel_repair(wheel_dir)
        collect_wheels(wheel_dir, args.output_dir)


if __name__ == "__main__":
    main()
