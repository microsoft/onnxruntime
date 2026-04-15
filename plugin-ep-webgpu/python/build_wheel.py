#!/usr/bin/env python3
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

# Patterns for binaries to include in the package
BINARY_PATTERNS = [
    "onnxruntime_providers_webgpu.dll",
    "libonnxruntime_providers_webgpu.so",
    "libonnxruntime_providers_webgpu.dylib",
    # DXC dependencies (Windows)
    "dxil.dll",
    "dxcompiler.dll",
    # Dawn shared library (if built as shared)
    "webgpu_dawn.dll",
    "libwebgpu_dawn.so",
    "libwebgpu_dawn.dylib",
]

# Libraries to exclude from auditwheel bundling (user-provided drivers)
AUDITWHEEL_EXCLUDE = [
    "libvulkan.so.1",
]


def prepare_staging_dir(staging_dir: Path, binary_dir: Path, version: str):
    """Copy the package source tree into staging_dir, copy binaries, and stamp the version."""
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Copy only the files needed to build the wheel
    shutil.copy2(SCRIPT_DIR / "pyproject.toml", staging_dir / "pyproject.toml")
    shutil.copy2(SCRIPT_DIR / "setup.py", staging_dir / "setup.py")
    shutil.copytree(SCRIPT_DIR / "onnxruntime_ep_webgpu", staging_dir / "onnxruntime_ep_webgpu")

    # Copy plugin binaries into the package directory
    package_dir = staging_dir / "onnxruntime_ep_webgpu"
    copied = []
    for pattern in BINARY_PATTERNS:
        for src in binary_dir.glob(pattern):
            dst = package_dir / src.name
            print(f"Copying {src} -> {dst}")
            shutil.copy2(src, dst)
            copied.append(dst)
    if not copied:
        print(f"ERROR: No plugin binaries found in {binary_dir}", file=sys.stderr)
        print(f"Looked for: {BINARY_PATTERNS}", file=sys.stderr)
        sys.exit(1)

    # Stamp the version in pyproject.toml
    pyproject_path = staging_dir / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")
    placeholder = 'version = "VERSION_PLACEHOLDER"'
    if placeholder not in content:
        print(f"ERROR: Version placeholder not found in pyproject.toml. Expected: {placeholder}", file=sys.stderr)
        sys.exit(1)
    updated = content.replace(placeholder, f'version = "{version}"')
    pyproject_path.write_text(updated, encoding="utf-8")


def build_wheel(source_dir: Path, wheel_dir: Path):
    """Build the wheel using pip."""
    wheel_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pip", "wheel",
        str(source_dir),
        "--wheel-dir", str(wheel_dir),
        "--no-deps",
        "--no-build-isolation",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def auditwheel_repair(wheel_dir: Path):
    """Run auditwheel repair on Linux to produce a manylinux-compliant wheel."""
    if platform.system() != "Linux":
        return

    raw_wheels = wheel_dir.glob("onnxruntime_ep_webgpu-*.whl")
    if not raw_wheels:
        return

    raw_wheel_list = list(raw_wheels)
    if not raw_wheel_list:
        return

    with tempfile.TemporaryDirectory() as repaired_dir_name:
        repaired_dir = Path(repaired_dir_name)

        for wheel in raw_wheel_list:
            cmd = [sys.executable, "-m", "auditwheel", "repair", str(wheel), "--wheel-dir", str(repaired_dir)]
            for lib in AUDITWHEEL_EXCLUDE:
                cmd.extend(["--exclude", lib])
            print(f"Running: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            # Remove the raw wheel so only the repaired one remains
            wheel.unlink()

        # Move repaired wheels into wheel_dir
        for repaired_wheel in repaired_dir.glob("*.whl"):
            repaired_wheel.replace(wheel_dir / repaired_wheel.name)


def collect_wheels(wheel_dir: Path, output_dir: Path):
    """Copy built wheels to the output directory and verify at least one was produced."""
    wheels = wheel_dir.glob("onnxruntime_ep_webgpu-*.whl")
    if not wheels:
        print("ERROR: No wheel was produced", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    for wheel in wheels:
        dest = output_dir / wheel.name
        shutil.copy2(wheel, dest)
        print(f"Built wheel: {dest}")


def main():
    parser = argparse.ArgumentParser(description="Build onnxruntime-ep-webgpu wheel")
    parser.add_argument("--binary_dir", required=True, type=Path,
                        help="Directory containing the built plugin EP binaries")
    parser.add_argument("--version", required=True,
                        help="Package version string (PEP 440 format)")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Directory to place the built wheel")
    args = parser.parse_args()

    if not args.binary_dir.is_dir():
        print(f"ERROR: Binary directory does not exist: {args.binary_dir}", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="ort_webgpu_wheel_") as tmp:
        staging_dir = Path(tmp) / "package"
        wheel_dir = Path(tmp) / "wheels"

        prepare_staging_dir(staging_dir, args.binary_dir, args.version)
        build_wheel(staging_dir, wheel_dir)
        auditwheel_repair(wheel_dir)
        collect_wheels(wheel_dir, args.output_dir)


if __name__ == "__main__":
    main()
