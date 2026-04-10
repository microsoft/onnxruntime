#!/usr/bin/env python3
"""Build a wheel for the onnxruntime-ep-webgpu package.

This script copies plugin EP binaries from a build directory into the package
source tree, sets the version in pyproject.toml, builds the wheel, optionally
runs auditwheel repair (Linux), verifies the output, and cleans up.

Usage:
    python build_wheel.py --binary_dir <path> --version <ver> --output_dir <path>
"""

import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR / "onnxruntime_ep_webgpu"

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


def copy_binaries(binary_dir: Path) -> list[Path]:
    """Copy plugin binaries from the build directory into the package directory."""
    copied = []
    for pattern in BINARY_PATTERNS:
        for src in binary_dir.glob(pattern):
            dst = PACKAGE_DIR / src.name
            print(f"Copying {src} -> {dst}")
            shutil.copy2(src, dst)
            copied.append(dst)
    if not copied:
        print(f"ERROR: No plugin binaries found in {binary_dir}", file=sys.stderr)
        print(f"Looked for: {BINARY_PATTERNS}", file=sys.stderr)
        sys.exit(1)
    return copied


def set_version(version: str) -> str:
    """Set the version in pyproject.toml. Returns the original content for restoration."""
    pyproject_path = SCRIPT_DIR / "pyproject.toml"
    original = pyproject_path.read_text(encoding="utf-8")
    updated = original.replace('version = "0.0.0"', f'version = "{version}"')
    if updated == original:
        print("WARNING: Could not find version placeholder in pyproject.toml", file=sys.stderr)
    pyproject_path.write_text(updated, encoding="utf-8")
    return original


def build_wheel(output_dir: Path):
    """Build the wheel using pip."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pip", "wheel",
        str(SCRIPT_DIR),
        "--wheel-dir", str(output_dir),
        "--no-deps",
        "--no-build-isolation",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def auditwheel_repair(output_dir: Path):
    """Run auditwheel repair on Linux to produce a manylinux-compliant wheel."""
    if platform.system() != "Linux":
        return

    raw_wheels = glob.glob(str(output_dir / "onnxruntime_ep_webgpu-*.whl"))
    if not raw_wheels:
        return

    repaired_dir = output_dir / "_repaired"
    repaired_dir.mkdir(parents=True, exist_ok=True)

    for wheel in raw_wheels:
        cmd = [sys.executable, "-m", "auditwheel", "repair", wheel,
               "--wheel-dir", str(repaired_dir)]
        for lib in AUDITWHEEL_EXCLUDE:
            cmd.extend(["--exclude", lib])
        print(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)

    # Replace raw wheels with repaired ones
    for wheel in raw_wheels:
        os.remove(wheel)
    for repaired_wheel in repaired_dir.glob("*.whl"):
        shutil.move(str(repaired_wheel), str(output_dir / repaired_wheel.name))
    repaired_dir.rmdir()


def verify_wheel(output_dir: Path):
    """Verify that at least one wheel was produced."""
    wheels = glob.glob(str(output_dir / "onnxruntime_ep_webgpu-*.whl"))
    if not wheels:
        print("ERROR: No wheel was produced", file=sys.stderr)
        sys.exit(1)
    for w in wheels:
        print(f"Built wheel: {w}")


def cleanup(copied_files: list[Path], original_pyproject: str):
    """Remove copied binaries and restore pyproject.toml."""
    for f in copied_files:
        if f.exists():
            f.unlink()
            print(f"Cleaned up: {f}")
    pyproject_path = SCRIPT_DIR / "pyproject.toml"
    pyproject_path.write_text(original_pyproject, encoding="utf-8")


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

    copied_files = copy_binaries(args.binary_dir)
    original_pyproject = set_version(args.version)
    try:
        build_wheel(args.output_dir)
        auditwheel_repair(args.output_dir)
        verify_wheel(args.output_dir)
    finally:
        cleanup(copied_files, original_pyproject)


if __name__ == "__main__":
    main()
