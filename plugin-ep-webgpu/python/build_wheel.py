#!/usr/bin/env python3
"""Build a wheel for the onnxruntime-ep-webgpu package.

Combines pre-built plugin EP binaries with the Python package source to produce
a platform-specific wheel.

Usage:
    python build_wheel.py --binary_dir <path> --version <ver> --output_dir <path>
"""

import argparse
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MIN_ONNXRUNTIME_VERSION_FILE = SCRIPT_DIR.parent / "MIN_ONNXRUNTIME_VERSION"

# Matches "@var@" template variables.
_TEMPLATE_VARIABLE_PATTERN = re.compile(r"@(\w+)@")


def gen_file_from_template(
    template_file: Path, output_file: Path, variable_substitutions: dict[str, str], strict: bool = True
) -> None:
    """Generate a file from a template by substituting "@var@" markers with provided values.

    If `strict` is True, raises ValueError when the set of "@var@" names found in the template
    does not match the keys of `variable_substitutions`.

    Note: substituted values are inserted verbatim with no awareness of the target file's syntax.
    The caller is responsible for any quoting/escaping required by the target format.
    """
    content = template_file.read_text(encoding="utf-8")

    variables_in_file: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        variables_in_file.add(name)
        return variable_substitutions.get(name, match.group(0))

    content = _TEMPLATE_VARIABLE_PATTERN.sub(replace, content)

    if strict and variables_in_file != variable_substitutions.keys():
        provided = set(variable_substitutions.keys())
        raise ValueError(
            f"Template variables and substitution keys do not match for {template_file}. "
            f"Only in template: {sorted(variables_in_file - provided)}. "
            f"Only in substitutions: {sorted(provided - variables_in_file)}."
        )

    output_file.write_text(content, encoding="utf-8")


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
        raise FileNotFoundError(f"No plugin binaries found in {binary_dir}. Looked for: {BINARY_PATTERNS}")

    # Render pyproject.toml from its template
    min_ort_version = MIN_ONNXRUNTIME_VERSION_FILE.read_text(encoding="utf-8").strip()
    if not min_ort_version:
        raise ValueError(f"{MIN_ONNXRUNTIME_VERSION_FILE} is empty")

    gen_file_from_template(
        SCRIPT_DIR / "pyproject.toml.in",
        staging_dir / "pyproject.toml",
        {"version": version, "min_onnxruntime_version": min_ort_version},
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
    args = parser.parse_args()

    if not args.binary_dir.is_dir():
        raise FileNotFoundError(f"Binary directory does not exist: {args.binary_dir}")

    with tempfile.TemporaryDirectory(prefix="ort_webgpu_wheel_") as tmp:
        staging_dir = Path(tmp) / "package"
        wheel_dir = Path(tmp) / "wheels"

        prepare_staging_dir(staging_dir, args.binary_dir, args.version)
        build_wheel(staging_dir, wheel_dir)
        auditwheel_repair(wheel_dir)
        collect_wheels(wheel_dir, args.output_dir)


if __name__ == "__main__":
    main()
