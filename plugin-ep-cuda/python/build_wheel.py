#!/usr/bin/env python3
"""Build a wheel for the onnxruntime-ep-cuda12 or onnxruntime-ep-cuda13 package."""

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

_TEMPLATE_VARIABLE_PATTERN = re.compile(r"@(\w+)@")
BINARY_PATTERNS = [
    "onnxruntime_providers_cuda_plugin.dll",
    "libonnxruntime_providers_cuda_plugin.so",
]
AUDITWHEEL_EXCLUDE = [
    "libcuda.so.1",
    "libnvidia-ml.so.1",
]


def gen_file_from_template(template_file: Path, output_file: Path, variable_substitutions: dict[str, str]) -> None:
    content = template_file.read_text(encoding="utf-8")
    variables_in_file: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        variables_in_file.add(name)
        return variable_substitutions.get(name, match.group(0))

    content = _TEMPLATE_VARIABLE_PATTERN.sub(replace, content)
    if variables_in_file != variable_substitutions.keys():
        provided = set(variable_substitutions.keys())
        raise ValueError(
            f"Template variables and substitution keys do not match for {template_file}. "
            f"Only in template: {sorted(variables_in_file - provided)}. "
            f"Only in substitutions: {sorted(provided - variables_in_file)}."
        )

    output_file.write_text(content, encoding="utf-8")


def prepare_staging_dir(staging_dir: Path, binary_dir: Path, version: str, package_name: str) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT_DIR / "setup.py", staging_dir / "setup.py")
    shutil.copytree(SCRIPT_DIR / "onnxruntime_ep_cuda", staging_dir / "onnxruntime_ep_cuda")

    package_dir = staging_dir / "onnxruntime_ep_cuda"
    copied = []
    for pattern in BINARY_PATTERNS:
        for src in binary_dir.glob(pattern):
            dst = package_dir / src.name
            print(f"Copying {src} -> {dst}")
            shutil.copy2(src, dst)
            copied.append(dst)
    if not copied:
        raise FileNotFoundError(f"No plugin binaries found in {binary_dir}. Looked for: {BINARY_PATTERNS}")

    min_ort_version = MIN_ONNXRUNTIME_VERSION_FILE.read_text(encoding="utf-8").strip()
    if not min_ort_version:
        raise ValueError(f"{MIN_ONNXRUNTIME_VERSION_FILE} is empty")

    gen_file_from_template(
        SCRIPT_DIR / "pyproject.toml.in",
        staging_dir / "pyproject.toml",
        {"package_name": package_name, "version": version, "min_onnxruntime_version": min_ort_version},
    )


def build_wheel(source_dir: Path, wheel_dir: Path) -> None:
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


def auditwheel_repair(wheel_dir: Path, wheel_name_prefix: str) -> None:
    if platform.system() != "Linux":
        return

    original_wheels = list(wheel_dir.glob(f"{wheel_name_prefix}-*.whl"))
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
            wheel.unlink()

        repaired_wheels = list(repaired_dir.glob("*.whl"))
        if not repaired_wheels:
            raise RuntimeError(f"auditwheel repair produced no wheels in {repaired_dir}")

        for repaired_wheel in repaired_wheels:
            repaired_wheel.replace(wheel_dir / repaired_wheel.name)


def collect_wheels(wheel_dir: Path, output_dir: Path, wheel_name_prefix: str) -> None:
    wheels = list(wheel_dir.glob(f"{wheel_name_prefix}-*.whl"))
    if not wheels:
        raise RuntimeError("No wheel was produced")
    output_dir.mkdir(parents=True, exist_ok=True)
    for wheel in wheels:
        dest = output_dir / wheel.name
        shutil.copy2(wheel, dest)
        print(f"Built wheel: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build onnxruntime-ep-cuda wheel")
    parser.add_argument("--binary_dir", required=True, type=Path, help="Directory containing built plugin EP binaries")
    parser.add_argument("--version", required=True, help="Package version string (PEP 440 format)")
    parser.add_argument("--package_name", required=True, help="Python distribution name to write into pyproject.toml")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to place the built wheel")
    args = parser.parse_args()

    if not args.binary_dir.is_dir():
        raise FileNotFoundError(f"Binary directory does not exist: {args.binary_dir}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.package_name):
        raise ValueError(f"Invalid package name: {args.package_name}")

    wheel_name_prefix = args.package_name.replace("-", "_").replace(".", "_")

    with tempfile.TemporaryDirectory(prefix="ort_cuda_wheel_") as tmp:
        staging_dir = Path(tmp) / "package"
        wheel_dir = Path(tmp) / "wheels"
        prepare_staging_dir(staging_dir, args.binary_dir, args.version, args.package_name)
        build_wheel(staging_dir, wheel_dir)
        auditwheel_repair(wheel_dir, wheel_name_prefix)
        collect_wheels(wheel_dir, args.output_dir, wheel_name_prefix)


if __name__ == "__main__":
    main()
