# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Create a deterministic native ONNX Runtime wheel without setuptools."""

import argparse
import base64
import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path

_WHEEL_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _metadata_name(distribution: str) -> str:
    return re.sub(r"[-_.]+", "-", distribution)


def _filename_name(distribution: str) -> str:
    return re.sub(r"[-_.]+", "_", distribution).lower()


def _read_requirements(path: Path) -> list[str]:
    return [
        line
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith(("#", "--"))
    ]


def _cuda_extra_requirements(cuda_version: str) -> dict[str, list[str]]:
    if not cuda_version:
        return {}
    major = cuda_version.split(".", maxsplit=1)[0]
    if not major.isdigit():
        raise ValueError(f"CUDA version must start with a numeric major version, got {cuda_version!r}.")
    suffix = "" if int(major) >= 13 else f"-cu{major}"
    cufft_version = "12.0" if major == "13" else "11.0"
    return {
        "cuda": [
            f"nvidia-cuda-nvrtc{suffix}~={major}.0",
            f"nvidia-cuda-runtime{suffix}~={major}.0",
            f"nvidia-cufft{suffix}~={cufft_version}",
            f"nvidia-curand{suffix}~=10.0",
        ],
        "cudnn": [f"nvidia-cudnn-cu{major}~=9.0"],
    }


def _metadata(
    *,
    distribution: str,
    version: str,
    python_requires: str,
    requirements: list[str],
    cuda_version: str,
    description: str,
) -> bytes:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {_metadata_name(distribution)}",
        f"Version: {version}",
        "Summary: ONNX Runtime is a runtime accelerator for Machine Learning models",
        "Home-page: https://onnxruntime.ai",
        "Author: Microsoft Corporation",
        "Author-email: onnxruntime@microsoft.com",
        "License: MIT License",
        f"Requires-Python: {python_requires}",
        "Keywords: onnx machine learning",
        "Description-Content-Type: text/x-rst",
        "Project-URL: Download, https://github.com/microsoft/onnxruntime/tags",
        "Classifier: Development Status :: 5 - Production/Stable",
        "Classifier: Intended Audience :: Developers",
        "Classifier: License :: OSI Approved :: MIT License",
        "Classifier: Operating System :: POSIX :: Linux",
        "Classifier: Operating System :: Microsoft :: Windows",
        "Classifier: Operating System :: MacOS",
        "Classifier: Topic :: Scientific/Engineering",
        "Classifier: Topic :: Scientific/Engineering :: Mathematics",
        "Classifier: Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Classifier: Topic :: Software Development",
        "Classifier: Topic :: Software Development :: Libraries",
        "Classifier: Topic :: Software Development :: Libraries :: Python Modules",
        "Classifier: Programming Language :: Python",
        "Classifier: Programming Language :: Python :: 3 :: Only",
    ]
    lines.extend(f"Classifier: Programming Language :: Python :: 3.{minor}" for minor in range(11, 15))
    lines.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    extras = {
        "quantization": ["ml_dtypes"],
        "symbolic": ["sympy"],
        **_cuda_extra_requirements(cuda_version),
    }
    for extra, extra_requirements in sorted(extras.items()):
        lines.append(f"Provides-Extra: {extra}")
        lines.extend(f"Requires-Dist: {requirement}; extra == '{extra}'" for requirement in extra_requirements)
    return ("\n".join(lines) + "\n\n" + description).encode("utf-8")


def _zip_info(name: str, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, _WHEEL_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o755 if executable else 0o644) << 16
    return info


def _record(rows: list[tuple[str, bytes]], record_name: str) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name, contents in rows:
        digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=").decode("ascii")
        writer.writerow((name, f"sha256={digest}", len(contents)))
    writer.writerow((record_name, "", ""))
    return stream.getvalue().encode("utf-8")


def _prepare_output(output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Wheel output exists but is not a directory: {output}")
        if next(output.iterdir(), None) is not None:
            raise ValueError(f"Wheel output directory must be empty: {output}")
        return
    output.mkdir(parents=True)


def build_wheel(args: argparse.Namespace) -> Path:
    version = args.version_file.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError("VERSION_NUMBER is empty.")
    distribution = _filename_name(args.distribution)
    dist_info = f"{distribution}-{version}.dist-info"
    tag = f"{args.python_tag}-{args.abi}-{args.platform}"
    wheel_name = f"{distribution}-{version}-{tag}.whl"
    output = args.output / wheel_name
    _prepare_output(args.output)

    rows = []
    for source in sorted(args.package_tree.rglob("*")):
        if not source.is_file():
            continue
        name = source.relative_to(args.package_tree).as_posix()
        rows.append((name, source.read_bytes()))

    description = args.description_file.read_text(encoding="utf-8")
    rows.extend(
        [
            (
                f"{dist_info}/METADATA",
                _metadata(
                    distribution=args.distribution,
                    version=version,
                    python_requires=args.python_requires,
                    requirements=_read_requirements(args.requirements_file),
                    cuda_version=args.cuda_version,
                    description=description,
                ),
            ),
            (
                f"{dist_info}/WHEEL",
                (f"Wheel-Version: 1.0\nGenerator: onnxruntime-bazel\nRoot-Is-Purelib: false\nTag: {tag}\n").encode(),
            ),
            (
                f"{dist_info}/entry_points.txt",
                b"[console_scripts]\nonnxruntime_test = onnxruntime.tools.onnxruntime_test:main\n",
            ),
        ]
    )
    for license_file in args.license_file:
        rows.append((f"{dist_info}/{license_file.name}", license_file.read_bytes()))
    rows.sort(key=lambda row: row[0])
    record_name = f"{dist_info}/RECORD"
    record = _record(rows, record_name)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as wheel:
        for name, contents in rows:
            wheel.writestr(_zip_info(name, executable=name.endswith((".dll", ".dylib", ".pyd", ".so"))), contents)
        wheel.writestr(_zip_info(record_name), record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-tree", type=Path, required=True)
    parser.add_argument("--version-file", type=Path, required=True)
    parser.add_argument("--requirements-file", type=Path, required=True)
    parser.add_argument("--description-file", type=Path, required=True)
    parser.add_argument("--license-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--python-tag", required=True)
    parser.add_argument("--abi", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--python-requires", default=">=3.11")
    parser.add_argument("--cuda-version", default="")
    args = parser.parse_args()
    build_wheel(args)


if __name__ == "__main__":
    main()
