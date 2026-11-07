# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Assemble the installable ONNX Runtime Python package tree."""

import argparse
import json
import re
import shutil
from pathlib import Path


def _safe_destination(root: Path, relative_path: str) -> Path:
    destination = (root / relative_path).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"Package destination escapes the output tree: {relative_path!r}")
    return destination


def _copy_file(source: Path, destination: Path, version: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.as_posix().endswith("onnxruntime/__init__.py"):
        contents = source.read_text(encoding="utf-8")
        contents, replacements = re.subn(
            r'^__version__ = "[^"]+"$',
            f'__version__ = "{version}"',
            contents,
            count=1,
            flags=re.MULTILINE,
        )
        if replacements != 1:
            raise ValueError(f"Could not update __version__ in {source}")
        destination.write_text(contents, encoding="utf-8")
        return
    if source.name == "_pybind_state.py.in":
        contents = source.read_text(encoding="utf-8")
        for placeholder in (
            "@ONNXRUNTIME_IMPORT_PYTORCH_TO_RESOLVE_DLLS@",
            "@ONNXRUNTIME_SETDLOPENFLAGS_GLOBAL@",
            "@ONNXRUNTIME_SETDLOPENFLAGS_LOCAL@",
        ):
            contents = contents.replace(placeholder, "")
        destination.write_text(contents, encoding="utf-8")
        return
    shutil.copyfile(source, destination)


def _prepare_output(output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Package output exists but is not a directory: {output}")
        if next(output.iterdir(), None) is not None:
            raise ValueError(f"Package output directory must be empty: {output}")
        return
    output.mkdir(parents=True)


def _append_linux_cuda_preload(preload_path: Path, cuda_version: str) -> None:
    cuda_major = cuda_version.split(".", maxsplit=1)[0]
    if cuda_major not in {"12", "13"}:
        raise ValueError(f"Linux CUDA wheels support CUDA major versions 12 and 13, got {cuda_version!r}.")
    with preload_path.open("a", encoding="utf-8") as preload:
        preload.write(
            "\nimport os\n"
            "from ctypes import CDLL, RTLD_GLOBAL\n"
            "_libcudart = None\n"
            f'for _cudart_lib in ["libcudart.so.{cuda_major}"]:\n'
            "    try:\n"
            "        _libcudart = CDLL(_cudart_lib, mode=RTLD_GLOBAL)\n"
            "        break\n"
            "    except OSError:\n"
            "        pass\n"
            "if _libcudart is None:\n"
            '    os.environ["ORT_CUDA_UNAVAILABLE"] = "1"\n'
        )


def assemble(manifest_path: Path, output: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = Path(manifest["version_file"]).read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError("VERSION_NUMBER is empty.")

    _prepare_output(output)
    destinations = set()
    for entry in manifest["files"]:
        relative_path = entry["destination"]
        if relative_path in destinations:
            raise ValueError(f"Duplicate package destination: {relative_path}")
        destinations.add(relative_path)
        _copy_file(
            Path(entry["source"]),
            _safe_destination(output, relative_path),
            version,
        )

    if manifest["cuda_enabled"] and manifest["linux"]:
        _append_linux_cuda_preload(
            _safe_destination(output, "onnxruntime/capi/_ld_preload.py"),
            manifest["cuda_version"],
        )

    tools_init = _safe_destination(output, "onnxruntime/tools/__init__.py")
    append_source = Path(manifest["tools_init_append"])
    with tools_init.open("a", encoding="utf-8") as destination:
        destination.write("\n")
        destination.write(append_source.read_text(encoding="utf-8"))

    capi = output / "onnxruntime" / "capi"
    capi.mkdir(parents=True, exist_ok=True)
    (capi / "version_info.py").write_text(
        f"use_cuda = {manifest['cuda_enabled']!r}\nvs2019 = False\n",
        encoding="utf-8",
    )
    package_info = [
        f"package_name = {manifest['package_name']!r}",
        f"__version__ = {version!r}",
    ]
    if manifest["cuda_version"]:
        package_info.append(f"cuda_version = {manifest['cuda_version']!r}")
    (capi / "build_and_package_info.py").write_text(
        "\n".join(package_info) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.manifest, args.output)


if __name__ == "__main__":
    main()
