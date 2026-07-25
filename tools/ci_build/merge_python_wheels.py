# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Merge several single-CPython onnxruntime wheels into one multi-CPython wheel.

    python merge_python_wheels.py --out-dir dist \
        onnxruntime_gpu-1.28.0-cp312-cp312-manylinux_2_28_x86_64.whl \
        onnxruntime_gpu-1.28.0-cp313-cp313-manylinux_2_28_x86_64.whl

produces ``onnxruntime_gpu-1.28.0-cp312.cp313-cp312.cp313-manylinux_2_28_x86_64.whl``.

The merged wheel keeps the exact tag of every input wheel, so pip installs it on
those interpreters and on no others. A compressed tag set expands to the cross
product of its parts, which is why the free-threaded build stays distinguishable:
``cp313-cp313`` matches the default 3.13 build while a free-threaded interpreter
only ever matches ``cp313-cp313t``. The wheels must all be built for the same
platform: a x86_64 wheel and an aarch64 wheel share no native library and must
stay separate packages.

Every file that is byte identical across the input wheels is stored once. That
is where the native libraries live, and they contain no CPython symbols, so a
single copy serves every interpreter. The only file that may differ is
``onnxruntime_pybind11_state``; it stays in ``onnxruntime/capi/`` but is renamed
to carry the interpreter specific extension suffix, for example
``onnxruntime_pybind11_state.cpython-312-x86_64-linux-gnu.so``.

No import time dispatcher is needed: CPython's own FileFinder tries every entry
of ``importlib.machinery.EXTENSION_SUFFIXES`` -- which starts with the ABI
specific one -- so ``from .onnxruntime_pybind11_state import *`` in
``onnxruntime/capi/_pybind_state.py`` loads the file matching the running
interpreter and ignores the others.

Keeping the extension inside ``onnxruntime/capi/`` also matters at runtime:
onnxruntime resolves ``onnxruntime_providers_*`` through
``Env::Default().GetRuntimePath()``, which is the directory of the module that
links the ORT core, i.e. the pybind extension itself. Moving the extension into
a per-interpreter sub-directory breaks the provider bridge.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import sys
import zipfile

EXTENSION_MODULE_SUFFIXES = (".so", ".pyd", ".dylib")

# The only file that is allowed to differ between the input wheels.
PYBIND_MODULE_DIRECTORY = "onnxruntime/capi"
PYBIND_MODULE_STEM = "onnxruntime_pybind11_state"

LINUX_ARCHITECTURES = ("x86_64", "aarch64", "armv7l", "i686", "ppc64le", "s390x", "riscv64")

CHUNK_SIZE = 1 << 20


def parse_wheel_name(path: str):
    """Split a wheel file name into (name, version, python_tag, abi_tag, platform_tag)."""
    stem = os.path.basename(path)
    if not stem.endswith(".whl"):
        raise ValueError(f"not a wheel: {path}")
    parts = stem[: -len(".whl")].split("-")
    if len(parts) not in (5, 6):  # the optional build tag makes it 6
        raise ValueError(f"unexpected wheel name: {stem}")
    return parts[0], parts[1], parts[-3], parts[-2], parts[-1]


def extension_suffix(abi_tag: str, platform_tag: str) -> str:
    """The value of ``sysconfig.get_config_var('EXT_SUFFIX')`` for a wheel tag.

    ``cp312`` / ``manylinux_2_28_x86_64`` -> ``.cpython-312-x86_64-linux-gnu.so``
    ``cp313t`` / ``manylinux_2_28_x86_64`` -> ``.cpython-313t-x86_64-linux-gnu.so``
    ``cp312`` / ``macosx_11_0_arm64``     -> ``.cpython-312-darwin.so``
    ``cp312`` / ``win_amd64``             -> ``.cp312-win_amd64.pyd``
    """
    match = re.fullmatch(r"cp(\d)(\d+)(t?)", abi_tag)
    if not match:
        raise ValueError(f"only CPython ABI tags are supported, got {abi_tag!r}")
    major, minor, freethreaded = match.groups()

    if platform_tag.startswith("win"):
        return f".{abi_tag}-{platform_tag}.pyd"
    if platform_tag.startswith("macosx"):
        return f".cpython-{major}{minor}{freethreaded}-darwin.so"
    for architecture in LINUX_ARCHITECTURES:
        if platform_tag.endswith("_" + architecture):
            return f".cpython-{major}{minor}{freethreaded}-{architecture}-linux-gnu.so"
    raise ValueError(f"cannot derive the extension suffix for platform tag {platform_tag!r}")


def is_pybind_module(path: str) -> bool:
    """Whether an archive member is the onnxruntime CPython extension module."""
    directory, _, filename = path.rpartition("/")
    return (
        directory == PYBIND_MODULE_DIRECTORY
        and filename.split(".", 1)[0] == PYBIND_MODULE_STEM
        and filename.endswith(EXTENSION_MODULE_SUFFIXES)
    )


def compress_tag_set(tags: list[str]) -> str:
    """Join interpreter tags the way a wheel file name does, lowest version first.

    ``["cp313t", "cp39", "cp310"]`` -> ``"cp39.cp310.cp313t"``
    """

    def sort_key(tag: str):
        match = re.fullmatch(r"cp(\d)(\d+)(t?)", tag)
        return (int(match.group(1)), int(match.group(2)), bool(match.group(3))) if match else (99, 99, False)

    return ".".join(sorted(set(tags), key=sort_key))


def file_hashes(wheel: str) -> dict[str, str]:
    hashes = {}
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            digest = hashlib.sha256()
            with archive.open(info) as handle:
                while chunk := handle.read(CHUNK_SIZE):
                    digest.update(chunk)
            hashes[info.filename] = digest.hexdigest()
    return hashes


def record_entry(name: str, data_hash, size: int):
    encoded = base64.urlsafe_b64encode(data_hash.digest()).decode("ascii").rstrip("=")
    return (name, "sha256=" + encoded, size)


def copy_member(source: zipfile.ZipFile, info: zipfile.ZipInfo, target: zipfile.ZipFile, name: str):
    """Stream one member into the output wheel, preserving its mode bits."""
    target_info = zipfile.ZipInfo(name, date_time=info.date_time)
    target_info.external_attr = info.external_attr
    target_info.internal_attr = info.internal_attr
    target_info.create_system = info.create_system
    target_info.compress_type = zipfile.ZIP_DEFLATED

    digest = hashlib.sha256()
    size = 0
    with source.open(info) as reader, target.open(target_info, "w") as writer:
        while True:
            chunk = reader.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            writer.write(chunk)
    return record_entry(name, digest, size)


def write_text_member(target: zipfile.ZipFile, name: str, text: str):
    data = text.encode("utf-8")
    target.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), data)
    return record_entry(name, hashlib.sha256(data), len(data))


def retag_wheel_metadata(text: str, tags: list[str]) -> str:
    """Replace the single ``cp3xx-cp3xx-<plat>`` tag of a WHEEL file with one line per input wheel."""
    lines = []
    for line in text.splitlines():
        if line.startswith("Tag:"):
            continue
        if line.startswith("Root-Is-Purelib:"):
            # Keep it a platform wheel so that pip installs it into platlib.
            lines.append("Root-Is-Purelib: false")
            continue
        lines.append(line)
    lines.extend(f"Tag: {tag}" for tag in tags)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wheels", nargs="+", help="the single-CPython wheels to merge")
    parser.add_argument("--out-dir", default="dist", help="where to write the merged wheel")
    args = parser.parse_args()

    if len(args.wheels) < 2:
        sys.exit("at least two wheels are required")

    metadata = [parse_wheel_name(wheel) for wheel in args.wheels]
    identities = {(name, version, platform_tag) for name, version, _py, _abi, platform_tag in metadata}
    if len(identities) != 1:
        sys.exit(f"the wheels must share name, version and platform tag, got: {sorted(identities)}")
    name, version, platform_tag = identities.pop()

    abi_tags = [abi_tag for _n, _v, _py, abi_tag, _p in metadata]
    if len(set(abi_tags)) != len(abi_tags):
        sys.exit(f"the same ABI tag appears twice: {abi_tags}")
    suffixes = [extension_suffix(abi_tag, platform_tag) for abi_tag in abi_tags]

    # Keep the exact tag of every input wheel. pip expands the compressed sets in the file name to
    # their cross product, so each interpreter still matches only the entry it has a binding for.
    python_tags = [python_tag for _n, _v, python_tag, _abi, _p in metadata]
    tags = [f"{python_tag}-{abi_tag}-{platform_tag}" for python_tag, abi_tag in zip(python_tags, abi_tags, strict=True)]

    hashes = [file_hashes(wheel) for wheel in args.wheels]
    all_paths = sorted(set().union(*[set(h) for h in hashes]))
    identical, differing = [], []
    for path in all_paths:
        if all(path in h for h in hashes) and len({h[path] for h in hashes}) == 1:
            identical.append(path)
        else:
            differing.append(path)

    dist_info = f"{name}-{version}.dist-info"
    # The dist-info files legitimately differ (WHEEL carries the tag); they are regenerated below.
    per_interpreter = [path for path in differing if not path.startswith(dist_info + "/")]
    unexpected = [path for path in per_interpreter if not is_pybind_module(path)]
    if unexpected:
        sys.exit(
            "only the onnxruntime pybind11 extension module may differ between the wheels, because "
            "it is the only file that is renamed per interpreter. These files differ too:\n    "
            + "\n    ".join(unexpected)
        )

    # Each wheel has to contribute its own binding, otherwise the merged wheel would advertise an
    # interpreter it cannot serve.
    pybind_paths = []
    for wheel, wheel_hashes in zip(args.wheels, hashes, strict=True):
        matches = sorted(path for path in wheel_hashes if is_pybind_module(path))
        if len(matches) != 1:
            sys.exit(f"expected exactly one {PYBIND_MODULE_DIRECTORY}/{PYBIND_MODULE_STEM} in {wheel}, got {matches}")
        if matches[0] not in per_interpreter:
            sys.exit(f"{matches[0]} is byte identical across the wheels, so they are not for different interpreters")
        pybind_paths.append(matches[0])

    print(f"identical files       : {len(identical)}")
    print(f"per-interpreter files : {len(per_interpreter)}")
    for path in per_interpreter:
        print(f"    {path}")

    os.makedirs(args.out_dir, exist_ok=True)
    merged_tag = f"{compress_tag_set(python_tags)}-{compress_tag_set(abi_tags)}-{platform_tag}"
    out_wheel = os.path.join(args.out_dir, f"{name}-{version}-{merged_tag}.whl")
    records = []
    with zipfile.ZipFile(out_wheel, "w", zipfile.ZIP_DEFLATED) as out_zip:
        # 1. Everything that is shared, taken from the first wheel.
        with zipfile.ZipFile(args.wheels[0]) as base_zip:
            for path in identical:
                if path.startswith(dist_info + "/"):
                    continue
                records.append(copy_member(base_zip, base_zip.getinfo(path), out_zip, path))

            # 2. The dist-info of the first wheel, with the WHEEL file retagged.
            for path in sorted(p for p in all_paths if p.startswith(dist_info + "/")):
                if path == f"{dist_info}/RECORD":
                    continue  # regenerated below
                if path not in base_zip.namelist():
                    sys.exit(f"{path} is missing from {args.wheels[0]}")
                if path == f"{dist_info}/WHEEL":
                    text = base_zip.read(path).decode("utf-8")
                    records.append(write_text_member(out_zip, path, retag_wheel_metadata(text, tags)))
                else:
                    records.append(copy_member(base_zip, base_zip.getinfo(path), out_zip, path))

        # 3. One copy of every per-interpreter extension, renamed with its ABI suffix.
        for wheel, suffix, pybind_path in zip(args.wheels, suffixes, pybind_paths, strict=True):
            with zipfile.ZipFile(wheel) as archive:
                target = f"{PYBIND_MODULE_DIRECTORY}/{PYBIND_MODULE_STEM}{suffix}"
                records.append(copy_member(archive, archive.getinfo(pybind_path), out_zip, target))

        # 4. RECORD must list every file in the wheel, itself included (without a hash).
        records.sort()
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerows(records)
        writer.writerow([f"{dist_info}/RECORD", "", ""])
        out_zip.writestr(
            zipfile.ZipInfo(f"{dist_info}/RECORD", date_time=(1980, 1, 1, 0, 0, 0)),
            buffer.getvalue().encode("utf-8"),
        )

    total_in = sum(os.path.getsize(wheel) for wheel in args.wheels)
    total_out = os.path.getsize(out_wheel)
    print(f"\ninput  : {len(args.wheels)} wheels, {total_in:,} bytes total")
    for wheel in args.wheels:
        print(f"    {os.path.basename(wheel):<70} {os.path.getsize(wheel):>14,}")
    print(f"output : {os.path.basename(out_wheel):<70} {total_out:>14,}")
    print(f"saving : {total_in - total_out:,} bytes ({100.0 * (total_in - total_out) / total_in:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
