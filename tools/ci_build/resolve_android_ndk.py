# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Resolve an installed NDK major without network access, or install an explicitly pinned revision."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def _read_revision(ndk_path: Path) -> str:
    properties = ndk_path / "source.properties"
    revision = re.search(r"^Pkg\.Revision\s*=\s*(.*?)\s*$", properties.read_text(encoding="utf-8"), re.MULTILINE)
    if revision is None:
        raise ValueError(f"NDK revision in {properties} does not match directory {ndk_path.name}")
    return revision.group(1)


def resolve_ndk(sdk_root: Path, version: str) -> Path:
    sdk_root = sdk_root.resolve(strict=True)
    ndk_root = sdk_root / "ndk"
    if re.fullmatch(r"[0-9]+", version):
        if not ndk_root.is_dir():
            raise ValueError(f"No NDK directory at {ndk_root}. Provision NDK {version} in the runner image.")
        candidates = [
            path
            for path in ndk_root.iterdir()
            if path.is_dir()
            and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", path.name)
            and path.name.split(".")[0] == version
        ]
        for candidate in sorted(
            candidates, key=lambda path: tuple(int(part) for part in path.name.split(".")), reverse=True
        ):
            revision = _read_revision(candidate)
            # SDK preview packages can also use numeric directory names.
            if re.fullmatch(rf"{re.escape(candidate.name)}[- ](?:alpha|beta|rc)[0-9]+", revision):
                continue
            ndk_path = candidate
            break
        else:
            raise ValueError(
                f"No stable NDK {version} is installed under {ndk_root}. Provision it in the runner image."
            )
    elif re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        ndk_path = ndk_root / version
        if not ndk_path.exists():
            sdkmanager = (
                sdk_root
                / "cmdline-tools"
                / "latest"
                / "bin"
                / ("sdkmanager.bat" if sys.platform == "win32" else "sdkmanager")
            )
            subprocess.run([str(sdkmanager), "--install", f"ndk;{version}"], check=True, stdout=sys.stderr)
        revision = _read_revision(ndk_path)
    else:
        raise ValueError(f"Expected an NDK major or full stable revision, got {version!r}")

    if revision != ndk_path.name:
        raise ValueError(f"NDK revision in {ndk_path / 'source.properties'} does not match directory {ndk_path.name}")
    toolchain = ndk_path / "build" / "cmake" / "android.toolchain.cmake"
    if not toolchain.is_file():
        raise ValueError(f"Missing NDK toolchain: {toolchain}")
    host = {"linux": "linux-x86_64", "darwin": "darwin-x86_64", "win32": "windows-x86_64"}[sys.platform]
    compiler = (
        ndk_path
        / "toolchains"
        / "llvm"
        / "prebuilt"
        / host
        / "bin"
        / ("clang.exe" if sys.platform == "win32" else "clang")
    )
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise ValueError(f"Missing or non-executable NDK compiler: {compiler}")
    subprocess.run([str(compiler), "--version"], check=True, stdout=sys.stderr)
    print(f"Using Android NDK {ndk_path.name}: {ndk_path}", file=sys.stderr)
    return ndk_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--android-sdk-root", type=Path, required=True)
    parser.add_argument(
        "--ndk-version", default="28", help="Installed major (offline), or exact revision (install if absent)"
    )
    args = parser.parse_args()
    print(resolve_ndk(args.android_sdk_root, args.ndk_version))


if __name__ == "__main__":
    main()
