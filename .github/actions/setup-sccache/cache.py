# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def append_values(file, values):
    with open(file, "a", encoding="utf-8") as output:
        for key, value in values.items():
            if "\n" in str(value) or "\r" in str(value):
                raise ValueError(f"Multiline value for {key}")
            output.write(f"{key}={value}\n")


def use_cache(ref, matched_key):
    return ref == "refs/heads/main" or bool(matched_key)


def detect():
    binary = shutil.which("sccache")
    if binary:
        print(f"Preinstalled compiler cache: {binary}")
    elif os.environ["RUNNER_OS"] == "Windows":
        raise FileNotFoundError(
            "sccache must be preinstalled on the Windows CI image and available in the compiler PATH"
        )
    append_values(os.environ["GITHUB_OUTPUT"], {"installed": str(bool(binary)).lower()})


def configure():
    binary = shutil.which("sccache")
    if binary is None:
        raise FileNotFoundError("sccache was not installed")
    version = subprocess.check_output([binary, "--version"], text=True).strip()
    print(f"Compiler-cache executable: {binary}\n{version}")
    match = re.fullmatch(r"sccache (\d+)\.(\d+)\.(\d+)(?:[-+]\S+)?", version)
    if not match:
        raise ValueError(f"Invalid sccache --version output: {version!r}")

    identity_text = f"{os.environ['CACHE_VARIANT']}\n{version}"
    if os.environ["RUNNER_OS"] == "Windows":
        if tuple(map(int, match.groups())) < (0, 19, 0):
            raise ValueError("Windows compiler interception requires sccache 0.19.0 or newer")
        compiler = shutil.which("cl")
        if compiler is None:
            raise FileNotFoundError("Configure the Visual C++ environment before setup-sccache")
        # Hash the selected compiler, not a version inferred from an image label.
        identity = {
            "compiler": compiler,
            "compiler-sha256": hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
            **{
                key: os.environ.get(key, "")
                for key in (
                    "VCToolsVersion",
                    "WindowsSDKVersion",
                    "VSCMD_ARG_HOST_ARCH",
                    "VSCMD_ARG_TGT_ARCH",
                    "ImageVersion",
                    "INCLUDE",
                    "LIB",
                )
            },
        }
        identity_text += "\n" + repr(identity)
        print(f"Windows compiler-cache identity: {identity}")
        nvcc = shutil.which("nvcc")
        if nvcc:
            cuda_version = subprocess.check_output([nvcc, "--version"], text=True).strip()
            identity_text += f"\n{nvcc}\n{cuda_version}"
            print(f"CUDA compiler-cache identity: {nvcc}\n{cuda_version}")
    prefix = "sccache-v1-{}-{}-{}-".format(
        os.environ["RUNNER_OS"], os.environ["RUNNER_ARCH"], hashlib.sha256(identity_text.encode()).hexdigest()
    )
    append_values(
        os.environ["GITHUB_ENV"],
        {
            "SCCACHE_DIR": Path(os.environ["GITHUB_WORKSPACE"]) / ".sccache",
            "SCCACHE_BASEDIRS": os.environ["GITHUB_WORKSPACE"],
            "SCCACHE_CACHE_SIZE": os.environ["CACHE_SIZE"],
            "SCCACHE_LOCAL_RW_MODE": "READ_WRITE" if os.environ["CACHE_REF"] == "refs/heads/main" else "READ_ONLY",
        },
    )
    append_values(os.environ["GITHUB_OUTPUT"], {"prefix": prefix})


def select():
    enabled = use_cache(os.environ["CACHE_REF"], os.environ.get("CACHE_MATCHED_KEY", ""))
    enabled_text = str(enabled).lower()
    append_values(os.environ["GITHUB_ENV"], {"SCCACHE_ENABLED": enabled_text})
    append_values(
        os.environ["GITHUB_OUTPUT"],
        {"enabled": enabled_text, "build-flags": "--use_cache --cache_tool sccache" if enabled else ""},
    )
    decision = (
        f"sccache enabled ({os.environ['SCCACHE_LOCAL_RW_MODE']}); "
        f"restored snapshot: {os.environ.get('CACHE_MATCHED_KEY') or 'none (populate on main)'}"
        if enabled
        else "sccache bypassed: nonpublishing run without a compatible snapshot"
    )
    print(decision)
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
        summary.write(f"### Compiler-cache decision\n{decision}\n")


if __name__ == "__main__":
    {"detect": detect, "configure": configure, "select": select}[sys.argv[1]]()
