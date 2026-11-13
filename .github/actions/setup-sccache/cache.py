# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import re
import shlex
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


def configure():
    image = os.environ.get("CACHE_CONTAINER_IMAGE", "")
    environment = ""
    if image:
        prefix = os.environ.get("CACHE_CONTAINER_PREFIX", "") or ":"
        probe = (
            "set -e; command -v sccache; sccache --version; "
            'command -v "${CC:-cc}"; "${CC:-cc}" --version; '
            'command -v "${CXX:-c++}"; "${CXX:-c++}" --version; '
            "cmake --version; python3 --version; cat /etc/os-release"
        )
        environment = subprocess.check_output(
            ["docker", "run", "--rm", image, "/bin/bash", "-c", f"set -e; {prefix}; {probe}"], text=True
        ).strip()
        # docker run also pulls a registry cache hit that is not present on this host yet.
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True
        ).strip()
        versions = re.findall(r"^sccache \S+$", environment, re.MULTILINE)
        if len(versions) != 1:
            raise ValueError("Expected exactly one installed sccache version inside the build image")
        version = versions[0]
        environment = f"{image_id}\n{environment}"
        print(f"Compiler-cache container identity:\n{environment}")
    else:
        binary = shutil.which("sccache")
        if binary is None:
            raise FileNotFoundError("sccache was not installed")
        version = subprocess.check_output([binary, "--version"], text=True).strip()
        print(f"Compiler-cache executable: {binary}\n{version}")
    if not re.fullmatch(r"sccache \d+\.\d+\.\d+(?:[-+]\S+)?", version):
        raise ValueError(f"Invalid sccache --version output: {version!r}")

    identity_text = f"{os.environ['CACHE_VARIANT']}\n{version}"
    if image:
        identity_text += f"\n{environment}"
    identity = hashlib.sha256(identity_text.encode()).hexdigest()
    prefix = f"{os.environ['CACHE_PREFIX']}-{os.environ['RUNNER_OS']}-{os.environ['RUNNER_ARCH']}-{identity}-"
    cache_dir = Path(os.environ["GITHUB_WORKSPACE"]) / ".sccache"
    cache_dir.mkdir(exist_ok=True)
    append_values(
        os.environ["GITHUB_ENV"],
        {
            "SCCACHE_DIR": cache_dir,
            "SCCACHE_BASEDIRS": os.environ["GITHUB_WORKSPACE"],
            "SCCACHE_CACHE_SIZE": os.environ["CACHE_SIZE"],
            "SCCACHE_LOCAL_RW_MODE": "READ_WRITE" if os.environ["CACHE_REF"] == "refs/heads/main" else "READ_ONLY",
            "SCCACHE_CONTAINER_IMAGE": image,
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
    if os.environ.get("SCCACHE_CONTAINER_IMAGE"):
        values = {
            "SCCACHE_ENABLED": enabled_text,
            "SCCACHE_DIR": "/onnxruntime_src/.sccache",
            "SCCACHE_BASEDIRS": "/onnxruntime_src",
            "SCCACHE_CACHE_SIZE": os.environ["SCCACHE_CACHE_SIZE"],
            "SCCACHE_LOCAL_RW_MODE": os.environ["SCCACHE_LOCAL_RW_MODE"],
        }
        # The pinned Docker runner mounts RUNNER_TEMP at /onnxruntime_src/build.
        with open(Path(os.environ["RUNNER_TEMP"]) / "ort-sccache.env", "w", encoding="utf-8", newline="\n") as output:
            output.writelines(f"export {key}={shlex.quote(value)}\n" for key, value in values.items())


def verify_launchers():
    cache = Path(os.environ["RUNNER_TEMP"]) / os.environ["BUILD_CONFIG"] / os.environ["BUILD_CONFIG"] / "CMakeCache.txt"
    entries = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            key, value = line.split("=", 1)
            entries[key.split(":", 1)[0]] = value
    expected = "sccache" if os.environ["SCCACHE_ENABLED"] == "true" else ""
    languages = ["C", "CXX"]
    if entries.get("CMAKE_CUDA_COMPILER"):
        languages.append("CUDA")
    elif entries.get("CMAKE_CUDA_COMPILER_LAUNCHER"):
        raise ValueError("CUDA compiler launcher configured without a CUDA compiler")
    for language in languages:
        key = f"CMAKE_{language}_COMPILER_LAUNCHER"
        actual = entries.get(key, "")
        if actual != expected:
            raise ValueError(f"{key}: expected {expected!r}, got {actual!r}")
        print(f"{key}={actual!r}; compiler={entries[f'CMAKE_{language}_COMPILER']}")


def report():
    if os.environ.get("SCCACHE_CONTAINER_IMAGE"):
        temp = Path(os.environ["RUNNER_TEMP"])
        if not (temp / "sccache-flushed").is_file():
            raise RuntimeError("The build container did not confirm a stopped/flushed sccache server")
        stats = (temp / "sccache-stats.txt").read_text(encoding="utf-8")
    else:
        stats = subprocess.check_output(["sccache", "--show-stats"], text=True)
    print(stats, end="")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
        summary.write(f"### Compiler cache\n```text\n{stats}\n```\n")


if __name__ == "__main__":
    {"configure": configure, "select": select, "verify": verify_launchers, "report": report}[sys.argv[1]]()
