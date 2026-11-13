# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Capture compiler-key inputs for explicit Windows cache experiments."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def command_records(text):
    source = None
    for line in text.splitlines():
        if line.startswith("^"):
            source = line[1:]
        elif line.strip():
            if source is None:
                raise ValueError("Compiler command without a source record")
            yield source, line


def compiler_responses(directory):
    responses = []
    for path in sorted(directory.rglob("*.rsp")):
        raw = path.read_bytes()
        command = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
        if re.search(r"(?<!\S)/c(?=\s|$)", command):
            responses.append({"file": str(path), "command": command})
    return responses


def collect(build_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("cl")
    if compiler is None:
        raise FileNotFoundError("Configure the Visual C++ environment before collecting diagnostics")
    responses = compiler_responses(Path(tempfile.gettempdir()) / "MSBuildTemp")
    if not responses:
        raise RuntimeError("No preserved compiler response files; set MSBUILDPRESERVETOOLTEMPFILES=1 before building")
    records = []
    selected = set()
    for log in sorted(build_dir.rglob("CL.command.1.tlog")):
        if "CompilerId" in str(log):
            continue
        # MSBuild's tracking logs use UTF-16 and uppercase path spellings.
        for source, command in command_records(log.read_text(encoding="utf-16")):
            record = {
                "source": source,
                "log": str(log.relative_to(build_dir)),
                "command": command,
                "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                "cwd": str(log.parents[3]),
            }
            records.append(record)
            if re.search(r"/(?:Yc|Yu|Fp)", command):
                continue
            target = log.parent.parent.parent.name
            if target in selected or len(selected) >= 4:
                continue
            selected.add(target)
            # Tracking logs normalize path casing and can omit per-invocation options.
            matches = [
                response
                for response in responses
                if source.replace("\\", "/").casefold() in response["command"].replace("\\", "/").casefold()
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one actual compiler response for {source}, found {len(matches)}")
            record["response"] = matches[0]
            # /EP suppresses compilation; use the real compiler, never the cache wrapper.
            preprocess = re.sub(r"(?<!\S)/c(?=\s|$)", "/EP", matches[0]["command"])
            result = subprocess.run(
                subprocess.list2cmdline([compiler]) + " " + preprocess + " /WX-",
                cwd=log.parents[3],
                capture_output=True,
                check=True,
            )
            filename = f"preprocessed-{len(selected)}.txt"
            (output_dir / filename).write_bytes(result.stdout)
            (output_dir / f"preprocessed-{len(selected)}.stderr.txt").write_bytes(result.stderr)
            record["preprocessed"] = filename
            record["preprocessed_sha256"] = hashlib.sha256(result.stdout).hexdigest()
    if not records or not selected:
        raise RuntimeError("No cacheable compiler command records found")
    result = {
        "compiler": compiler,
        "compiler_sha256": hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
        "responses": responses,
        # Only cache-key inputs, not the complete environment (which can contain credentials).
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CL",
                "_CL_",
                "INCLUDE",
                "LIB",
                "SCCACHE_BASEDIRS",
                "SCCACHE_C_CUSTOM_CACHE_BUSTER",
                "SCCACHE_EXTRAFILES",
                "SDKROOT",
                "CCC_OVERRIDE_OPTIONS",
                "MACOSX_DEPLOYMENT_TARGET",
                "IPHONEOS_DEPLOYMENT_TARGET",
                "TVOS_DEPLOYMENT_TARGET",
                "WATCHOS_DEPLOYMENT_TARGET",
            )
        },
        "commands": records,
    }
    (output_dir / "compiler-inputs.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Captured {len(records)} compiler commands and {len(selected)} preprocessed translation units")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    collect(args.build_dir.resolve(), args.output_dir.resolve())
