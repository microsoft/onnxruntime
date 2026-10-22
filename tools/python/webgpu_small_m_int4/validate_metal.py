# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Resolve actual diagnostic WGSL and emit MSL using ORT's pinned Linux Tint."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tint", type=Path, required=True)
    parser.add_argument("--dump", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dependency = next(line for line in (ROOT / "cmake/deps.txt").read_text().splitlines() if line.startswith("dawn;"))
    records = []
    sources = {}
    for dump in args.dump:
        for code in re.findall(
            r"=== WebGPU Shader code \[DiagnosticSmallMSimt,[^\n]*\] Start ===\n\n"
            r"(.*?)\n=== WebGPU Shader code \[DiagnosticSmallMSimt,[^\n]*\] End ===",
            dump.read_text(),
            re.DOTALL,
        ):
            digest = hashlib.sha256(code.encode()).hexdigest()
            sources.setdefault(digest, (code, []))[1].append(str(dump))
    if not sources:
        raise ValueError("No actual DiagnosticSmallMSimt shader dumps were found.")
    report = {
        "dawn_dependency": dependency,
        "tint": str(args.tint.resolve()),
        "tint_sha256": hashlib.sha256(args.tint.read_bytes()).hexdigest(),
        "hardware_execution": False,
        "metal_sdk_compilation": False,
        "shaders": records,
    }
    for i, (digest, (code, dumps)) in enumerate(sources.items()):
        extensions = re.findall(r"^enable ([^;]+);", code, re.MULTILINE)
        if extensions != ["f16"] or re.search(r"\bsubgroup\w*\b", code):
            raise AssertionError("The actual diagnostic shader must require only f16 and no subgroup builtins.")
        source = args.output / f"simt-{i}.wgsl"
        source.write_text(code)
        a_size = int(re.search(r"var<workgroup> a_stage: array<vec4<f16>, (\d+)>;", code)[1]) * 8
        partial_size = int(re.search(r"var<workgroup> partial: array<f32, (\d+)>;", code)[1]) * 4
        record = {
            "sha256": digest,
            "origin_dumps": dumps,
            "declared_workgroup_storage_bytes": a_size + partial_size,
            "extensions": extensions,
            "phases": [],
        }
        for fmt in ("wgsl", "msl"):
            destination = args.output / f"simt-{i}.{'resolved.wgsl' if fmt == 'wgsl' else 'metal'}"
            command = [
                str(args.tint.resolve()),
                str(source.resolve()),
                "--format",
                fmt,
                "-o",
                str(destination.resolve()),
            ]
            if fmt == "wgsl":
                command.extend(["--validate", "true"])
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            record["phases"].append(
                {
                    "command": command,
                    "returncode": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                }
            )
            if process.returncode:
                records.append(record)
                (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                raise RuntimeError(f"Tint {fmt} failed: {process.stdout}{process.stderr}")
        records.append(record)
        print(f"shader {i}: WGSL and MSL passed; {a_size + partial_size} shared bytes; f16 only", flush=True)
    report["status"] = "passed"
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
