import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

import numpy as np
import onnx
from generate_cases import generate
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(runner: Path, runtime: Path, output: Path):
    runner = runner.resolve(strict=True)
    runtime = runtime.resolve(strict=True)
    if output.exists():
        raise ValueError("Output already exists; use a fresh bundle directory")
    if runner.suffix.lower() != ".exe" or runtime.name.lower() != "onnxruntime.dll":
        raise ValueError("CPU CI currently stages a Windows executable and onnxruntime.dll")
    output.mkdir(parents=True)
    try:
        binary = output / "bin"
        binary.mkdir()
        shutil.copy2(runner, binary / "onnxruntime_ep_op_conformance_test.exe")
        shutil.copy2(runtime, binary / "onnxruntime.dll")
        shared = runtime.with_name("onnxruntime_providers_shared.dll")
        if shared.is_file():
            shutil.copy2(shared, binary / shared.name)
        shutil.copy2(ROOT / "tools/run_ci.ps1", output / "run_ci.ps1")
        (output / "reports").mkdir()

        definitions = ROOT / "cases/pilot.json"
        generate(definitions, output / "cases/positive")
        case_ids = sorted(path.name for path in (output / "cases/positive").iterdir() if path.is_dir())
        mutation_id = json.loads(definitions.read_text(encoding="utf-8"))["cases"][0]["id"]
        original = output / "cases/positive" / mutation_id
        for name in ("wrong-output", "malformed-artifact", "unknown-case-field"):
            shutil.copytree(original, output / "cases" / name)

        manifest = json.loads((original / "case.json").read_text(encoding="utf-8"))
        expected_path = output / "cases/wrong-output" / manifest["datasets"][0]["outputs"][0]
        expected = onnx.load_tensor(expected_path)
        wrong = numpy_helper.from_array(numpy_helper.to_array(expected).copy() + np.float32(1), name=expected.name)
        expected_path.write_bytes(wrong.SerializeToString(deterministic=True))
        malformed = output / "cases/malformed-artifact" / manifest["datasets"][0]["outputs"][0]
        malformed.write_bytes(b"\x80")
        write_json(output / "cases/unknown-case-field/case.json", {**manifest, "ignored_assertion": True})

        cpu = {"schema_version": 1, "kind": "builtin", "ep_name": "CPUExecutionProvider"}
        profiles = {
            "cpu": cpu,
            "invalid-profile": {**cpu, "kind": "cpu"},
            "unknown-profile-field": {**cpu, "provider_name": "CPU"},
            "partial-skip": {**cpu, "skips": {mutation_id: "CI explicit partial skip"}},
            "all-skipped": {**cpu, "skips": dict.fromkeys(case_ids, "CI explicit all-skipped check")},
            "unknown-skip": {**cpu, "skips": {"not-a-selected-case": "CI invalid skip check"}},
        }
        for name, profile in profiles.items():
            write_json(output / "profiles" / f"{name}.json", profile)

        source_files = [
            definitions,
            ROOT / "CMakeLists.txt",
            ROOT / "runner/main.cc",
            ROOT / "runner/case_io.h",
            ROOT / "runner/comparison.h",
            ROOT / "tools/generate_cases.py",
            Path(__file__).resolve(),
            ROOT / "tools/run_ci.ps1",
        ]
        public_headers = ROOT.parents[2] / "include/onnxruntime/core/session"
        source_files.extend(sorted(public_headers.glob("*.h")))
        metadata = {
            "schema_version": 1,
            "source_revision": os.environ.get("GITHUB_SHA"),
            "source_hashes": {path.relative_to(ROOT.parents[2]).as_posix(): sha256(path) for path in source_files},
            "producer_packages": {name: importlib.metadata.version(name) for name in ("numpy", "onnx", "protobuf")},
            "runtime_sha256": sha256(runtime),
            "runtime_prerequisite": "Microsoft Visual C++ 2015-2022 x64 Redistributable (on windows-2022)",
            "positive_case_ids": case_ids,
            "mutation_case_id": mutation_id,
            "files": {
                path.relative_to(output).as_posix(): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        }
        write_json(output / "prepared.json", metadata)
    except Exception:
        shutil.rmtree(output)
        raise
    print(f"Prepared {len(case_ids)} canonical CPU cases and negative checks in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage an immutable, Python-free Windows CPU CI bundle")
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    prepare(arguments.runner, arguments.runtime, arguments.output.resolve())
