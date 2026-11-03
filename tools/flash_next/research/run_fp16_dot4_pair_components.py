import json
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np
import onnx
from onnx import numpy_helper

from build_fp16_dot4_pair_components import FLAG
from build_qmoe_node_replay import sha
from prepare_fp16_dot4_pair_matrix_fixtures import prepare
from run_fp16_dot4_tree_probe import guarded, workspace, UUID


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-pair-components/build-v1")
FIXTURES = BUILD.parent / "fixtures-v1"
PREFIX = ROOT / "fp16-dot4-pair-components-v1-native-gpu7"


def prepare_components(output):
    prepare(output=output)
    records = json.loads((output / "manifest.json").read_text())
    for seed in range(8):
        path = output / f"seed{seed}.onnx"
        model = onnx.load(path)
        tensors = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
        del model.graph.initializer[:]
        for name, values in tensors.items():
            if name in ("b", "scales"):
                values = np.concatenate([values[:16]] * 10)
            model.graph.initializer.append(numpy_helper.from_array(values, name))
        node, = model.graph.node
        attribute, = [v for v in node.attribute if v.name == "N"]
        if attribute.i != 64:
            raise RuntimeError("Unexpected inherited fixture")
        attribute.i = 160
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 160
        onnx.checker.check_model(model)
        onnx.save(model, path)
        records[path.name]["fixture_sha256"] = sha(path)
    (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    outcome = ROOT / "fp16-dot4-pair-components-v1-outcome.json"
    if outcome.exists():
        raise FileExistsError(outcome)
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["source_sha256"].items():
        if sha(BUILD / name) != digest:
            raise RuntimeError(f"Changed build input: {name}")
    provider, runner = BUILD / "libonnxruntime_providers_webgpu.so", BUILD / "pair_matrix_compare"
    if sha(provider) != manifest["provider_sha256"] or sha(runner) != manifest["diagnostic_runner_sha256"]:
        raise RuntimeError("Changed provider or comparator")
    prepare_components(FIXTURES)
    paths = [FIXTURES / f"seed{seed}.onnx" for seed in range(8)]
    subprocess.run(["nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader"], check=True)
    head = "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " + UUID
    tail = "; exec " + shlex.join([workspace(runner), workspace(provider), *map(workspace, paths)])
    code, log = guarded(head + f"; export {FLAG}=1" + tail, PREFIX)
    if code not in (0, 2) or "FAIL" in log:
        raise RuntimeError(f"Diagnostic failed with {code}: {PREFIX.with_suffix('.log')}")
    groups = {}
    for label, index in (("PAIR", "pattern"), ("COMPONENT", "pair"), ("TREE", "kind")):
        entries = re.findall(rf"^{label} {index}=(\d+) comparisons=(\d+) mismatches=(\d+)$", log, re.M)
        if len(entries) != 2 or {int(p) for p, _, _ in entries} != {0, 1}:
            raise RuntimeError(f"Incomplete {label} evidence")
        groups[label] = {int(p): dict(comparisons=int(n), mismatches=int(m)) for p, n, m in entries}
        if any(v["comparisons"] != 8192 or not 0 <= v["mismatches"] <= 8192 for v in groups[label].values()):
            raise RuntimeError(f"Invalid {label} counts")
    if code != (0 if groups["TREE"][0]["mismatches"] == 0 else 2):
        raise RuntimeError("Scalar tree comparator exit disagrees")
    invalid_prefix = ROOT / "fp16-dot4-pair-components-v1-invalid-gpu7"
    invalid_code, invalid_log = guarded(head + f"; export {FLAG}=2" + tail, invalid_prefix)
    if invalid_code != 1 or f"{FLAG} must be 0 or 1" not in invalid_log:
        raise RuntimeError("Invalid selector not explicitly rejected")
    evidence = [manifest_path, provider, runner, FIXTURES / "manifest.json", *paths,
                PREFIX.with_suffix(".log"), PREFIX.with_suffix(".guard.json"),
                invalid_prefix.with_suffix(".log"), invalid_prefix.with_suffix(".guard.json"),
                Path(__file__), ROOT / "build_fp16_dot4_pair_components.py",
                ROOT / "test_fp16_dot4_pair_components.py",
                ROOT / "build_fp16_dot4_pair_matrix_materialized.py",
                ROOT / "build_fp16_dot4_pair_matrix_probe.py",
                ROOT / "build_fp16_dot4_matrix_f32_probe.py",
                ROOT / "fp16_dot4_pair_matrix_compare.cpp",
                ROOT / "prepare_fp16_dot4_pair_matrix_fixtures.py",
                ROOT / "build_scoped_webgpu_source.py",
                ROOT / "run_fp16_dot4_tree_probe.py", ROOT / "shared_activation_guard.py"]
    report = dict(status="pair_components_characterized", groups=groups,
                  default_enabled=False, model_eligible=False, model_timing_performed=False,
                  invalid_selector_explicitly_rejected=True,
                  provider_sha256=manifest["provider_sha256"],
                  source_and_evidence_sha256={str(p): sha(p) for p in evidence})
    outcome.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], groups=groups)), flush=True)
