import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


def main():
    parent = Path("/workspace/qwen38-int4-qmoe-fc2-final-mix/build-v2")
    output = Path("/workspace/qwen38-int4-qmoe-subgroup-gate-compare/build-v1")
    output.mkdir(parents=True, exist_ok=False)
    original = parent / "native_compare.cpp"
    source = output / original.name
    source.write_text(replace_once(original.read_text(),
        'set_flag("ORT_WEBGPU_QMOE_FC2_FINAL_MIX", "0");',
        'set_flag("ORT_WEBGPU_QMOE_FC2_FINAL_MIX", "1");'))
    comparator = output / "native_compare"
    subprocess.run(["c++", "-O3", "-std=c++20", str(source),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(comparator)], check=True)
    report = dict(status="built_not_qualified", comparator_sha256=sha(comparator),
        source_sha256=sha(source), original_source_sha256=sha(original),
        fc2_fusion_fixed_enabled=True, requires_ort_test_compare_gate=True)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
