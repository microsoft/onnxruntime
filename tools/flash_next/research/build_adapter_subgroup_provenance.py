import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path("/workspace/qwen38-int4-adapter-tensor-reuse")
OUTPUT = ROOT / "probe-v8"
SOURCE = Path("/workspace/onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.cc")


def recover(source):
    return replace_once(source,
        "                             WGSL_TEMPLATE_PARAMETER(fp32_accumulation, config.resultComponentType == wgpu::SubgroupMatrixComponentType::F32),\n", "")


def main():
    failure = json.loads((ROOT / "build-v6/baseline/object-144.failure.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    source = OUTPUT / SOURCE.name
    source.write_text(SOURCE.read_text())
    obj = source.with_suffix(".cc.o")
    command = failure["compile_command"].copy()
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(obj)
    command.insert(1, "-I/workspace/onnxruntime-webgpu-h200-perf/build-int4-uniform-arena")
    command.insert(1, f"-fmacro-prefix-map={source}={SOURCE}")
    for flag in ("-MF", "-MT"):
        if flag in command:
            command[command.index(flag) + 1] = str(obj) + ".d" if flag == "-MF" else str(obj)
    subprocess.run(command,
        cwd="/workspace/onnxruntime-webgpu-fp8-checkpoint/build-webgpu-fp8/Release", check=True)
    report = dict(status="subgroup_original_source_probe_completed", source_sha256=sha(source),
        object_sha256=sha(obj), expected_sha256=failure["expected_sha256"],
        byte_identical=sha(obj) == failure["expected_sha256"], compile_command=command,
        candidate_provider_built=False)
    (OUTPUT / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "source_sha256", "byte_identical")}))
    assert report["byte_identical"], "Original subgroup source remains unresolved"


if __name__ == "__main__":
    main()
