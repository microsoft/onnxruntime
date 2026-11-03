import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha


ROOT = Path("/workspace/qwen38-int4-adapter-tensor-reuse")
OUTPUT = ROOT / "probe-v4"
MAP = ("-fmacro-prefix-map=/workspace/onnxruntime-webgpu-fp8-checkpoint/include/onnxruntime/"
       "=/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/")


def main():
    failure_path = ROOT / "build-v1/baseline/object-2.failure.json"
    failure = json.loads(failure_path.read_text())
    command = failure["compile_command"].copy()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    obj = OUTPUT / "compute_context.cc.o"
    command[command.index("-o") + 1] = str(obj)
    for flag in ("-MF", "-MT"):
        if flag in command:
            command[command.index(flag) + 1] = str(obj.with_suffix(".d") if flag == "-MF" else obj)
    command.insert(1, MAP)
    subprocess.run(command,
        cwd="/workspace/onnxruntime-webgpu-fp8-checkpoint/build-webgpu-fp8/Release", check=True)
    actual = sha(obj)
    original = Path(failure["original_object"])
    report = dict(status="original_context_macro_path_probe_completed",
        original_object_sha256=sha(original), rebuilt_object_sha256=actual,
        byte_identical=actual == sha(original), compile_command=command,
        failure_evidence_sha256=sha(failure_path),
        source_sha256=sha(Path(failure["source"])), canonical_sdk_definitions_unchanged=True,
        macro_diagnostic_path_map_only=MAP, candidate_provider_built=False)
    (OUTPUT / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    assert report["byte_identical"], "Diagnostic path mapping did not reproduce the original object"


if __name__ == "__main__":
    main()
