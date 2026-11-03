import json
from pathlib import Path
import shlex
import subprocess

from build_dense_columns8_double_buffer import FLAG
from build_qmoe_node_replay import sha
from run_dense_columns8_double_buffer_batch import run_guard, workspace, UUID


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-dense-columns8-double-buffer/build-v1")


if __name__ == "__main__":
    prerequisite = ROOT / "dense-columns8-double-buffer-actual-batch-v1-outcome.json"
    proof = json.loads(prerequisite.read_text())
    if proof["status"] != "columns8_double_buffer_batched_actual_and_default_supported":
        raise RuntimeError("Actual-weight gate not supported")
    for path, digest in proof["source_and_evidence_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Changed actual qualification: {path}")
    prefix = ROOT / "dense-columns8-double-buffer-v1-invalid-gpu7"
    outcome = ROOT / "dense-columns8-double-buffer-v1-invalid-outcome.json"
    invalid_log = ROOT / "dense-columns8-double-buffer-v1-invalid-child.log"
    if outcome.exists() or invalid_log.exists():
        raise FileExistsError("Existing invalid-selector evidence")
    fixture = Path("/home/sayanshaw/repos/qwen38-int4-dense-paired-matrix-reset/actual-fixtures-v1/k640-n2560.onnx")
    reference = fixture.with_name("k640-n2560-reference.f16")
    target = "/evidence/webgpu-perf-oct20/" + invalid_log.name
    args = [workspace(BUILD / "paired_matrix_compare"), "compare",
            workspace(BUILD / "libonnxruntime_providers_webgpu.so"),
            workspace(fixture), workspace(reference),
            "/evidence/webgpu-perf-oct20/dense-columns8-double-buffer-invalid-profile"]
    command = (f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
               f"if env {FLAG}=2 " + shlex.join(args) + f" > {target} 2>&1; then "
               "echo 'FAIL: invalid selector accepted'; exit 1; fi; "
               f"grep -F '{FLAG} must be 0 or 1' {target}")
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                    "--format=csv,noheader"], check=True)
    code, log = run_guard(command, prefix)
    if code != 0 or FLAG + " must be 0 or 1" not in log:
        raise RuntimeError("Fresh invalid selector rejection not proved")
    evidence = [Path(__file__), prerequisite, prefix.with_suffix(".log"),
                prefix.with_suffix(".guard.json"), invalid_log, fixture, reference,
                BUILD / "paired_matrix_compare", BUILD / "libonnxruntime_providers_webgpu.so"]
    report = dict(status="fresh_invalid_columns8_double_buffer_selector_rejected",
                  fresh_process=True, model_timing_performed=False,
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    outcome.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"])), flush=True)
