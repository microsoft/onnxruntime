import json
from pathlib import Path
import shlex
import subprocess

from build_hc_narrow_rms_broadcast import FLAG
from build_qmoe_node_replay import sha
from run_hc_narrow_rms_broadcast_native import BUILD, DATA, FLAGS, PROOF, ROOT, run_guard, UUID, workspace


if __name__ == "__main__":
    prerequisite = ROOT / "hc-narrow-rms-broadcast-v3-native-outcome.json"
    proof = json.loads(prerequisite.read_text())
    if proof["status"] != "hc_separate_original_outputs_supported":
        raise RuntimeError("Independent native arithmetic not supported")
    for name, digest in proof["source_and_evidence_sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Changed native qualification: {name}")
    outcome = ROOT / "hc-narrow-rms-broadcast-v3-invalid-outcome.json"
    prefix = ROOT / "hc-narrow-rms-broadcast-v3-invalid-gpu7"
    child_log = ROOT / "hc-narrow-rms-broadcast-v3-invalid-child.log"
    if outcome.exists() or child_log.exists() or prefix.with_suffix(".guard.json").exists():
        raise FileExistsError("Existing fresh selector evidence")
    fixtures = sorted(Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/fixtures-v1").glob("*.onnx"))
    if len(fixtures) != 36:
        raise RuntimeError("Missing inherited fixture")
    command = (f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
               f"if env {FLAGS} {FLAG}=2 " +
               shlex.join([workspace(PROOF / "independent_compare"),
                           workspace(BUILD / "libonnxruntime_providers_webgpu.so"),
                           "compare", workspace(DATA), workspace(fixtures[0])]) +
               f" > /evidence/webgpu-perf-oct20/{child_log.name} 2>&1; then "
               "echo 'FAIL: invalid selector accepted'; exit 1; fi; "
               f"grep -F '{FLAG} must be 0 or 1' /evidence/webgpu-perf-oct20/{child_log.name}")
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                    "--format=csv,noheader"], check=True)
    code, log = run_guard(command, prefix)
    if code != 0 or f"{FLAG} must be 0 or 1" not in log:
        raise RuntimeError("Fresh invalid selector rejection not proved")
    evidence = [Path(__file__), prerequisite, child_log, prefix.with_suffix(".log"),
                prefix.with_suffix(".guard.json")]
    report = dict(status="fresh_invalid_hc_narrow_rms_broadcast_selector_rejected",
                  fresh_process=True, natural_model_timing_performed=False,
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    outcome.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], flush=True)
