import json
from pathlib import Path
import shlex
import subprocess

from build_hc_owned_vector import FLAG
from build_qmoe_node_replay import sha
from run_hc_owned_vector_native import BUILD, DATA, FLAGS, PROOF, ROOT, UUID, run_guard, workspace


if __name__ == "__main__":
    prerequisite = ROOT / "hc-owned-vector-v1-native-outcome.json"
    proof = json.loads(prerequisite.read_text())
    assert proof["status"] == "hc_separate_original_outputs_supported"
    for name, digest in proof["source_and_evidence_sha256"].items():
        assert sha(Path(name)) == digest, name
    outcome = ROOT / "hc-owned-vector-v1-invalid-outcome.json"
    prefix = ROOT / "hc-owned-vector-v1-invalid-gpu7"
    child_log = ROOT / "hc-owned-vector-v1-invalid-child.log"
    if outcome.exists() or child_log.exists():
        raise FileExistsError("Existing invalid-selector evidence")
    fixture = sorted(Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/fixtures-v1").glob("*.onnx"))[0]
    command = (f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
        f"if env {FLAGS} {FLAG}=2 " +
        shlex.join([workspace(PROOF / "independent_compare"),
                    workspace(BUILD / "libonnxruntime_providers_webgpu.so"), "compare",
                    workspace(DATA), workspace(fixture)]) +
        f" > /evidence/webgpu-perf-oct20/{child_log.name} 2>&1; then "
        "echo 'FAIL: invalid selector accepted'; exit 1; fi; "
        f"grep -F '{FLAG} must be 0 or 1' /evidence/webgpu-perf-oct20/{child_log.name}")
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                    "--format=csv,noheader"], check=True)
    code, log = run_guard(command, prefix)
    assert code == 0 and f"{FLAG} must be 0 or 1" in log
    evidence = [Path(__file__), prerequisite, child_log, prefix.with_suffix(".log"),
                prefix.with_suffix(".guard.json")]
    report = dict(status="fresh_invalid_hc_owned_vector_selector_rejected", fresh_process=True,
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    outcome.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], flush=True)
