import json
from pathlib import Path
import shlex
import subprocess

from build_int8_lm_head_full_a import FLAG
from build_qmoe_node_replay import sha
from prepare_int8_lm_head_full_a_native_v3 import DATA, PROOF
from run_int8_lm_head_full_a_native import BUILD, FLAGS, ROOT, run_guard, UUID, workspace


if __name__ == "__main__":
    prerequisite = ROOT / "int8-lm-head-full-a-v4-native-outcome.json"
    proof = json.loads(prerequisite.read_text())
    assert proof["status"] == "int8_head_separate_original_outputs_supported"
    for name, digest in proof["source_and_evidence_sha256"].items():
        assert sha(Path(name)) == digest, name
    prefix = ROOT / "int8-lm-head-full-a-v1-invalid-gpu7"
    outcome = ROOT / "int8-lm-head-full-a-v1-invalid-outcome.json"
    child_log = ROOT / "int8-lm-head-full-a-v1-invalid-child.log"
    if outcome.exists() or child_log.exists():
        raise FileExistsError("Existing invalid-selector evidence")
    manifest = json.loads((DATA / "manifest.json").read_text())
    fixture = Path(manifest["fixtures"][0]["path"])
    reference = PROOF / "original-references-v4" / (fixture.stem + ".reference.bin")
    command = (f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
        f"if env {FLAGS} {FLAG}=2 " +
        shlex.join([workspace(PROOF / "independent_compare"), "compare",
                    workspace(BUILD / "libonnxruntime_providers_webgpu.so"),
                    workspace(fixture), workspace(reference),
                    "/evidence/webgpu-perf-oct20/int8-head-invalid-profile"]) +
        f" > /evidence/webgpu-perf-oct20/{child_log.name} 2>&1; then "
        "echo 'FAIL: invalid selector accepted'; exit 1; fi; "
        f"grep -F '{FLAG} must be 0 or 1' /evidence/webgpu-perf-oct20/{child_log.name}")
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                    "--format=csv,noheader"], check=True)
    code, log = run_guard(command, prefix)
    assert code == 0 and f"{FLAG} must be 0 or 1" in log
    evidence = [Path(__file__), prerequisite, child_log, prefix.with_suffix(".log"),
                prefix.with_suffix(".guard.json")]
    report = dict(status="fresh_invalid_int8_lm_head_full_a_selector_rejected", fresh_process=True,
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    outcome.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], flush=True)
