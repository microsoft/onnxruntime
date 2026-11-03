import json
from pathlib import Path
import re

from build_dense_paired_matrix_batch8 import STORAGE_BYTES
from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-dense-paired-matrix-batch8/build-v1")


if __name__ == "__main__":
    target = ROOT / "dense-paired-matrix-batch8-probe-v1-admission-outcome.json"
    if target.exists():
        raise FileExistsError(target)
    limits_path = ROOT / "native-matrix-limit-probe-v1-outcome.json"
    limits = json.loads(limits_path.read_text())
    if limits["status"] != "actual_device_limits_observed" or limits["workgroup_storage_bytes"] >= STORAGE_BYTES:
        raise RuntimeError("Storage-limit hold not established")
    for name, digest in limits["source_and_evidence_sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Changed actual-device proof: {name}")
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    if sha(provider) != manifest["provider_sha256"]:
        raise RuntimeError("Changed batch8 provider")
    evidence = [Path(__file__), limits_path, manifest_path, provider]
    for mode in ("capture", "compare"):
        prefix = ROOT / f"dense-paired-matrix-batch8-probe-v1-k640-n2560-{mode}-gpu7"
        guard_path, log_path = prefix.with_suffix(".guard.json"), prefix.with_suffix(".log")
        guard = json.loads(guard_path.read_text())
        if guard["status"] != "completed" or guard["exit_code"] != 0 or guard["gpu_uuid"] != limits["gpu_uuid"]:
            raise RuntimeError("Probe child did not complete on the diagnosed GPU")
        log = log_path.read_text()
        if "DENSE comparisons=163840 mismatches=0 K=640 N=2560" not in log:
            raise RuntimeError("Missing original/fallback arithmetic coverage")
        name = re.search(r"^PROFILE /evidence/webgpu-perf-oct20/(.+)$", log, re.M)[1]
        profile_path = ROOT / name
        events = json.loads(profile_path.read_text())
        kernels = [event for event in events if event.get("cat") == "Kernel"
                   and "&MatMulNBits&" in event.get("name", "")]
        if len(kernels) != 64 or any("DensePairedMatrixBatch8" in event.get("name", "") for event in events):
            raise RuntimeError("Fallback-only dispatch proof is incomplete")
        evidence.extend((guard_path, log_path, profile_path))
    state_path = ROOT / "dense-paired-matrix-batch8-probe-v1-execution.json"
    state = json.loads(state_path.read_text())
    if state["status"] != "running" or len(state["records"]) != 2:
        raise RuntimeError("Unexpected interrupted probe journal")
    state["status"] = "held_hardware_storage_not_admitted"
    state["runner_exception"] = "Native matrix dispatch coverage=0, expected64"
    state["journal_finalized_after_explicit_device_limit_diagnosis"] = True
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    evidence.append(state_path)
    result = dict(status=state["status"], required_storage_bytes=STORAGE_BYTES,
                  observed_storage_bytes=limits["workgroup_storage_bytes"],
                  selected_native_dispatches=0, fallback_only_comparisons=163840,
                  native_arithmetic_qualified=False, model_eligible=False, default_enabled=False,
                  natural_model_timing_performed=False,
                  provider_sha256=manifest["provider_sha256"],
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
