import json
from pathlib import Path
import re
import statistics

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent


if __name__ == "__main__":
    proof_path = ROOT / "dense-paired-matrix-batch4-probe-v1-outcome.json"
    proof = json.loads(proof_path.read_text())
    if proof["status"] != "packed_dense_original_provider_synthetic_shapes_supported":
        raise RuntimeError("Selected-kernel arithmetic probe did not pass")
    for path, digest in proof["source_and_evidence_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Changed probe evidence: {path}")
    if len(proof["shapes"]) != 1 or proof["shapes"][0]["phases"][1]["native_dispatches"] != 64:
        raise RuntimeError("Unexpected probe coverage")
    durations, evidence = [], [proof_path, Path(__file__)]
    for mode in ("capture", "compare"):
        log = ROOT / f"dense-paired-matrix-batch4-probe-v1-k640-n2560-{mode}-gpu7.log"
        profile_name = re.search(r"^PROFILE /evidence/webgpu-perf-oct20/(.+)$", log.read_text(), re.M)[1]
        profile = ROOT / profile_name
        events = json.loads(profile.read_text())
        kernels = [event for event in events if event.get("cat") == "Kernel"
                   and "&MatMulNBits&" in event.get("name", "")]
        if len(kernels) != 64 or any(event.get("dur", 0) <= 0 for event in kernels):
            raise RuntimeError("Incomplete isolated kernel timings")
        durations.append(statistics.median(event["dur"] for event in kernels))
        evidence.extend((log, profile))
    if durations[1] <= durations[0]:
        raise RuntimeError("Regression hold is not supported by observed timings")
    target = ROOT / "dense-paired-matrix-batch4-probe-v1-kernel-verdict.json"
    if target.exists():
        raise FileExistsError(target)
    result = dict(status="held_isolated_kernel_regression", k=640, n=2560,
                  original_median_us=durations[0], candidate_median_us=durations[1],
                  candidate_over_original=durations[1] / durations[0],
                  selected_dispatches=64, finite_bitwise_comparisons=163840, mismatches=0,
                  instrumented_single_shape_probe_only=True, model_eligible=False,
                  default_enabled=False, natural_tps_measured=False,
                  broader_suite_run=False, unchanged_retry_allowed=False,
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
