import json
from pathlib import Path
import re
import statistics

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent


if __name__ == "__main__":
    proof_path = ROOT / "dense-paired-matrix-reset-v1-outcome.json"
    proof = json.loads(proof_path.read_text())
    if proof["status"] != "packed_dense_original_provider_synthetic_shapes_supported":
        raise RuntimeError("Synthetic arithmetic not supported")
    for path, digest in proof["source_and_evidence_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Changed qualification: {path}")
    rows = []
    for shape in proof["shapes"]:
        k, n = shape["k"], shape["n"]
        durations = []
        for mode in ("capture", "compare"):
            log = ROOT / f"dense-paired-matrix-reset-v1-k{k}-n{n}-{mode}-gpu7.log"
            profile_name = re.search(
                r"^PROFILE /evidence/webgpu-perf-oct20/(.+)$", log.read_text(), re.M)[1]
            events = json.loads((ROOT / profile_name).read_text())
            selected = [event for event in events if event.get("cat") == "Kernel"
                        and "&MatMulNBits&" in event.get("name", "")]
            if len(selected) != 64 or any(event.get("dur", 0) <= 0 for event in selected):
                raise RuntimeError("Invalid isolated kernel profiling")
            durations.append(statistics.median(event["dur"] for event in selected))
        rows.append(dict(k=k, n=n, original_median_us=durations[0],
                         candidate_median_us=durations[1],
                         candidate_over_original=durations[1] / durations[0],
                         fallback=n == 1))
    report = dict(status="held_isolated_kernel_profile_regressions", rows=rows,
                  natural_model_tps_measured=False, model_timing_performed=False,
                  arithmetic_synthetic_comparisons=sum(s["phases"][1]["comparisons"] for s in proof["shapes"]),
                  model_eligible=False, default_enabled=False,
                  source_and_evidence_sha256={str(proof_path): sha(proof_path), str(Path(__file__)): sha(Path(__file__))})
    path = ROOT / "dense-paired-matrix-reset-v1-kernel-profile-outcome.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
