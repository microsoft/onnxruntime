import json
from pathlib import Path

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent


if __name__ == "__main__":
    path = ROOT / "dense-columns8-swizzle-actual-batch-v1-outcome.json"
    proof = json.loads(path.read_text())
    if proof["status"] != "columns8_swizzle_batched_actual_and_default_supported":
        raise RuntimeError("Actual-weight arithmetic did not pass")
    for name, digest in proof["source_and_evidence_sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Changed swizzle proof: {name}")
    rows = []
    for baseline, candidate in zip(proof["phases"][1]["shapes"], proof["phases"][2]["shapes"], strict=True):
        if (baseline["k"], baseline["n"]) != (candidate["k"], candidate["n"]) or candidate["mismatches"]:
            raise RuntimeError("Unmatched shape or numerical mismatch")
        if candidate["selected_dispatches"]:
            rows.append(dict(k=baseline["k"], n=baseline["n"],
                             supported_parent_median_us=baseline["instrumented_gpu_median_us"],
                             candidate_median_us=candidate["instrumented_gpu_median_us"]))
    if len(rows) != 7 or not all(row["candidate_median_us"] >= row["supported_parent_median_us"] for row in rows):
        raise RuntimeError("No-isolated-gain verdict is not established")
    target = ROOT / "dense-columns8-swizzle-v1-kernel-verdict.json"
    if target.exists():
        raise FileExistsError(target)
    report = dict(status="held_no_isolated_actual_weight_gain", rows=rows,
                  instrumented_timings_only=True, model_eligible=False, default_enabled=False,
                  natural_model_timing_performed=False, unchanged_retry_allowed=False,
                  source_and_evidence_sha256={str(path): sha(path), str(Path(__file__)): sha(Path(__file__))})
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
