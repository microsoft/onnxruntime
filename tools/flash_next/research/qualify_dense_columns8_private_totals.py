import json
from pathlib import Path

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent


if __name__ == "__main__":
    proof_path = ROOT / "dense-columns8-private-totals-actual-batch-v1-outcome.json"
    proof = json.loads(proof_path.read_text())
    if proof["status"] != "columns8_private_totals_batched_actual_and_default_supported":
        raise RuntimeError("Private-total actual arithmetic not supported")
    for path, digest in proof["source_and_evidence_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Changed private-total proof: {path}")
    rows = []
    for before, selected in zip(proof["phases"][1]["shapes"], proof["phases"][2]["shapes"], strict=True):
        if (before["k"], before["n"]) != (selected["k"], selected["n"]) or selected["mismatches"]:
            raise RuntimeError("Unmatched shape or numerical mismatch")
        if selected["selected_dispatches"]:
            rows.append(dict(k=before["k"], n=before["n"],
                             supported_parent_median_us=before["instrumented_gpu_median_us"],
                             selected_median_us=selected["instrumented_gpu_median_us"]))
    if len(rows) != 7 or not all(row["selected_median_us"] >= row["supported_parent_median_us"] for row in rows):
        raise RuntimeError("No-isolated-gain hold not established")
    target = ROOT / "dense-columns8-private-totals-v1-kernel-verdict.json"
    if target.exists():
        raise FileExistsError(target)
    result = dict(status="held_no_isolated_actual_weight_gain", rows=rows,
                  instrumented_timings_only=True, default_enabled=False, model_eligible=False,
                  natural_model_timing_performed=False, unchanged_retry_allowed=False,
                  source_and_evidence_sha256={str(proof_path): sha(proof_path), str(Path(__file__)): sha(Path(__file__))})
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
