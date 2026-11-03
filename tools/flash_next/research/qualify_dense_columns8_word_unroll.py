import json
from pathlib import Path

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent


if __name__ == "__main__":
    target = ROOT / "dense-columns8-word-unroll-v1-kernel-verdict.json"
    if target.exists():
        raise FileExistsError(target)
    evidence = [Path(__file__)]
    proofs = []
    for label in ("batch", "actual-batch"):
        path = ROOT / f"dense-columns8-word-unroll-{label}-v1-outcome.json"
        proof = json.loads(path.read_text())
        expected = ("columns8_word_unroll_batched_actual_and_default_supported" if label == "actual-batch"
                    else "columns8_word_unroll_batched_synthetic_and_default_supported")
        if proof["status"] != expected:
            raise RuntimeError("Arithmetic batch not supported")
        for name, digest in proof["source_and_evidence_sha256"].items():
            if sha(Path(name)) != digest:
                raise RuntimeError(f"Changed batch proof: {name}")
        proofs.append(proof)
        evidence.append(path)
    totals = []
    for proof, expected_count in zip(proofs, (3067968, 3067904), strict=True):
        observed = [sum(shape["comparisons"] for shape in phase["shapes"])
                    for phase in proof["phases"]]
        if observed != [expected_count] * 3:
            raise RuntimeError(f"Incorrect original/default/candidate coverage: {observed}")
        if any(shape["mismatches"] for phase in proof["phases"] for shape in phase["shapes"]):
            raise RuntimeError("Numerical mismatch in sealed phases")
        totals.append(observed[0])
    rows = []
    for baseline, selected in zip(proofs[1]["phases"][1]["shapes"], proofs[1]["phases"][2]["shapes"], strict=True):
        if (baseline["k"], baseline["n"]) != (selected["k"], selected["n"]) or selected["mismatches"]:
            raise RuntimeError("Unmatched shape or numerical failure")
        if selected["selected_dispatches"]:
            rows.append(dict(k=baseline["k"], n=baseline["n"],
                             supported_parent_median_us=baseline["instrumented_gpu_median_us"],
                             selected_median_us=selected["instrumented_gpu_median_us"]))
    if len(rows) != 7 or not any(row["selected_median_us"] >= row["supported_parent_median_us"] for row in rows):
        raise RuntimeError("Lack-of-consistent-gain hold not established")
    result = dict(status="held_no_consistent_isolated_actual_weight_gain", rows=rows,
                  synthetic_comparisons_per_phase=totals[0], actual_comparisons_per_phase=totals[1],
                  mismatches=0, default_enabled=False, model_eligible=False,
                  natural_model_timing_performed=False, word_order_preserved=True,
                  instrumented_timings_only=True, unchanged_retry_allowed=False,
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
