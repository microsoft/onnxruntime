import json

from qualify_qsa_score16_standard import all_contexts_reach_target
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_qsa_prefill_score16_standard import VARIANT, prerequisites, verify_case


def main():
    output = ROOT / f"{VARIANT}-contexts-20261031.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    native, parent, gate, sources = prerequisites()
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    controls = json.loads(controls_path.read_text())
    sources[controls_path.name] = sha(controls_path)
    previous = {r["context"]: r for r in parent["rows"]}
    rows = []
    for context in (2048, 8192, 32768):
        results = {}
        references = [s for s in controls["samples"] if s["context"] == context]
        for full in (True, False):
            prefix = ROOT / f"{VARIANT}-chunk1024-gpu7-{'full-' if full else ''}standard{context}-20261031"
            proof_path = prefix.with_suffix(".qualification.json")
            proof, raw, guard = (json.loads(p.read_text()) for p in
                                 (proof_path, prefix.with_suffix(".json"), prefix.with_suffix(".guard.json")))
            result = verify_case(raw, guard, native, parent, references, context, full)
            assert proof["status"] == "original_sequences_qualified" and proof["phase"] == "standard"
            assert proof["context"] == context and proof["full_validation"] == full and proof["chunk"] == 1024
            assert proof["zero_major_faults"] and proof["original_output_prefixes_match"]
            assert proof["provider_sha256"] == native["provider_sha256"] and proof["result"] == result
            assert proof["screen_gate"] == gate
            assert len(raw["samples"]) == 4
            assert [s["warmup"] for s in raw["samples"]] == [True, False, False, False]
            results["full_validation" if full else "candidate"] = result
            sources[proof_path.name] = sha(proof_path)
            for name, digest in proof["source_sha256"].items():
                assert sha(ROOT / name) == digest
                sources[name] = digest
        rows.append(dict(context=context, previous=previous[context]["candidate"], **results,
                         standard_comparison_kind="separate_run_observation_not_isolated_causal_pair"))
    report = dict(status="prefill_score16_native_and_standard_original_sequences_qualified",
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
        gpu_index=7, contexts=[2048, 8192, 32768], chunk=1024, pending_dispatches=64,
        output_tokens=256, measured_repetitions=3, excluded_warmup_requests_per_case=1,
        original_sequence_comparisons=6144, all_original_sequences_match=True, zero_major_faults=True,
        all_contexts_at_least_100_decode_tps=all_contexts_reach_target(rows),
        full_validation_all_contexts_at_least_100_decode_tps=all(
            r["full_validation"]["median_decode_tps"] >= 100 for r in rows),
        screen_gate=gate, rows=rows, default_off=True, provider_promoted=False,
        quality_qualified=False, headline_eligible=False, source_sha256=sources)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "all_contexts_at_least_100_decode_tps", "rows")}), flush=True)


if __name__ == "__main__":
    main()
