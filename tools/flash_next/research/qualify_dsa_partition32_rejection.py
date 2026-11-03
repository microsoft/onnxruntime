import collections
import json

from qualify_quickgelu_mul_long_context import ROOT, sha


if __name__ == "__main__":
    prefix = ROOT / "dsa-partition32-v1-chunk512-gpu7-full-profile2048-20261030"
    output = prefix.with_suffix(".rejection.json")
    if output.exists():
        raise FileExistsError(output)
    raw = json.loads(prefix.with_suffix(".json").read_text())
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    native_path = ROOT / "dsa-partition32-v1-native-gpu7-20261030.qualification.json"
    native = json.loads(native_path.read_text())
    assert raw["status"] == guard["status"] == "completed" and guard["exit_code"] == 0
    assert raw["plugin_sha256"] == native["provider_sha256"]
    assert raw["webgpu_validation_mode_override"] == "full"
    assert raw["load_experiment_flags"]["ORT_WEBGPU_DSA_PARTITION32"] == "1"
    assert raw["load_experiment_flags"]["ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SPLIT_KV"] == "1"
    assert guard["gpu_index"] == 7 and raw["profiling_scope"] == "decode_only"
    assert raw["profile_decode_after_tokens"] == 35
    controls = json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())
    references = [sample for sample in controls["samples"] if sample["context"] == 2048]
    assert len(raw["samples"]) == 2
    mismatches = []
    for sample, reference in zip(raw["samples"], references):
        assert sample["prompt_sha256"] == reference["prompt_sha256"]
        assert sample["output_tokens"] == len(sample["output_token_ids"]) == 36
        assert all(usage["major_page_faults"] == 0 for usage in sample["process_resource_usage"].values())
        indices = [index for index, (candidate, control) in enumerate(
            zip(sample["output_token_ids"], reference["output_token_ids"][:36])) if candidate != control]
        mismatches.append(dict(warmup=sample["warmup"], mismatching_token_positions=len(indices),
                               first_mismatch_zero_based=indices[0] if indices else None))
    assert any(sample["mismatching_token_positions"] > 0 for sample in mismatches)
    profiles = list(ROOT.glob(prefix.name + "-profile-ctx2048-rep0-configured_*.json"))
    assert len(profiles) == 1
    counts = collections.Counter(event["name"].rsplit("&", 1)[-1]
        for event in json.loads(profiles[0].read_text()) if event.get("cat") == "Kernel")
    assert counts["DynamicSparseAttentionPartition32"] == 12 and counts["DynamicSparseAttention"] == 0
    sources = [native_path, prefix.with_suffix(".json"), prefix.with_suffix(".guard.json"), *profiles]
    report = dict(status="rejected_original_generated_ids_changed", provider_sha256=native["provider_sha256"],
                  candidate_program_dispatches=12, mismatches=mismatches, zero_major_faults=True,
                  native_cpu_reference_executions=native["cpu_reference_executions"],
                  partition_grouping_changes_fp32_softmax_association=True,
                  promoted=False, quality_qualified=False, headline_eligible=False,
                  source_sha256={path.name: sha(path) for path in sources})
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "mismatches")}))
