import json
import math
from pathlib import Path
import statistics

from benchmark_webgpu_int4 import token_hash


ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def cache_stats(prefix):
    records = [json.loads(line) for line in
               (ROOT / f"{prefix}.stats.jsonl").read_text().splitlines()]
    for record in records:
        assert 0 <= record["peak_referenced_bytes"] <= 256 * 1024 * 1024
        assert 0 <= record["peak_entries"] <= 8192
    return records


def successful_guard(prefix):
    guard = read(f"{prefix}.guard.json")
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    return guard


def native(prefix, fixtures):
    guard = successful_guard(prefix)
    lines = [line for line in (ROOT / f"{prefix}.log").read_text().splitlines()
             if line.startswith("PASS case")]
    assert len(lines) == fixtures
    assert all("comparisons=8" in line and "exact_off_on=true" in line for line in lines)
    return {
        "guard": f"{prefix}.guard.json",
        "elapsed_seconds": guard["elapsed_seconds"],
        "successful_fixture_runs": fixtures,
        "gpu_reference_and_exact_parity_executions": fixtures * 8,
        "cache_statistics": cache_stats(prefix),
    }


def model(prefix, quality_name, expected_hash, comparison="bind_group_cache_only"):
    guard = successful_guard(prefix)
    result = read(f"{prefix}.json")
    assert result["status"] == "completed"
    assert result["plugin_sha256"] == expected_hash
    assert result["comparison"] == comparison
    assert result["webgpu_max_pending_dispatches_override"] == 64
    assert result["graph_sha256"] == "fd3a152c8f4dbf93aae8d12060d16b0714dc5977f1d8c7e3eea19098f9268426"
    assert result["config_sha256"] == "8b878a2c79a0763448f02a5105aa98a3e32f43ded848f93a1df9a071607e16b6"
    samples = result["samples"]
    assert len(samples) == 8 and sum(sample["warmup"] for sample in samples) == 2
    pairs = {}
    for sample in samples:
        assert sample["context"] == sample["input_tokens"] == 2048
        assert sample["output_tokens"] == len(sample["output_token_ids"]) == 256
        assert sample["output_sha256"] == token_hash(sample["output_token_ids"])
        assert math.isclose(sample["decode_tps"], 255 / sample["decode_seconds"], rel_tol=1e-12)
        assert sample["bind_group_cache_entries"] == {"control": "0", "candidate": "8192"}[sample["mode"]]
        assert sample["paired_output_matches"]
        if comparison == "dynamic_uniform_cache_bundle":
            assert sample["dynamic_uniforms_flag"] == {"control": "0", "candidate": "1"}[sample["mode"]]
        pairs.setdefault(sample["repetition"], {})[sample["mode"]] = sample["output_token_ids"]
    assert len(pairs) == 4
    assert all(pair["control"] == pair["candidate"] for pair in pairs.values())
    quality = read(quality_name)
    baseline = read("int4-bindgroup-cache-quality.json")
    assert len(quality["samples"]) == len(baseline["samples"]) == 9
    assert all(left["input_tokens"] == right["input_tokens"] and
               left["output_tokens"] == right["output_tokens"]
               for left, right in zip(quality["samples"], baseline["samples"]))
    summary = {
        "status": "completed_not_promoted",
        "provider_sha256": expected_hash,
        "comparison": result["comparison"],
        "context": 2048,
        "output_tokens": 256,
        "decode_intervals": 255,
        "max_pending_dispatches": 64,
        "measured_request_pairs": 3,
        "excluded_warmup_pairs": 1,
        "one_model_load": True,
        "all_four_paired_output_sequences_exactly_equal": True,
        "all_nine_quality_sequences_exactly_match_prior_fast_stack": True,
        "strict_exact_answer_checks_passed": sum(sample["exact_answer_passed"]
                                                for sample in quality["samples"]),
        "strict_exact_answer_checks_total": 9,
        "quality_qualified": False,
        "guard_elapsed_seconds": guard["elapsed_seconds"],
        "shared_host_not_isolated": True,
        "comparison_scope": (
            "Static uniforms/cache0 vs dynamic uniforms/cache8192 within the same provider build."
            if comparison == "dynamic_uniform_cache_bundle" else
            "Cache0/8192; dynamic uniforms held at the load configuration in both modes."),
        "sanity_dynamic_uniforms_flag": result["load_experiment_flags"].get("ORT_WEBGPU_DYNAMIC_UNIFORMS", "0"),
        "sanity_cache_entries": result["load_experiment_flags"].get("ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES", "0"),
    }
    for mode in ("control", "candidate"):
        measured = [sample for sample in samples if sample["mode"] == mode and not sample["warmup"]]
        assert len(measured) == 3
        summary[f"{mode}_decode_tps"] = [sample["decode_tps"] for sample in measured]
        summary[f"{mode}_median_decode_tps"] = statistics.median(summary[f"{mode}_decode_tps"])
        summary[f"{mode}_aggregate_decode_tps"] = 765 / sum(sample["decode_seconds"] for sample in measured)
        summary[f"{mode}_median_ttft_seconds"] = statistics.median(sample["ttft_seconds"] for sample in measured)
    stats_file = ROOT / f"{prefix}.stats.jsonl"
    if stats_file.exists():
        records = cache_stats(prefix)
        assert len(records) == 1
        stats = records[0]
        summary["cache_statistics"] = stats
        summary["cache_hit_fraction"] = stats["hits"] / (stats["hits"] + stats["misses"])
        summary["eligible_cache_hit_fraction"] = stats["hits"] / (
            stats["hits"] + stats["misses"] - stats["skipped_large"])
        summary["cache_statistics_scope"] = (
            "Whole process: cache-enabled sanity, prefill, warmup and measured decode; not decode-only.")
    (ROOT / f"{prefix}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def fused_followup():
    prefix = "int4-mlp-fast-stack"
    successful_guard(prefix)
    result = read(f"{prefix}.json")
    assert result["status"] == "completed" and result["comparison"] is None
    assert result["graph_sha256"] == "bd6d338361552f31469fd7d845b3c99b5fe1b582ae54251b288e083ade19579c"
    assert result["plugin_sha256"] == "a9b04ee22290dad61e98d7d85b46aa85de1351c1146be3ac29ec9fb4a97766c5"
    assert result["config_sha256"] == "8b878a2c79a0763448f02a5105aa98a3e32f43ded848f93a1df9a071607e16b6"
    samples = result["samples"]
    assert len(samples) == 4 and sum(sample["warmup"] for sample in samples) == 1
    for sample in samples:
        assert sample["input_tokens"] == sample["context"] == 2048
        assert sample["output_tokens"] == len(sample["output_token_ids"]) == 256
        assert sample["output_sha256"] == token_hash(sample["output_token_ids"])
        assert math.isclose(sample["decode_tps"], 255 / sample["decode_seconds"], rel_tol=1e-12)
    fixtures = [line for line in (ROOT / f"{prefix}.log").read_text().splitlines()
                if line.startswith("PASS mlp_converted_")]
    assert len(fixtures) == 3 and all("validationMode=full" in line for line in fixtures)
    measured = [sample for sample in samples if not sample["warmup"]]
    quality = read(f"{prefix}-quality.json")
    assert quality["status"] == "completed" and len(quality["samples"]) == 9
    summary = {
        "status": "standalone_derived_graph_not_promoted",
        "graph_sha256": result["graph_sha256"],
        "provider_sha256": result["plugin_sha256"],
        "median_decode_tps": statistics.median(sample["decode_tps"] for sample in measured),
        "aggregate_decode_tps": 765 / sum(sample["decode_seconds"] for sample in measured),
        "measured_ttft_seconds": [sample["ttft_seconds"] for sample in measured],
        "native_actual_converter_fixtures_full_validation_passed": 3,
        "strict_answer_checks_passed": sum(sample["exact_answer_passed"] for sample in quality["samples"]),
        "strict_answer_checks_total": 9,
        "quality_qualified": False,
        "causal_speedup_qualified": False,
        "caveats": ["Shared host; anomalous third TTFT.",
                    "Not a counterbalanced original/derived graph comparison.",
                    "Does not replace the complete original-graph context table."],
    }
    (ROOT / f"{prefix}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def hyper_skinny():
    prefix = "int4-hyper-skinny-gpu5"
    successful_guard(prefix)
    result = read(f"{prefix}.json")
    assert result["status"] == "completed" and result["comparison"] == "hyper_skinny_only"
    assert result["plugin_sha256"] == "6be27643e88b58ea22872079643f7cd43ebd3e1f48e82ea4cc0d338910df313b"
    assert result["graph_sha256"] == "fd3a152c8f4dbf93aae8d12060d16b0714dc5977f1d8c7e3eea19098f9268426"
    assert result["config_sha256"] == "8b878a2c79a0763448f02a5105aa98a3e32f43ded848f93a1df9a071607e16b6"
    assert result["webgpu_max_pending_dispatches_override"] == 64
    samples = result["samples"]
    assert len(samples) == 8 and sum(sample["warmup"] for sample in samples) == 2
    pairs = {}
    for sample in samples:
        assert sample["input_tokens"] == sample["context"] == 2048
        assert sample["output_tokens"] == len(sample["output_token_ids"]) == 256
        assert sample["output_sha256"] == token_hash(sample["output_token_ids"])
        assert math.isclose(sample["decode_tps"], 255 / sample["decode_seconds"], rel_tol=1e-12)
        assert sample["hyper_skinny_gemv_flag"] == {"control": "0", "candidate": "1"}[sample["mode"]]
        assert sample["skinny_gemv_flag"] == sample["dynamic_uniforms_flag"] == sample["bind_group_cache_entries"] == "0"
        pairs.setdefault(sample["repetition"], {})[sample["mode"]] = sample
    assert len(pairs) == 4 and all(set(pair) == {"control", "candidate"} for pair in pairs.values())
    divergences = {}
    for repetition, pair in pairs.items():
        left, right = pair["control"]["output_token_ids"], pair["candidate"]["output_token_ids"]
        difference = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
        assert all(sample["paired_output_matches"] == (difference is None) for sample in pair.values())
        divergences[repetition] = difference
    native_lines = [line for line in (ROOT / f"{prefix}.log").read_text().splitlines()
                    if line.startswith("PASS dense_")]
    assert len(native_lines) == 35 and all("validationMode=full" in line for line in native_lines)
    quality = read(f"{prefix}-quality.json")
    baseline_quality = read("int4-bindgroup-cache-quality.json")
    assert quality["status"] == "completed" and len(quality["samples"]) == 9
    summary = {
        "status": "completed_not_promoted",
        "provider_sha256": result["plugin_sha256"],
        "comparison": result["comparison"],
        "context": 2048,
        "output_tokens": 256,
        "decode_intervals": 255,
        "measured_request_pairs": 3,
        "excluded_warmup_pairs": 1,
        "one_model_load": True,
        "max_pending_dispatches": 64,
        "candidate_native_fixture_runs_full_validation": 35,
        "baseline_native_limitation": "Prior flag0 actual-hyper M2 FP16 reference failure retained in int4-hyper-skinny.log; not a full baseline native pass.",
        "interrupted_gpu0_attempt": "Ownership guard yielded to foreign process after excluded warmups; no headline data taken.",
        "first_differing_output_index_by_repetition": divergences,
        "quality_qualified": False,
        "strict_answer_checks_passed": sum(sample["exact_answer_passed"] for sample in quality["samples"]),
        "strict_answer_checks_total": 9,
        "sanity_sequences_match_prior_fast_stack": all(
            a["input_tokens"] == b["input_tokens"] and a["output_tokens"] == b["output_tokens"]
            for a, b in zip(quality["samples"], baseline_quality["samples"])),
        "caveats": ["Shared host; GPU5 rather than original context-table GPU0.",
                    "Generated sequences differ; this is not an identical-token workload speedup.",
                    "No whole-model accuracy qualification or replacement context table."],
    }
    for mode in ("control", "candidate"):
        measured = [sample for sample in samples if sample["mode"] == mode and not sample["warmup"]]
        assert len(measured) == 3
        summary[f"{mode}_decode_tps"] = [sample["decode_tps"] for sample in measured]
        summary[f"{mode}_median_decode_tps"] = statistics.median(summary[f"{mode}_decode_tps"])
        summary[f"{mode}_aggregate_decode_tps"] = 765 / sum(sample["decode_seconds"] for sample in measured)
        summary[f"{mode}_ttft_seconds"] = [sample["ttft_seconds"] for sample in measured]
    (ROOT / f"{prefix}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(prefix, summary["control_median_decode_tps"], summary["candidate_median_decode_tps"], divergences)


def main():
    versions = [
        ("int4-bindgroup-cache-model-pairs", "int4-bindgroup-cache-quality.json",
         "79e2acc8807666d71ecb014af9fece84eee0104ad81cfba0edfeed793db9c071"),
        ("int4-bindgroup-cache-model-v3", "int4-bindgroup-cache-model-v3-quality.json",
         "b3629dfd0a9e2883905d5ff3cca7740e3755bc5412cec78072af4864618e4b29"),
        ("int4-bindgroup-cache-model-v4", "int4-bindgroup-cache-model-v4-quality.json",
         "b2928305cd51bc1a2802a1072401f5926442cb1ca89122da1f6db8415f8e9b44"),
        ("int4-dynamic-uniform-model", "int4-dynamic-uniform-model-quality.json",
         "c267a40c54c10ed06496b9fc3c61581200d8761c52734ee06c56cf71d2d51e66"),
    ]
    for prefix, quality_name, expected_hash in versions:
        summary = model(prefix, quality_name, expected_hash)
        print(prefix, summary["control_median_decode_tps"], summary["candidate_median_decode_tps"])
    fused_followup()
    hyper_guard = ROOT / "int4-hyper-skinny-gpu5.guard.json"
    if hyper_guard.exists() and read(hyper_guard.name)["status"] == "completed":
        hyper_skinny()
    bundle = ROOT / "int4-dynamic-uniform-bundle.guard.json"
    if bundle.exists() and read(bundle.name)["status"] == "completed":
        model("int4-dynamic-uniform-bundle", "int4-dynamic-uniform-bundle-quality.json",
              versions[-1][2], comparison="dynamic_uniform_cache_bundle")
    report = {"status": "native_qualified_model_not_promoted", "quality_qualified": False}
    for version, sessions in (("v3", 220), ("v4", 1200)):
        ordinary = native(f"int4-bindgroup-cache-native-{version}", 30)
        pressure_prefix = "int4-bindgroup-cache-byte-pressure" + ("-v4" if version == "v4" else "")
        pressure = native(pressure_prefix, sessions)
        stats = pressure["cache_statistics"]
        assert len(stats) == 1 and stats[0]["clears"] > 0 and stats[0]["peak_entries"] < 8192
        ordinary["byte_pressure"] = pressure
        ordinary["total_gpu_reference_and_exact_parity_executions"] = 240 + sessions * 8
        ordinary["accounting"] = ("Unique whole buffers across entries" if version == "v4" else
                                  "Conservative sum of whole buffers repeated per entry")
        report[version] = ordinary
    negative_log = (ROOT / "int4-bindgroup-cache-native-v4.log").read_text()
    assert negative_log.count("PASS-explicit-negative-error-exit1-") == 12
    report["v4"]["negative_capacities"] = {
        "values": ["-1", "8193", "abc"], "repetitions_each": 4,
        "expected_error_messages_and_exit_one_verified": True,
        "validation_moved_before_pipeline_and_uniform_work": True,
        "prior_v3_shutdown_crash": "Owned -1 test emitted expected error then SIGSEGV; not erased.",
        "limitation": "Twelve clean v4 exits do not prove elimination of every asynchronous shutdown race.",
    }
    dynamic = native("int4-dynamic-uniform-native", 45)
    text = (ROOT / "int4-dynamic-uniform-native.log").read_text()
    additional = [line for line in text.splitlines()
                  if line.startswith("PASS ") and not line.startswith(("PASS case", "PASS physical"))]
    assert len(additional) == 37
    uniform = [line for line in additional if line.startswith("PASS uniform_offsets ")]
    assert len(uniform) == 2 and all("max_abs_errors=nan,0,nan,nan" in line for line in uniform)
    assert all("samples_per_mode=7" in line and "validationMode=full" in line for line in additional)
    negative_text = (ROOT / "int4-dynamic-uniform-model.log").read_text()
    assert negative_text.count("PASS-explicit-dynamic-uniform-error-exit1-") == 12
    result = read("int4-dynamic-uniform-model.json")
    assert result["load_experiment_flags"]["ORT_WEBGPU_DYNAMIC_UNIFORMS"] == "1"
    assert all(sample["dynamic_uniforms_flag"] == "1" for sample in result["samples"])
    dynamic.update(
        provider_sha256=versions[-1][2],
        additional_reference_fixture_runs=37,
        additional_fixture_scope="35 dense fixtures; varying uniforms with interleaved sessions; exact capture/replay",
        additional_timing_samples_per_fixture=7,
        additional_full_validation=True,
        varying_uniform_and_capture_max_abs_error=0,
        explicit_invalid_flag_exit_one_checks=12,
        pipeline_cache_separates_dynamic_and_static_layouts=True,
        whole_feature_vs_original_A9_binary_model_comparison_qualified=False,
        same_provider_whole_bundle_comparison_completed=(
            bundle.exists() and read(bundle.name)["status"] == "completed"),
        model_comparison_scope="Cache0/8192 at dynamic_uniforms1 in both modes; not a whole-feature static-layout A/B.",
    )
    report["dynamic_uniforms"] = dynamic
    (ROOT / "int4-bindgroup-cache-native-qualification.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
