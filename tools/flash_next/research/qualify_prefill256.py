import collections
import json
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_prefill_flags, verify_tokens
from run_qmoe_prefill256_native import gate_rows


def main(variant="prefill256-v1", stamp="20261029", parent_variant="qsa-indexer-stack-v1",
         parent_stamp="20261029", previous_chunk=128, chunk=256):
    family, version = variant.rsplit("-v", 1)
    output = ROOT / f"{variant}-contexts-{stamp}.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    directory = Path(f"/home/sayanshaw/repos/qwen38-int4-qmoe-{family}/build-v{version}")
    native_path = ROOT / f"qmoe-{variant}-native-gpu7-{stamp}.qualification.json"
    native = json.loads(native_path.read_text())
    parent_path = ROOT / f"{parent_variant}-contexts-{parent_stamp}.qualification.json"
    parent = json.loads(parent_path.read_text())
    assert native["status"] == f"prefill{chunk}_retained_chunk16_finite_bitwise_and_dispatch_qualified"
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    if chunk == 256:
        assert native["finite_bitwise_comparisons"] == 68843520 and native["fallback_rows"] == [129, 255, 257]
    elif chunk == 512:
        assert native["finite_bitwise_comparisons"] == 143523840 and native["fallback_rows"] == [511, 513]
        assert native["rows_tested"] == [16, 128, 256, 511, 512, 513]
    elif chunk == 1024:
        assert native["finite_bitwise_comparisons"] == 300810240 and native["fallback_rows"] == [1023, 1025]
        assert native["rows_tested"] == [16, 128, 256, 512, 1023, 1024, 1025]
    elif chunk == 1280:
        assert native["finite_bitwise_comparisons"] == 568197120 and native["fallback_rows"] == [1279, 1281]
        assert native["rows_tested"] == [16, 128, 256, 512, 768, 1024, 1279, 1280, 1281]
    else:
        raise ValueError(f"Unsupported prefill chunk: {chunk}")
    assert native["invalid_flag_explicitly_rejected"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == native["provider_sha256"]
    assert sha(directory / "native_compare") == native["comparator_sha256"]
    sources = {p.name: sha(p) for p in (native_path, parent_path)}
    for name, digest in native["source_sha256"].items():
        assert sha(ROOT / name) == digest
        sources[name] = digest
    if chunk in (512, 1024, 1280):
        fixture_version = "v4" if chunk == 512 else "v1"
        attention_path = ROOT / f"{variant}-dsa-native-gpu7-fixtures{fixture_version}-{stamp}.qualification.json"
        attention = json.loads(attention_path.read_text())
        assert attention["status"] == f"prefill{chunk}_dsa_cpu_reference_and_admission_qualified"
        assert attention["provider_sha256"] == native["provider_sha256"]
        assert attention["finite_cpu_reference_comparisons"] == {
            512: 83238912, 1024: 168256512, 1280: 338789376}[chunk]
        assert attention["expected_rejections"] == 12
        assert attention["full_validation"] and attention["pending_windows_tested"] == [1, 64]
        assert attention["original_dsa_object_byte_identical"] and attention["shader_arithmetic_unchanged"]
        assert attention["cache_outputs_checked"] and attention["invalid_selector_rejected"]
        assert attention["default_sequence_bound_preserved"]
        sources[attention_path.name] = sha(attention_path)
        for name, digest in attention["source_sha256"].items():
            assert sha(ROOT / name) == digest
            sources[name] = digest
        build_path = directory / "build-manifest.json"
        assert sha(build_path) == attention["build_manifest_sha256"]
        build = json.loads(build_path.read_text())
        for name, digest in build["source_sha256"].items():
            assert sha(directory / name) == digest
        fixtures = directory.parent / f"fixtures-dsa-{fixture_version}"
        fixture_manifest = fixtures / "manifest.json"
        assert sha(fixture_manifest) == attention["fixture_manifest_sha256"]
        for name, evidence in json.loads(fixture_manifest.read_text())["fixtures"].items():
            assert sha(fixtures / name) == evidence["sha256"]
    profile_prefix = ROOT / f"{variant}-gpu7-full-profile2048-{stamp}"
    profile_proof = json.loads(profile_prefix.with_suffix(".qualification.json").read_text())
    assert profile_proof["provider_sha256"] == native["provider_sha256"]
    assert profile_proof[f"active_prefill{chunk}_programs"] == {"QmoeGate1Token": 2 * (2048 // chunk) * 48}
    profile_raw = json.loads(profile_prefix.with_suffix(".json").read_text())
    profile_guard = json.loads(profile_prefix.with_suffix(".guard.json").read_text())
    verify_metadata(profile_raw, profile_guard, native, parent, True, chunk)
    if chunk == 1280:
        verify_prefill_flags(profile_raw, chunk)
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    controls = json.loads(controls_path.read_text())
    sources[controls_path.name] = sha(controls_path)
    profile_refs = [s for s in controls["samples"] if s["context"] == 2048]
    verify_tokens(profile_raw, profile_refs, 2048, 2)
    for name, digest in profile_proof["source_sha256"].items():
        assert sha(ROOT / name) == digest
        sources[name] = digest
    if chunk in (512, 1024, 1280):
        profiles = [ROOT / name for name in profile_proof["source_sha256"]
                    if name.startswith(profile_prefix.name + "-profile_")]
        assert len(profiles) == 1
        events = json.loads(profiles[0].read_text())
        coverage = collections.Counter(event["name"].rsplit("&", 1)[-1] for event in events
            if event.get("cat") == "Kernel" and event["name"].endswith("&QmoeGate1Token") and gate_rows(event) == chunk)
        assert coverage == {"QmoeGate1Token": 2 * (2048 // chunk) * 48}
        if chunk == 1280:
            shapes = collections.Counter(gate_rows(event) for event in events
                if event.get("cat") == "Kernel" and event["name"].endswith("&QmoeGate1Token"))
            assert shapes == {1280: 96, 768: 96}, shapes
        sources[profile_prefix.with_suffix(".qualification.json").name] = sha(profile_prefix.with_suffix(".qualification.json"))
    previous = {r["context"]: r["candidate"] for r in parent["rows"]}
    rows = []
    for context in (2048, 8192, 32768):
        references = [s for s in controls["samples"] if s["context"] == context]
        results = {}
        for full in (True, False):
            prefix = ROOT / f"{variant}-gpu7-{'full-' if full else ''}standard{context}-{stamp}"
            proof_path = prefix.with_suffix(".qualification.json")
            proof = json.loads(proof_path.read_text())
            raw = json.loads(prefix.with_suffix(".json").read_text())
            guard = json.loads(prefix.with_suffix(".guard.json").read_text())
            verify_metadata(raw, guard, native, parent, full, chunk)
            if chunk in (512, 1024, 1280):
                verify_prefill_flags(raw, chunk)
                assert not raw["profiling"] and raw["profiling_scope"] is None
                assert raw["comparison"] is None and raw["profile_decode_after_tokens"] is None
            assert proof["status"] == "original_sequences_qualified"
            assert proof["context"] == context and proof["phase"] == "standard"
            assert proof["full_validation"] == full and proof["zero_major_faults"]
            assert proof["provider_sha256"] == native["provider_sha256"]
            result = verify_tokens(raw, references, context, 256)
            assert result == proof["result"]
            results["full_validation" if full else "candidate"] = result
            sources[proof_path.name] = sha(proof_path)
            for name, digest in proof["source_sha256"].items():
                assert sha(ROOT / name) == digest
                sources[name] = digest
        candidate = results["candidate"]
        rows.append(dict(context=context, previous_chunk=previous_chunk, candidate_chunk=chunk,
            previous=previous[context], **results,
            ttft_reduction_percent=100 * (1 - candidate["mean_ttft_seconds"] / previous[context]["mean_ttft_seconds"]),
            standard_decode_tps_ratio=candidate["median_decode_tps"] / previous[context]["median_decode_tps"],
            comparison_kind="separate_run_observation_not_isolated_causal_pair"))
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    report = dict(status=f"prefill{chunk}_native_and_original_sequences_qualified",
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"], gpu_index=7,
        contexts=[2048, 8192, 32768], candidate_chunk=chunk, pending_dispatches=64,
        output_tokens=256, measured_repetitions=3, excluded_warmup_requests=1,
        original_sequence_comparisons=6144, all_original_sequences_match=True, zero_major_faults=True,
        model_and_weights_unchanged=True, default_off=True,
        native_finite_bitwise_comparisons=native["finite_bitwise_comparisons"],
        all_contexts_at_least_100_decode_tps=all(r["candidate"]["median_decode_tps"] >= 100 for r in rows),
        quality_qualified=False, headline_eligible=False, rows=rows, source_sha256=sources)
    if chunk == 512:
        report["native_dsa_finite_cpu_reference_comparisons"] = attention["finite_cpu_reference_comparisons"]
        report["build_manifest_sha256"] = sha(build_path)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "all_original_sequences_match", "rows")}), flush=True)


if __name__ == "__main__":
    main()
