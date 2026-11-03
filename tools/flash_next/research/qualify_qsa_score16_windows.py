import argparse
import collections
import json
from pathlib import Path

from build_qsa_score16 import FLAG
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_tokens
from run_qsa_score16_model import verify_paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, choices=(2, 5), default=2)
    args = parser.parse_args()
    variant = f"qsa-score16-v{args.version}"
    program_name = "QsaPackedScore16" if args.version == 2 else "QsaPackedScore16QueryWide"
    output = ROOT / f"{variant}-windows-contexts-20261029.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    native_path = ROOT / f"{variant}-native-gpu7-20261029.qualification.json"
    parent_path = ROOT / "prefill256-v1-contexts-20261029.qualification.json"
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    native, parent, controls = (json.loads(p.read_text()) for p in (native_path, parent_path, controls_path))
    assert native["status"] == "exact_reference_and_score16_dispatch_qualified"
    assert native["minimum_score_stride"] == (8192 if args.version == 2 else 2048)
    assert native["reference_executions"] == 2484
    assert native["score16_dispatches"] == (162 if args.version == 2 else 198)
    assert len(native["exercised"]) == (9 if args.version == 2 else 11)
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    provider = Path(f"/home/sayanshaw/repos/qwen38-int4-qsa-score16/build-v{args.version}/libonnxruntime_providers_webgpu.so")
    assert sha(provider) == native["provider_sha256"]
    sources = {p.name: sha(p) for p in (native_path, parent_path, controls_path)}
    for name, digest in native["source_sha256"].items():
        assert sha(ROOT / name) == digest
        sources[name] = digest
    for path, digest in (*native["fixture_sha256"].items(), *native["reference_sha256"].items()):
        assert sha(Path(path)) == digest
    assert sha(Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf/build-int4-qsa-rank/qsa_rank_compare")) == native["comparator_sha256"]
    rows = []
    cases = [(32768, True, "profile"), (32768, True, "windows"),
             (2048, False, "windows"), (8192, False, "windows"), (32768, False, "windows")]
    if args.version == 5:
        cases = [(8192, True, "profile"), (8192, True, "windows"), *cases]
    for context, full, phase in cases:
        prefix = ROOT / f"{variant}-chunk256-gpu7-{'full-' if full else ''}{phase}{context}-20261029"
        proof_path = prefix.with_suffix(".qualification.json")
        proof = json.loads(proof_path.read_text())
        raw = json.loads(prefix.with_suffix(".json").read_text())
        guard = json.loads(prefix.with_suffix(".guard.json").read_text())
        verify_metadata(raw, guard, native, parent, full)
        assert raw["load_experiment_flags"][FLAG] == "1"
        assert proof["status"] == "original_sequences_qualified"
        assert proof["provider_sha256"] == native["provider_sha256"]
        assert proof["context"] == context and proof["phase"] == phase and proof["full_validation"] == full
        assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
        references = [s for s in controls["samples"] if s["context"] == context]
        if phase == "profile":
            verify_tokens(raw, references, context, 36)
            assert raw["profiling_scope"] == "decode_only" and raw["profile_decode_after_tokens"] == 35
            profiles = [ROOT / name for name in proof["source_sha256"] if f"-profile-ctx{context}-rep0-configured_" in name]
            assert len(profiles) == 1
            counts = collections.Counter(e["name"].rsplit("&", 1)[-1] for e in json.loads(profiles[0].read_text())
                                         if e.get("cat") == "Kernel")
            assert counts[program_name] == 12 and counts["QsaPackedScore4"] == 0
        else:
            result = verify_paired(raw, references, context)
            assert result == proof["result"]
            rows.append(dict(context=context, full_validation=full, result=result))
        sources[proof_path.name] = sha(proof_path)
        for name, digest in proof["source_sha256"].items():
            assert sha(ROOT / name) == digest
            sources[name] = digest
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    report = dict(status="score16_native_and_paired_original_sequences_qualified",
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
        gpu_index=7, chunk=256, pending_dispatches=64, contexts=[2048, 8192, 32768],
        native_reference_executions=2484, window_original_prefix_comparisons=256 * 4 * len(rows),
        all_original_prefixes_match=True, all_repeated_288_tokens_match=True, zero_major_faults=True,
        comparison_kind="controlled_paired_windows_not_standard_decode_headline",
        default_off=True, quality_qualified=False, headline_eligible=False,
        rows=rows, source_sha256=sources)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "all_original_prefixes_match", "rows")}), flush=True)


if __name__ == "__main__":
    main()
