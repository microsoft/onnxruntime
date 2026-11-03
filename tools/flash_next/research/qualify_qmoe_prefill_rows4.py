import json
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT, sha
from qualify_qsa_prefill_score16_screens import ttft_metrics


VARIANT = "qmoe-prefill-rows4-v2"


def verify_screen(proof, native, context, full):
    assert proof["status"] == "paired_ttft_win_qualified"
    assert proof["context"] == context and proof["full_validation"] == full
    assert proof["provider_sha256"] == native["provider_sha256"]
    assert proof["all_original_sequences_match"] and proof["repeated_sequences_match"]
    assert proof["zero_major_faults"] and proof["original_sequence_comparisons"] == 288
    assert proof["output_tokens"] == 36 and proof["order"] == [0, 1, 1, 0]
    assert proof["matched_pairs"] == 2 and proof["all_pairs_favor_candidate"]
    expected = ttft_metrics([row["ttft_seconds"] for row in proof["trials"]])
    assert expected["ttft_win_screen_qualified"]
    for name, value in expected.items():
        assert proof[name] == value, name
    return expected


def main():
    output = ROOT / f"{VARIANT}-contexts-20261031.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    directory = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2")
    build_path = directory / "build-manifest.json"
    native_path = ROOT / f"{VARIANT}-native-gpu7-20261031.qualification.json"
    native = json.loads(native_path.read_text())
    build = json.loads(build_path.read_text())
    assert native["finite_bitwise_comparisons"] == 300810240 and native["original_layers"] == 6
    assert native["status"] == "grouped_rows4_prefill_retained_chunk16_finite_bitwise_and_dispatch_qualified"
    assert build["baseline_byte_identical"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == native["provider_sha256"] == build["provider_sha256"]
    assert sha(directory / "native_compare") == native["comparator_sha256"] == build["comparator_sha256"]
    assert build["baseline_provider_sha256"] == native["baseline_provider_sha256"]
    sources = {str(path): sha(path) for path in (build_path, native_path)}
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest
        sources[str(directory / name)] = digest
    for name, digest in native["source_sha256"].items():
        assert sha(ROOT / name) == digest
        sources[str(ROOT / name)] = digest
    contexts, identity = {}, None
    for context in (2048, 8192, 32768):
        profile_path = ROOT / f"{VARIANT}-mode1-gpu7-full-profile{context}-20261031.qualification.json"
        profile = json.loads(profile_path.read_text())
        assert profile["provider_sha256"] == native["provider_sha256"]
        assert profile["original_output_prefixes_match"] and profile["zero_major_faults"]
        assert profile["full_validation"] and profile["selector_value"] == 1
        assert profile["active_grouped_prefill_programs"] == 96 * (context // 1024)
        assert profile["active_rows4_prefill_projections"] == 192 * (context // 1024)
        records = {"profile": profile["active_rows4_prefill_projections"]}
        for full in (True, False):
            path = ROOT / (f"{VARIANT}-gpu7-{'full-' if full else ''}"
                           f"paired-screen{context}-20261031.qualification.json")
            proof = json.loads(path.read_text())
            records["full" if full else "retained"] = verify_screen(proof, native, context, full)
            current = (proof["graph_sha256"], proof["config_sha256"], proof["baseline_provider_sha256"])
            identity = current if identity is None else identity
            assert current == identity
            sources[str(path)] = sha(path)
            for name, digest in proof["source_sha256"].items():
                assert sha(ROOT / name) == digest
                sources[str(ROOT / name)] = digest
        sources[str(profile_path)] = sha(profile_path)
        for name, digest in profile["source_sha256"].items():
            assert sha(ROOT / name) == digest
            sources[str(ROOT / name)] = digest
        contexts[str(context)] = records
    assert identity[2] == native["baseline_provider_sha256"]
    report = dict(status="controlled_ttft_screens_all_contexts_qualified_not_standard_decode",
                  provider_sha256=native["provider_sha256"], baseline_provider_sha256=identity[2],
                  graph_sha256=identity[0], config_sha256=identity[1], contexts=contexts,
                  all_original_sequences_match=True, zero_major_faults=True,
                  frozen_prefix_comparisons=1728, quality_qualified=False, headline_eligible=False,
                  standard_decode_qualified=False, default_off=True, source_sha256=sources)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "contexts": contexts}), flush=True)


if __name__ == "__main__":
    main()
