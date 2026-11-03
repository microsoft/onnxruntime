import json
from pathlib import Path

from qualify_qmoe_prefill_rows4 import VARIANT
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_qmoe_prefill_rows4_standard import DIRECTORY, TEMPLATE, verify_result


def verify_proof(proof, result, native, parent, context, full):
    assert proof["status"] == "original_sequences_qualified" and proof["phase"] == "standard"
    assert proof["context"] == context and proof["full_validation"] == full
    assert proof["provider_sha256"] == native["provider_sha256"] == parent["provider_sha256"]
    assert proof["graph_sha256"] == parent["graph_sha256"]
    assert proof["config_sha256"] == parent["config_sha256"]
    assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
    assert proof["chunk"] == 1024 and proof["output_tokens"] == 256
    assert proof["measured_repetitions"] == 3 and proof["excluded_warmup_requests"] == 1
    assert proof["original_sequence_comparisons"] == 1024 and proof["result"] == result
    assert proof["fixed_grouped_prefill_selector"] == 1 and proof["rejected_fc2_fold_selector"] == 0
    assert proof["comparison_kind"] == "standalone_standard_not_isolated_causal_pair"
    assert not proof["quality_qualified"] and not proof["headline_eligible"]


def main():
    output = ROOT / f"{VARIANT}-standard-contexts-20261031.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    parent_path = ROOT / f"{VARIANT}-contexts-20261031.qualification.json"
    native_path = ROOT / f"{VARIANT}-native-gpu7-20261031.qualification.json"
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    template_path = ROOT / f"{TEMPLATE}.guard.json"
    parent, native, controls = (json.loads(path.read_text()) for path in
                               (parent_path, native_path, controls_path))
    assert parent["status"] == "controlled_ttft_screens_all_contexts_qualified_not_standard_decode"
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert native["finite_bitwise_comparisons"] == 300810240
    assert sha(DIRECTORY / "libonnxruntime_providers_webgpu.so") == native["provider_sha256"]
    sources = {str(path): sha(path) for path in (parent_path, native_path, controls_path, template_path)}
    for path, digest in parent["source_sha256"].items():
        assert sha(Path(path)) == digest, path
        sources[path] = digest
    rows = []
    for context in (2048, 8192, 32768):
        references = [sample for sample in controls["samples"] if sample["context"] == context]
        row = {"context": context}
        for full in (True, False):
            prefix = ROOT / (f"{VARIANT}-chunk1024-gpu7-{'full-' if full else ''}"
                             f"standard{context}-20261031")
            proof_path = prefix.with_suffix(".qualification.json")
            proof, raw, guard = (json.loads(prefix.with_suffix(suffix).read_text()) for suffix in
                                 (".qualification.json", ".json", ".guard.json"))
            result = verify_result(raw, guard, native, parent, references, context, full)
            verify_proof(proof, result, native, parent, context, full)
            for key, path in (("native_proof_sha256", native_path), ("parent_proof_sha256", parent_path),
                              ("controls_sha256", controls_path), ("command_template_sha256", template_path)):
                assert proof[key] == sha(path), key
            for name, digest in proof["source_sha256"].items():
                path = ROOT / name
                assert sha(path) == digest, path
                sources[str(path)] = digest
            sources[str(proof_path)] = sha(proof_path)
            row["full_validation" if full else "candidate"] = result
        rows.append(row)
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    report = dict(status="rows4_native_and_standard_original_sequences_qualified",
                  provider_sha256=native["provider_sha256"], graph_sha256=parent["graph_sha256"],
                  config_sha256=parent["config_sha256"], contexts=[2048, 8192, 32768],
                  output_tokens=256, measured_repetitions=3, excluded_warmup_requests=1,
                  original_sequence_comparisons=6144, all_original_sequences_match=True,
                  zero_major_faults=True, standard_decode_qualified=True,
                  all_contexts_exceed_100_decode_tps=all(row["candidate"]["median_decode_tps"] > 100
                                                       for row in rows),
                  controlled_ttft_parent_sha256=sha(parent_path), fixed_grouped_prefill_selector=1,
                  rejected_fc2_fold_selector=0, default_off=True, model_and_weights_unchanged=True,
                  quality_qualified=False, headline_eligible=False, rows=rows, source_sha256=sources)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "all_contexts_exceed_100_decode_tps", "rows")}), flush=True)


if __name__ == "__main__":
    main()
