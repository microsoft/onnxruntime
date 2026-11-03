import collections
import hashlib
import json
from pathlib import Path

import onnx


def main():
    root = Path(__file__).parent
    repo = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf")
    result = json.loads((root / "int4-hc-fusion-2k-20261026.json").read_text())
    control = json.loads((root / "int4-hot-engram-2k-20261026.json").read_text())
    build = json.loads((repo / "build-int4-hc-fusion/build-manifest.json").read_text())
    if result["status"] != "completed" or result["plugin_sha256"] != build["provider_sha256"]:
        raise ValueError("Missing completed benchmark for the exact built provider")
    library = repo / "build-int4-hc-fusion/libonnxruntime_providers_webgpu.so"
    if hashlib.sha256(library.read_bytes()).hexdigest() != result["plugin_sha256"]:
        raise ValueError("Built provider no longer matches benchmark")
    original_path = Path("/home/sayanshaw/repos/qwen38-int4-webgpu-perf/model.onnx")
    graph_path = root / "int4-hc-fusion-model/model.onnx"
    if hashlib.sha256(graph_path.read_bytes()).hexdigest() != result["graph_sha256"]:
        raise ValueError("Derived graph no longer matches benchmark")
    original = onnx.load(original_path, load_external_data=False)
    graph = onnx.load(graph_path, load_external_data=False)
    if [v.SerializeToString() for v in original.graph.initializer] != [
        v.SerializeToString() for v in graph.graph.initializer
    ]:
        raise ValueError("Persisted initializer descriptors differ")
    for location in {
        e.value for v in graph.graph.initializer for e in v.external_data if e.key == "location"
    }:
        if not (original_path.parent / location).samefile(graph_path.parent / location):
            raise ValueError(f"External weights do not share original inode: {location}")
    for path, digest in build["source_sha256"].items():
        local = Path(path.replace("/workspace/", "/home/sayanshaw/repos/"))
        if hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Built source changed: {local}")
    samples = result["samples"]
    if len(samples) != 4 or sum(s["warmup"] for s in samples) != 1:
        raise ValueError("Expected one excluded warmup and three measured requests")
    measured = [s for s in samples if not s["warmup"]]
    if any(s["context"] != 2048 or len(s["output_token_ids"]) != 256 for s in samples):
        raise ValueError("Expected complete 2K/256-token requests")
    if result["config_sha256"] != control["config_sha256"]:
        raise ValueError("Original configuration changed")
    if len(control["samples"]) != len(samples):
        raise ValueError("Missing full original-graph sequence controls")
    for actual, expected in zip(samples, control["samples"]):
        if (actual["repetition"] != expected["repetition"] or
                actual["prompt_sha256"] != expected["prompt_sha256"] or
                actual["output_token_ids"] != expected["output_token_ids"]):
            raise ValueError("Full-model sequence parity failed")
    profiles = sorted(root.glob("int4-hc-fusion-profile-gpu1-20261026-*configured*.json"))
    if len(profiles) != 2:
        raise ValueError("Expected two bounded full-model decode profiles")
    for path in profiles:
        events = json.loads(path.read_text())
        kernels = [e for e in events if e.get("cat") == "Kernel"]
        counts = collections.Counter(e["name"].split("&")[1] for e in kernels)
        if (len(kernels) != 2037 or counts["HyperConnectionPreMixSigmoidF16"] != 97 or
                counts["HyperConnectionPostMixGateF16"] != 96 or counts["Div"] != 0):
            raise ValueError(f"Full-model fusion dispatch proof failed: {path}")
    native = json.loads((root / "hc-fusion-exact-native.guard.json").read_text())
    if native["status"] != "completed" or native["exit_code"] != 0:
        raise ValueError("Strict native comparison did not complete")
    native_log = (root / "hc-fusion-exact-native.log").read_text()
    if native_log.count("PASS exact finite FP16 ") != 8 or "total_exact_elements=653070" not in native_log:
        raise ValueError("Missing all exact native comparison cases")
    report = {
        "status": "native_and_2k_sequence_qualified_not_whole_model_quality_qualified",
        "provider_sha256": result["plugin_sha256"],
        "graph_sha256": result["graph_sha256"],
        "config_sha256": result["config_sha256"],
        "baseline_relink_byte_identical": build["baseline_byte_identical"],
        "original_weight_descriptors_and_data": "Unchanged; staged external data are same-inode hardlinks",
        "model_scope": "Explicit FP16 HC function-call derivative; original graph/config untouched",
        "native_exact_comparisons": {"fixture_pairs": 8, "runs_each": 3, "finite_bitwise_elements": 653070,
                                     "validation": "full", "CPU_fallback": False},
        "decode_dispatches": {"before": 2422, "after": 2037, "removed": 385,
                              "profiled_requests": 2, "profiling_not_headline": True},
        "standard_context_three_sample_row": {
            "context": 2048, "excluded_warmups": 1, "measured_requests": 3, "outputs_each": 256,
            "timed_decode_intervals_each": 255,
            "sample_decode_tps": [s["decode_tps"] for s in measured],
            "median_decode_tps": result["summary"]["2048:configured"]["median_decode_tps"],
            "aggregate_decode_tps": result["summary"]["2048:configured"]["decode_tps"],
            "sample_TTFT_seconds": [s["ttft_seconds"] for s in measured],
            "exact_original_graph_sequences_including_warmup": 4,
            "process_major_page_faults": [
                {phase: usage["major_page_faults"] for phase, usage in s["process_resource_usage"].items()}
                for s in samples
            ],
        },
        "headline_eligible": False,
        "quality_qualified": False,
        "limitations": [
            "Not a counterbalanced same-GPU graph comparison; do not infer an isolated percentage speedup.",
            "Complete original-graph 2K/8K/32K table remains separate and unchanged.",
            "No new fused-graph 8K/32K measurements; all GPUs subsequently had foreign work.",
            "Independent CPU function-expansion parity is tested; CPU/GPU numerical parity is not qualified.",
            "Earlier CPU-oracle native run failed at hc_post_m2_h7; tolerances were not relaxed.",
            "Strict GPU reference is the unfused path in the new provider; original-provider full sequences also match.",
            "Unsupported scalar-parameter runtime negatives are prepared but GPU execution was ownership-blocked.",
            "Unmodified guard unit tests have four stale subprocess-mock/NVML UUID failures; no guard code changed.",
            "100 tok/s is not reached; this is a targeted fusion improvement, not whole-model quality promotion.",
        ],
    }
    output = root / "hc-fusion-qualification.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "decode_dispatches")}))


if __name__ == "__main__":
    main()
