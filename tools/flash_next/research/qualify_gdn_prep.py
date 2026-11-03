import collections
import hashlib
import json
from pathlib import Path

import onnx


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).parent
    repo = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf")
    build_dir = repo / "build-int4-gdn-prep"
    build = load(build_dir / "build-manifest.json")
    library = build_dir / "libonnxruntime_providers_webgpu.so"
    graph_path = Path("/home/sayanshaw/repos/qwen38-int4-gdn-prep/model.onnx")
    original_path = Path("/home/sayanshaw/repos/qwen38-int4-webgpu-perf/model.onnx")
    graph = onnx.load(graph_path, load_external_data=False)
    original = onnx.load(original_path, load_external_data=False)
    if [v.SerializeToString() for v in graph.graph.initializer] != [
        v.SerializeToString() for v in original.graph.initializer
    ]:
        raise ValueError("Initializer descriptors changed")
    for location in {
        e.value for v in graph.graph.initializer for e in v.external_data if e.key == "location"
    }:
        if not (graph_path.parent / location).samefile(original_path.parent / location):
            raise ValueError(f"Original weight inode changed: {location}")
    if not (graph_path.parent / "genai_config.json").samefile(original_path.parent / "genai_config.json"):
        raise ValueError("Original configuration inode changed")
    if digest(library) != build["provider_sha256"] or not build["baseline_byte_identical"]:
        raise ValueError("Missing exact provider baseline relink gate")
    for name, sha in build["source_sha256"].items():
        if digest(Path(name.replace("/workspace/", "/home/sayanshaw/repos/"))) != sha:
            raise ValueError(f"Built source changed: {name}")
    if collections.Counter(n.op_type for n in graph.graph.node)["GdnProjPrepF16"] != 36:
        raise ValueError("Missing 36 opted-in GDN preparation chains")
    native = load(root / "gdn-prep-native.guard.json")
    log = (root / "gdn-prep-native.log").read_text()
    if (native["status"] != "completed" or native["exit_code"] != 0 or
            "FAIL" in log or "GetCapability()" in log or
            log.count("PASS exact finite outputs ") != 7 or "total_exact_elements=2121600" not in log):
        raise ValueError("Missing clean mixed FP16/FP32 native comparisons")
    profiles = sorted(root.glob("gdn-prep-exact-*fused*.json"))
    if len(profiles) != 7:
        raise ValueError("Expected seven fused fixture profiles")
    for path in profiles:
        count = collections.Counter(e["name"].split("&")[1] for e in load(path) if e.get("cat") == "Kernel")
        expected = {} if "b1_l0" in path.name else {"GdnProjPrepF16": 3}
        if count != expected:
            raise ValueError(f"Unexpected fused fixture dispatches: {path}")
    for name in ("bad_state_width", "bad_state_channels"):
        prefix = root / ("gdn-prep-negative-v2-" + name)
        state = load(prefix.with_suffix(".guard.json"))
        if (state["status"] != "failed" or state["exit_code"] != 1 or
                "GdnProjPrepF16 requires ab [B,L,96]" not in prefix.with_suffix(".log").read_text()):
            raise ValueError(f"Missing native state-layout rejection: {name}")
    name = "int4-gdn-prep-2k-v1-20261026"
    data = load(root / (name + ".json"))
    guard = load(root / (name + ".guard.json"))
    control = load(root / "int4-hot-engram-2k-20261026.json")
    if (data["status"] != "completed" or guard["status"] != "completed" or guard["exit_code"] != 0 or
            data["profiling"] or data["plugin_sha256"] != digest(library) or
            data["graph_sha256"] != digest(graph_path) or data["config_sha256"] != control["config_sha256"] or
            data["webgpu_max_pending_dispatches_override"] != 64):
        raise ValueError("Benchmark identity or submission protocol changed")
    if len(data["samples"]) != 4 or sum(s["warmup"] for s in data["samples"]) != 1:
        raise ValueError("Expected one excluded warmup and three measured requests")
    for sample, reference in zip(data["samples"], control["samples"]):
        if (sample["context"] != 2048 or sample["output_tokens"] != 256 or
                len(sample["output_token_ids"]) != 256 or sample["prompt_sha256"] != reference["prompt_sha256"] or
                sample["output_token_ids"] != reference["output_token_ids"]):
            raise ValueError("Full original-graph sequence parity failed")
    profiles = sorted(root.glob("int4-gdn-prep-profile-20261026-*configured*.json"))
    if len(profiles) != 2:
        raise ValueError("Expected two bounded full-model decode profiles")
    for path in profiles:
        counts = collections.Counter(e["name"].split("&")[1] for e in load(path) if e.get("cat") == "Kernel")
        if (sum(counts.values()) != 1725 or counts["GdnProjPrepF16"] != 36 or counts["Split"] or
                counts["Transpose"] or counts["Cast"] != 2 or counts["CausalConvWithState"] != 1 or
                counts["SplitScaledSiLUF16"] != 96 or counts["HyperConnectionPreMixSigmoidF16"] != 97 or
                counts["HyperConnectionPostMixGateF16"] != 96):
            raise ValueError(f"Full-model fused dispatch proof failed: {path}")
    profiled = load(root / "int4-gdn-prep-profile-20261026.json")
    if any(s["output_token_ids"] != b["output_token_ids"][:32]
           for s, b in zip(profiled["samples"], data["samples"])):
        raise ValueError("Profiled output-prefix parity failed")
    measured = [s for s in data["samples"] if not s["warmup"]]
    report = {
        "status": "native_and_2k_sequence_qualified_not_whole_model_quality_qualified",
        "provider_sha256": digest(library), "graph_sha256": digest(graph_path),
        "original_graph_sha256": digest(original_path),
        "baseline_provider_sha256": build["baseline_provider_sha256"],
        "baseline_relink_byte_identical": True,
        "weights_and_config": "Unchanged descriptors and same-inode original external data and config",
        "fusion": {"op": "GdnProjPrepF16", "chains": 36,
                   "replaced_each": "2 Splits + 2 Casts + 2 Transposes + CausalConvWithState",
                   "inputs": "FP16 packed projection [B,L,16384], ab [B,L,96], weights, bias, CF state",
                   "outputs": "FP16 channels-last conv and z, FP32 a and b, FP16 channels-first state",
                   "GatedDeltaNet_itself": "Unchanged packed-input kernel and cache layout"},
        "native_exact_comparisons": {"fixture_pairs": 7, "outputs_each": 5, "runs_each": 3,
                                     "finite_bitwise_elements": 2121600, "validation": "full",
                                     "CPU_fallback": False, "actual_fused_dispatch_proof": True,
                                     "lengths": [0, 1, 2, 3, 4, 16], "batches": [1, 2]},
        "negative_checks": "Two invalid CF state layouts explicitly reject in the fused kernel, exit1",
        "full_model_dispatches": {"original": 2422, "prior_HC_and_split_fused": 1941,
                                  "current": 1725, "removed_this_stage": 216, "removed_total": 697},
        "standard_context_row": {
            "context": 2048, "window": 64, "measured_requests": 3, "outputs_each": 256,
            "excluded_warmups": 1, "timed_decode_intervals_each": 255,
            "sample_decode_tps": [s["decode_tps"] for s in measured],
            "median_decode_tps": data["summary"]["2048:configured"]["median_decode_tps"],
            "aggregate_decode_tps": data["summary"]["2048:configured"]["decode_tps"],
            "sample_TTFT_seconds": [s["ttft_seconds"] for s in measured],
            "exact_original_graph_sequences_including_warmup": 4,
            "process_major_page_faults": [
                {phase: usage["major_page_faults"] for phase, usage in s["process_resource_usage"].items()}
                for s in data["samples"]],
        },
        "limitations": [
            "Not a counterbalanced causal speedup; shared-host timing variation remains",
            "Strict GPU arithmetic preservation, not a passed CPU/GPU numerical-quality gate",
            "No new 8K/32K context table, whole-model quality promotion, CUDA transfer completeness, or 100TPS",
            "Only explicit local-function derivative opts in; original graph and generic convolution remain unchanged",
        ],
        "headline_eligible": False, "quality_qualified": False,
    }
    output = root / "gdn-prep-qualification.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "median_2k_tps":
                      report["standard_context_row"]["median_decode_tps"], "output": str(output)}))


if __name__ == "__main__":
    main()
