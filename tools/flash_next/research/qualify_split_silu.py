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
    build_dir = repo / "build-int4-split-silu-v3"
    build = load(build_dir / "build-manifest.json")
    library = build_dir / "libonnxruntime_providers_webgpu.so"
    graph_path = Path("/home/sayanshaw/repos/qwen38-int4-split-silu/model.onnx")
    original_path = Path("/home/sayanshaw/repos/qwen38-int4-webgpu-perf/model.onnx")
    original = onnx.load(original_path, load_external_data=False)
    graph = onnx.load(graph_path, load_external_data=False)
    if [v.SerializeToString() for v in graph.graph.initializer] != [
        v.SerializeToString() for v in original.graph.initializer
    ]:
        raise ValueError("Initializer descriptors changed")
    for location in {
        e.value for v in graph.graph.initializer for e in v.external_data if e.key == "location"
    }:
        if not (graph_path.parent / location).samefile(original_path.parent / location):
            raise ValueError(f"External data no longer share original inode: {location}")
    if not (graph_path.parent / "genai_config.json").samefile(original_path.parent / "genai_config.json"):
        raise ValueError("Configuration no longer shares original inode")
    if digest(library) != build["provider_sha256"] or not build["baseline_byte_identical"]:
        raise ValueError("Missing exact provider/baseline relink qualification")
    for name, sha in build["source_sha256"].items():
        if digest(Path(name.replace("/workspace/", "/home/sayanshaw/repos/"))) != sha:
            raise ValueError(f"Built source changed: {name}")
    control = load(root / "int4-hot-engram-2k-20261026.json")
    rows = {}
    for window, name in (
        (64, "int4-split-silu-2k-20261026"),
        (128, "int4-split-silu-window128-2k-20261026"),
        (32, "int4-split-silu-window32-2k-20261026"),
    ):
        data = load(root / (name + ".json"))
        guard = load(root / (name + ".guard.json"))
        if (data["status"] != "completed" or guard["status"] != "completed" or guard["exit_code"] != 0 or
                data["profiling"] or data["plugin_sha256"] != digest(library) or
                data["graph_sha256"] != digest(graph_path) or
                data["config_sha256"] != control["config_sha256"] or
                data["webgpu_max_pending_dispatches_override"] != window):
            raise ValueError(f"Benchmark identities/protocol changed: {name}")
        samples = data["samples"]
        if len(samples) != 4 or sum(s["warmup"] for s in samples) != 1:
            raise ValueError(f"Expected excluded warmup plus three measured requests: {name}")
        for sample, reference in zip(samples, control["samples"]):
            if (sample["context"] != 2048 or sample["output_tokens"] != 256 or
                    len(sample["output_token_ids"]) != 256 or
                    sample["prompt_sha256"] != reference["prompt_sha256"] or
                    sample["output_token_ids"] != reference["output_token_ids"]):
                raise ValueError(f"Full original-model sequence parity failed: {name}")
        measured = [s for s in samples if not s["warmup"]]
        rows[str(window)] = {
            "context": 2048, "measured_requests": 3, "outputs_each": 256,
            "excluded_warmups": 1, "timed_decode_intervals_each": 255,
            "sample_decode_tps": [s["decode_tps"] for s in measured],
            "median_decode_tps": data["summary"]["2048:configured"]["median_decode_tps"],
            "aggregate_decode_tps": data["summary"]["2048:configured"]["decode_tps"],
            "sample_TTFT_seconds": [s["ttft_seconds"] for s in measured],
            "exact_original_graph_sequences_including_warmup": 4,
            "process_major_page_faults": [
                {phase: usage["major_page_faults"] for phase, usage in s["process_resource_usage"].items()}
                for s in samples
            ],
        }
    profiles = sorted(root.glob("int4-split-silu-profile-20261026-*configured*.json"))
    if len(profiles) != 2:
        raise ValueError("Expected two bounded full-model decode profiles")
    for path in profiles:
        kernels = [e for e in load(path) if e.get("cat") == "Kernel"]
        counts = collections.Counter(e["name"].split("&")[1] for e in kernels)
        if (len(kernels) != 1941 or counts["SplitScaledSiLUF16"] != 96 or counts["ScaledSiLU"] != 1 or
                counts["HyperConnectionPreMixSigmoidF16"] != 97 or
                counts["HyperConnectionPostMixGateF16"] != 96):
            raise ValueError(f"Full-model kernel proof failed: {path}")
    native = load(root / "split-silu-native-v3.guard.json")
    log = (root / "split-silu-native-v3.log").read_text()
    if (native["status"] != "completed" or native["exit_code"] != 0 or
            "FAIL" in log or "GetCapability()" in log or
            log.count("PASS exact finite FP16 ") != 6 or "total_exact_elements=19506" not in log):
        raise ValueError("Missing clean strict two-output native comparison")
    native_profiles = sorted(root.glob("split-silu-exact-v3-*fused*.json"))
    if len(native_profiles) != 6:
        raise ValueError("Expected six fused fixture profiles")
    for path in native_profiles:
        counts = collections.Counter(e["name"].split("&")[1] for e in load(path) if e.get("cat") == "Kernel")
        if counts != {"SplitScaledSiLUF16": 3}:
            raise ValueError(f"Fixture inlined instead of executing fused kernel: {path}")
    for name, expected in (
        ("bad_total", "Mismatch between the sum of 'split' (325) and the split dimension of the input (324)"),
        ("empty_prefix", "SplitScaledSiLUF16 requires two positive sizes summing to the last dimension"),
    ):
        prefix = root / ("split-silu-negative-" + name)
        state = load(prefix.with_suffix(".guard.json"))
        if state["status"] != "failed" or state["exit_code"] != 1 or expected not in prefix.with_suffix(".log").read_text():
            raise ValueError(f"Missing explicit expected negative rejection: {name}")
    profiled = load(root / "int4-split-silu-profile-20261026.json")
    candidate = load(root / "int4-split-silu-2k-20261026.json")
    if any(s["output_token_ids"] != b["output_token_ids"][:32]
           for s, b in zip(profiled["samples"], candidate["samples"])):
        raise ValueError("Profiled output-prefix parity failed")
    report = {
        "status": "native_and_2k_sequence_qualified_not_whole_model_quality_qualified",
        "provider_sha256": digest(library), "graph_sha256": digest(graph_path),
        "original_graph_sha256": digest(original_path),
        "baseline_provider_sha256": build["baseline_provider_sha256"],
        "baseline_relink_byte_identical": True,
        "weights_and_config": "Unchanged descriptors, same-inode external-data and config hardlinks",
        "fusion": {"op": "SplitScaledSiLUF16", "eligible_chains": 96,
                   "scope": "Two positive last-axis split widths, first output FP16 ScaledSiLU with alpha, second raw tail",
                   "remaining_unfused_ScaledSiLU": "Final HC mixer has no matching Split chain"},
        "native_exact_comparisons": {"fixture_pairs": 6, "outputs_each": 2, "runs_each": 3,
                                     "finite_bitwise_elements": 19506, "validation": "full",
                                     "CPU_fallback": False, "actual_fused_dispatch_proof": True},
        "negative_checks": {"invalid_total": "Model-local function shape inference rejects, exit1",
                             "empty_prefix": "Fused kernel rejects explicitly, exit1"},
        "full_model_dispatches": {"original": 2422, "prior_HC_fused": 2037, "current": 1941,
                                  "removed_this_stage": 96, "removed_total": 481,
                                  "profiled_requests": 2, "profiling_not_headline": True},
        "standard_context_rows_by_submission_window": rows,
        "selected_submission_window": 64,
        "headline_eligible": False, "quality_qualified": False, "target_100_tps_met": False,
        "limitations": [
            "Sequential submission-window checks, not a counterbalanced causal speedup qualification.",
            "No new 8K/32K table; original complete context table remains unchanged.",
            "Earlier registration attempts v1/v2 silently inlined after logged capability errors; excluded from fused qualification.",
            "Final v3 qualification requires actual fused-only dispatches, no capability errors and exact finite outputs.",
            "Whole-model quality and original CUDA/Tianlei parity remain unqualified; INT3 is a different architecture/arithmetic.",
            "Window64 TTFT variation 20.37/23.36/25.82s is retained; no cause attributed on shared host.",
        ],
    }
    (root / "split-silu-qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "median_2k_tps": rows["64"]["median_decode_tps"],
                      "dispatches": report["full_model_dispatches"]}))


if __name__ == "__main__":
    main()
