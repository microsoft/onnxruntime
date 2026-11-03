"""Recheck retained shared-post evidence; never infer quality from timing."""
import collections
import hashlib
import json
from pathlib import Path

import onnx


ROOT = Path(__file__).parent
REPOS = Path("/home/sayanshaw/repos")


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kernels(path):
    return collections.Counter(
        e["name"].split("&")[1] for e in load(path) if e.get("cat") == "Kernel")


def main():
    build_dir = REPOS / "onnxruntime-webgpu-h200-perf/build-int4-shared-post-v4"
    build = load(build_dir / "build-manifest.json")
    library = build_dir / "libonnxruntime_providers_webgpu.so"
    graph_path = REPOS / "qwen38-int4-shared-post/model.onnx"
    original_path = REPOS / "qwen38-int4-webgpu-perf/model.onnx"
    graph = onnx.load(graph_path, load_external_data=False)
    original = onnx.load(original_path, load_external_data=False)
    assert [v.SerializeToString() for v in graph.graph.initializer] == [
        v.SerializeToString() for v in original.graph.initializer]
    for location in {e.value for v in graph.graph.initializer for e in v.external_data
                     if e.key == "location"} | {"genai_config.json"}:
        assert (graph_path.parent / location).samefile(original_path.parent / location), location
    assert build["baseline_byte_identical"] and sha(library) == build["provider_sha256"]
    for source, expected in build["source_sha256"].items():
        assert sha(Path(source.replace("/workspace/", str(REPOS) + "/"))) == expected, source
    assert collections.Counter(n.op_type for n in graph.graph.node)["HyperConnectionSharedPostMixF16"] == 48
    native = load(ROOT / "shared-post-native-v4.guard.json")
    log = (ROOT / "shared-post-native-v4.log").read_text()
    assert native["status"] == "completed" and native["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability()" not in log
    assert log.count("PASS exact finite outputs ") == 16
    assert "total_exact_elements=1107510" in log
    fixture_profiles = sorted(ROOT.glob("shared-post-exact-v4-*-fused*.json"))
    assert len(fixture_profiles) == 16
    for path in fixture_profiles:
        assert kernels(path) == ({} if "shared_m0_" in path.name else {
            "HyperConnectionSharedPostMixF16": 3}), path
    for name, message in [
        ("gate_last", "requires shared gate last dimension 1"),
        ("shared_width", "requires shared output and gate with matching reduced rank"),
    ]:
        prefix = ROOT / ("shared-post-v4-negative-" + name)
        guard = load(prefix.with_suffix(".guard.json"))
        assert guard["status"] == "failed" and guard["exit_code"] == 1
        assert message in prefix.with_suffix(".log").read_text()
    original_control = load(ROOT / "int4-hot-engram-2k-20261026.json")
    rows = {}
    for mode in ("candidate", "control"):
        prefix = ROOT / ("int4-shared-post-v4-" + mode + "-2k-20261027")
        data, guard = load(prefix.with_suffix(".json")), load(prefix.with_suffix(".guard.json"))
        assert data["status"] == guard["status"] == "completed" and guard["exit_code"] == 0
        assert not data["profiling"] and not data["headline_eligible"] and not data["quality_qualified"]
        assert data["plugin_sha256"] == sha(library)
        assert data["graph_sha256"] == sha(graph_path if mode == "candidate" else
                                           REPOS / "qwen38-int4-gdn-prep/model.onnx")
        assert data["config_sha256"] == original_control["config_sha256"]
        assert data["webgpu_max_pending_dispatches_override"] == 64
        assert len(data["samples"]) == 4 and sum(s["warmup"] for s in data["samples"]) == 1
        command = " ".join(guard["command"])
        assert "ORT_WEBGPU_HC_EXACT_DENSE=1" in command and "ORT_WEBGPU_EXACT_DENSE_MID=1" in command
        for sample, reference in zip(data["samples"], original_control["samples"]):
            assert sample["context"] == 2048 and sample["output_tokens"] == 256
            assert len(sample["output_token_ids"]) == 256
            assert sample["prompt_sha256"] == reference["prompt_sha256"]
            assert sample["output_token_ids"] == reference["output_token_ids"]
            assert all(p["major_page_faults"] == 0 for p in sample["process_resource_usage"].values())
        measured = [s for s in data["samples"] if not s["warmup"]]
        rows[mode] = dict(data["summary"]["2048:configured"],
                         sample_decode_tps=[s["decode_tps"] for s in measured],
                         exact_original_sequences=4)
    profiles = sorted(ROOT.glob("int4-shared-post-v4-profile-20261027-*configured*.json"))
    assert len(profiles) == 2
    for path in profiles:
        count = kernels(path)
        assert sum(count.values()) == 1629 and count["HyperConnectionSharedPostMixF16"] == 48
        assert count["GatedAdd"] == 0 and count["Sigmoid"] == 12
    report = {
        "status": "native_and_original_2k_sequence_qualified_not_whole_model_quality",
        "provider_sha256": sha(library), "graph_sha256": sha(graph_path),
        "baseline_relink_byte_identical": True,
        "weights_and_config": "Unchanged initializer descriptors; original same-inode data/config",
        "fusion_chains": 48, "kernels_per_token": 1629, "prior_kernels": 1725,
        "native_fixture_pairs": 16, "native_runs_each": 3,
        "native_finite_bitwise_elements": 1107510, "native_shape_negatives": 2,
        "validation": "full; CPU fallback disabled; actual fused-only dispatches checked",
        "ordered_2k_rows": rows,
        "limitations": [
            "Candidate-before-control provider blocks, not counterbalanced causal speedup",
            "Shared-host timing variation; scalar64 retained, scalar256 and vec4 evidence preserved",
            "No new 8K/32K shared-fusion table; 100TPS not achieved",
            "Strict GPU arithmetic preservation, not CPU/GPU or whole-model quality qualification",
            "Earlier compiler/shape failures and blocked GPU attempts remain in original evidence",
        ],
        "headline_eligible": False, "quality_qualified": False,
    }
    output = ROOT / "shared-post-qualification.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
