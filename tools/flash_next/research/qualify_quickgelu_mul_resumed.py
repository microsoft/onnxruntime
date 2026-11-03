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
    overlay = REPOS / "qwen38-int4-quickgelu-next"
    build = load(overlay / "build/build-manifest.json")
    library = overlay / "build/libonnxruntime_providers_webgpu.so"
    assert build["baseline_byte_identical"] and sha(library) == build["provider_sha256"]
    assert build["baseline_provider_sha256"] == (
        "f65f28b1521f0106f14b49dea79047387f451ef4d3c3294b75bc87f7d946913d")
    assert sha(REPOS / "onnxruntime-webgpu-h200-perf/build-int4-shared-post-v4/"
               "libonnxruntime_providers_webgpu.so") == build["baseline_provider_sha256"]
    for path, digest in build["source_sha256"].items():
        assert sha(Path(path.replace("/workspace/", str(REPOS) + "/"))) == digest, path
    graph_path = overlay / "model/model.onnx"
    original_path = REPOS / "qwen38-int4-webgpu-perf/model.onnx"
    graph = onnx.load(graph_path, load_external_data=False)
    original = onnx.load(original_path, load_external_data=False)
    assert [v.SerializeToString() for v in graph.graph.initializer] == [
        v.SerializeToString() for v in original.graph.initializer]
    for location in {e.value for v in graph.graph.initializer for e in v.external_data
                     if e.key == "location"} | {"genai_config.json"}:
        assert (graph_path.parent / location).samefile(original_path.parent / location), location
    assert collections.Counter(n.op_type for n in graph.graph.node)["QuickGeluMulF16"] == 48
    native = load(ROOT / "quickgelu-mul-native.guard.json")
    log = (ROOT / "quickgelu-mul-native.log").read_text()
    assert native["status"] == "completed" and native["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability()" not in log
    assert log.count("PASS exact finite outputs ") == 16 and "total_exact_elements=153684" in log
    profiles = sorted(ROOT.glob("quickgelu-mul-exact-*-fused*.json"))
    assert len(profiles) == 16
    for path in profiles:
        assert kernels(path) == ({} if "qgm_0_" in path.name else {"QuickGeluMulF16": 3}), path
    for name in ("broadcast", "rank"):
        prefix = ROOT / ("quickgelu-mul-resumed-negative-" + name)
        guard = load(prefix.with_suffix(".guard.json"))
        assert guard["status"] == "failed" and guard["exit_code"] == 1
        assert "QuickGeluMulF16 requires identical input shapes" in prefix.with_suffix(".log").read_text()
    profiles = sorted(ROOT.glob("quickgelu-mul-profile-20261027-*configured*.json"))
    assert len(profiles) == 2
    for path in profiles:
        count = kernels(path)
        assert sum(count.values()) == 1581 and count["QuickGeluMulF16"] == 48
        assert count["QuickGelu"] == 0 and count["Mul"] == 12
    profiled = load(ROOT / "quickgelu-mul-profile-20261027.json")
    assert profiled["webgpu_validation_mode_override"] == "full"
    original_control = load(ROOT / "int4-hot-engram-2k-20261026.json")
    for sample, reference in zip(profiled["samples"], original_control["samples"]):
        assert sample["output_token_ids"] == reference["output_token_ids"][:32]
    extended = load(ROOT / "quickgelu-mul-resumed-native.guard.json")
    extended_log = (ROOT / "quickgelu-mul-resumed-native.log").read_text()
    assert extended["status"] == "completed" and extended["exit_code"] == 0
    assert "FAIL" not in extended_log and "GetCapability()" not in extended_log
    assert extended_log.count("PASS exact finite outputs ") == 10
    assert "total_exact_elements=277242" in extended_log
    extended_profiles = sorted(ROOT.glob("quickgelu-mul-resumed-exact-*-fused*.json"))
    assert len(extended_profiles) == 10
    for path in extended_profiles:
        assert kernels(path) == {"QuickGeluMulF16": 3}, path
    abba = load(ROOT / "quickgelu-mul-resumed-abba-summary.json")
    assert abba["status"] == "completed" and abba["same_provider"]
    assert [row["mode"] for row in abba["rows"]] == ["control", "candidate", "candidate", "control"]
    for row in abba["rows"]:
        path = Path(row["path"])
        data, guard = load(path), load(path.with_suffix(".guard.json"))
        assert data["status"] == guard["status"] == "completed" and guard["exit_code"] == 0
        assert data["plugin_sha256"] == sha(library) and not data["profiling"]
        expected_graph = graph_path if row["mode"] == "candidate" else REPOS / "qwen38-int4-shared-post/model.onnx"
        assert data["graph_sha256"] == sha(expected_graph)
        assert data["config_sha256"] == original_control["config_sha256"]
        assert data["webgpu_max_pending_dispatches_override"] == 64
        assert len(data["samples"]) == 4 and sum(s["warmup"] for s in data["samples"]) == 1
        command = " ".join(guard["command"])
        assert "ORT_WEBGPU_HC_EXACT_DENSE=1" in command and "ORT_WEBGPU_EXACT_DENSE_MID=1" in command
        assert "GetCapability()" not in path.with_suffix(".log").read_text()
        for sample, reference in zip(data["samples"], original_control["samples"]):
            assert sample["output_tokens"] == len(sample["output_token_ids"]) == 256
            assert sample["prompt_sha256"] == reference["prompt_sha256"]
            assert sample["output_token_ids"] == reference["output_token_ids"]
            assert all(p["major_page_faults"] == 0 for p in sample["process_resource_usage"].values())
    performance = {}
    for metric in ("median_decode_tps", "aggregate_decode_tps"):
        control = abba["summary"]["control"][metric]
        candidate = abba["summary"]["candidate"][metric]
        performance[metric] = {"control": control, "candidate": candidate,
                               "gain_percent": 100 * (candidate / control - 1)}
    positive = all(v["gain_percent"] > 0 for v in performance.values())
    report = {
        "status": "native_and_original_2k_sequence_qualified_isolated_overlay",
        "provider_sha256": sha(library), "graph_sha256": sha(graph_path),
        "baseline_provider_sha256": build["baseline_provider_sha256"],
        "source_sha256": build["source_sha256"],
        "reproduction_sha256": {
            name: sha(ROOT / name) for name in (
                "build_quickgelu_mul.py", "fuse_quickgelu_mul.py",
                "resume_quickgelu_mul_abba.py", "extend_quickgelu_mul_fixtures.py",
                "shared_activation_guard.py", "run_shared_activation_container.sh")
        },
        "baseline_relink_byte_identical": True,
        "weights_and_config": "Unchanged descriptors and original same-inode data/config",
        "reference_scope": (
            "Existing ORT production optimizer's QuickGelu(alpha=1) then Mul, including its "
            "negative-input sigmoid branch and FP16 storage boundary; not raw unfused Sigmoid/Mul"),
        "native_fixture_pairs": 26, "native_runs_each": 3, "finite_bitwise_elements": 430926,
        "explicit_shape_rejections": 2, "validation": "full; CPU fallback disabled",
        "actual_fused_dispatch_proof": True, "full_model_kernels": 1581,
        "prior_shared_post_kernels": 1629, "new_fused_dispatches": 48,
        "original_full_sequences_exact": 16, "ABBA": abba,
        "performance": performance,
        "decision": ("Measured positive ABBA result; retain isolated overlay, no main-source integration"
                     if positive else
                     "No consistent measured ABBA gain; retain negative evidence, do not integrate"),
        "continuation": {
            "inherited_overlay_rebuilt": False,
            "prior_cancelled_run_preserved": "quickgelu-mul-abba-0-control-2k-20261027",
            "prior_run_status": "running metadata but no process/container; only warmup result persisted",
            "new_measurements": "Separate resumed ABBA evidence; no prior result overwritten",
            "fixed_alpha": 1.0,
            "resources": "GPU1 model runs; GPU0 added native tests; GPU5 reserved; GPUs6/7 untouched",
        },
        "limitations": [
            "Same-provider ABBA process blocks, not randomized within-request comparisons",
            "Shared-host clocks/power and page cache are not fixed; no causal timing guarantee",
            "Not a raw Sigmoid-chain numerical-equivalence claim or whole-model quality qualification",
            "No new 8K/32K table; original source and qualified baseline provider remain unchanged",
            "No 100TPS, publication, commit or integration claim",
        ],
        "headline_eligible": False, "quality_qualified": False,
    }
    path = overlay / "qualification-resumed.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
