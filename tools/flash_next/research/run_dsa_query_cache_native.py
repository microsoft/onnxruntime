import argparse
import collections
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from build_dsa_query_cache import FLAG
from build_qmoe_node_replay import sha


ROOT = Path(__file__).resolve().parent
PREFIX = "dsa-query-cache-v2-native-gpu7-20261030"


def run_logged(path, command, environment):
    with path.open("x") as stream:
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    text = path.read_text()
    assert "FAIL" not in text, path
    return text


def inside(directory=None, prefix_name=PREFIX, parent_variant="prefill1024-v2", flag=FLAG,
           program_name="DynamicSparseAttentionQueryCache", extra_flags=None,
           report_status="dsa_query_cache_finite_bitwise_and_cpu_reference_qualified"):
    directory = directory or Path("/workspace/qwen38-int4-dsa-query-cache/build-v2")
    provider = directory / "libonnxruntime_providers_webgpu.so"
    build = json.loads((directory / "build-manifest.json").read_text())
    parent = json.loads((ROOT / f"{parent_variant}-contexts-20261030.qualification.json").read_text())
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert sha(provider) == build["provider_sha256"] and build["baseline_byte_identical"]
    assert build["baseline_provider_sha256"] == parent["provider_sha256"]
    assert build["fp16_only"] and build["fp32_original_fallback_preserved"]
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest
    fixtures_dir = ROOT / "dsa-split-kv-fixtures"
    manifest_path = ROOT / f"{prefix_name}-fixtures.json"
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["fixture_sha256"].items():
        assert sha(fixtures_dir / name) == digest
    fixtures = [fixtures_dir / name for name in manifest["output_elements"]]
    assert len(fixtures) == 27 and len(manifest["fixture_sha256"]) == 54
    comparator = Path("/workspace/onnxruntime-webgpu-h200-perf/build-int4-dsa-split-kv/dispatch_compare")
    sources, program_counts, timings = {manifest_path.name: sha(manifest_path)}, {}, {}
    expected_elements = 9 * sum(manifest["output_elements"].values())
    for pending in (1, 64):
        prefix = ROOT / f"{prefix_name}-pending{pending}"
        environment = dict(os.environ, ORT_TEST_MAX_PENDING_DISPATCHES=str(pending),
                           ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SPLIT_KV="1",
                           ORT_WEBGPU_QMOE_PREFILL512="1", ORT_WEBGPU_QMOE_PREFILL1024="1",
                           ORT_WEBGPU_DYNAMIC_UNIFORMS="1", ORT_WEBGPU_UNIFORM_ARENA="1",
                           ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES="8192")
        environment.update(extra_flags or {})
        path = ROOT / f"{prefix.name}-bitwise.log"
        text = run_logged(path, [str(directory / "native_compare"), str(provider), *map(str, fixtures)], environment)
        assert text.count("PASS finite_bitwise ") == 27
        assert f"PASS total_exact_elements={expected_elements}; CPU fallback disabled" in text
        sources[path.name] = sha(path)
        environment.update(ORT_TEST_ONLY_MODE="1", ORT_TEST_ATTENTION_SPLIT_COMPARE="1",
                           ORT_TEST_PROFILE_PREFIX=str(prefix))
        environment[flag] = "1"
        path = ROOT / f"{prefix.name}-reference.log"
        text = run_logged(path, [str(comparator), str(provider), *map(str, fixtures)], environment)
        assert "validationMode=full" in text and text.count("samples_per_mode=7 selected_mode=1") == 27
        profiles = [Path(line.removeprefix("PROFILE ")) for line in text.splitlines() if line.startswith("PROFILE ")]
        assert len(profiles) == 27
        events = [event for profile in profiles for event in json.loads(profile.read_text())]
        counts = collections.Counter(event["name"].rsplit("&", 1)[-1]
            for event in events if event.get("cat") == "Kernel")
        assert counts[program_name] == 18, counts
        assert counts["DynamicSparseAttention"] == 225, counts
        model_events = [event for event in events if event.get("cat") == "Kernel" and
                        event["name"].endswith("&" + program_name) and
                        "outputs[0] = {24,33,258}" in event["args"].get("shapes", "")]
        assert len(model_events) == 9
        timings[f"pending{pending}"] = dict(
            candidate_attention_median_gpu_us=statistics.median(event["dur"] for event in model_events[2:]),
            instrumented_not_model_tps=True)
        program_counts[f"pending{pending}"] = dict(counts)
        sources[path.name] = sha(path)
        sources.update({profile.name: sha(profile) for profile in profiles})
    environment.pop("ORT_TEST_PROFILE_PREFIX")
    environment["ORT_TEST_ENABLE_GRAPH_CAPTURE"] = "1"
    path = ROOT / f"{prefix_name}-capture.log"
    text = run_logged(path, [str(comparator), str(provider), str(fixtures_dir / "model_2k.onnx")], environment)
    assert "PASS model_2k " in text and "samples_per_mode=7 selected_mode=1" in text
    sources[path.name] = sha(path)
    report = dict(status=report_status,
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        fixtures=27, finite_bitwise_comparisons=2 * expected_elements,
        cpu_reference_executions=486, fixed_flag_capture_executions=9,
        pending_windows_tested=[1, 64], full_validation=True, invalid_selector_rejected=True,
        original_partition_size_preserved=64, original_dot_order_preserved=True,
        original_merge_unchanged=True, cache_outputs_checked=True,
        fp16_only=True, fp32_original_fallback_preserved=True,
        program_counts=program_counts, model_shaped_gpu_diagnostic=timings,
        fixture_sha256=manifest["fixture_sha256"], comparator_sha256=build["comparator_sha256"],
        reference_comparator_sha256=sha(comparator), source_sha256=sources, runner_sha256=sha(Path(__file__)),
        quality_qualified=False, headline_eligible=False)
    path = ROOT / f"{prefix_name}.inner.json"
    with path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "finite_bitwise_comparisons")}), flush=True)


def main(variant="dsa-query-cache-v2", parent_variant="prefill1024-v2", flag=FLAG,
         program_name="DynamicSparseAttentionQueryCache", extra_flags=None,
         report_status="dsa_query_cache_finite_bitwise_and_cpu_reference_qualified"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside", action="store_true")
    prefix_name = f"{variant}-native-gpu7-20261030"
    family, version = variant.rsplit("-v", 1)
    if parser.parse_args().inside:
        inside(directory=Path(f"/workspace/qwen38-int4-{family}/build-v{version}"),
               prefix_name=prefix_name, parent_variant=parent_variant, flag=flag,
               program_name=program_name, extra_flags=extra_flags, report_status=report_status)
    else:
        prefix = ROOT / prefix_name
        assert (ROOT / f"{parent_variant}-contexts-20261030.qualification.json").is_file()
        if any(prefix.with_suffix(suffix).exists() for suffix in (".guard.json", ".qualification.json", ".inner.json")):
            raise FileExistsError(prefix)
        import onnx

        fixtures_dir = ROOT / "dsa-split-kv-fixtures"
        manifest = dict(fixture_sha256={}, output_elements={})
        for path in sorted(fixtures_dir.glob("*.onnx")):
            manifest["fixture_sha256"][path.name] = sha(path)
            if not path.stem.endswith("_reference"):
                model = onnx.load(path)
                shape = [dimension.dim_value for dimension in model.graph.output[0].type.tensor_type.shape.dim]
                assert all(dimension > 0 for dimension in shape)
                manifest["output_elements"][path.name] = math.prod(shape)
        assert len(manifest["fixture_sha256"]) == 54 and len(manifest["output_elements"]) == 27
        with (ROOT / f"{prefix_name}-fixtures.json").open("x") as stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")
        command = ("set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
                   "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; exec python3 "
                   f"/evidence/webgpu-perf-oct20/{Path(sys.argv[0]).name} --inside")
        subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
            "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "1200",
            "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
        guard = json.loads(prefix.with_suffix(".guard.json").read_text())
        assert guard["status"] == "completed" and guard["exit_code"] == 0 and guard["gpu_index"] == 7
        assert guard["gpu_uuid"] == "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
        report = json.loads(prefix.with_suffix(".inner.json").read_text())
        for name, digest in report["source_sha256"].items():
            assert sha(ROOT / name) == digest
        for suffix in (".guard.json", ".log", ".inner.json"):
            path = prefix.with_suffix(suffix)
            report["source_sha256"][path.name] = sha(path)
        prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("status", "finite_bitwise_comparisons")}), flush=True)


if __name__ == "__main__":
    main()
