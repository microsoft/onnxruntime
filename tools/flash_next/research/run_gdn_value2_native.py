import argparse
import collections
import json
from pathlib import Path
import shlex
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-gdn-value2/build-v1")
FIXTURES = Path("/home/sayanshaw/repos/qwen38-int4-gdn-decode-view")
PREFIX = "gdn-value2-v1-native-gpu7-20261030"


def main(directory=DIRECTORY, prefix_name=PREFIX, parent_variant="prefill1024-v2",
         program_name="GatedDeltaNetValue2", toggle_flag="ORT_TEST_TOGGLE_GDN_VALUE2",
         report_status="gdn_value2_both_outputs_finite_bitwise_and_dispatch_qualified",
         parent_stamp="20261030", norm_only=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    output = ROOT / f"{prefix_name}.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    build = json.loads((directory / "build-manifest.json").read_text())
    parent = json.loads((ROOT / f"{parent_variant}-contexts-{parent_stamp}.qualification.json").read_text())
    assert build["baseline_byte_identical"] and build["original_gdn_object_byte_identical"]
    assert build["baseline_provider_sha256"] == parent["provider_sha256"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    assert sha(directory / "native_compare") == build["comparator_sha256"]
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest
    manifest = json.loads((FIXTURES / "fixtures.json").read_text())
    assert len(manifest) == 36
    fixtures = [FIXTURES / "fixtures" / entry["fixture"] for entry in manifest]
    eligible = [entry for entry in manifest if entry["decode_view"] and
                "_dk128_dv128_" in entry["fixture"] and entry["fixture"].endswith("float16.onnx") and
                (not norm_only or "_n1_" in entry["fixture"])]
    assert len(eligible) == (1 if norm_only else 2)
    hashes = {}
    for fixture in fixtures:
        reference = fixture.with_name(fixture.stem + "_reference.onnx")
        assert sha(fixture) == sha(reference)
        hashes.update({path.name: sha(path) for path in (fixture, reference)})
    sources, counts_by_pending = {}, {}

    def run(suffix, selected, pending, capture=False):
        prefix = ROOT / f"{prefix_name}-{suffix}"
        reuse = args.verify_existing and not capture
        if not reuse and any(prefix.with_suffix(ext).exists() for ext in (".guard.json", ".log")):
            raise FileExistsError(prefix)
        comparator = directory.parent / "capture-compare-v1/native_compare" if capture else directory / "native_compare"
        command_args = [comparator, directory / "libonnxruntime_providers_webgpu.so", *selected]
        command = (
            "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
            "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export "
            "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
            "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_GDN_PACKED_DECODE_VIEW=1 "
            f"{toggle_flag}=1 "
            f"ORT_TEST_MAX_PENDING_DISPATCHES={pending} ORT_TEST_ENABLE_GRAPH_CAPTURE={int(capture)}; ")
        if not capture:
            command += f"export ORT_TEST_PROFILE_PREFIX=/evidence/webgpu-perf-oct20/{prefix.name}-profile; "
        command += "exec " + shlex.join(str(path).replace("/home/sayanshaw/repos", "/workspace") for path in command_args)
        if not reuse:
            subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "600",
                "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
        guard = json.loads(prefix.with_suffix(".guard.json").read_text())
        assert guard["status"] == "completed" and guard["exit_code"] == 0 and guard["gpu_index"] == 7
        assert guard["gpu_uuid"] == "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
        assert guard["command"] == ["bash", "-lc", command]
        text = prefix.with_suffix(".log").read_text()
        assert "FAIL" not in text and "GetCapability" not in text
        assert text.count("PASS exact finite outputs ") == len(selected)
        assert text.count("PASS invalid_selector ") == (0 if capture else len(selected))
        elements = sum(entry["elements_per_run"] for entry in manifest
                       if FIXTURES / "fixtures" / entry["fixture"] in selected) * 9
        assert f"PASS total_exact_elements={elements}; CPU fallback disabled" in text
        for ext in (".guard.json", ".log"):
            path = prefix.with_suffix(ext)
            sources[path.name] = sha(path)
        return prefix, elements

    comparisons = 0
    for pending in (1, 64):
        prefix, elements = run(f"pending{pending}", fixtures, pending)
        comparisons += elements
        totals = collections.Counter()
        for entry in manifest:
            stem = Path(entry["fixture"]).stem
            for mode in ("reference", "fused"):
                profiles = list(ROOT.glob(f"{prefix.name}-profile-{stem}-{mode}*.json"))
                assert len(profiles) == 1
                counts = collections.Counter(event["name"].rsplit("&", 1)[-1]
                    for event in json.loads(profiles[0].read_text()) if event.get("cat") == "Kernel")
                active = mode == "fused" and entry in eligible
                assert counts[program_name] == (9 if active else 0), (stem, mode, counts)
                assert counts["GatedDeltaNet"] == (0 if active else 9), (stem, mode, counts)
                totals.update(counts)
                sources[profiles[0].name] = sha(profiles[0])
        assert totals[program_name] == len(eligible) * 9
        assert totals["GatedDeltaNet"] == (len(fixtures) * 2 - len(eligible)) * 9
        counts_by_pending[str(pending)] = dict(totals)
    capture_fixture = FIXTURES / "fixtures" / eligible[-1]["fixture"]
    capture_directory = directory.parent / "capture-compare-v1"
    capture_build = json.loads((capture_directory / "build-manifest.json").read_text())
    assert capture_build["provider_sha256"] == build["provider_sha256"] and capture_build["fixed_output_io_binding"]
    assert sha(capture_directory / "native_compare") == capture_build["comparator_sha256"]
    assert sha(capture_directory / "native_compare.cpp") == capture_build["source_sha256"]["native_compare.cpp"]
    assert capture_build["original_comparator_source_sha256"] == build["source_sha256"]["native_compare.cpp"]
    _, elements = run("capture-v2", [capture_fixture], 64, capture=True)
    comparisons += elements
    report = dict(status=report_status,
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=build["comparator_sha256"], fixtures=36, optimized_fixtures=len(eligible),
        capture_comparator_sha256=capture_build["comparator_sha256"],
        build_manifest_sha256=sha(directory / "build-manifest.json"),
        capture_build_manifest_sha256=sha(capture_directory / "build-manifest.json"),
        rounds_per_fixture=9, pending_windows_tested=[1, 64], fixed_flag_capture_candidate_executions=9,
        finite_bitwise_comparisons=comparisons, both_output_and_final_state_checked=True,
        invalid_selector_rejected=True, original_per_channel_arithmetic_preserved=True,
        multi_token_batch_dtype_and_odd_shape_fallbacks_checked=True,
        program_counts=counts_by_pending, fixture_sha256=hashes, source_sha256=sources,
        quality_qualified=False, headline_eligible=False)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "finite_bitwise_comparisons")}), flush=True)


if __name__ == "__main__":
    main()
