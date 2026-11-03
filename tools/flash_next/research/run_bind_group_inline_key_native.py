import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_fc2_rank_waves_native import guarded
from run_gpu_slice_model import verify_sources


BUILD = Path("/home/sayanshaw/repos/qwen38-int4-bind-group-inline-key/build-v1")
FLAG = "ORT_WEBGPU_BIND_GROUP_INLINE_KEY"


if __name__ == "__main__":
    output = ROOT / "bind-group-inline-key-v1-native-outcome.json"
    if output.exists():
        raise FileExistsError(output)
    parent_path = ROOT / "fc2-private-a-v1-native-outcome.json"
    parent = verify_sources(parent_path, "fc2_private_a_native_finite_bitwise_and_shader_routing_qualified")
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path, digest in manifest["source_sha256"].items():
        if sha(host_path(path)) != digest:
            raise RuntimeError(f"Changed build input: {path}")
    if (sha(BUILD / "libonnxruntime_providers_webgpu.so") != manifest["provider_sha256"] or
            parent["provider_sha256"] != manifest["parent_provider_sha256"]):
        raise RuntimeError("Wrong parent or inline-key provider")
    template = ROOT / "fc2-private-a-v1-native-flag0-gpu0.guard.json"
    original = json.loads(template.read_text())
    phases = []
    evidence = [Path(__file__), GUARD, parent_path, template, manifest_path,
                BUILD / "libonnxruntime_providers_webgpu.so"]
    for flag, capacity in ((0, 8192), (1, 8192), (1, 1), (2, 8192)):
        prefix = ROOT / f"bind-group-inline-key-v1-native-flag{flag}-capacity{capacity}-gpu0"
        stats_path = Path(str(prefix) + "-cache-stats.jsonl")
        dump = Path(str(prefix) + "-shaders.txt")
        for path in (stats_path, dump):
            if path.exists():
                raise FileExistsError(path)
        command = original["command"].copy()
        command[-1] = replace_once(command[-1],
            "/workspace/qwen38-int4-fc2-private-a/build-v1/libonnxruntime_providers_webgpu.so",
            "/workspace/qwen38-int4-bind-group-inline-key/build-v1/libonnxruntime_providers_webgpu.so")
        command[-1] = replace_once(command[-1], "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192",
                                  f"ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES={capacity}")
        command[-1] = command[-1].replace("fc2-private-a-v1-native-flag0-gpu0", prefix.name)
        command[-1] = (f"export {FLAG}={flag} ORT_WEBGPU_QMOE_SUBGROUP_GATE=1 "
            f"ORT_WEBGPU_BIND_GROUP_CACHE_STATS_FILE=/evidence/webgpu-perf-oct20/{stats_path.name}; " + command[-1])
        child, guard = guarded(prefix, command)
        log = prefix.with_suffix(".log").read_text()
        if guard["gpu_index"] != 0 or guard["gpu_uuid"] != original["gpu_uuid"]:
            raise RuntimeError("Wrong native GPU identity")
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".log", ".guard.json"))
        if flag == 2:
            if child.returncode != 1 or f"{FLAG} must be 0 or 1" not in log:
                raise RuntimeError("Invalid inline-key selector did not reject")
            phases.append(dict(flag=flag, invalid_selector_rejected=True))
            continue
        child.check_returncode()
        if (guard["status"] != "completed" or log.count("PASS changing_inputs_and_routes ") != 23 or
                "PASS total_exact_elements=9748480; CPU fallback disabled" not in log):
            raise RuntimeError("Incomplete original-unfused-GPU arithmetic qualification")
        shader = dump.read_text()
        if "QmoeFC2PrivateABitcastF16" in shader or "QmoeHierarchicalGate1Token" not in shader:
            raise RuntimeError("Private FC2 enabled or actual hierarchical gate missing")
        stats = [json.loads(line) for line in stats_path.read_text().splitlines()]
        if (not stats or any(row["peak_entries"] > capacity or row["peak_referenced_bytes"] > 256 * 1024**2
                             for row in stats) or not sum(row["misses"] for row in stats)):
            raise RuntimeError("Missing real cache activity or cache memory bounds exceeded")
        hits = sum(row["hits"] for row in stats)
        clears = sum(row["clears"] for row in stats)
        if capacity == 8192 and hits == 0:
            raise RuntimeError("No actual bind-group cache hits qualified")
        if capacity == 1 and clears == 0:
            raise RuntimeError("One-entry cache eviction was not exercised")
        evidence.extend((dump, stats_path))
        phases.append(dict(flag=flag, capacity=capacity, finite_bitwise_elements=9748480, fixtures=23,
            cpu_fallback_disabled=True, cache_hits=hits, cache_clears=clears,
            max_referenced_bytes=max(row["peak_referenced_bytes"] for row in stats),
            sampled_peak_gpu_gib=guard["peak_gpu_used_bytes"] / 2**30))
    result = dict(status="bind_group_inline_key_native_finite_bitwise_qualified",
        phases=phases, flag=FLAG, default_enabled=False, provider_sha256=manifest["provider_sha256"],
        reference="original unfused GPU FP16 arithmetic; not independent whole-model CPU",
        no_shader_or_arithmetic_changes=True, cache_entry_and_byte_limits_unchanged=True,
        model_eligible=True, performance_qualified=False, quality_qualified=False,
        source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "source_and_evidence_sha256"}, indent=2))
