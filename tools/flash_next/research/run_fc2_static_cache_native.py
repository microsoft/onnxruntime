import json
import math
from pathlib import Path
import re
import statistics

from build_qmoe_node_replay import replace_once, sha
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_fc2_rank_waves_native import guarded
from run_gpu_slice_model import verify_sources


BUILD = Path("/home/sayanshaw/repos/qwen38-int4-fc2-static-cache/build-v1")
FLAG = "ORT_WEBGPU_QMOE_FC2_STATIC_CACHE"


def command_from(path, flag):
    command = json.loads(path.read_text())["command"].copy()
    command[-1] = replace_once(command[-1],
        "/workspace/qwen38-int4-fc2-private-a/build-v1/libonnxruntime_providers_webgpu.so",
        "/workspace/qwen38-int4-fc2-static-cache/build-v1/libonnxruntime_providers_webgpu.so")
    command[-1], count = re.subn(r"ORT_WEBGPU_QMOE_FC2_PRIVATE_A=[01]", "ORT_WEBGPU_QMOE_FC2_PRIVATE_A=1", command[-1])
    if count != 1:
        raise RuntimeError("Nonunique private-A selector")
    command[-1] = f"export {FLAG}={flag} ORT_WEBGPU_QMOE_SUBGROUP_GATE=1; " + command[-1]
    return command


if __name__ == "__main__":
    output = ROOT / "fc2-static-cache-v1-native-operator-outcome.json"
    if output.exists():
        raise FileExistsError(output)
    parent_native = ROOT / "fc2-private-a-v1-native-outcome.json"
    native = verify_sources(parent_native, "fc2_private_a_native_finite_bitwise_and_shader_routing_qualified")
    parent_operator = ROOT / "fc2-private-a-v1-operator-screen.json"
    operator = verify_sources(parent_operator, "fc2_private_a_operator_screen_completed")
    manifest = json.loads((BUILD / "build-manifest.json").read_text())
    for p, digest in manifest["source_sha256"].items():
        if sha(host_path(p)) != digest:
            raise RuntimeError(f"Changed static-cache build source: {p}")
    if (sha(BUILD / "libonnxruntime_providers_webgpu.so") != manifest["provider_sha256"] or
            manifest["parent_provider_sha256"] != native["provider_sha256"]):
        raise RuntimeError("Wrong candidate/parent provider")
    evidence = [Path(__file__), ROOT / "run_fc2_rank_waves_native.py", GUARD, parent_native, parent_operator,
                BUILD / "build-manifest.json", BUILD / "libonnxruntime_providers_webgpu.so"]
    phases = []
    template = ROOT / "fc2-private-a-v1-native-flag0-gpu0.guard.json"
    for flag in (0, 1, 2):
        prefix = ROOT / f"fc2-static-cache-v1-native-flag{flag}-gpu0"
        dump = Path(str(prefix) + "-shaders.txt")
        if dump.exists():
            raise FileExistsError(dump)
        command = command_from(template, flag)
        command[-1] = command[-1].replace("fc2-private-a-v1-native-flag0-gpu0", prefix.name)
        child, guard = guarded(prefix, command)
        log = prefix.with_suffix(".log").read_text()
        evidence.extend(prefix.with_suffix(s) for s in (".log", ".guard.json"))
        if flag == 2:
            if child.returncode != 1 or f"{FLAG} must be 0 or 1" not in log:
                raise RuntimeError("Invalid static-cache selector did not reject")
            phases.append(dict(flag=flag, invalid_selector_rejected=True))
            continue
        child.check_returncode()
        if guard["status"] != "completed" or log.count("PASS changing_inputs_and_routes ") != 23 or "PASS total_exact_elements=9748480; CPU fallback disabled" not in log:
            raise RuntimeError("Incomplete original-unfused-GPU arithmetic qualification")
        shader = dump.read_text()
        marker = "=== WebGPU Shader code [QmoeFC2StaticCacheBitcastF16"
        if (marker in shader) != bool(flag):
            raise RuntimeError("Wrong static-cache shader routing")
        if flag:
            begin = shader.index(marker)
            selected = shader[begin:shader.index("] End ===", begin)]
            if (any(f"let cached_a{i} =" not in selected for i in range(8)) or
                    selected.count("sum += dot(cached_a") != 4 or
                    "var cached_a:" in selected or "var<workgroup> tile_a" in selected):
                raise RuntimeError("Named vectors/ordered constant-word updates missing")
        evidence.append(dump)
        phases.append(dict(flag=flag, finite_bitwise_elements=9748480, fixtures=23,
            reference="original unfused GPU FP16, not independent whole-model CPU",
            sampled_peak_gpu_gib=guard["peak_gpu_used_bytes"] / 1024**3))
    wall, profiles = [], []
    for flag, phase, profiling in ((0, "before", False), (1, "candidate", False), (0, "after", False),
                                   (0, "profile-control", True), (1, "profile-candidate", True), (0, "profile-after", True)):
        prefix = ROOT / f"fc2-static-cache-v1-{phase}-gpu0"
        source_phase = phase if phase != "profile-after" else "profile-control"
        source_prefix = ROOT / f"fc2-private-a-v1-{source_phase}-gpu0"
        command = command_from(source_prefix.with_suffix(".guard.json"), flag)
        command[-1] = command[-1].replace(source_prefix.name, prefix.name)
        child, guard = guarded(prefix, command)
        child.check_returncode()
        raw = json.loads(prefix.with_suffix(".json").read_text())
        if (guard["status"] != "completed" or raw["status"] != "qmoe_operator_benchmark_completed" or
                raw["profiling"] != profiling or raw["measured_runs"] != 128 or raw["warmups"] != 16 or
                raw["measured_major_faults"] != 0 or raw["output_hash"] != operator["output_hash"]):
            raise RuntimeError("Wrong operator benchmark scope/faults/output bytes")
        evidence.extend(prefix.with_suffix(s) for s in (".json", ".log", ".guard.json"))
        if not profiling:
            wall.append(raw["run_seconds"])
            continue
        profile = host_path(raw["profile_path"])
        events = json.loads(profile.read_text())
        name = "QmoeFC2StaticCacheBitcastF16" if flag else "QmoeFC2PrivateABitcastF16"
        fc2 = [e for e in events if e.get("cat") == "Kernel" and e.get("name", "").endswith("&" + name)]
        gates = [e for e in events if e.get("cat") == "Kernel" and e.get("name", "").endswith("&QmoeHierarchicalGate1Token")]
        if len(fc2) != 144 or len(gates) != 144 or any(e["pid"] != -1 or e["tid"] != -1 or e["dur"] <= 0 or
                                                      ":320,1,1:" not in e["args"]["cache_key"] for e in fc2):
            raise RuntimeError("Wrong actual FC2 GPU event count/workgroup/gate")
        durations = [e["dur"] for e in fc2[16:]]
        profiles.append(dict(phase=phase, flag=flag, measured_gpu_runs=128,
            median_gpu_us=statistics.median(durations), mean_gpu_us=statistics.mean(durations),
            gpu_us=durations))
        evidence.append(profile)
    wall_ratios = [math.sqrt(b * a) / c for b, c, a in zip(*wall)]
    gpu_ratios = [math.sqrt(b * a) / c for b, c, a in zip(*[p["gpu_us"] for p in profiles])]
    gpu_speed = statistics.geometric_mean(gpu_ratios)
    supported = gpu_speed > 1.01 and sum(r > 1 for r in gpu_ratios) >= 120
    result = dict(status="fc2_static_cache_native_operator_screen_completed", native_phases=phases,
        provider_sha256=manifest["provider_sha256"], output_hash=operator["output_hash"],
        profiles=profiles, matched_gpu_speed_ratio=gpu_speed, positive_gpu_runs=sum(r > 1 for r in gpu_ratios),
        matched_operator_wall_speed_ratio=statistics.geometric_mean(wall_ratios),
        operator_support=supported, held_no_unchanged_retry=not supported,
        physical_register_residency_verified=False, actual_hierarchical_gate_routing_verified=True,
        model_eligible=False, performance_qualified=False, quality_qualified=False,
        operator_ratios_are_not_model_decode_ratios=True,
        source_and_evidence_sha256={str(p): sha(p) for p in evidence})
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("profiles", "source_and_evidence_sha256", "native_phases")}), flush=True)
