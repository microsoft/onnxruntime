import json
import math
from pathlib import Path
import shlex
import statistics
import time

from build_qmoe_node_replay import sha
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_fc2_rank_waves_native import guarded


BUILD = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-small-cta/build-v2")
BENCHMARK = BUILD.parent / "operator-v1/native_benchmark"
FIXTURE = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/fixtures-v1/"
               "postnorm_shared0_rows1_scale2_alpha1.0.onnx")
NATIVE = ROOT / "hc-post-norm-small-cta-v2-native-gpu7.qualification.json"
FLAG = "ORT_WEBGPU_HC_POST_NORM_SMALL_CTA"
VARIANT = "hc-small-cta-v2-operator-v1"


def command_for(prefix, flag, profiling):
    if flag not in (0, 1):
        raise ValueError("Operator timing accepts only valid selectors")
    def container(path):
        path = str(path)
        if path.startswith("/home/sayanshaw/repos/"):
            return path.replace("/home/sayanshaw/repos/", "/workspace/", 1)
        if path.startswith(str(ROOT.parent) + "/"):
            return path.replace(str(ROOT.parent) + "/", "/evidence/", 1)
        raise ValueError("Path is outside approved mounts")
    arguments = [BENCHMARK, BUILD / "libonnxruntime_providers_webgpu.so", FIXTURE, prefix.with_suffix(".json")]
    profile = container(Path(str(prefix) + "-ort-profile")) if profiling else "-"
    shell = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd; export "
        "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
        "ORT_WEBGPU_HC_POST_NORM_PROJECTION=1 ORT_WEBGPU_HC_PROJECTION_SILU=1 "
        "ORT_WEBGPU_HC_EXACT_DENSE=1 ORT_WEBGPU_EXACT_DENSE_MID=1 "
        f"ORT_WEBGPU_QMOE_FC2_PRIVATE_A=0 {FLAG}={flag}; exec "
        + shlex.join([*(container(path) for path in arguments), profile]))
    return ["taskset", "-c", "0-7", "bash", "-lc", shell]


def measured_gpu(events, flag):
    name = "HCPostNormProjectionSmallCtaF16" if flag else "HCPostNormProjectionF16"
    selected = [event for event in events if event.get("cat") == "Kernel"
                and event.get("name", "").endswith("&" + name)]
    expected_size = ":128,1,1:" if flag else ":256,1,1:"
    if len(selected) != 144 or any(
            event.get("pid") != -1 or event.get("tid") != -1 or event.get("ph") != "X"
            or not math.isfinite(event["dur"]) or event["dur"] <= 0
            or expected_size not in event["args"]["cache_key"] for event in selected):
        raise RuntimeError("Incorrect actual HC GPU program/count/workgroup")
    return [event["dur"] for event in selected[16:]]


def main():
    output = ROOT / (VARIANT + "-outcome.json")
    if output.exists():
        raise FileExistsError(output)
    deadline = time.monotonic() + 1200
    while not NATIVE.exists():
        guard_path = NATIVE.with_suffix("").with_suffix(".guard.json")
        if guard_path.exists():
            native_guard = json.loads(guard_path.read_text())
            if native_guard["status"] not in ("running", "completed", "waiting"):
                raise RuntimeError("Native qualification stopped before operator admission")
        if time.monotonic() >= deadline:
            raise TimeoutError("Native qualification did not produce an admission within budget")
        time.sleep(1)
    native = json.loads(NATIVE.read_text())
    build = json.loads((BUILD / "build-manifest.json").read_text())
    if (native["status"] != "hc_post_norm_small_cta_four_outputs_finite_bitwise_and_replay_qualified"
            or native["finite_bitwise_comparisons"] != 607143936 or native["fixtures"] != 36
            or native["provider_sha256"] != build["provider_sha256"]
            or not build["original_hc_object_byte_identical"] or not build["baseline_byte_identical"]
            or sha(BUILD / "libonnxruntime_providers_webgpu.so") != build["provider_sha256"]
            or sha(FIXTURE) != native["fixture_sha256"][FIXTURE.name]):
        raise RuntimeError("Incomplete native/original-object/provider/fixture admission")
    for path, digest in build["source_sha256"].items():
        if sha(host_path(path)) != digest:
            raise RuntimeError(f"Changed build source: {path}")
    evidence = [Path(__file__), ROOT / "hc_small_cta_operator_benchmark.cpp", BENCHMARK,
                GUARD, ROOT / "run_fc2_rank_waves_native.py", NATIVE,
                BUILD / "build-manifest.json", BUILD / "libonnxruntime_providers_webgpu.so", FIXTURE]
    wall = []
    profiles = []
    output_hash = None
    for flag, phase, profiling in ((0, "before", False), (1, "candidate", False), (0, "after", False),
                                   (0, "profile-before", True), (1, "profile-candidate", True),
                                   (0, "profile-after", True)):
        prefix = ROOT / f"{VARIANT}-{phase}-gpu0"
        command = command_for(prefix, flag, profiling)
        child, guard = guarded(prefix, command)
        child.check_returncode()
        raw = json.loads(prefix.with_suffix(".json").read_text())
        if (guard["status"] != "completed" or raw["status"] != "hc_operator_benchmark_completed"
                or raw["warmups"] != 16 or raw["measured_runs"] != 128
                or len(raw["run_seconds"]) != 128 or raw["profiling"] != profiling
                or raw["measured_major_faults"] != 0
                or any(not math.isfinite(value) or value <= 0 for value in raw["run_seconds"])):
            raise RuntimeError("Wrong operator sample scope, faults, or completion")
        if output_hash is None:
            output_hash = raw["output_hash"]
        if raw["output_hash"] != output_hash:
            raise RuntimeError("All-four-output byte hash differs between operator brackets")
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".json", ".log", ".guard.json"))
        if not profiling:
            wall.append(raw["run_seconds"])
        else:
            path = host_path(raw["profile_path"])
            durations = measured_gpu(json.loads(path.read_text()), flag)
            profiles.append(dict(phase=phase, flag=flag, median_gpu_us=statistics.median(durations),
                                 gpu_us=durations, measured_gpu_runs=128))
            evidence.append(path)
    gpu_ratios = [math.sqrt(before * after) / candidate
                  for before, candidate, after in zip(*(profile["gpu_us"] for profile in profiles), strict=True)]
    wall_ratios = [math.sqrt(before * after) / candidate for before, candidate, after in zip(*wall, strict=True)]
    speed = statistics.geometric_mean(gpu_ratios)
    positives = sum(ratio > 1 for ratio in gpu_ratios)
    supported = speed > 1.01 and positives >= 120
    report = dict(status="hc_small_cta_native_operator_screen_completed", provider_sha256=build["provider_sha256"],
                  output_hash=output_hash, profiles=profiles, matched_gpu_speed_ratio=speed,
                  positive_gpu_runs=positives, matched_operator_wall_speed_ratio=statistics.geometric_mean(wall_ratios),
                  operator_support=supported, held_no_unchanged_retry=not supported, model_eligible=False,
                  performance_qualified=False, quality_qualified=False, default_enabled=False,
                  operator_ratios_are_not_model_decode_ratios=True,
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("source_and_evidence_sha256", "profiles")}), flush=True)


if __name__ == "__main__":
    main()
