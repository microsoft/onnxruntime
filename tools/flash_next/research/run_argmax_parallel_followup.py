import json
import math
from pathlib import Path
import re
import shlex
import statistics

import onnx
from onnx import TensorProto, helper

from build_argmax_parallel_bitwise import FLAG
from build_qmoe_node_replay import sha
from run_argmax_parallel_bitwise import BUILD, DATA, ROOT, UUID, guarded, host_path, profile_counts, workspace


def main():
    output = ROOT / "argmax-parallel-bitwise-v4-followup-outcome.json"
    if output.exists():
        raise FileExistsError(output)
    native_path = ROOT / "argmax-parallel-bitwise-v4-native-outcome.json"
    native = json.loads(native_path.read_text())
    if native["status"] != "argmax_parallel_native_exact_and_dispatch_supported":
        raise RuntimeError("Native exactness/dispatch prerequisite missing")
    for path, digest in native["source_and_evidence_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Changed native evidence: {path}")
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    provider, binary = BUILD / "libonnxruntime_providers_webgpu.so", BUILD / "bitwise_argmax_compare"
    if sha(provider) != native["provider_sha256"] or sha(binary) != manifest["bitwise_comparator_sha256"]:
        raise RuntimeError("Changed provider/comparator")
    evidence = [Path(__file__), native_path, manifest_path, provider, binary,
                ROOT / "run_argmax_parallel_bitwise.py"]
    head = f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
    memory_flags = (" ORT_WEBGPU_SMALL_UPLOAD_WRITE_BUFFER=1 ORT_WEBGPU_SMALL_READBACK_CACHE=1"
                    " ORT_WEBGPU_UNIFORM_ARENA=1")
    pipeline = DATA / "f16-cast-f32-argmax-rows1-n248320.onnx"
    if pipeline.exists():
        raise FileExistsError(pipeline)
    graph = helper.make_graph(
        [helper.make_node("Cast", ["x"], ["cast"], to=TensorProto.FLOAT),
         helper.make_node("ArgMax", ["cast"], ["y"], axis=1, keepdims=0, select_last_index=0)],
        "cast-argmax-cpu-semantics", [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 248320])],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, pipeline)
    evidence.append(pipeline)
    prefix = ROOT / "argmax-parallel-bitwise-v4-cast-pipeline-gpu4"
    args = [workspace(binary), "compare", workspace(provider), workspace(pipeline),
            f"/evidence/webgpu-perf-oct20/{prefix.name}-profile"]
    code, log = guarded(head + f"export {FLAG}=1{memory_flags}; taskset -c 48 " + shlex.join(args), prefix)
    evidence.extend((prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")))
    if code != 0 or "ARGMAX rows=1 N=248320 comparisons=64 mismatches=0" not in log:
        raise RuntimeError(f"GPU F16->F32 conversion/argmax does not preserve CPU IDs: {prefix}")
    profiles = re.findall(r"^PROFILE (.+)$", log, re.M)
    if len(profiles) != 1:
        raise RuntimeError("Missing cast pipeline profile")
    profile = host_path(profiles[0])
    if profile_counts(profile) != {"ArgMaxParallelPartial": 64, "ArgMaxParallelFinish": 64}:
        raise RuntimeError("Missing cast pipeline selected dispatch coverage")
    evidence.append(profile)
    invalid_prefix = ROOT / "argmax-parallel-bitwise-v4-invalid-gpu4"
    invalid_code, invalid_log = guarded(head + f"export {FLAG}=2; " + shlex.join(args), invalid_prefix)
    evidence.extend((invalid_prefix.with_suffix(".log"), invalid_prefix.with_suffix(".guard.json")))
    if invalid_code != 1 or f"{FLAG} must be 0 or 1" not in invalid_log:
        raise RuntimeError("Invalid selector did not explicitly fail")
    timings = {"f16": [], "f32": []}
    for trial in range(3):
        prefix = ROOT / f"argmax-parallel-bitwise-v4-unprofiled-trial{trial}-gpu4"
        commands = []
        for dtype in timings:
            fixture = DATA / f"{dtype}-rows1-n248320.onnx"
            evidence.append(fixture)
            for mode in (0, 1, 0):
                commands.append(shlex.join(["env", f"{FLAG}={mode}", "taskset", "-c", "48",
                    workspace(binary), "timing", workspace(provider), workspace(fixture), "unused"]))
        code, log = guarded(head + f"export {memory_flags.strip()}; " + "; ".join(commands), prefix)
        evidence.extend((prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")))
        matches = re.findall(r"^ARGMAX rows=1 N=248320 comparisons=1032 mismatches=0 iterations=1024 "
                             r"warmup=8 mean_us=(\S+) median_us=(\S+)$", log, re.M)
        if code != 0 or "FAIL" in log or len(matches) != 6:
            raise RuntimeError(f"Unprofiled primitive timing/CPU-ID validation failed: {prefix}")
        for offset, dtype in enumerate(timings):
            means = [float(match[0]) for match in matches[offset * 3:offset * 3 + 3]]
            if not all(math.isfinite(x) and x > 0 for x in means):
                raise RuntimeError("Invalid primitive timings")
            timings[dtype].append(dict(before_mean_us=means[0], candidate_mean_us=means[1],
                after_mean_us=means[2], ratio=math.sqrt(means[0] * means[2]) / means[1]))
        print(json.dumps(dict(trial=trial, timings={k: v[-1] for k, v in timings.items()})), flush=True)
    result = dict(status="primitive_followup_completed", gpu_uuid=UUID, cast_pipeline_cpu_ids_exact=True,
        cast_pipeline_comparisons=64, cast_pipeline_includes_subnormals=True, invalid_selector_rejected=True,
        timings=timings, matched_speed_ratios={k: statistics.geometric_mean(x["ratio"] for x in v)
                                             for k, v in timings.items()},
        cpu_affinity="48", same_provider_both_sides=True, profiling=False,
        warmup_runs_per_process=8, measured_runs_per_process=1024,
        timings_include_host_upload_run_and_readback=True,
        control_is_serial_gpu_argmax_not_current_model_cpu_greedy=True,
        natural_model_tps_measured=False, runtime_wiring_complete=False,
        model_eligible=False, default_enabled=False, provider_sha256=sha(provider),
        source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], ratios=result["matched_speed_ratios"])), flush=True)


if __name__ == "__main__":
    main()
