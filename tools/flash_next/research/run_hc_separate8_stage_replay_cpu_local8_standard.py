import argparse
import json
from pathlib import Path
import subprocess
import time

from build_cpu_greedy_argmax_lanes8 import FLAG as CPU_FLAG
from build_hc_separate8_stage_replay import FLAG
from prepare_hc_separate8_stage_replay_proofs import benchmark_command
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_genai_allocator_workers import OLD_RUNTIME
from run_row4_default_cpu_policy import verify_cpu_policy
from run_row4_cpu_locality import placement_command
from run_qmoe_prefill_rows4_standard import DIRECTORY, TEMPLATE, standard_command, verify_result


RUNTIME = Path("/home/sayanshaw/repos/qwen38-int4-cpu-greedy-argmax/build-v1")
VARIANT = "hc-separate8-stage-replay-cpu-local8-v1"
STAMP = "20261103"



def verify_workers(raw, mode):
    assert mode == "control"
    verify_cpu_policy(raw, "local8")
    masks = raw["thread_cpu_affinity"]
    return dict(total_observed_threads=len(masks),
                individually_pinned_threads=sum(len(mask) == 1 for mask in masks.values()))


def prefix_for(mode, context, full, trial):
    return ROOT / f"{VARIANT}-{mode}-gpu7-{'full-' if full else ''}standard{context}-trial{trial}-{STAMP}"


def model_command(template, provider, prefix, context, full, mode):
    assert mode in ("control", "candidate")
    command = benchmark_command(standard_command(template, provider, prefix, context, full))
    assert command[-1].count(OLD_RUNTIME) == 1
    command[-1] = command[-1].replace(OLD_RUNTIME, str(RUNTIME).replace("/home/sayanshaw/repos", "/workspace"))
    command[-1] = f"export {CPU_FLAG}=1; " + command[-1]
    command[-1] = f"export {FLAG}={int(mode == 'candidate')}; " + command[-1]
    command[-1] += " --inherit-worker-cpu-affinity"
    return placement_command(command, "local8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("control", "candidate"), required=True)
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--trial", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if args.mode == "candidate" and args.trial == 2:
        parser.error("Trial2 is reserved for the final control")
    prefix = prefix_for(args.mode, args.context, args.full_validation, args.trial)
    for suffix in (".json", ".guard.json", ".log", ".qualification.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    build_path = RUNTIME / "build-manifest.json"
    build = json.loads(build_path.read_text())
    assert build["baseline_relink_byte_identical"] and build["baseline_search_rebuild_byte_identical"]
    runtime = RUNTIME / "libonnxruntime-genai.so"
    runtime_hash = build["runtime_sha256"]
    assert sha(runtime) == runtime_hash
    cpu_path = ROOT / "cpu-greedy-argmax-lanes8-v1-cpu.qualification.json"
    cpu = json.loads(cpu_path.read_text())
    assert cpu["status"] == "cpu_greedy_lanes8_selected_ids_exact" and cpu["runtime_sha256"] == build["runtime_sha256"]
    for name, digest in cpu["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    invalid_path = ROOT / "cpu-greedy-argmax-lanes8-v1-invalid-gpu7-attempt1.qualification.json"
    invalid = json.loads(invalid_path.read_text())
    assert invalid["status"] == "cpu_greedy_lanes8_invalid_selector_rejected"
    for name, digest in invalid["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    native_path = ROOT / "qmoe-prefill-rows4-v2-native-gpu7-20261031.qualification.json"
    parent, native = (json.loads(path.read_text()) for path in (parent_path, native_path))
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == parent["provider_sha256"] == native["provider_sha256"]
    for name, digest in parent["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    locality_path = ROOT / "row4-default-cpu-policy-v1-all-contexts-20261031.qualification.json"
    locality = json.loads(locality_path.read_text())
    assert locality["status"] == "repeated_default_policy_cpu_locality_supported_all_contexts"
    for name, digest in locality["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    gpu_path = ROOT / "hc-separate8-stage-replay-v1-candidate-gpu7-full-profile2048-trial0-20261103.qualification.json"
    gpu = json.loads(gpu_path.read_text())
    assert gpu["status"] == "original_sequences_qualified"
    assert gpu["phase"] == "profile" and gpu["context"] == 2048
    assert gpu["mode"] == "candidate" and gpu["full_validation"]
    assert gpu["experiment_flag"] == FLAG and gpu["experiment_flag_value"] == 1
    assert gpu["result"]["program_counts"]["HCSeparate8ProjectionF16"] == 94
    assert gpu["result"]["program_counts"]["HCSeparate8NormF16"] == 94
    assert gpu["result"]["program_counts"].get("HCPostNormProjectionF16", 0) == 0
    assert gpu["result"]["program_counts"]["MatMulNBitsBlock32F16"] == 159
    cpu_all_path = ROOT / "cpu-greedy-argmax-lanes8-local8-v1-all-contexts-20261101.qualification.json"
    cpu_all = json.loads(cpu_all_path.read_text())
    assert cpu_all["status"] == "repeated_cpu_greedy_local8_supported_all_contexts"
    for proof in (gpu, cpu_all):
        for name, digest in proof["source_sha256"].items():
            assert sha(Path(name)) == digest, name
    gpu_build = Path("/home/sayanshaw/repos/qwen38-int4-hc-separate8-stage-replay/build-v1")
    gpu_build_path = gpu_build / "build-manifest.json"
    gpu_manifest = json.loads(gpu_build_path.read_text())
    assert gpu_manifest["provider_sha256"] == gpu["provider_sha256"]
    for name, digest in gpu_manifest["source_sha256"].items():
        assert sha(gpu_build / name) == digest, name
    if args.mode == "candidate":
        provider = gpu_build / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == (gpu_manifest["provider_sha256"] if args.mode == "candidate" else parent["provider_sha256"])
    native = dict(provider_sha256=sha(provider), baseline_provider_sha256=parent["provider_sha256"])
    prerequisites = []
    if not args.full_validation:
        prerequisites.append(prefix_for(args.mode, args.context, True, args.trial).with_suffix(".qualification.json"))
    if args.mode == "candidate" and args.trial == 1:
        prerequisites.append(ROOT / f"{VARIANT}-context{args.context}-screen-{STAMP}.qualification.json")
    if args.context != 2048:
        previous = 2048 if args.context == 8192 else 8192
        prerequisites.append(ROOT / f"{VARIANT}-context{previous}-repeat-screen-{STAMP}.qualification.json")
    for path in prerequisites:
        proof = json.loads(path.read_text())
        expected = ("repeated_hc_separate8_stage_replay_cpu_local8_decode_gain_both_modes" if "repeat-screen" in path.name else
                    "initial_hc_separate8_stage_replay_cpu_local8_decode_gain_both_modes" if "-screen-" in path.name else
                    "hc_separate8_stage_replay_cpu_local8_original_prefixes_qualified")
        assert proof["status"] == expected
        expected_provider = gpu_manifest["provider_sha256"] if "-screen-" in path.name else native["provider_sha256"]
        assert proof["provider_sha256"] == expected_provider
        if "-screen-" in path.name:
            assert proof["runtime_sha256"] == build["runtime_sha256"]
        else:
            assert proof["runtime_sha256"] == runtime_hash
        for name, digest in proof["source_sha256"].items():
            assert sha(Path(name)) == digest, name
    template_path = ROOT / f"{TEMPLATE}.guard.json"
    command = model_command(json.loads(template_path.read_text())["command"], provider, prefix,
                            args.context, args.full_validation, args.mode)
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    cache = prepare_model_cache(model, prefix)
    started_ns = time.time_ns()
    subprocess.run(["python3", str(ROOT / "shared_activation_metrics_guard.py"), "--gpu-index", "7",
                    "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
                    "--output-prefix", str(prefix), "--", *command], check=True)
    completed_ns = time.time_ns()
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = (json.loads(prefix.with_suffix(suffix).read_text()) for suffix in (".json", ".guard.json"))
    assert guard["command"] == command and raw["genai_native"]["sha256"] == runtime_hash
    assert raw["comparison"] is None and not raw["profiling"] and raw["profiling_scope"] is None
    workers = verify_workers(raw, "control")
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [sample for sample in json.loads(controls_path.read_text())["samples"]
                  if sample["context"] == args.context]
    result = verify_result(raw, guard, native, parent, references, args.context, args.full_validation)
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    sources = [Path(__file__), ROOT / "build_cpu_greedy_argmax_lanes8.py",
               ROOT / "run_genai_allocator_workers.py", ROOT / "benchmark_webgpu_int4_hc_separate8_stage_replay.py",
               ROOT / "prepare_cpu_greedy_argmax_lanes8_local8.py",
               ROOT / "run_row4_default_cpu_policy.py", ROOT / "run_row4_cpu_locality.py",
               ROOT / "cpu_local_windows.py",
               parent_path, native_path, build_path, runtime, cpu_path, invalid_path, locality_path, gpu_path, cpu_all_path, gpu_build_path, template_path, controls_path,
               *prerequisites, *[prefix.with_suffix(suffix) for suffix in (".json", ".guard.json", ".log")]]
    report = dict(status="hc_separate8_stage_replay_cpu_local8_original_prefixes_qualified", mode=args.mode,
        context=args.context, trial=args.trial, full_validation=args.full_validation,
        runtime_sha256=runtime_hash, provider_sha256=native["provider_sha256"],
        original_runtime_control=False, original_provider_control=args.mode == "control",
        cpu_argmax_flag=CPU_FLAG, cpu_argmax_flag_value=1,
        experiment_flag=FLAG, experiment_flag_value=int(args.mode == "candidate"),
        workers=workers, original_output_prefixes_match=True, zero_major_faults=True,
        main_affinity_unchanged=True, manual_affinity_override=True,
        model_and_arithmetic_unchanged=True, cpu_placement="local8",
        matched_cpu_and_worker_policy_in_both_modes=True, quality_qualified=False, headline_eligible=False,
        result=result, started_ns=started_ns, completed_ns=completed_ns, cache_preflight=cache,
        source_sha256={str(path): sha(path) for path in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
        ("status", "mode", "context", "full_validation", "result", "workers")}), flush=True)


if __name__ == "__main__":
    main()
