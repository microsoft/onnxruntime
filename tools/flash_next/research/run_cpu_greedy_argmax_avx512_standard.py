import argparse
import json
from pathlib import Path
import subprocess
import time

from build_qmoe_node_replay import sha
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from prepare_webgpu_int4_metrics import BENCHMARK, GUARD, ROOT
from qualify_cpu_greedy_argmax_avx512 import BUILD
from run_cpu_greedy_argmax_avx512_initial_model import FLAG
from run_webgpu_int4_metrics import PROVIDER, RUNTIME, TEMPLATE, model_command, verify_result, verify_workers


VARIANT = "cpu-greedy-argmax-avx512-local8-v1"
STAMP = "20261103"


def prefix_for(mode, context, full, trial):
    return ROOT / f"{VARIANT}-{mode}-gpu7-{'full-' if full else ''}standard{context}-trial{trial}-{STAMP}"


def command_for(template, prefix, context, full, mode):
    if mode not in ("control", "avx512"):
        raise ValueError(f"Unknown CPU argmax mode: {mode}")
    command = model_command(template, PROVIDER, prefix, context, full, "columns8")
    parent = str(RUNTIME).replace("/home/sayanshaw/repos", "/workspace")
    assert command[-1].count(parent) == 1
    command[-1] = command[-1].replace(parent, str(BUILD).replace("/home/sayanshaw/repos", "/workspace"))
    benchmark = "/evidence/webgpu-perf-oct20/benchmark_webgpu_int4.py"
    assert command[-1].count(benchmark) == 1
    command[-1] = command[-1].replace(benchmark, "/evidence/webgpu-perf-oct20/" + BENCHMARK.name)
    command[-1] = f"export {FLAG}={int(mode == 'avx512')}; " + command[-1]
    return command


def verify_proof(path, expected_status=None):
    proof = json.loads(path.read_text())
    if expected_status is not None:
        assert proof["status"] == expected_status, path
    for name, digest in proof["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("control", "avx512"), required=True)
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--trial", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if args.mode == "avx512" and args.trial == 2:
        parser.error("Trial2 is reserved for the final control")
    prefix = prefix_for(args.mode, args.context, args.full_validation, args.trial)
    for suffix in (".json", ".guard.json", ".log", ".qualification.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    cpu_path = ROOT / "cpu-greedy-argmax-avx512-v1-cpu.qualification.json"
    invalid_path = ROOT / "cpu-greedy-argmax-avx512-v1-invalid-gpu7-attempt2.qualification.json"
    initial_path = ROOT / "cpu-greedy-argmax-avx512-v1-full-prefix2048-gpu7-attempt2.qualification.json"
    best_path = ROOT / "dense-columns8-double-buffer-cpu-local8-v1-all-contexts-20261101.qualification.json"
    cpu = verify_proof(cpu_path, "cpu_greedy_avx512_selected_ids_exact")
    verify_proof(invalid_path, "cpu_greedy_avx512_invalid_selector_rejected")
    verify_proof(initial_path, "cpu_greedy_avx512_original_prefixes_qualified")
    best = verify_proof(best_path, "repeated_columns8_double_buffer_cpu_local8_supported_all_contexts")
    assert sha(BUILD / "libonnxruntime-genai.so") == cpu["runtime_sha256"]
    assert sha(PROVIDER) == best["provider_sha256"]
    prerequisites = []
    if not args.full_validation:
        prerequisites.append(prefix_for(args.mode, args.context, True, args.trial).with_suffix(".qualification.json"))
    if args.mode == "avx512" and args.trial == 1:
        prerequisites.append(ROOT / f"{VARIANT}-context{args.context}-screen-{STAMP}.qualification.json")
    if args.context != 2048:
        previous = 2048 if args.context == 8192 else 8192
        prerequisites.append(ROOT / f"{VARIANT}-context{previous}-repeat-screen-{STAMP}.qualification.json")
    for path in prerequisites:
        expected = ("repeated_cpu_greedy_decode_gain_both_modes" if "repeat-screen" in path.name else
                    "initial_cpu_greedy_decode_gain_both_modes" if "-screen-" in path.name else
                    "cpu_greedy_avx512_original_prefixes_qualified")
        proof = verify_proof(path, expected)
        assert proof["runtime_sha256"] == cpu["runtime_sha256"]
        assert proof["provider_sha256"] == best["provider_sha256"]
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    parent = json.loads(parent_path.read_text())
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [sample for sample in json.loads(controls_path.read_text())["samples"]
                  if sample["context"] == args.context]
    template_path = ROOT / f"{TEMPLATE}.guard.json"
    command = command_for(json.loads(template_path.read_text())["command"], prefix,
                          args.context, args.full_validation, args.mode)
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    for native_execution in ROOT.glob("hc-parallel16-v1-native*execution.json"):
        assert json.loads(native_execution.read_text())["status"] != "running", "Owned native proof still active"
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["free", "-h"], check=True)
    subprocess.run(["df", "-h", str(model)], check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                    "--format=csv,noheader"], check=True)
    started_ns = time.time_ns()
    subprocess.run(["python3", str(GUARD), "--gpu-index", "7", "--idle-seconds", "3",
                    "--wait-seconds", "15", "--run-seconds", "2400",
                    "--output-prefix", str(prefix), "--", *command], check=True)
    completed_ns = time.time_ns()
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = [json.loads(prefix.with_suffix(suffix).read_text()) for suffix in (".json", ".guard.json")]
    assert guard["command"] == command
    assert raw["genai_native"]["sha256"] == cpu["runtime_sha256"]
    assert raw["load_experiment_flags"][FLAG] == str(int(args.mode == "avx512"))
    assert raw["load_experiment_flags"]["ORT_GENAI_CPU_GREEDY_ARGMAX_LANES8"] == "1"
    assert not raw["profiling"] and raw["comparison"] is None
    workers = verify_workers(raw, "control")
    result = verify_result(raw, guard, dict(provider_sha256=best["provider_sha256"]),
                           parent, references, args.context, args.full_validation)
    sources = [Path(__file__), ROOT / "run_cpu_greedy_argmax_avx512_initial_model.py",
               ROOT / "run_webgpu_int4_metrics.py", BENCHMARK, GUARD, cpu_path,
               invalid_path, initial_path, best_path, parent_path, controls_path, template_path,
               BUILD / "libonnxruntime-genai.so", BUILD / "build-manifest.json", PROVIDER,
               Path(cache["evidence_path"]), *prerequisites,
               *[prefix.with_suffix(suffix) for suffix in (".json", ".guard.json", ".log")]]
    report = dict(status="cpu_greedy_avx512_original_prefixes_qualified",
        context=args.context, full_validation=args.full_validation, mode=args.mode, trial=args.trial,
        runtime_sha256=cpu["runtime_sha256"], provider_sha256=best["provider_sha256"],
        experiment_flag=FLAG, experiment_flag_value=int(args.mode == "avx512"),
        same_runtime_both_sides=True, control_is_lanes8_with_avx512_disabled=True,
        cpu_placement="local8", workers=workers, result=result,
        started_ns=started_ns, completed_ns=completed_ns,
        original_output_prefixes_match=True, zero_major_faults=True, quality_qualified=False,
        default_enabled=False, source_sha256={str(path): sha(path) for path in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "mode", "context", "result")}), flush=True)


if __name__ == "__main__":
    main()
