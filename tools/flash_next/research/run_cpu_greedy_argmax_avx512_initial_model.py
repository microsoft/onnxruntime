import argparse
import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from prepare_webgpu_int4_metrics import BENCHMARK, GUARD, ROOT
from qualify_cpu_greedy_argmax_avx512 import BUILD
from run_webgpu_int4_metrics import PROVIDER, RUNTIME, TEMPLATE, model_command, verify_result, verify_workers


FLAG = "ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512"


def command_for(prefix, flag):
    template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
    command = model_command(template, PROVIDER, prefix, 2048, True, "columns8")
    parent_runtime = str(RUNTIME).replace("/home/sayanshaw/repos", "/workspace")
    assert command[-1].count(parent_runtime) == 1
    command[-1] = command[-1].replace(parent_runtime, str(BUILD).replace("/home/sayanshaw/repos", "/workspace"))
    old = "/evidence/webgpu-perf-oct20/benchmark_webgpu_int4.py"
    assert command[-1].count(old) == 1
    command[-1] = f"export {FLAG}={flag}; " + command[-1].replace(old, "/evidence/webgpu-perf-oct20/" + BENCHMARK.name)
    if flag == 2:
        assert command[-1].count("--tokens 256") == command[-1].count("--repetitions 3") == 1
        command[-1] = command[-1].replace("--tokens 256", "--tokens 2").replace("--repetitions 3", "--repetitions 1")
    return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("--attempt must be positive")
    cpu_path = ROOT / "cpu-greedy-argmax-avx512-v1-cpu.qualification.json"
    cpu = json.loads(cpu_path.read_text())
    assert cpu["status"] == "cpu_greedy_avx512_selected_ids_exact"
    best_path = ROOT / "dense-columns8-double-buffer-cpu-local8-v1-all-contexts-20261101.qualification.json"
    best = json.loads(best_path.read_text())
    for proof in (cpu, best):
        for name, digest in proof["source_sha256"].items():
            assert sha(Path(name)) == digest, name
    assert sha(BUILD / "libonnxruntime-genai.so") == cpu["runtime_sha256"]
    assert sha(PROVIDER) == best["provider_sha256"]
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    parent = json.loads(parent_path.read_text())
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [sample for sample in json.loads(controls_path.read_text())["samples"] if sample["context"] == 2048]
    sources = [Path(__file__), cpu_path, best_path, parent_path, controls_path, BENCHMARK, GUARD,
               BUILD / "libonnxruntime-genai.so", BUILD / "build-manifest.json", PROVIDER]
    for flag, phase in ((2, "invalid"), (1, "full-prefix2048")):
        prefix = ROOT / f"cpu-greedy-argmax-avx512-v1-{phase}-gpu7-attempt{args.attempt}"
        for suffix in (".json", ".guard.json", ".log", ".qualification.json"):
            if prefix.with_suffix(suffix).exists():
                raise FileExistsError(prefix.with_suffix(suffix))
        command = command_for(prefix, flag)
        cache = prepare_model_cache(model, prefix)
        subprocess.run(["free", "-h"], check=True)
        subprocess.run(["df", "-h", str(model)], check=True)
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                        "--format=csv,noheader"], check=True)
        result = subprocess.run(["python3", str(GUARD), "--gpu-index", "7", "--idle-seconds", "3",
                                 "--wait-seconds", "15", "--run-seconds", "2400",
                                 "--output-prefix", str(prefix), "--", *command])
        cache["after_run"] = snapshot_model_cache(model)
        Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
        raw, guard = [json.loads(prefix.with_suffix(suffix).read_text()) for suffix in (".json", ".guard.json")]
        assert guard["command"] == command
        assert raw["genai_native"]["sha256"] == cpu["runtime_sha256"]
        assert raw["plugin_sha256"] == best["provider_sha256"]
        assert raw["load_experiment_flags"][FLAG] == str(flag)
        if flag == 2:
            assert result.returncode == guard["exit_code"] == 1
            assert guard["status"] == raw["status"] == "failed"
            assert f"Invalid value for environment variable {FLAG}: 2." in raw["error"]
            status = "cpu_greedy_avx512_invalid_selector_rejected"
        else:
            result.check_returncode()
            verify_workers(raw, "control")
            verify_result(raw, guard, dict(provider_sha256=best["provider_sha256"]), parent,
                          references, 2048, True)
            status = "cpu_greedy_avx512_original_prefixes_qualified"
        sources.extend(prefix.with_suffix(suffix) for suffix in (".json", ".guard.json", ".log"))
        proof = dict(status=status, command=command, runtime_sha256=cpu["runtime_sha256"],
            provider_sha256=best["provider_sha256"], cpu_argmax_flag=FLAG, cpu_argmax_flag_value=flag,
            phase=phase, context=2048, full_validation=True, cpu_placement="local8",
            original_output_prefixes_match=flag == 1, quality_qualified=False,
            isolated_model_speedup_claimed=False, default_enabled=False,
            source_sha256={str(path): sha(path) for path in sources})
        with prefix.with_suffix(".qualification.json").open("x") as stream:
            stream.write(json.dumps(proof, indent=2) + "\n")
        print(json.dumps({key: proof[key] for key in ("status", "phase", "runtime_sha256")}), flush=True)
