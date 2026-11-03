import json
from pathlib import Path
import re
import subprocess

from build_qmoe_node_replay import sha
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_webgpu_int4_metrics import RUNTIME, TEMPLATE, model_command, verify_result, verify_workers


BUILD = Path("/home/sayanshaw/repos/qwen38-int4-gpu-greedy-argmax/build-v3")
BENCHMARK = ROOT / "benchmark_gpu_greedy_argmax.py"
FLAG = "ORT_GENAI_WEBGPU_GREEDY_ARGMAX"


def command_for(prefix, flag, trace):
    template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    command = model_command(template, provider, prefix, 2048, True, "columns8")
    parent_runtime = "/workspace/" + str(RUNTIME.relative_to("/home/sayanshaw/repos"))
    if command[-1].count(parent_runtime) != 1:
        raise RuntimeError("Parent runtime replacement not unique")
    command[-1] = command[-1].replace(parent_runtime, "/workspace/" + str(BUILD.relative_to("/home/sayanshaw/repos")))
    benchmark = "/evidence/webgpu-perf-oct20/benchmark_webgpu_int4.py"
    if command[-1].count(benchmark) != 1:
        raise RuntimeError("Benchmark replacement not unique")
    command[-1] = command[-1].replace(benchmark, "/evidence/webgpu-perf-oct20/" + BENCHMARK.name)
    command[-1] = (f"export {FLAG}={flag} ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE={trace} "
                   "ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE=1 ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512=0; " + command[-1])
    if flag == 2:
        if command[-1].count("--tokens 256") != 1 or command[-1].count("--repetitions 3") != 1:
            raise RuntimeError("Invalid selector short probe replacement not unique")
        command[-1] = command[-1].replace("--tokens 256", "--tokens 2").replace("--repetitions 3", "--repetitions 1")
    return command


def main():
    output = ROOT / "gpu-greedy-argmax-v3-initial-model-outcome.json"
    if output.exists():
        raise FileExistsError(output)
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["supported_runtime_and_provider_relinked_identically"]:
        raise RuntimeError("Supported parent relink provenance missing")
    for path, digest in manifest["source_sha256"].items():
        if sha(host_path(path)) != digest:
            raise RuntimeError(f"Changed runtime/provider source: {path}")
    for filename, key in (("libonnxruntime-genai.so", "runtime_sha256"),
                          ("libonnxruntime_providers_webgpu.so", "provider_sha256")):
        if sha(BUILD / filename) != manifest[key]:
            raise RuntimeError(f"Changed built binary: {filename}")
    primitive_path = ROOT / "argmax-parallel-bitwise-v4-followup-outcome.json"
    primitive = json.loads(primitive_path.read_text())
    if primitive["status"] != "primitive_followup_completed" or not primitive["cast_pipeline_cpu_ids_exact"]:
        raise RuntimeError("Primitive/cast/selector prerequisite missing")
    for path, digest in primitive["source_and_evidence_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Changed primitive evidence: {path}")
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    parent = json.loads(parent_path.read_text())
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    if sha(model / "model.onnx") != parent["graph_sha256"] or sha(model / "genai_config.json") != parent["config_sha256"]:
        raise RuntimeError("Frozen model changed")
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [x for x in json.loads(controls_path.read_text())["samples"] if x["context"] == 2048]
    sources = [Path(__file__), BENCHMARK, GUARD, ROOT / "prepare_gpu_greedy_argmax_benchmark.py",
               ROOT / "prepare_gpu_greedy_argmax_v3.py", ROOT / "test_gpu_greedy_argmax.py",
               manifest_path, primitive_path, parent_path, controls_path,
               BUILD / "libonnxruntime-genai.so", BUILD / "libonnxruntime_providers_webgpu.so"]
    phases = []
    for flag, phase in ((2, "invalid"), (0, "default-full-prefix2048"), (1, "selected-full-prefix2048")):
        prefix = ROOT / f"gpu-greedy-argmax-v3-{phase}-gpu7"
        for suffix in (".json", ".guard.json", ".log", ".qualification.json"):
            if prefix.with_suffix(suffix).exists():
                raise FileExistsError(prefix.with_suffix(suffix))
        command = command_for(prefix, flag, 1)
        cache = prepare_model_cache(model, prefix)
        subprocess.run(["free", "-h"], check=True)
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader"], check=True)
        child = subprocess.run(["python3", str(GUARD), "--gpu-index", "7", "--idle-seconds", "3",
            "--wait-seconds", "15", "--run-seconds", "2400", "--output-prefix", str(prefix), "--", *command])
        cache["after_run"] = snapshot_model_cache(model)
        Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
        raw, guard = [json.loads(prefix.with_suffix(suffix).read_text()) for suffix in (".json", ".guard.json")]
        if guard["command"] != command or raw["genai_native"]["sha256"] != manifest["runtime_sha256"]:
            raise RuntimeError("Wrong runtime/command in model evidence")
        if raw["plugin_sha256"] != manifest["provider_sha256"] or raw["load_experiment_flags"][FLAG] != str(flag):
            raise RuntimeError("Wrong provider/selector in model evidence")
        sources.extend(prefix.with_suffix(suffix) for suffix in (".json", ".guard.json", ".log"))
        if flag == 2:
            if child.returncode != 1 or f"Invalid value for environment variable {FLAG}: 2." not in raw["error"]:
                raise RuntimeError("Runtime invalid selector did not explicitly fail")
            phases.append(dict(phase=phase, invalid_selector_rejected=True))
            continue
        child.check_returncode()
        verify_workers(raw, "control")
        verify_result(raw, guard, dict(provider_sha256=manifest["provider_sha256"]),
                      parent, references, 2048, True)
        log = prefix.with_suffix(".log").read_text()
        selected = re.findall(r"^WEBGPU_GREEDY_ARGMAX rows=1 vocabulary=248320 token_bytes=8$", log, re.M)
        fallbacks = log.count("WEBGPU_GREEDY_ARGMAX_CPU_FALLBACK")
        expected = sum(sample["output_tokens"] for sample in raw["samples"]) if flag else 0
        if len(selected) != expected or fallbacks != 0:
            raise RuntimeError(f"Missing complete selected/default routing: selected={len(selected)} expected={expected} fallback={fallbacks}")
        phases.append(dict(phase=phase, original_output_prefixes_match=True,
                           selected_argmax_calls=len(selected), cpu_fallback_calls=fallbacks,
                           trace_enabled=True, natural_speedup_claimed=False))
    result = dict(status="gpu_greedy_argmax_default_selected_prefixes_and_routing_qualified", phases=phases,
        runtime_sha256=manifest["runtime_sha256"], provider_sha256=manifest["provider_sha256"],
        context=2048, trace_enabled=True, quality_qualified=False, model_eligible=False,
        performance_qualified=False, default_enabled=False,
        source_and_evidence_sha256={str(path): sha(path) for path in sources})
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], phases=phases)), flush=True)


if __name__ == "__main__":
    main()
