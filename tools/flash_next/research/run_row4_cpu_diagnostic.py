import json
import os
from pathlib import Path
import signal
import subprocess
import time

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_prefill_flags
from run_qmoe_prefill_rows4_standard import DIRECTORY, TEMPLATE, standard_command


PREFIX = ROOT / "row4-cpu-decode-diagnostic-v3-gpu7-20261031"


def benchmark_pid(prefix):
    candidates = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.geteuid():
                continue
            arguments = (directory / "cmdline").read_bytes().decode().split("\0")
        except FileNotFoundError:
            continue
        if (any(argument.endswith("/benchmark_row4_cpu_diagnostic.py") for argument in arguments) and
                any(argument.endswith(prefix.name + ".json") for argument in arguments)):
            candidates.append(int(directory.name))
    if len(candidates) > 1:
        raise RuntimeError(f"Ambiguous owned diagnostic process: {candidates}")
    return candidates[0] if candidates else None


def diagnostic_command(template, provider, prefix):
    command = standard_command(template, provider, prefix, 2048, False)
    marker = "--tokens 256 --repetitions 3"
    assert command[-1].count(marker) == 1
    command[-1] = command[-1].replace(marker, "--tokens 768 --repetitions 1")
    assert command[-1].count("/benchmark_webgpu_int4.py") == 1
    command[-1] = command[-1].replace("/benchmark_webgpu_int4.py", "/benchmark_row4_cpu_diagnostic.py")
    return command


def main():
    for suffix in (".json", ".guard.json", ".log", ".perf.data", ".perf.log", ".diagnostic.json",
                   ".cpu-phases.jsonl", ".cpu-maps.txt"):
        if PREFIX.with_suffix(suffix).exists():
            raise FileExistsError(PREFIX.with_suffix(suffix))
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    native_path = ROOT / "qmoe-prefill-rows4-v2-native-gpu7-20261031.qualification.json"
    parent, native = (json.loads(path.read_text()) for path in (parent_path, native_path))
    provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == native["provider_sha256"] == parent["provider_sha256"]
    template_path = ROOT / f"{TEMPLATE}.guard.json"
    command = diagnostic_command(json.loads(template_path.read_text())["command"], provider, PREFIX)
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    cache = prepare_model_cache(model, PREFIX)
    guard_command = ["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                     "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
                     "--output-prefix", str(PREFIX), "--", *command]
    guard_process = subprocess.Popen(guard_command)
    perf_process = None
    perf_path = PREFIX.with_suffix(".perf.data")
    try:
        deadline = time.monotonic() + 180
        pid = None
        while guard_process.poll() is None and time.monotonic() < deadline:
            pid = benchmark_pid(PREFIX)
            if pid is not None:
                break
            time.sleep(0.25)
        if pid is None:
            raise RuntimeError("Owned benchmark process was not found for CPU profiling")
        with PREFIX.with_suffix(".perf.log").open("x") as perf_log:
            perf_process = subprocess.Popen(
                ["perf", "record", "-e", "cpu-clock:u", "-F", "199", "--clockid", "mono",
                 "--call-graph", "dwarf,8192", "-p", str(pid),
                 "-o", str(perf_path)], stdout=perf_log, stderr=subprocess.STDOUT)
            code = guard_process.wait(timeout=900)
            if code != 0:
                raise subprocess.CalledProcessError(code, guard_command)
            if perf_process.poll() is None:
                perf_process.send_signal(signal.SIGINT)
            perf_code = perf_process.wait(timeout=30)
            if perf_code != 0:
                raise subprocess.CalledProcessError(perf_code, perf_process.args)
    finally:
        if perf_process is not None and perf_process.poll() is None:
            perf_process.send_signal(signal.SIGINT)
            perf_process.wait(timeout=30)
        if guard_process.poll() is None:
            guard_process.wait(timeout=900)
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = (json.loads(PREFIX.with_suffix(suffix).read_text()) for suffix in (".json", ".guard.json"))
    verify_metadata(raw, guard, native, parent, False, chunk=1024)
    verify_prefill_flags(raw, 1024)
    assert raw["load_experiment_flags"]["ORT_WEBGPU_QMOE_GROUPED_PREFILL"] == "1"
    assert raw["load_experiment_flags"]["ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD"] == "0"
    assert guard["command"] == command and perf_path.stat().st_size > 0
    references = [sample for sample in json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())["samples"]
                  if sample["context"] == 2048]
    assert len(raw["samples"]) == 2
    phases_path = PREFIX.with_suffix(".cpu-phases.jsonl")
    phases = [json.loads(line) for line in phases_path.read_text().splitlines()]
    assert len(phases) == len(raw["samples"])
    for phase, sample in zip(phases, raw["samples"]):
        assert sample["context"] == 2048 and len(sample["output_token_ids"]) == sample["output_tokens"] == 768
        reference = next(reference for reference in references if reference["warmup"] == sample["warmup"])
        assert sample["prompt_sha256"] == reference["prompt_sha256"]
        assert sample["output_token_ids"][:256] == reference["output_token_ids"][:256]
        assert phase["context"] == sample["context"] and phase["output_tokens"] == sample["output_tokens"]
        assert phase["request_start"] < phase["decode_start"] < phase["decode_end"]
        assert abs(sample["decode_seconds"] - (phase["decode_end"] - phase["decode_start"])) < 1e-8
    report = dict(status="cpu_diagnostic_original_prefixes_qualified_not_performance_standard",
                  cpu_profiled=True, benchmark_pid=pid, gpu_index=7, output_tokens=768,
                  original_prefix_comparisons=512, original_output_prefixes_match=True,
                  zero_major_faults=True, provider_sha256=native["provider_sha256"],
                  headline_eligible=False, quality_qualified=False,
                  decode_phases=phases, perf_clock="CLOCK_MONOTONIC",
                  source_sha256={str(path): sha(path) for path in
                                 (PREFIX.with_suffix(".json"), PREFIX.with_suffix(".guard.json"),
                                  perf_path, PREFIX.with_suffix(".perf.log"), phases_path, Path(__file__),
                                  ROOT / "benchmark_row4_cpu_diagnostic.py", PREFIX.with_suffix(".cpu-maps.txt"))})
    with PREFIX.with_suffix(".diagnostic.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "benchmark_pid", "original_prefix_comparisons")}),
          flush=True)


if __name__ == "__main__":
    main()
