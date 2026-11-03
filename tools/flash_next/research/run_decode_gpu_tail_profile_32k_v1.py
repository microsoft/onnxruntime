import collections
import json
from pathlib import Path
import subprocess

import run_cpu_decode_sample as parent
from cpu_decode_diagnostic_checks import verify_two_requests
from run_cpu_decode_32k_v2 import verify_placement
from run_decode_gpu_tail_profile_v1 import gpu_events


VARIANT = "decode-gpu-tail-profile-32k-v1"
TEMPLATE = "gpu-slice-combined32k-v1-context32768-retained-flag1-trial0-trace0-gpu7"


def profile_command(template):
    command = list(template)
    if command[:5] != ["taskset", "-c", "48-55", "bash", "-lc"] or command[-1].count(TEMPLATE) != 3:
        raise RuntimeError("Unexpected sealed GPU7 launch template")
    shell = command[-1].replace(TEMPLATE, VARIANT)
    if shell.count("--contexts 32768") != 1:
        raise RuntimeError("Expected the original 32K context")
    shell = parent.replace_once(shell, "--repetitions 3", "--repetitions 1")
    if "--profile-prefix" in shell or "--webgpu-validation-mode" in shell:
        raise RuntimeError("Template already contains profile or validation overrides")
    command[-1] = shell + (
        " --webgpu-validation-mode full"
        f" --profile-prefix /evidence/webgpu-perf-oct20/{VARIANT}-timestamps"
        " --profile-decode-after 252"
    )
    return command


def main():
    root = parent.ROOT
    prefix = root / VARIANT
    for path in root.glob(VARIANT + "*"):
        if path.name != VARIANT + "-run.log":
            raise FileExistsError(path)
    parent.verify_sources(root / (TEMPLATE + ".qualification.json"), "decode_stack_original_prefixes_qualified")
    template = json.loads((root / (TEMPLATE + ".guard.json")).read_text())
    baseline = json.loads((root / (TEMPLATE + ".json")).read_text())
    if template["status"] != "completed" or baseline["status"] != "completed":
        raise RuntimeError("Incomplete source template")
    command = profile_command(template["command"])
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    for name, key in (("model.onnx", "graph_sha256"), ("genai_config.json", "config_sha256")):
        if parent.sha(model / name) != baseline[key]:
            raise RuntimeError("Frozen model changed")
    subprocess.run(["free", "-h"], check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                    "--format=csv"], check=True)
    cache = parent.prepare_model_cache(model, prefix)
    guard = subprocess.Popen(["python3", str(parent.GUARD), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
        "--output-prefix", str(prefix), "--", *command])
    try:
        guard.wait(timeout=2500)
        if guard.returncode != 0:
            raise RuntimeError(f"GPU profiling guard failed: {guard.returncode}")
    finally:
        if guard.poll() is None:
            state_path = prefix.with_suffix(".guard.json")
            state = json.loads(state_path.read_text()) if state_path.exists() else None
            if state is not None and state["command"] == command and state.get("container_id"):
                with Path(str(prefix) + ".cleanup.log").open("a") as log:
                    parent.stop_container(state["container_id"], log)
            else:
                guard.terminate()
            guard.wait(timeout=60)
    state = json.loads(prefix.with_suffix(".guard.json").read_text())
    raw = json.loads(prefix.with_suffix(".json").read_text())
    verify_placement(raw, state)
    for key in ("genai_native", "plugin_sha256", "graph_sha256", "config_sha256", "load_experiment_flags"):
        if raw[key] != baseline[key]:
            raise RuntimeError(f"GPU diagnostic changed source configuration: {key}")
    if (raw["status"] != "completed" or raw["headline_eligible"] or not raw["profiling"]
            or raw["profiling_scope"] != "decode_only" or raw["profile_decode_after_tokens"] != 252
            or raw["webgpu_validation_mode_override"] != "full"):
        raise RuntimeError("Incorrect diagnostic completion or profile scope")
    references = json.loads((root / "int4-pending64-standard-contexts.json").read_text())["samples"]
    verify_two_requests(raw, [sample for sample in references if sample["context"] == 32768], 32768)
    profile_paths, summaries = [], []
    totals = collections.Counter()
    for sample in raw["samples"]:
        files = sample["decode_profile_files"]
        if sample["profiled_decode_tokens"] != 4 or len(files) < 4:
            raise RuntimeError("Missing four-token tail profiles")
        for file in files:
            path = Path(file.replace("/evidence/", str(root.parent) + "/", 1))
            if path.parent != root or not path.name.startswith(VARIANT + "-timestamps-"):
                raise RuntimeError("Profile escaped the diagnostic prefix")
            events = gpu_events(json.loads(path.read_text()))
            profile_paths.append(path)
            summaries.append(dict(file=str(path), warmup=sample["warmup"],
                                  gpu_events=len(events), summed_gpu_us=sum(event["dur"] for event in events)))
            if not sample["warmup"]:
                for event in events:
                    totals[event["name"]] += event["dur"]
    cache["after_run"] = parent.snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    sources = [Path(__file__), root / "run_decode_gpu_tail_profile_v1.py",
               root / "benchmark_gpu_slice.py", root / "benchmark_gpu_decode_stack.py",
               root / "cpu_decode_diagnostic_checks.py", prefix.with_suffix(".json"),
               prefix.with_suffix(".guard.json"), *profile_paths]
    outcome = dict(status="decode_gpu_tail_profile_32k_completed", diagnostic_only=True,
                   performance_qualified=False, quality_qualified=False, model_original_prefixes_agree=True,
                   context=32768, profile_after_tokens=252, profiled_tokens_per_request=4,
                   profiles=summaries, measured_kernel_us=totals.most_common(),
                   scope="native model GPU events only; not end-to-end latency or necessarily greedy helper events",
                   sampled_peak_gpu_gib=state["peak_gpu_used_bytes"] / 2**30,
                   source_and_evidence_sha256={str(path): parent.sha(path) for path in sources})
    with Path(str(prefix) + "-outcome.json").open("x") as stream:
        stream.write(json.dumps(outcome, indent=2) + "\n")


if __name__ == "__main__":
    main()
