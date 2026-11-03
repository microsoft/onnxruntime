import json
from pathlib import Path
import re
import signal
import subprocess
import time

import run_cpu_decode_sample as parent


VARIANT = "cpu-decode-deep-stack-v1"


def main():
    root = parent.ROOT
    prefix = root / VARIANT
    for path in root.glob(VARIANT + "*"):
        if path.name != VARIANT + "-run.log":
            raise FileExistsError(path)
    template = json.loads((root / (parent.TEMPLATE + ".guard.json")).read_text())
    baseline = json.loads((root / (parent.TEMPLATE + ".json")).read_text())
    parent.verify_sources(root / (parent.TEMPLATE + ".qualification.json"), "fused_slice_original_prefixes_qualified")
    command = list(template["command"])
    if command[:5] != ["taskset", "-c", "16-23", "bash", "-lc"] or command[-1].count(parent.TEMPLATE) != 3:
        raise RuntimeError("Unexpected baseline launch template")
    command[-1] = parent.replace_once(command[-1], "benchmark_gpu_decode_stack.py", "benchmark_cpu_decode_sample.py")
    command[-1] = command[-1].replace(parent.TEMPLATE, VARIANT)
    command[-1] = f"export ORT_WEBGPU_CPU_PERF_MARKER_PREFIX=/evidence/webgpu-perf-oct20/{VARIANT}-markers; " + command[-1]
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    for name, key in (("model.onnx", "graph_sha256"), ("genai_config.json", "config_sha256")):
        if parent.sha(model / name) != baseline[key]:
            raise RuntimeError("Frozen model changed")
    cache = parent.prepare_model_cache(model, prefix)
    subprocess.run(["free", "-h"], check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv"], check=True)
    guard = subprocess.Popen(["python3", str(parent.GUARD), "--gpu-index", "1", "--idle-seconds", "3",
        "--wait-seconds", "15", "--run-seconds", "1200", "--output-prefix", str(prefix), "--", *command])
    perf = None
    container_id = None
    deadline = time.monotonic() + 1200
    markers = {phase: root / f"{VARIANT}-markers.{phase}.json" for phase in ("start", "stop")}
    try:
        while not markers["start"].exists():
            if guard.poll() is not None:
                raise RuntimeError(f"Guard exited before the decode marker: {guard.returncode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("No decode marker within budget")
            time.sleep(.02)
        state = json.loads(prefix.with_suffix(".guard.json").read_text())
        if (state["status"] != "running" or state["command"] != command or state["gpu_index"] != 1 or
                state["gpu_uuid"] != template["gpu_uuid"] or
                not state["container_name"].startswith("webgpu-nvfp4-")):
            raise RuntimeError("Unexpected owned guard identity")
        container_id = state["container_id"]
        start = json.loads(markers["start"].read_text())
        pid = parent.owned_model_pid(container_id, start)
        perf_command = ["taskset", "-c", "8-15", "perf", "record", "-e", "cpu-clock:u",
            "-F", "199", "--clockid", "mono", "--call-graph", "dwarf,32768", "-p", str(pid),
            "-o", str(prefix) + ".perf.data"]
        with Path(str(prefix) + ".perf.log").open("x") as log:
            perf = subprocess.Popen(perf_command, stdout=log, stderr=log)
            while not markers["stop"].exists():
                if perf.poll() is not None or guard.poll() is not None:
                    raise RuntimeError("Sampler or model exited before the decode end marker")
                if time.monotonic() >= deadline:
                    raise TimeoutError("No decode end marker within budget")
                time.sleep(.02)
            stop = json.loads(markers["stop"].read_text())
            perf.send_signal(signal.SIGINT)
            perf_returncode = perf.wait(timeout=30)
            if perf_returncode not in (0, -signal.SIGINT, 128 + signal.SIGINT):
                raise RuntimeError(f"perf record failed with {perf_returncode}; see its diagnostic log")
        interval = f'{start["monotonic_ns"] / 1e9:.9f},{stop["monotonic_ns"] / 1e9:.9f}'
        report_path = Path(str(prefix) + ".perf-report.txt")
        with report_path.open("x") as report:
            subprocess.run(["perf", "report", "--stdio", "--no-children", "--call-graph", "none",
                "--sort", "dso,symbol", "--percent-limit", "0.5", "--time", interval,
                "--symfs", f"/proc/{pid}/root", "-i", str(prefix) + ".perf.data"],
                stdout=report, check=True, timeout=30)
        text = report_path.read_text()
        samples = re.search(r"# Samples: ([0-9]+)\s+of event 'cpu-clock:u'", text)
        if not samples or int(samples[1]) < 100 or "# Total Lost Samples: 0" not in text:
            raise RuntimeError("Insufficient or lost user-space CPU samples")
        guard.wait(timeout=max(1, deadline - time.monotonic()))
        if guard.returncode != 0:
            raise RuntimeError(f"Diagnostic guard failed: {guard.returncode}")
    finally:
        if perf is not None and perf.poll() is None:
            perf.send_signal(signal.SIGINT)
            perf.wait(timeout=30)
        if guard.poll() is None:
            if container_id is None and prefix.with_suffix(".guard.json").exists():
                state = json.loads(prefix.with_suffix(".guard.json").read_text())
                if state["command"] == command:
                    container_id = state.get("container_id")
            if container_id:
                with Path(str(prefix) + ".cleanup.log").open("a") as log:
                    parent.stop_container(container_id, log)
            else:
                guard.terminate()
            guard.wait(timeout=60)
    raw = json.loads(prefix.with_suffix(".json").read_text())
    state = json.loads(prefix.with_suffix(".guard.json").read_text())
    parent.verify_placement(raw, state)
    for key in ("genai_native", "plugin_sha256", "graph_sha256", "config_sha256", "load_experiment_flags"):
        if raw[key] != baseline[key]:
            raise RuntimeError(f"CPU diagnostic changed the supported baseline: {key}")
    if raw["status"] != "completed" or not raw["cpu_sampling"]["enabled"] or raw["headline_eligible"]:
        raise RuntimeError("Incomplete diagnostic or natural timing incorrectly enabled")
    references = json.loads((root / "int4-pending64-standard-contexts.json").read_text())["samples"]
    parent.verify_tokens(raw, [s for s in references if s["context"] == 2048], 2048, 256)
    cache["after_run"] = parent.snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    outcome = dict(status="cpu_decode_sampling_completed", host_pid=pid, owned_container_id=container_id,
        perf_command=perf_command, perf_returncode=perf_returncode, interval=interval, samples=int(samples[1]),
        gpu_index=1, gpu_uuid=state["gpu_uuid"], diagnostic_only=True, performance_qualified=False,
        quality_qualified=False, sampled_peak_gpu_gib=state["peak_gpu_used_bytes"] / 2**30,
        model_original_prefixes_agree=True,
        source_and_evidence_sha256={str(path): parent.sha(path) for path in (
            Path(__file__), Path(parent.__file__), root / "benchmark_cpu_decode_sample.py",
            root / "benchmark_gpu_slice.py", root / (parent.TEMPLATE + ".guard.json"),
            prefix.with_suffix(".json"), prefix.with_suffix(".guard.json"),
            Path(str(prefix) + ".perf.data"), report_path, *markers.values())})
    Path(str(prefix) + "-outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps({key: value for key, value in outcome.items() if key != "source_and_evidence_sha256"}, indent=2))


if __name__ == "__main__":
    main()
