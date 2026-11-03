import json
from pathlib import Path
import subprocess
import time

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_tokens
from shared_activation_guard import container_process_ids, gpu_compute_pids, stop_container


UUID = "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"


def read_live(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        print(f"Waiting for complete live snapshot {path.name}: {error}", flush=True)
        return None


def owned_profile_pid(state, container_pids, gpu_pids):
    assert state["status"] == "running" and state["gpu_index"] == 7 and state["gpu_uuid"] == UUID
    owners = set(state["own_compute_pids"])
    assert len(owners) == 1 and owners == set(gpu_pids)
    assert owners <= set(container_pids)
    return next(iter(owners))


def clone_command(command, baseline, name):
    for suffix in (".json", "-stats.jsonl", "-static-stats.jsonl"):
        assert command[-1].count(f"/evidence/webgpu-perf-oct20/{baseline}{suffix}") == 1
    return [*command[:-1], command[-1].replace(baseline, name)]


def main():
    prefix = ROOT / "cpu-perf-hc-up-parallel2-v1-gpu7-ctx2048-20261030"
    paths = [prefix.with_suffix(suffix) for suffix in
             (".json", ".guard.json", ".log", ".data", ".controller.log", ".qualification.json",
              ".perf.log", ".self.txt", ".inclusive.txt")]
    if any(path.exists() for path in paths):
        raise FileExistsError(prefix)
    baseline = "hc-up-parallel2-v1-chunk256-gpu7-standard2048-20261029"
    parent = json.loads((ROOT / "hc-up-parallel2-v1-contexts-20261029.qualification.json").read_text())
    native = json.loads((ROOT / "hc-up-parallel2-v1-native-gpu7-20261029.qualification.json").read_text())
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert parent["provider_sha256"] == native["provider_sha256"]
    command = clone_command(json.loads((ROOT / f"{baseline}.guard.json").read_text())["command"],
                            baseline, prefix.name)
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    cache = prepare_model_cache(model, prefix)
    guard_command = ["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                     "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
                     "--output-prefix", str(prefix), "--", *command]
    with prefix.with_suffix(".controller.log").open("x") as controller:
        process = subprocess.Popen(guard_command, stdout=controller, stderr=controller)
        profile_ok = False
        try:
            deadline = time.monotonic() + 600
            while process.poll() is None and time.monotonic() < deadline:
                raw = read_live(prefix.with_suffix(".json"))
                if raw and raw.get("samples"):
                    assert raw["samples"][0]["warmup"]
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError("Owned workload ended or timed out before the excluded warmup completed")
            state = read_live(prefix.with_suffix(".guard.json"))
            assert state is not None
            pid = owned_profile_pid(state, container_process_ids(state["container_id"]), gpu_compute_pids(UUID))
            perf_command = ["perf", "record", "-e", "cycles:u", "-F", "199",
                            "--call-graph", "dwarf,8192", "--pid", str(pid),
                            "--output", str(prefix.with_suffix(".data")), "--", "sleep", "12"]
            print(json.dumps(dict(stage="profiling_owned_post_warmup_workload", pid=pid,
                                  container_id=state["container_id"],
                                  includes_prefill_and_decode=True, timings_not_a_speed_claim=True)), flush=True)
            with prefix.with_suffix(".perf.log").open("x") as log:
                subprocess.run(perf_command, stdout=log, stderr=log, check=True)
            profile_ok = True
        finally:
            if not profile_ok and process.poll() is None:
                state = read_live(prefix.with_suffix(".guard.json"))
                if state and state.get("container_id"):
                    stop_container(state["container_id"], controller)
            try:
                process.wait(timeout=960)
            except subprocess.TimeoutExpired:
                print("Owned profiling guard did not finish within its deadline; stopping its container", flush=True)
                state = read_live(prefix.with_suffix(".guard.json"))
                if state and state.get("container_id"):
                    stop_container(state["container_id"], controller)
                process.wait(timeout=60)
                raise
        assert process.returncode == 0
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw = json.loads(prefix.with_suffix(".json").read_text())
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    verify_metadata(raw, guard, native, parent, False)
    assert guard["command"] == command
    for flag in ("ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_HC_UP_PREMIX", "ORT_WEBGPU_QSA_SCORE16"):
        assert raw["load_experiment_flags"][flag] == "1"
    controls = json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())
    verify_tokens(raw, [sample for sample in controls["samples"] if sample["context"] == 2048], 2048, 256)
    data = prefix.with_suffix(".data")
    assert data.stat().st_size > 0
    for name, mode in ((".self.txt", "--no-children"), (".inclusive.txt", "--children")):
        with prefix.with_suffix(name).open("x") as report, prefix.with_suffix(".perf.log").open("a") as log:
            subprocess.run(["perf", "report", "--stdio", mode, "--call-graph", "none",
                            "--sort", "dso,symbol", "--percent-limit", "0.5", "--input", str(data)],
                           stdout=report, stderr=log, check=True)
    evidence = [path for path in paths if path != prefix.with_suffix(".qualification.json")]
    report = dict(status="post_warmup_cpu_profile_original_sequences_qualified", provider_sha256=native["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
        gpu_index=7, gpu_uuid=UUID, sampled_pid=pid, perf_command=perf_command,
        scope="12-second post-warmup repeated requests; includes prefill and decode",
        original_sequence_comparisons=1024, all_original_sequences_match=True, zero_major_faults=True,
        instrumented_not_a_speed_claim=True, standard_headline_eligible=False,
        source_sha256={path.name: sha(path) for path in evidence})
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "scope", "all_original_sequences_match")}), flush=True)


if __name__ == "__main__":
    main()
