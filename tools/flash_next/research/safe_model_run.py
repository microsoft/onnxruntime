import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from nvml_monitor import get_nvml

DOCKER = ["sudo", "-n", "docker", "-H", "unix:///home/sayanshaw/docker-qwen38.sock"]


def gpu_compute_pids(gpu_uuid):
    return get_nvml().compute_pids(gpu_uuid)


def container_process_ids(container_id):
    output = subprocess.check_output([*DOCKER, "top", container_id, "-eo", "pid"], text=True)
    return {int(line.strip()) for line in output.splitlines()[1:] if line.strip()}


def stop_container(container_id, log):
    for attempt in range(3):
        stopped = subprocess.run([*DOCKER, "stop", "--time", "0", container_id],
                                 capture_output=True, text=True)
        log.write(stopped.stdout + stopped.stderr)
        log.flush()
        if stopped.returncode == 0:
            return True
        if f"No such container: {container_id}" in stopped.stderr:
            return False
        if "sudo: PAM account management error:" in stopped.stderr and attempt < 2:
            time.sleep(1)
            continue
        stopped.check_returncode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--idle-seconds", type=float, default=60)
    parser.add_argument("--wait-seconds", type=float, default=1800)
    parser.add_argument("--run-seconds", type=float, default=3600)
    parser.add_argument("--gpu-index", type=int, choices=range(8), default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or min(args.idle_seconds, args.wait_seconds, args.run_seconds) < 0:
        parser.error("A command and nonnegative time limits are required")
    docker = DOCKER
    monitor = get_nvml()
    name = "webgpu-nvfp4-" + uuid.uuid4().hex[:12]
    status_path = args.output_prefix.with_suffix(".guard.json")
    log_path = args.output_prefix.with_suffix(".log")
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "--id=" + str(args.gpu_index), "--query-gpu=uuid", "--format=csv,noheader"], text=True,
    ).strip()
    state = {"status": "waiting_for_idle_gpu", "container_name": name, "command": command,
             "gpu_index": args.gpu_index, "gpu_uuid": gpu_uuid,
             "gpu_monitor": "process-lifetime read-only NVML; no clock/power changes",
             "guard_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "monitor_sha256": hashlib.sha256(
                 Path(__file__).with_name("nvml_monitor.py").read_bytes()).hexdigest(),
             "runtime_image": os.environ.get("WEBGPU_RUNTIME_IMAGE",
                                            "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel")}

    def persist():
        status_path.write_text(json.dumps(state, indent=2) + "\n")

    def compute_pids():
        return gpu_compute_pids(gpu_uuid)

    persist()
    deadline = time.monotonic() + args.wait_seconds
    idle_since = None
    while True:
        free = monitor.free_memory_bytes(gpu_uuid) // (1024 * 1024)
        if free >= 130000 and not compute_pids():
            if idle_since is None:
                idle_since = time.monotonic()
            if time.monotonic() - idle_since >= args.idle_seconds:
                break
        else:
            idle_since = None
        if time.monotonic() >= deadline:
            state.update(status="blocked_gpu_busy", gpu_free_mib=free)
            persist()
            return 75
        time.sleep(1)
    state["gpu_snapshot"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv"], text=True,
    )
    state["status"] = "running"
    persist()
    environment = dict(os.environ, WEBGPU_CONTAINER_NAME=name, WEBGPU_CREATE_ONLY="1",
                       WEBGPU_DEVICE_INDEX=str(args.gpu_index))
    runner = Path(__file__).resolve().parent.parent / "run_fp8_webgpu_container.sh"
    started = time.monotonic()
    own_pid = None
    with log_path.open("w") as log:
        created = subprocess.run(["bash", str(runner), *command], env=environment,
                                 capture_output=True, text=True)
        log.write(created.stdout + created.stderr)
        log.flush()
        created.check_returncode()
        container_id = created.stdout.strip().splitlines()[-1]
        state["container_id"] = container_id
        persist()
        started = time.monotonic()
        process = subprocess.Popen([*docker, "start", "--attach", container_id], stdout=log, stderr=log)

        def stop_owned():
            stopped = stop_container(container_id, log)
            process.wait(timeout=60)
            state["cleanup_status"] = "stopped" if stopped else "already_exited"
            persist()
            return stopped

        try:
            while process.poll() is None:
                if not own_pid:
                    result = subprocess.run(
                        [*docker, "inspect", container_id, "--format", "{{.State.Pid}}"],
                        capture_output=True, text=True, check=True,
                    )
                    own_pid = int(result.stdout.strip())
                foreign = set()
                own_compute = set()
                container_pids = None
                for pid in compute_pids():
                    if pid == own_pid:
                        own_compute.add(pid)
                        continue
                    try:
                        cgroup = Path(f"/proc/{pid}/cgroup").read_text()
                    except FileNotFoundError:
                        log.write(f"GPU process {pid} exited during ownership check\n")
                        log.flush()
                        continue
                    if container_id in cgroup:
                        own_compute.add(pid)
                    else:
                        if container_pids is None:
                            container_pids = container_process_ids(container_id)
                        if pid in container_pids:
                            own_compute.add(pid)
                        elif Path(f"/proc/{pid}").exists():
                            foreign.add(pid)
                if state.get("own_compute_pids") != sorted(own_compute):
                    state["own_compute_pids"] = sorted(own_compute)
                    state["container_init_pid"] = own_pid
                    persist()
                timed_out = time.monotonic() - started > args.run_seconds
                if foreign or timed_out:
                    state.update(status="yielded_to_training" if foreign else "run_timed_out",
                                 foreign_compute_pids=sorted(foreign))
                    if foreign:
                        state["container_process_ids_at_yield"] = sorted(
                            container_pids if container_pids is not None else container_process_ids(container_id)
                        )
                        state["foreign_process_metadata"] = {}
                        for pid in foreign:
                            try:
                                metadata = Path(f"/proc/{pid}/status").read_text().splitlines()
                            except FileNotFoundError:
                                metadata = ["Process exited before metadata capture"]
                            state["foreign_process_metadata"][str(pid)] = [
                                line for line in metadata if line.startswith(("Name:", "PPid:", "NSpid:", "Process exited"))
                            ]
                        state["compute_apps_at_yield"] = subprocess.check_output(
                            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
                             "--format=csv,noheader"], text=True,
                        )
                    persist()
                    if not stop_owned():
                        state.update(status="completed" if process.returncode == 0 else "failed",
                                     exit_code=process.returncode)
                        persist()
                        return process.returncode
                    persist()
                    return 75
                time.sleep(.5)
            state.update(status="completed" if process.returncode == 0 else "failed",
                         exit_code=process.returncode, elapsed_seconds=time.monotonic() - started)
            persist()
            return process.returncode
        finally:
            try:
                if process.poll() is None:
                    stop_owned()
            finally:
                if process.poll() is None:
                    state["cleanup_status"] = "failed_process_still_running"
                if state["status"] == "running":
                    state["status"] = "guard_failed"
                persist()


if __name__ == "__main__":
    raise SystemExit(main())
