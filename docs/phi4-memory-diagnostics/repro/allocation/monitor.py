# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Bounded child controller, independent of the original benchmark framework."""

from __future__ import annotations

import json
import queue
import shlex
import statistics
import subprocess
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import config as settings
import support as s


def gpu_sample(nv, handle, pid):
    begin = time.monotonic_ns()
    memory = nv.nvmlDeviceGetMemoryInfo(handle)
    memory_end = time.monotonic_ns()
    processes, errors = {}, []
    for kind, query in (
        ("compute", nv.nvmlDeviceGetComputeRunningProcesses),
        ("graphics", nv.nvmlDeviceGetGraphicsRunningProcesses),
    ):
        try:
            entries = query(handle)
        except (nv.NVMLError_NotSupported, nv.NVMLError_NoPermission) as error:
            errors.append({"kind": kind, "error": str(error)})
            continue
        for entry in entries:
            used = entry.usedGpuMemory
            valid = isinstance(used, int) and 0 <= used <= memory.total
            record = processes.setdefault(
                entry.pid, {"pid": entry.pid, "readings": [], "kinds": [], "unavailable": False}
            )
            record["kinds"].append(kind)
            record["unavailable"] |= not valid
            if valid:
                record["readings"].append(int(used))
    for record in processes.values():
        record["used_bytes"] = None if record["unavailable"] or not record["readings"] else max(record["readings"])
    target = processes.get(pid)
    used = target["used_bytes"] if target is not None and not errors else None
    utilization = nv.nvmlDeviceGetUtilizationRates(handle)
    return {
        "monotonic_ns": (begin + memory_end) // 2,
        "memory_query_start_ns": begin,
        "memory_query_end_ns": memory_end,
        "query_end_ns": time.monotonic_ns(),
        "whole_device_used_bytes": int(memory.used),
        "gpu_free_bytes": int(memory.free),
        "process_used_bytes": used,
        "process_present": target is not None,
        "process_query_errors": errors,
        "gpu_utilization_percent": int(utilization.gpu),
        "processes": list(processes.values()),
        "foreign_pids": sorted(other for other in processes if other != pid),
    }


def idle_baseline(nv, handle, directory, config):
    time.sleep(2)
    samples = []
    with (directory / "pre-run-nvml.jsonl").open("x") as stream:
        for _ in range(50):
            sample = gpu_sample(nv, handle, None)
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            samples.append(sample)
            s.check_interference(sample)
            s.require(sample["gpu_utilization_percent"] == 0, "Selected GPU is not idle")
            s.require(
                sample["gpu_free_bytes"] >= config["resource_limits"]["min_gpu_free_bytes"],
                "GPU free-memory resource gate failed",
            )
            time.sleep(0.01)
    return samples


def launch(config, directory, command, variant, nv, handle, phase=None, ready_check=None, cwd=None):
    """Never authorizes inference unless input/config/maps and live guards pass."""
    directory = Path(directory)
    env, overrides, removed = settings.environment(config, variant, phase == "b")
    cwd = Path(cwd) if cwd is not None else directory
    s.save(
        directory / "command.json",
        {
            "argv": command,
            "cwd": str(cwd),
            "environment_overrides": overrides,
            "environment_removed": removed,
            "watchdog_seconds": 300,
            "termination_grace_seconds": 5,
            "preflight_cooldown_seconds": 2,
            "sampling_requested_ms": 10,
            "executable_sha256": s.sha256(command[0]),
        },
    )
    (directory / "command.txt").write_text(
        "env "
        + " ".join(f"-u {key}" for key in removed)
        + " "
        + " ".join(shlex.quote(f"{key}={value}") for key, value in overrides.items())
        + " "
        + shlex.join(command)
        + "\n"
    )
    state = {"status": "preflight", "pid": None, "exit_code": None}
    s.save(directory / "status.json", state)
    threads, samples = [], []
    errors, events = queue.Queue(), queue.Queue()
    stop, expired = threading.Event(), threading.Event()
    child = timer = None
    protocol = {}
    try:
        s.save(directory / "resources-before.json", settings.host_resources(config))
        before = idle_baseline(nv, handle, directory, config)
        state["pre_run_whole_device_mib"] = (
            statistics.median(sample["whole_device_used_bytes"] for sample in before) / s.MIB
        )
        with ExitStack() as stack:
            stdout = stack.enter_context((directory / "stdout.log").open("x"))
            stderr = stack.enter_context((directory / "stderr.log").open("x"))
            nvml_log = stack.enter_context((directory / "nvml.jsonl").open("x"))
            resources_log = stack.enter_context((directory / "resources.jsonl").open("x"))
            child = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                text=True,
                bufsize=1,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            state.update(status="running", pid=child.pid, launched_ns=time.monotonic_ns())
            s.save(directory / "status.json", state)

            def terminate():
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)

            def cleanup():
                stop.set()
                if timer is not None:
                    timer.cancel()
                terminate()
                for thread in threads:
                    thread.join(timeout=5)
                for pipe in (child.stdout, child.stderr, child.stdin):
                    pipe.close()

            stack.callback(cleanup)

            def deadline():
                expired.set()
                terminate()

            def collect(pipe, output):
                try:
                    for line in pipe:
                        output.write(line)
                        output.flush()
                        if line.startswith("[ CONTROL ]"):
                            events.put(line.strip())
                except (OSError, ValueError) as error:
                    errors.put(f"Log collector: {error}")

            def sample_gpu():
                try:
                    next_sample, next_resource = time.monotonic_ns(), 0
                    while not stop.is_set():
                        if stop.wait(max(0, (next_sample - time.monotonic_ns()) / 1e9)):
                            break
                        sample = gpu_sample(nv, handle, child.pid)
                        samples.append(sample)
                        nvml_log.write(json.dumps(sample) + "\n")
                        nvml_log.flush()
                        s.check_interference(sample)
                        if time.monotonic_ns() >= next_resource:
                            resources = settings.host_resources(config)
                            resources_log.write(json.dumps(resources) + "\n")
                            resources_log.flush()
                            next_resource = time.monotonic_ns() + 1_000_000_000
                        next_sample = max(next_sample + 10_000_000, time.monotonic_ns())
                except (nv.NVMLError, RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
                    errors.put(f"GPU/resource monitor: {error}")

            timer = threading.Timer(300, deadline)
            timer.daemon = True
            timer.start()
            for target, args in (
                (collect, (child.stdout, stdout)),
                (collect, (child.stderr, stderr)),
                (sample_gpu, ()),
            ):
                thread = threading.Thread(target=target, args=args, daemon=True)
                threads.append(thread)
                thread.start()
            while child.poll() is None:
                s.require(not expired.is_set(), "300-second watchdog expired")
                if not errors.empty():
                    raise RuntimeError(errors.get_nowait())
                while not events.empty():
                    event = events.get_nowait()
                    if event == "[ CONTROL ] ready":
                        s.require(not protocol and ready_check is not None, "Duplicate/unexpected ready event")
                        ready_check()
                        s.require(errors.empty() and not expired.is_set(), "Monitor failed before GO")
                        s.check_interference(gpu_sample(nv, handle, child.pid))
                        protocol["go_ns"] = time.monotonic_ns()
                        response = "GO"
                    elif event == "[ CONTROL ] ended":
                        s.require(phase == "b" and set(protocol) == {"go_ns"}, "Unexpected end event")
                        s.timing_metrics(s.load(directory / "timing.json"))
                        protocol["ack_ns"] = time.monotonic_ns()
                        response = "ACK"
                    else:
                        raise RuntimeError(f"Unknown control event: {event}")
                    s.require(child.poll() is None and not expired.is_set(), "Child exited before authorization")
                    child.stdin.write(response + "\n")
                    child.stdin.flush()
                    s.save(directory / "protocol.json", protocol)
                time.sleep(0.002)
            stop.set()
            for thread in threads:
                thread.join(timeout=5)
            s.require(not expired.is_set(), "300-second watchdog expired")
            s.require(all(not thread.is_alive() for thread in threads), "Incomplete log/sampling capture")
            if not errors.empty():
                raise RuntimeError(errors.get_nowait())
            s.require(events.empty(), "Unprocessed control event")
            expected = {"go_ns", "ack_ns"} if phase == "b" else {"go_ns"} if phase == "a" else set()
            s.require(set(protocol) == expected, "Missing verification/timing handshake")
            state.update(status="exited", exit_code=child.returncode, protocol=protocol)
        s.require(len(samples) >= 2, "Missing live monitoring samples")
        s.save(directory / "sampling-summary.json", s.interval_stats(samples))
        return state, samples
    finally:
        if timer is not None:
            timer.cancel()
        if child is not None:
            state["exit_code"] = child.poll()
        if state["status"] != "exited":
            state["status"] = "failed"
        s.save(directory / "status.json", state)
