import argparse
import json
import subprocess
import time
from pathlib import Path


def read_json(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(0.05)


def process_sample(pid):
    try:
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return {"exited_before_sample": True}
    return {
        "status": [line for line in status if line.startswith(("VmRSS:", "VmSwap:", "Threads:"))],
        "minor_faults": int(fields[7]), "major_faults": int(fields[9]),
        "user_ticks": int(fields[11]), "system_ticks": int(fields[12]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2100)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval <= 0:
        parser.error("Observation duration and interval must be positive")
    fields = "timestamp,index,uuid,memory.used,memory.total,utilization.gpu,clocks.sm,power.draw,temperature.gpu"
    active = {"waiting_for_idle_gpu", "running"}
    terminal = {"completed", "failed", "run_timed_out", "yielded_to_training",
                "blocked_gpu_busy", "guard_failed"}
    deadline = time.monotonic() + args.seconds
    with args.prefix.with_suffix(".telemetry.jsonl").open("x") as output:
        while time.monotonic() < deadline:
            started = time.monotonic()
            guard = read_json(args.prefix.with_suffix(".guard.json"))
            row = {"time_ns": time.time_ns(), "guard_status": guard["status"] if guard else None}
            if guard:
                if guard["status"] not in active | terminal:
                    raise RuntimeError(f"Unknown guard state: {guard['status']}")
                model = read_json(args.prefix.with_suffix(".json"))
                if model is not None:
                    row.update(model_status=model["status"], phase=model.get("active_phase"),
                               completed_windows=len(model.get("window_samples", [])))
                row["owned_processes"] = {}
                if guard["status"] == "running":
                    for pid in guard.get("own_compute_pids", []):
                        row["owned_processes"][str(pid)] = process_sample(pid)
                    query_start = time.perf_counter()
                    row["gpu_csv"] = subprocess.check_output([
                        "nvidia-smi", "--id=" + guard["gpu_uuid"],
                        "--query-gpu=" + fields, "--format=csv,noheader,nounits",
                    ], text=True, timeout=30).strip()
                    row["gpu_query_seconds"] = time.perf_counter() - query_start
                    row["gpu_query_fields"] = fields
            output.write(json.dumps(row) + "\n")
            output.flush()
            if guard and guard["status"] in terminal:
                print("Telemetry completed with guard", guard["status"])
                return
            time.sleep(max(0, args.interval - (time.monotonic() - started)))
    raise TimeoutError("Observation bound reached before guard termination")


if __name__ == "__main__":
    main()
