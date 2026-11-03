import argparse
import json
import os
from pathlib import Path
import time

from safe_model_run import container_process_ids, gpu_compute_pids


parser = argparse.ArgumentParser()
parser.add_argument("--guard", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
args = parser.parse_args()
root = args.model
started = time.monotonic()
last_check = -1
total = 0
reported = 0
chunk = bytearray(8 * 1024 * 1024)
for name in ("model.onnx.data",):
    with (root / name).open("rb", buffering=0) as source:
        os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        while True:
            now = time.monotonic()
            if now - started > 600:
                print(json.dumps({"status": "bounded_timeout", "bytes": total,
                                  "seconds": now - started}), flush=True)
                raise SystemExit(75)
            if now - last_check >= 1:
                state = json.loads(args.guard.read_text())
                if state["status"] == "waiting_for_idle_gpu":
                    time.sleep(1)
                    continue
                if state["status"] != "running":
                    print(json.dumps({"status": "stopped_model_no_longer_running", "bytes": total}), flush=True)
                    raise SystemExit(0)
                pids = gpu_compute_pids(state["gpu_uuid"])
                foreign = set()
                container_pids = None
                for pid in pids:
                    if pid == state.get("container_init_pid"):
                        continue
                    try:
                        cgroup = Path(f"/proc/{pid}/cgroup").read_text()
                    except FileNotFoundError:
                        print(f"Compute process {pid} exited during ownership check", flush=True)
                        continue
                    if state["container_id"] not in cgroup:
                        if container_pids is None:
                            container_pids = container_process_ids(state["container_id"])
                        if pid not in container_pids and Path(f"/proc/{pid}").exists():
                            foreign.add(pid)
                available_kib = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                                     if line.startswith("MemAvailable:"))
                if foreign or available_kib < 200 * 1024 * 1024:
                    raise RuntimeError(f"Stopping cache warming: foreign GPU processes={sorted(foreign)}, "
                                       f"available memory KiB={available_kib}")
                last_check = now
            count = source.readinto(chunk)
            if not count:
                break
            total += count
            if total - reported >= 8 * 1024**3:
                reported = total
                print(json.dumps({"bytes_total": total, "seconds": time.monotonic() - started}), flush=True)
    print(json.dumps({"file": name, "bytes_total": total, "seconds": time.monotonic() - started}), flush=True)
print(json.dumps({"status": "completed", "bytes": total, "seconds": time.monotonic() - started}), flush=True)
