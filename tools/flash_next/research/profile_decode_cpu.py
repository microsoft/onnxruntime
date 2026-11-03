import json
import os
from pathlib import Path
import subprocess
import time

from run_int4_candidate import GPU_UUIDS, ROOT


DOCKER = ["sudo", "-n", "docker", "-H", "unix:///home/sayanshaw/docker-qwen38.sock"]


def main():
    prefix = ROOT / "generic-matmul-replay-v1-gpu4-cpu-diagnostic-v2-20261029"
    assert not prefix.with_suffix(".json").exists()
    data_file = prefix.with_suffix(".perf.data")
    assert not data_file.exists()
    parse_retries = 0
    with prefix.with_suffix(".driver.log").open("w") as log:
        driver = subprocess.Popen(
            ["python3", str(ROOT / "run_int4_candidate.py"), "--kind", "generic-replay",
             "--phase", "windows", "--context", "2048", "--gpu-index", "4",
             "--output-prefix", str(prefix)], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if driver.poll() is not None:
                    raise RuntimeError(f"Diagnostic exited before sampling, code {driver.returncode}")
                if prefix.with_suffix(".json").exists():
                    try:
                        state = json.loads(prefix.with_suffix(".json").read_text())
                    except json.JSONDecodeError:
                        parse_retries += 1
                    else:
                        phase = state.get("active_phase", {})
                        if phase.get("repetition") == 0 and phase.get("phase") in (
                                "excluded_decode_warmup", "measured_decode"):
                            break
                time.sleep(0.05)
            else:
                raise TimeoutError("No decode sampling window within 300 seconds")
            guard = json.loads(prefix.with_suffix(".guard.json").read_text())
            assert guard["status"] == "running" and guard["gpu_uuid"] == GPU_UUIDS[4]
            container = json.loads(subprocess.check_output(
                DOCKER + ["inspect", guard["container_id"]], text=True))[0]
            assert prefix.name in " ".join(container["Config"]["Cmd"])
            pid = container["State"]["Pid"]
            gpu_pids = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
            assert any(line.strip() == f"{GPU_UUIDS[4]}, {pid}" for line in gpu_pids.splitlines())
            start = dict(phase)
            subprocess.run(
                ["sudo", "-n", "perf", "record", "-e", "cpu-clock:u", "-F", "499",
                 "--call-graph", "dwarf,8192", "-p", str(pid), "-o", str(data_file),
                 "--", "sleep", "2"], check=True,
                stdout=log, stderr=subprocess.STDOUT)
            end = json.loads(prefix.with_suffix(".json").read_text())["active_phase"]
            assert end["repetition"] == 0 and end["phase"] == "measured_decode", end
            report_file = prefix.with_suffix(".cpu-report.txt")
            with report_file.open("w") as report:
                subprocess.run(
                    ["sudo", "-n", "perf", "report", "--stdio", "--no-children",
                     "--sort", "dso,symbol", "--percent-limit", "1",
                     "-i", str(data_file)], stdout=report, check=True)
            metadata = dict(status="decode_cpu_samples_collected", pid=pid,
                owned_container=container["Id"], gpu_uuid=GPU_UUIDS[4],
                start_phase=start, end_phase=end, transient_json_retries=parse_retries,
                data_bytes=os.stat(data_file).st_size,
                performance_scope="CPU diagnostic only; its window token rates must not be promoted",
                quality_qualified=False, headline_eligible=False)
            prefix.with_suffix(".cpu-profile.json").write_text(json.dumps(metadata, indent=2) + "\n")
        finally:
            code = driver.wait(timeout=1400)
        if code != 0:
            raise RuntimeError(f"Diagnostic model gate failed with code {code}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
