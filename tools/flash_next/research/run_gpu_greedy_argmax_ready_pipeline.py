import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha
from report_webgpu_int4_metrics import metrics, table
from run_gpu_greedy_argmax_ready_standard import BUILD, ROOT, VARIANT, prefix_for, verify_proof


def main():
    output = ROOT / f"{VARIANT}-progressive-pipeline.json"
    if output.exists():
        raise FileExistsError(output)
    build = json.loads((BUILD / "build-manifest.json").read_text())
    records, reports, rows = [], [], []
    evidence = [Path(__file__), ROOT / "prepare_gpu_greedy_argmax_ready.py",
                ROOT / "test_gpu_greedy_argmax_ready.py",
                ROOT / "run_gpu_greedy_argmax_ready_standard.py",
                ROOT / "run_gpu_greedy_argmax_ready_screen.py",
                ROOT / "benchmark_gpu_greedy_argmax_ready.py",
                ROOT / "report_webgpu_int4_metrics.py", BUILD / "build-manifest.json"]
    state = dict(status="running", records=records, reports=reports, candidate_full_metrics=rows,
                 runtime_sha256=build["runtime_sha256"], provider_sha256=build["provider_sha256"],
                 measured_requests_per_run=3, excluded_warmups_per_run=1,
                 prefill_definition="input_tokens/append_tokens_through_first_token_selection",
                 prefill_includes_first_token_selection=True, quality_qualified=False,
                 default_enabled=False, goal_reached=False)

    def persist():
        state["source_and_evidence_sha256"] = {str(path): sha(path) for path in evidence}
        output.write_text(json.dumps(state, indent=2) + "\n")

    persist()
    for context in (2048, 8192, 32768):
        for repeat in (False, True):
            command = ["python3", "-u", str(ROOT / "run_gpu_greedy_argmax_ready_screen.py"),
                       "--context", str(context)]
            if repeat:
                command.append("--repeat")
            record = dict(command=command, context=context, repeat=repeat, completed=False)
            records.append(record)
            persist()
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as error:
                state.update(status="failed", error_command=error.cmd, error_returncode=error.returncode)
                persist()
                raise
            record["completed"] = True
            path = ROOT / f"{VARIANT}-context{context}-{'repeat-' if repeat else ''}screen.qualification.json"
            report = json.loads(path.read_text())
            supported_status = ("repeated_" if repeat else "initial_") + "gpu_greedy_decode_gain_both_modes"
            if report["status"] not in (supported_status, "no_supported_decode_gain"):
                raise RuntimeError(f"Unexpected screen status: {report['status']}")
            verify_proof(path, report["status"])
            reports.append(dict(path=str(path), status=report["status"], context=context,
                                repeat=repeat, rows=report["rows"]))
            evidence.append(path)
            prefix = prefix_for("gpu", context, True, int(repeat))
            raw_path, guard_path = [prefix.with_suffix(suffix) for suffix in (".json", ".guard.json")]
            row = metrics(json.loads(raw_path.read_text()), json.loads(guard_path.read_text()), context, 3)
            row.update(repeat=repeat, performance_supported=report["status"] == supported_status)
            rows.append(row)
            evidence.extend((raw_path, guard_path))
            print(table([row]), flush=True)
            if report["status"] != supported_status:
                state["status"] = "held_no_supported_gain"
                persist()
                return
            persist()
    state["status"] = "repeated_gpu_greedy_supported_all_contexts"
    state["goal_reached"] = all(row["candidate"]["median_decode_tps"] > 100
                                for report in reports for row in report["rows"])
    persist()


if __name__ == "__main__":
    main()
