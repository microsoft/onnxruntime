import json
from pathlib import Path
import re

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
GUARD_HELPER = '''
def run_guard(command, prefix):
    from run_fp16_dot4_tree_probe import GUARD
    for suffix in (".log", ".guard.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    child = subprocess.run(["python3", str(GUARD), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command])
    proof = json.loads(prefix.with_suffix(".guard.json").read_text())
    valid = ((proof["status"] == "completed" and child.returncode == 0) or
             (proof["status"] == "failed" and child.returncode == 2))
    if not valid or proof.get("exit_code") != child.returncode or proof["gpu_uuid"] != UUID:
        raise RuntimeError(f"GPU guard did not prove completed child: {prefix.with_suffix('.guard.json')}")
    return child.returncode, prefix.with_suffix(".log").read_text()

'''


def make_runner():
    source = (ROOT / "run_dense_paired_matrix.py").read_text()
    source = replace_once(source, "from run_fp16_dot4_dynamic_sweep import run_guard\n", "")
    source = source.replace("build_dense_paired_matrix", "build_dense_paired_matrix_reset")
    source = source.replace("test_dense_paired_matrix.py", "test_dense_paired_matrix_reset.py")
    source = source.replace("qwen38-int4-dense-paired-matrix", "qwen38-int4-dense-paired-matrix-reset")
    source = source.replace("dense-paired-matrix-v1", "dense-paired-matrix-reset-v1")
    source = source.replace('"DensePairedMatrix"', '"DensePairedMatrixReset"')
    return replace_once(source, "\ndef fixture(k, n):", GUARD_HELPER + "\ndef fixture(k, n):")


def seal_failed_v1():
    state_path = ROOT / "dense-paired-matrix-v1-execution.json"
    state = json.loads(state_path.read_text())
    missing_prefix = ROOT / "dense-paired-matrix-v1-k2560-n2560-compare-gpu7"
    if not any(row["prefix"] == str(missing_prefix) for row in state["records"]):
        state["records"].append(dict(k=2560, n=2560, mode="compare", exit_code=2,
                                    prefix=str(missing_prefix)))
    evidence = [state_path, ROOT / "run_dense_paired_matrix.py",
                ROOT / "build_dense_paired_matrix.py", ROOT / "test_dense_paired_matrix.py",
                ROOT / "dense_paired_matrix_compare.cpp",
                Path("/home/sayanshaw/repos/qwen38-int4-dense-paired-matrix/build-v1/build-manifest.json")]
    phases = []
    for row in state["records"]:
        prefix = Path(row["prefix"])
        proof = json.loads(prefix.with_suffix(".guard.json").read_text())
        if proof["status"] not in ("completed", "failed") or proof["exit_code"] != row["exit_code"]:
            raise RuntimeError("Unproved finished child")
        log = prefix.with_suffix(".log").read_text()
        match = re.search(r"^DENSE comparisons=(\d+) mismatches=(\d+) K=(\d+) N=(\d+)$", log, re.M)
        if not match:
            raise RuntimeError("Incomplete arithmetic verdict")
        comparisons, mismatches, k, n = map(int, match.groups())
        if comparisons != n * 64 or (k, n) != (row["k"], row["n"]):
            raise RuntimeError("Wrong failed-attempt coverage")
        profile_name = re.search(r"^PROFILE /evidence/webgpu-perf-oct20/(.+)$", log, re.M)[1]
        profile = ROOT / profile_name
        events = json.loads(profile.read_text())
        native = sum("DensePairedMatrix" in event.get("name", "") for event in events)
        if native != (64 if row["mode"] == "compare" else 0):
            raise RuntimeError("Missing original/native production dispatch coverage")
        phases.append(dict(k=k, n=n, mode=row["mode"], comparisons=comparisons,
                           mismatches=mismatches, native_dispatches=native))
        evidence.extend((prefix.with_suffix(".log"), prefix.with_suffix(".guard.json"), profile))
        directory = Path("/home/sayanshaw/repos/qwen38-int4-dense-paired-matrix/fixtures-v1")
        evidence.extend((directory / f"k{k}-n{n}.onnx", directory / f"k{k}-n{n}-reference.f16"))
    if phases[-1]["mismatches"] != 163840:
        raise RuntimeError("Unexpected original failed attempt")
    state["status"] = "held_original_production_kernel_bitwise_mismatches"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    report = dict(status=state["status"], phases=phases, model_eligible=False,
                  actual_model_weights_used=False, model_timing_performed=False,
                  runner_exception="completed child exit2 classified failed by GPU guard",
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    path = ROOT / "dense-paired-matrix-v1-outcome.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    seal_failed_v1()
    target = ROOT / "run_dense_paired_matrix_reset.py"
    with target.open("x") as stream:
        stream.write(make_runner())
    print(target.name)
