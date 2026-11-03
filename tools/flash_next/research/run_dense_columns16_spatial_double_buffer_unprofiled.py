import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess

from build_dense_columns16_spatial_double_buffer import FLAG
from build_qmoe_node_replay import sha
from run_dense_columns16_synthetic import FLAGS as PARENT_FLAGS, UUID, run_guard, workspace


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-dense-columns16-spatial-double-buffer/build-v1")
ITERATIONS = 1024
FLAGS = PARENT_FLAGS + " ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER=1"


def matched_shape(triplets):
    ratios = [math.sqrt(row["before"]["mean_us"] * row["after"]["mean_us"]) /
              row["candidate"]["mean_us"] for row in triplets]
    return dict(matched_speed_ratio=statistics.geometric_mean(ratios),
                positive=sum(ratio > 1 for ratio in ratios), requests=len(ratios),
                ratios=ratios)


if __name__ == "__main__":
    outcome = ROOT / "dense-columns16-spatial-double-buffer-unprofiled-v1-outcome.json"
    if outcome.exists():
        raise FileExistsError(outcome)
    actual_path = ROOT / "dense-columns16-spatial-double-buffer-actual-batch-v1-outcome.json"
    actual = json.loads(actual_path.read_text())
    if actual["status"] != "columns16_spatial_double_buffer_batched_actual_and_default_supported":
        raise RuntimeError("Fresh actual arithmetic/default gate not supported")
    for path, digest in actual["source_and_evidence_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Changed actual proof: {path}")
    selected = actual["phases"][-1]["shapes"]
    shapes = [(row["k"], row["n"]) for row in selected if row["selected_dispatches"] == 64]
    if len(shapes) != 7 or any(row["mismatches"] for row in selected):
        raise RuntimeError("Incorrect selected actual-weight coverage")
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, digest in manifest["source_sha256"].items():
        if sha(BUILD / relative) != digest:
            raise RuntimeError(f"Changed source/SDK: {relative}")
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    binary = BUILD / "unprofiled_compare"
    if sha(provider) != manifest["provider_sha256"] or sha(binary) != manifest["unprofiled_comparator_sha256"]:
        raise RuntimeError("Changed timing provider or comparator")
    evidence = [Path(__file__), actual_path, manifest_path, provider, binary]
    trials = []
    for trial in range(3):
        prefix = ROOT / f"dense-columns16-spatial-double-buffer-unprofiled-v1-trial{trial}-gpu7"
        commands = []
        expected = []
        for k, n in shapes:
            fixture = Path("/home/sayanshaw/repos/qwen38-int4-dense-paired-matrix-reset/actual-fixtures-v1") / f"k{k}-n{n}.onnx"
            reference = BUILD.parent / "actual-batch-v1" / f"k{k}-n{n}-reference.f16"
            evidence.extend((fixture, reference))
            for phase, mode in (("before", 0), ("candidate", 1), ("after", 0)):
                args = ["env", f"{FLAG}={mode}", "taskset", "-c", "48",
                        workspace(binary), "timing", workspace(provider),
                        workspace(fixture), workspace(reference), str(ITERATIONS)]
                commands.append(shlex.join(args))
                expected.append((k, n, phase))
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                        "--format=csv,noheader"], check=True)
        command = (f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
                   f"export {FLAGS}; " + "; ".join(commands))
        code, log = run_guard(command, prefix)
        matches = re.findall(r"^TIMING K=(\d+) N=(\d+) iterations=(\d+) warmup=8 comparisons=(\d+) "
                             r"mismatches=0 mean_us=(\S+) median_us=(\S+) profiling=0$", log, re.M)
        if code != 0 or "FAIL" in log or len(matches) != len(expected):
            raise RuntimeError(f"Unprofiled finite-output timing failed: {prefix.with_suffix('.log')}")
        rows = {}
        for match, (k, n, phase) in zip(matches, expected, strict=True):
            observed_k, observed_n, iterations, comparisons = map(int, match[:4])
            mean, median = map(float, match[4:])
            if ((observed_k, observed_n, iterations, comparisons) != (k, n, ITERATIONS, n * ITERATIONS)
                    or not all(math.isfinite(value) and value > 0 for value in (mean, median))):
                raise RuntimeError("Incorrect unprofiled timing/finite-output coverage")
            rows.setdefault((k, n), {})[phase] = dict(mean_us=mean, median_us=median, comparisons=comparisons)
        trials.append(rows)
        evidence.extend((prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")))
        print(json.dumps(dict(trial=trial, complete_triplets=len(rows))), flush=True)
    rows = [dict(k=k, n=n, **matched_shape([trial[(k, n)] for trial in trials])) for k, n in shapes]
    supported = all(row["matched_speed_ratio"] > 1 and row["positive"] == 3 for row in rows)
    result = dict(status="unprofiled_actual_shape_gain_supported" if supported else "held_unprofiled_actual_shape_consistency",
                  rows=rows, same_provider_both_sides=True, cpu_affinity="48",
                  profiling=False, timings_include_host_run_and_readback=True,
                  warmup_runs_per_process=8, measured_runs_per_process=ITERATIONS,
                  actual_weights=True, model_eligible=False, default_enabled=False,
                  natural_model_tps_measured=False, provider_sha256=manifest["provider_sha256"],
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    with outcome.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], rows=rows)), flush=True)
