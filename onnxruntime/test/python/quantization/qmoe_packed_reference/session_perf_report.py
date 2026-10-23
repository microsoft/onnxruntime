"""Validate session_perf output and generate a self-contained analysis and report."""

import argparse
import json
import statistics
from pathlib import Path

from session_perf import MODES, ROOT, digest

LIMITS = [
    "One QMoE node, synthetic deterministic repeated-pattern nonzero weights, fixed input/router, rows=1; not a full model, prefill, MMLU, or quality evaluation.",
    "Constant and dynamic graphs use identical numerical tensors but different control paths. No historical runtime or old full-model baseline is used.",
    "Init is synchronized Session construction: model file read/parse, CPU initialization, allocation, host-to-device initializer copies, private transforms, validation and synchronization. It is prepack-inclusive, not an isolated prepack timer.",
    "GPU feed setup (disk read, host allocation and H2D copies) is outside init and reported separately; graph generation, CUDA context/provider setup and process startup are outside both. Total setup is feed setup plus init, not process startup latency.",
    "First run includes lazy loading, allocations, validation if dynamic, routing, compute and small CPU output copy. Steady is synchronized CPU wall time per Run, not CUDA event kernel time. Session optimization is disabled and CPU EP fallback forbidden.",
    "Dynamic metadata retains private weight layouts but bypasses cached metadata validation. Its delta includes validation kernels, host synchronization and dynamic-feed bookkeeping; it is not a pure validation-kernel timer. Layout effects are only isolated by the separate existing kernel benchmark.",
    "All three modes execute the same SoftmaxTopK routing. No routing speedup or launch-count reduction was independently measured; no routing savings are claimed.",
    "cudaMemGetInfo snapshots are device-wide, synchronized sampled residency, not allocator counters or allocation peaks. Baseline is after CUDA/provider setup. Before-feed to steady includes device feeds and ORT arenas; after-destroy retains device feeds. Other GPU processes can perturb deltas. No NVML per-process or transient GPU-peak sampling was performed.",
    "Peak host RSS uses Linux getrusage and covers the process, including model loading and feed staging; it is not incremental host prepack memory.",
    "Four fresh processes per mode, alternating forward/reverse order; 10 warmups and 30 samples/process. Pooled medians and per-process medians are descriptive, not confidence intervals. Clocks are not locked, file cache is not flushed, and init/first-run variability is retained in raw data.",
    "Correctness means exact finite nonzero output agreement across modes/repetitions, not an independent mathematical oracle or new runtime correctness test.",
    "Clean core/provider build only, not whole CI. Static Session archives and the clean shared CUDA provider are linked without pybind. ONNX source override differs from the pinned gitlink; no global headers, runtime sources or dependencies were edited by this experiment.",
    "Existing kernel results are read-only contextual data supplied before this run, not rerun here and not proven to share the exact measured binary. Their GPU-event timings cannot be subtracted from new Session wall timings.",
    "No full-model throughput, memory or target-performance completion claim follows from these results.",
]


def median(records, action):
    return statistics.median(action(record) for record in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--kernel-results", type=Path, help="Optional kernel benchmark JSONL for contextual comparison")
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    parameters = data["parameters"]
    for path, expected in data["source_sha256"].items():
        assert digest(ROOT / path) == expected, f"Source changed since measurement: {path}"
    kernel_path = args.kernel_results
    kernel_rows = [json.loads(line) for line in kernel_path.read_text().splitlines()] if kernel_path else []
    cases = []
    for case in data["cases"]:
        assert case["output_max_abs_difference"] == 0 and case["output_nonzero_count"] > 0
        rows = {}
        for mode in MODES:
            records = [record for record in case["records"] if record["mode"] == mode]
            assert len(records) == parameters["repeats"]
            assert all(len(record["steady_ms"]) == parameters["samples"] for record in records)
            assert {record["position"] for record in records} == ({1} if mode == "dynamic_metadata" else {0, 2})
            summary = dict(case["summary"][mode])
            summary.update(
                feed_setup_median_ms=median(records, lambda record: record["feed_setup_ms"]),
                feed_plus_init_median_ms=median(records, lambda record: record["feed_setup_ms"] + record["init_ms"]),
                host_peak_rss_median_bytes=median(records, lambda record: record["peak_host_rss_bytes"]),
                memory_stage_median_bytes={
                    stage: median(records, lambda record, stage=stage: record["device_used_bytes"][stage])
                    for stage in records[0]["device_used_bytes"]
                },
                init_increment_median_bytes=median(
                    records,
                    lambda record: record["device_used_bytes"]["after_init"]
                    - record["device_used_bytes"]["before_init"],
                ),
            )
            rows[mode] = summary
        cases.append(
            {
                "bits": case["bits"],
                "dtype": case["dtype"],
                "modes": rows,
                "validation_path_wall_saving_ms": rows["dynamic_metadata"]["steady_median_ms"]
                - rows["constant"]["steady_median_ms"],
                "output_max_abs_difference": case["output_max_abs_difference"],
            }
        )
    device = data["cases"][0]["records"][0]
    analysis = {
        "schema_version": 1,
        "measurement_source": str(args.results.resolve()),
        "measurement_sha256": digest(args.results),
        "provenance": {
            key: data[key]
            for key in (
                "started_utc",
                "command",
                "parameters",
                "git_head",
                "git_status",
                "onnx_source_head",
                "onnx_pinned_gitlink",
                "python_dependencies",
                "gpu_inventory",
                "source_sha256",
                "binary_sha256",
            )
        },
        "device": {key: device[key] for key in ("device", "uuid_hex", "cuda_runtime", "cuda_driver", "ort_version")},
        "limitations_and_method": LIMITS,
        "cases": cases,
        "contextual_kernel_results": {
            "path": str(kernel_path.resolve()) if kernel_path else None,
            "sha256": digest(kernel_path) if kernel_path else None,
            "metadata": kernel_rows[0] if kernel_rows else None,
            "decode_rows": [row for row in kernel_rows[1:] if row["rows"] == 1],
        },
    }
    args.prefix.with_suffix(".json").write_text(json.dumps(analysis, indent=2) + "\n")
    lines = [
        "# PR2/PR3 QMoE Session Performance",
        "",
        f"Measured {data['started_utc']}; performance analysis only.",
        "",
        "## Result",
        "",
        "All six requested cases completed with exact finite nonzero output agreement across three modes and four fresh processes/mode: 72 Sessions and 2,160 measured decode calls. No speed-goal or full-model claim is made.",
        "",
        "H=2048, I=768, E=128, k=8, rows=1, block=64. FP16 and BF16; bit pairs 3/3, 3/4, 4/3. Input [1,1,H], router [1,E], interleaved SwiGLU, normalized routing, explicit packed zero points, no biases; deterministic codes cover the full bit range, scale=1/64. Graphs are matched by raw tensor hashes in the measurement JSON.",
        "",
        "Modes: `constant` caches all six weight/metadata initializers; `dynamic_metadata` keeps weights constant but feeds scales/zero points; `dynamic_all` feeds all six. Input/router are GPU feeds in every mode. No per-run weight transfers. Output is copied to CPU in every mode.",
        "",
        "## Session Timings",
        "",
        "Milliseconds; medians. Steady median pools 120 samples/mode/case. Raw per-process samples and launch positions are retained.",
        "",
        "| Bits / Dtype | Mode | Feed Setup | Init (Prepack-Inclusive) | Feed + Init | First Run | Steady |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        label = f"{'/'.join(map(str, case['bits']))} {case['dtype']}"
        for mode, row in case["modes"].items():
            lines.append(
                f"| {label} | {mode} | {row['feed_setup_median_ms']:.3f} | {row['init_median_ms']:.3f} | {row['feed_plus_init_median_ms']:.3f} | {row['first_run_median_ms']:.3f} | {row['steady_median_ms']:.6f} |"
            )
    lines += [
        "",
        "## Validation And Layout",
        "",
        "Positive validation-path deltas are `dynamic_metadata - constant` and include feed bookkeeping and synchronization. Both retain private weights. Routing is unchanged; no routing savings are claimed.",
        "",
        "| Bits / Dtype | Validation-Path Wall Delta (ms) | Dynamic-All / Constant Steady Ratio |",
        "|---|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {'/'.join(map(str, case['bits']))} {case['dtype']} | {case['validation_path_wall_saving_ms']:.6f} | {case['modes']['dynamic_all']['steady_median_ms'] / case['modes']['constant']['steady_median_ms']:.3f}x |"
        )
    if kernel_rows:
        lines += [
            "",
            "Read-only existing kernel benchmark, same nominal decode shape; not rerun here, CUDA-event timings only:",
            "",
            "| Bits / Dtype | Canonical (ms) | Private (ms) | GPU Transform (ms) |",
            "|---|---:|---:|---:|",
        ]
    for row in analysis["contextual_kernel_results"]["decode_rows"]:
        lines.append(
            f"| {'/'.join(map(str, row['bits']))} {row['dtype']} | {row['canonical_median_ms']:.6f} | {row['private_median_ms']:.6f} | {row['prepack_gpu_ms']:.3f} |"
        )
    lines += [
        "",
        "Private layout is not uniformly faster in that isolated benchmark (4/3 is slightly slower). New Session results show a benefit from the combined cached control path, not proof of universal layout acceleration. Init overhead is substantial and must be amortized; these runs do not establish a full-model break-even point.",
        "",
        "## Memory",
        "",
        "MiB; medians of per-process measurements. Init delta is after-init minus before-init. Resident delta is steady minus before-feeds. Device figures are sampled device-wide residency, not peaks. Host RSS is whole-process high-water RSS.",
        "",
        "| Bits / Dtype | Mode | Init GPU Delta | Steady GPU Delta Including Feeds | Peak Host RSS |",
        "|---|---|---:|---:|---:|",
    ]
    for case in cases:
        for mode, row in case["modes"].items():
            lines.append(
                f"| {'/'.join(map(str, case['bits']))} {case['dtype']} | {mode} | {row['init_increment_median_bytes'] / 2**20:.1f} | {row['resident_increment_median_bytes'] / 2**20:.1f} | {row['host_peak_rss_median_bytes'] / 2**20:.1f} |"
            )
    lines += [
        "",
        "Constant-session residency includes ORT arena retention and temporary-allocation reuse; it must not be equated to private packed payload size. Raw JSON includes before-feeds, before-init, after-init, after-first, after-steady and after-session-destroy snapshots.",
        "",
        "## Reproduce",
        "",
        "From the worktree root on the host, with `jiafa-dev` running and the accuracy mount visible at the same path:",
        "",
        "```bash",
        "python3 onnxruntime/test/python/quantization/qmoe_packed_reference/session_perf.py \\",
        "  --work /datadisks/disk1/jiafa/accuracy/int3-session-perf-full-scratch \\",
        "  --output /tmp/int3-qmoe-pr3-session-results.json",
        "python3 onnxruntime/test/python/quantization/qmoe_packed_reference/session_perf_report.py \\",
        "  /tmp/int3-qmoe-pr3-session-results.json \\",
        "  --prefix /tmp/int3-qmoe-pr3-session-analysis",
        "```",
        "",
        "For a quick probe append `--quick --hidden 128 --intermediate 64 --experts 8 --top-k 2 --repeats 2 --warmup 2 --samples 3` and use a different output path. The runner links a standalone C++17 client using the clean provider-test link dependencies; provider library symlinks are created only in scratch. It does not build or modify runtime. Generated fixtures/raw worker files remain in scratch; final JSON is self-contained.",
        "",
        "Dependencies: host Python 3.12, NumPy/ONNX/ml_dtypes (versions below), Docker CLI, git, nvidia-smi; container g++, Ninja, CUDA 13 headers/runtime, cuDNN 9, clean ORT static archives/shared CUDA provider. Existing MLAS assembly emits a GNU-stack linker warning, but linking and execution succeed. Building target `onnxruntime` was not possible in this static configuration; pybind was neither required nor built.",
        "",
        "## Provenance",
        "",
        f"- Worktree HEAD: `{data['git_head']}`; dirty source identities are SHA256-recorded, not inferred from HEAD.",
        f"- Build: `{parameters['build_dir']}`, Release, shared core OFF, CUDA provider ON; clean core/provider build only.",
        f"- ORT {device['ort_version']}; {device['device']}; UUID hex `{device['uuid_hex']}`; runtime/driver API versions {device['cuda_runtime']}/{device['cuda_driver']}.",
        f"- Local ONNX source: `{data['onnx_source_head']}`; differs from pinned `{data['onnx_pinned_gitlink']}`. CMake uses `FETCHCONTENT_SOURCE_DIR_ONNX=/datadisks/disk1/jiafa/accuracy/onnxruntime/cmake/external/onnx`; declared deps archive is ONNX v1.23.2.",
        f"- Host dependency versions: `{json.dumps(data['python_dependencies'], sort_keys=True)}`.",
        f"- Measurement SHA256: `{digest(args.results)}`.",
        "",
        "Clean artifact SHA256 identities:",
        "",
        "```text",
        data["binary_sha256"],
        "```",
        "",
        "Source dependencies: `qmoe_int3.cc` (`PrePackInt3` and `ComputeInt3`) controls private-layout caching, metadata validation and routing; `qmoe_int3.cu` implements kernels; `moe_quantization.cc` dispatches QMoE; `make_cuda_session_smoke.py` supplies the canonical row packer; `session_perf.cc` and `session_perf.py` implement this experiment. Existing `qmoe_int3_benchmark.cu` and its JSONL are read-only context. Full paths/hashes, tensor hashes, launch order and raw timing samples are retained in the results JSON; generated analysis JSON includes definitions/limitations and contextual kernel data.",
        "",
        "## Limits",
        "",
        *[f"- {item}" for item in LIMITS],
        "",
    ]
    args.prefix.with_suffix(".md").write_text("\n".join(lines))
    print(args.prefix.with_suffix(".json"))
    print(args.prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
