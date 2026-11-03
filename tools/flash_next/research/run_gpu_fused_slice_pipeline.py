import json
from pathlib import Path
import statistics
import time

from build_qmoe_node_replay import sha
from prepare_webgpu_int4_metrics import ROOT
from run_gpu_decode_stack_pipeline import compare
from run_gpu_fused_slice_model import VARIANT, prefix_for, run
from run_gpu_slice_model import verify_sources
from run_gpu_slice_pipeline import percentile


def screen(context, repeat):
    baseline = int(repeat)
    rows, sources = [], [Path(__file__), ROOT / "run_gpu_decode_stack_pipeline.py"]
    for full in (True, False):
        groups, times = [], []
        for flag, trial in ((0, baseline), (1, baseline), (0, baseline + 1)):
            if not (repeat and flag == 0 and trial == baseline):
                run(context, full, flag, trial, 0)
            prefix = prefix_for(context, full, flag, trial, 0)
            path = prefix.with_suffix(".qualification.json")
            proof = verify_sources(path, "fused_slice_original_prefixes_qualified")
            raw, guard = [json.loads(prefix.with_suffix(s).read_text()) for s in (".json", ".guard.json")]
            samples = [s for s in raw["samples"] if not s["warmup"]]
            if (proof["context"] != context or proof["full_validation"] != full or proof["flag"] != flag or
                    proof["trial"] != trial or proof["trace"] != 0 or proof["gpu_index"] != 1 or
                    guard["status"] != "completed" or len(samples) != 3 or len(raw["samples"]) != 4):
                raise RuntimeError("Wrong bracketed fused-helper sample")
            groups.append(samples)
            times.append((proof["started_ns"], proof["completed_ns"]))
            sources.append(path)
            if flag:
                candidate = dict(full_validation=full, decode_tps=3 * 255 / sum(s["decode_seconds"] for s in samples),
                    median_decode_tps=statistics.median(s["decode_tps"] for s in samples),
                    prefill_tps=sum(s["input_tokens"] for s in samples) / sum(s["prefill_seconds"] for s in samples),
                    ttft_p50_seconds=percentile([s["ttft_seconds"] for s in samples], 0.5),
                    ttft_p95_seconds=percentile([s["ttft_seconds"] for s in samples], 0.95),
                    sampled_peak_gpu_gib=guard["peak_gpu_used_bytes"] / 1024**3)
        if not times[0][1] <= times[1][0] <= times[1][1] <= times[2][0]:
            raise RuntimeError("Not time-bracketed")
        rows.append(dict(candidate=candidate, comparison=compare(groups[1], groups[0], groups[2])))
    supported = all(row["comparison"]["supported"] for row in rows)
    result = dict(status="fused_slice_decode_gain_both_modes" if supported else "fused_slice_no_supported_gain",
        context=context, repeat=repeat, rows=rows, gpu_index=1, reused_initial_ending_controls=repeat,
        quality_qualified=False, default_enabled=False, cross_gpu_absolute_comparison_qualified=False,
        prefill_scope="append_tokens_through_first_token_selection",
        source_and_evidence_sha256={str(p): sha(p) for p in sources})
    path = ROOT / f"{VARIANT}-context{context}-{'repeat-' if repeat else ''}gpu1-screen.json"
    with path.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return path, result


if __name__ == "__main__":
    state_path = ROOT / f"{VARIANT}-gpu1-pipeline.json"
    if state_path.exists():
        raise FileExistsError(state_path)
    state = dict(status="starting", reports=[], goal_reached=False, started_ns=time.time_ns())

    def persist():
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    persist()
    try:
        for context in (2048, 8192):
            state.update(status="frozen_admission", context=context)
            persist()
            admission_path = ROOT / f"{VARIANT}-context{context}-gpu1-admission.json"
            if admission_path.exists():
                raise FileExistsError(admission_path)
            phases = [run(context, True, flag, 0, 1) for flag in (0, 1)]
            sources = [Path(__file__), *[prefix_for(context, True, flag, 0, 1).with_suffix(".qualification.json") for flag in (0, 1)]]
            admission = dict(status="fused_slice_frozen_prefixes_and_routing_admitted",
                context=context, gpu_index=1, phases=phases, quality_qualified=False, performance_qualified=False,
                source_and_evidence_sha256={str(p): sha(p) for p in sources})
            with admission_path.open("x") as stream:
                stream.write(json.dumps(admission, indent=2) + "\n")
            for repeat in (False, True):
                state.update(status="natural_decode_screen", repeat=repeat)
                persist()
                path, result = screen(context, repeat)
                state["reports"].append(dict(path=str(path), **{k: result[k] for k in ("context", "repeat", "status", "rows")}))
                if result["status"] != "fused_slice_decode_gain_both_modes":
                    state["status"] = "held_no_unchanged_retry"
                    persist()
                    raise SystemExit(0)
                persist()
        state["status"] = "bounded_2k_8k_fused_qualification_complete_32k_requires_supported_control"
        persist()
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        persist()
        raise
