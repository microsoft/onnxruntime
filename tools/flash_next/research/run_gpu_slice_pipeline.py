import json
import math
from pathlib import Path
import statistics
import time

from build_qmoe_node_replay import sha
from prepare_webgpu_int4_metrics import ROOT
from run_gpu_slice_model import VARIANT, prefix_for, run, verify_sources


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def compare(candidate, before, after):
    if not len(candidate) == len(before) == len(after) == 3:
        raise RuntimeError("Expected three matched requests")
    for group in (candidate, before, after):
        if any(s["output_tokens"] != 256 or
               abs(s["decode_tps"] * s["decode_seconds"] - 255) >= 1e-8 for s in group):
            raise RuntimeError("Wrong decode denominator or output count")
    ttft = [c["ttft_seconds"] / math.sqrt(b["ttft_seconds"] * a["ttft_seconds"])
            for c, b, a in zip(candidate, before, after)]
    decode = [c["decode_tps"] / math.sqrt(b["decode_tps"] * a["decode_tps"])
              for c, b, a in zip(candidate, before, after)]
    ttft_ratio = statistics.geometric_mean(ttft)
    decode_ratio = statistics.geometric_mean(decode)
    return dict(matched_ttft_ratio=ttft_ratio, matched_decode_ratio=decode_ratio,
        ttft_request_ratios=ttft, decode_request_ratios=decode,
        supported=ttft_ratio < 0.99 and all(x < 1 for x in ttft) and decode_ratio >= 0.98)


def load(context, full, flag, trial):
    prefix = prefix_for(context, full, flag, trial, 0)
    proof = verify_sources(prefix.with_suffix(".qualification.json"), "gpu_slice_frozen_original_prefixes_qualified")
    raw = json.loads(prefix.with_suffix(".json").read_text())
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    if (proof["context"] != context or proof["full_validation"] != full or proof["flag"] != flag or
            proof["trial"] != trial or proof["trace"] != 0 or guard["status"] != "completed"):
        raise RuntimeError("Wrong bracketed sample")
    measured = [s for s in raw["samples"] if not s["warmup"]]
    if len(raw["samples"]) != 4 or len(measured) != 3:
        raise RuntimeError("Wrong warmup/request count")
    return proof, measured, guard


def screen(context, repeat):
    baseline = int(repeat)
    rows, proofs = [], []
    for full in (True, False):
        groups, times, candidates = [], [], None
        for flag, trial in ((0, baseline), (1, baseline), (0, baseline + 1)):
            if not (repeat and flag == 0 and trial == baseline):
                run(context, full, flag, trial, 0)
            proof, samples, guard = load(context, full, flag, trial)
            proofs.append(prefix_for(context, full, flag, trial, 0).with_suffix(".qualification.json"))
            groups.append(samples)
            times.append((proof["started_ns"], proof["completed_ns"]))
            if flag:
                candidates = dict(full_validation=full,
                    ttft_p50_seconds=percentile([s["ttft_seconds"] for s in samples], 0.5),
                    ttft_p95_seconds=percentile([s["ttft_seconds"] for s in samples], 0.95),
                    decode_tps=3 * 255 / sum(s["decode_seconds"] for s in samples),
                    median_decode_tps=statistics.median(s["decode_tps"] for s in samples),
                    prefill_tps=sum(s["input_tokens"] for s in samples) / sum(s["prefill_seconds"] for s in samples),
                    sampled_peak_gpu_gib=guard["peak_gpu_used_bytes"] / 1024**3)
        if not (times[0][1] <= times[1][0] and times[1][1] <= times[2][0]):
            raise RuntimeError("Not time-bracketed")
        rows.append(dict(candidate=candidates, comparison=compare(groups[1], groups[0], groups[2])))
    supported = all(r["comparison"]["supported"] for r in rows)
    report = dict(status="gpu_slice_ttft_gain_both_modes" if supported else "gpu_slice_no_supported_ttft_gain",
        context=context, repeat=repeat, rows=rows, reused_initial_ending_controls=repeat,
        quality_qualified=False, default_enabled=False,
        prefill_scope="append_tokens_through_first_token_selection",
        all_candidate_modes_above_100_decode_tps=all(r["candidate"]["median_decode_tps"] > 100 for r in rows),
        source_and_evidence_sha256={str(p): sha(p) for p in [Path(__file__), *proofs]})
    path = ROOT / f"{VARIANT}-context{context}-{'repeat-' if repeat else ''}ttft-screen.json"
    with path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return path, report


if __name__ == "__main__":
    state_path = ROOT / f"{VARIANT}-queued-pipeline.json"
    if state_path.exists():
        raise FileExistsError(state_path)
    original = ROOT / "gpu-greedy-argmax-local8-v4-ready1-progressive-pipeline.json"
    state = dict(status="waiting_for_original_pipeline", reports=[], goal_reached=False,
                 wait_started_ns=time.time_ns())

    def persist():
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    persist()
    deadline = time.monotonic() + 4 * 3600
    while json.loads(original.read_text())["status"] == "running":
        if time.monotonic() > deadline:
            state["status"] = "blocked_original_pipeline_wait_timeout"
            persist()
            raise TimeoutError("Original pipeline did not finish within four hours")
        time.sleep(20)
    try:
        wide = ROOT / "gpu-slice-v6-r2-wide-gpu4.qualification.json"
        verify_sources(wide, "segmented_large_unsigned_index_gpu_slice_exact")
        state["status"] = "frozen2k_admission"
        persist()
        phases = [run(2048, True, flag, 0, 1) for flag in (0, 1)]
        sources = [wide, *[prefix_for(2048, True, flag, 0, 1).with_suffix(".qualification.json") for flag in (0, 1)]]
        admission = dict(status="gpu_slice_frozen2k_prefixes_and_routing_admitted", phases=phases,
            quality_qualified=False, performance_qualified=False,
            source_and_evidence_sha256={str(p): sha(p) for p in sources})
        with (ROOT / f"{VARIANT}-model-admission.json").open("x") as stream:
            stream.write(json.dumps(admission, indent=2) + "\n")
        for context in (2048, 8192, 32768):
            for repeat in (False, True):
                state.update(status="natural_ttft_screen", context=context, repeat=repeat)
                persist()
                path, report = screen(context, repeat)
                state["reports"].append(dict(path=str(path), context=context, repeat=repeat,
                    supported=report["status"] == "gpu_slice_ttft_gain_both_modes", rows=report["rows"]))
                if report["status"] != "gpu_slice_ttft_gain_both_modes":
                    state["status"] = "held_no_unchanged_retry"
                    persist()
                    raise SystemExit(0)
        state["status"] = "all_contexts_ttft_repeat_qualified"
        state["goal_reached"] = all(row["candidate"]["median_decode_tps"] > 100
            for report in state["reports"] for row in report["rows"])
        persist()
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        persist()
        raise
