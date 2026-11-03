import json
import math
from pathlib import Path
import statistics
import time

from build_qmoe_node_replay import sha
from prepare_webgpu_int4_metrics import ROOT
from run_gpu_decode_stack_model import prefix_for, run
from run_gpu_slice_model import verify_sources
from run_gpu_slice_pipeline import percentile


def compare(candidate, before, after):
    if not len(candidate) == len(before) == len(after) == 3:
        raise RuntimeError("Expected three matched requests")
    for group in (candidate, before, after):
        if any(s["output_tokens"] != 256 or abs(s["decode_tps"] * s["decode_seconds"] - 255) >= 1e-8 for s in group):
            raise RuntimeError("Wrong decode denominator or output count")
    decode = [c["decode_tps"] / math.sqrt(b["decode_tps"] * a["decode_tps"]) for c, b, a in zip(candidate, before, after)]
    ttft = [c["ttft_seconds"] / math.sqrt(b["ttft_seconds"] * a["ttft_seconds"]) for c, b, a in zip(candidate, before, after)]
    speed, latency = statistics.geometric_mean(decode), statistics.geometric_mean(ttft)
    return dict(matched_decode_ratio=speed, matched_ttft_ratio=latency,
                decode_request_ratios=decode, ttft_request_ratios=ttft,
                supported=speed > 1 and all(r > 1 for r in decode) and latency <= 1.02)


def admission(variant, context):
    path = ROOT / f"{variant}-context{context}-admission.json"
    if path.exists():
        raise FileExistsError(path)
    phases = [run(variant, context, True, flag, 0, 1) for flag in (0, 1)]
    sources = [Path(__file__), *[prefix_for(variant, context, True, flag, 0, 1).with_suffix(".qualification.json") for flag in (0, 1)]]
    result = dict(status="decode_stack_frozen_prefixes_and_routing_admitted",
        variant=variant, context=context, phases=phases, quality_qualified=False, performance_qualified=False,
        source_and_evidence_sha256={str(p): sha(p) for p in sources})
    with path.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


def screen(variant, context, repeat):
    rows, sources = [], [Path(__file__)]
    baseline = int(repeat)
    for full in (True, False):
        groups, times = [], []
        for flag, trial in ((0, baseline), (1, baseline), (0, baseline + 1)):
            if not (repeat and flag == 0 and trial == baseline):
                run(variant, context, full, flag, trial, 0)
            prefix = prefix_for(variant, context, full, flag, trial, 0)
            proof_path = prefix.with_suffix(".qualification.json")
            proof = verify_sources(proof_path, "decode_stack_original_prefixes_qualified")
            raw, guard = [json.loads(prefix.with_suffix(s).read_text()) for s in (".json", ".guard.json")]
            samples = [s for s in raw["samples"] if not s["warmup"]]
            if (proof["variant"] != variant or proof["context"] != context or proof["full_validation"] != full or
                    proof["flag"] != flag or proof["trial"] != trial or proof["trace"] != 0 or
                    guard["status"] != "completed" or len(raw["samples"]) != 4 or len(samples) != 3):
                raise RuntimeError("Wrong natural bracketed sample")
            groups.append(samples)
            times.append((proof["started_ns"], proof["completed_ns"]))
            sources.append(proof_path)
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
    supported = all(r["comparison"]["supported"] for r in rows)
    result = dict(status="decode_stack_gain_both_modes" if supported else "decode_stack_no_supported_gain",
        variant=variant, context=context, repeat=repeat, rows=rows, reused_initial_ending_controls=repeat,
        quality_qualified=False, default_enabled=False,
        prefill_scope="append_tokens_through_first_token_selection",
        all_candidate_modes_above_100_decode_tps=all(r["candidate"]["median_decode_tps"] > 100 for r in rows),
        source_and_evidence_sha256={str(p): sha(p) for p in sources})
    path = ROOT / f"{variant}-context{context}-{'repeat-' if repeat else ''}screen.json"
    with path.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return path, result


if __name__ == "__main__":
    state_path = ROOT / "gpu-decode-stack-v1-queued-pipeline.json"
    if state_path.exists():
        raise FileExistsError(state_path)
    state = dict(status="waiting_for_slice_pipeline", reports=[], goal_reached=False, started_ns=time.time_ns())

    def persist():
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    def qualify(variant, context):
        state.update(status="frozen_admission", variant=variant, context=context)
        persist()
        admission(variant, context)
        for repeat in (False, True):
            state.update(status="natural_decode_screen", repeat=repeat)
            persist()
            path, result = screen(variant, context, repeat)
            state["reports"].append(dict(path=str(path), **{k: result[k] for k in ("variant", "context", "repeat", "status", "rows")}))
            persist()
            if result["status"] != "decode_stack_gain_both_modes":
                return False
        return True

    persist()
    try:
        deadline = time.monotonic() + 4 * 3600
        while True:
            previous = json.loads((ROOT / "gpu-slice-v6-r2-queued-pipeline.json").read_text())
            if previous["status"] not in ("waiting_for_original_pipeline", "frozen2k_admission", "natural_ttft_screen"):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Slice pipeline wait exceeded four hours")
            time.sleep(20)
        state["preceding_slice_terminal"] = previous
        repeat2k = verify_sources(ROOT / "gpu-slice-v6-r2-context2048-repeat-ttft-screen.json", "gpu_slice_ttft_gain_both_modes")
        state["preceding_2k_slice_repeat_supported"] = all(r["comparison"]["supported"] for r in repeat2k["rows"])
        fc2 = "fc2-private-a-stack-v1"
        fc2_supported = qualify(fc2, 2048)
        combined = "gpu-slice-combined32k-v1"
        for context in (2048, 8192):
            verify_sources(ROOT / f"gpu-slice-v6-r2-context{context}-repeat-ttft-screen.json", "gpu_slice_ttft_gain_both_modes")
        combined_supported = qualify(combined, 32768)
        if fc2_supported:
            if qualify(fc2, 8192) and combined_supported:
                qualify(fc2, 32768)
        state["status"] = "bounded_stack_qualification_complete"
        state["goal_reached"] = all(any(r["context"] == context and r["repeat"] and
            r["status"] == "decode_stack_gain_both_modes" and all(row["candidate"]["median_decode_tps"] > 100 for row in r["rows"])
            for r in state["reports"]) for context in (2048, 8192, 32768))
        persist()
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        persist()
        raise
