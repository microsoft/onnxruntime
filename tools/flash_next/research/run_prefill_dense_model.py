import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from qsa_rank_decode_windows import WINDOW_PATTERN


def verify_windows(data, context, prefix="dense_bitcast_unpack"):
    mode_field = prefix + "_flag"
    rows = data["window_samples"]
    assert len(rows) == 32
    keyed = {(r["repetition"], r["window"]): r for r in rows}
    assert set(keyed) == {(r, w) for r in range(4) for w in range(8)}
    for (repetition, window), row in keyed.items():
        assert row["context"] == context
        assert row[mode_field] == WINDOW_PATTERN[window] ^ (repetition % 2)
        assert row["decode_tokens"] == 32 and row["decode_seconds"] > 0
        assert row["logical_sequence_tokens_before"] == context + 32 + window * 32
        assert math.isclose(row["decode_tps"], 32 / row["decode_seconds"], rel_tol=1e-12)
    ratios = []
    for repetition in (0, 2):
        for window in range(8):
            pair = {keyed[(r, window)][mode_field]: keyed[(r, window)]["decode_tps"]
                    for r in (repetition, repetition + 1)}
            ratios.append(pair[1] / pair[0])
    summary = data["window_summary"][str(context)]
    assert summary["matched_window_pairs"] == 16 and summary["all_request_output_tokens_match"]
    expected = math.exp(statistics.mean(math.log(r) for r in ratios))
    assert math.isclose(summary["geometric_mean_matched_window_ratio"], expected, rel_tol=1e-12)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--phase", choices=("windows", "standard"), default="windows")
    parser.add_argument("--mode", type=int, choices=(0, 1), default=1)
    parser.add_argument("--pending", type=int, choices=(64, 128, 256, 512), default=64)
    parser.add_argument("--chunk", type=int, choices=(64, 128), default=64)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    directory = Path("/home/sayanshaw/repos/qwen38-int4-prefill-dense-stack/build-v1")
    provider = directory / "libonnxruntime_providers_webgpu.so"
    proof = json.loads((ROOT / "prefill-dense-stack-v1-native-gpu7-20261029.qualification.json").read_text())
    parent = json.loads((ROOT / "qmoe-extended-prefill-v1-contexts-20261029.qualification.json").read_text())
    assert proof["status"] == "combined_dense_and_prefill_native_qualified"
    assert sha(provider) == proof["provider_sha256"]
    assert proof["baseline_provider_sha256"] == parent["provider_sha256"]
    if args.chunk == 128:
        prefill = json.loads((ROOT / "prefill128-dense-stack-v1-contexts-20261029.qualification.json").read_text())
        assert prefill["provider_sha256"] == proof["provider_sha256"]
        assert prefill["all_original_sequences_match"] and prefill["zero_major_faults"]
    baseline = "dense-bitcast-unpack-v1-gpu7-windows-20261029"
    command = json.loads((ROOT / (baseline + ".guard.json")).read_text())["command"]
    suffix = "" if args.phase == "windows" else f"-mode{args.mode}"
    tuning = f"-chunk{args.chunk}-pending{args.pending}" if (args.chunk, args.pending) != (64, 64) else ""
    prefix = args.output_prefix or ROOT / f"prefill-dense-stack-v1{tuning}-gpu7-{args.phase}{args.context}{suffix}-20261029"
    assert prefix.parent == ROOT
    if any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json", ".qualification.json")):
        raise FileExistsError(prefix)
    original_provider = "/workspace/qwen38-int4-dense-bitcast-unpack/build-v1/libonnxruntime_providers_webgpu.so"
    assert command[-1].count(original_provider) == 1
    command[-1] = command[-1].replace(original_provider, str(provider).replace(
        "/home/sayanshaw/repos", "/workspace")).replace(baseline, prefix.name).replace(
        "--contexts 2048", f"--contexts {args.context}")
    if args.phase == "standard":
        marker = "--tokens 288 --repetitions 4 --compare-dense-bitcast-unpack-windows"
        assert marker in command[-1]
        command[-1] = command[-1].replace(marker, "--tokens 256 --repetitions 3").replace(
            "ORT_WEBGPU_DENSE_BITCAST_UNPACK=1", f"ORT_WEBGPU_DENSE_BITCAST_UNPACK={args.mode}")
    pending_marker = "--webgpu-max-pending-dispatches 64"
    assert command[-1].count(pending_marker) == 1
    command[-1] = command[-1].replace(pending_marker, f"--webgpu-max-pending-dispatches {args.pending}")
    command[-1] = "export ORT_WEBGPU_QMOE_EXTENDED_PREFILL=1; " + command[-1] + f" --chunk {args.chunk}"
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "1800",
        "--output-prefix", str(prefix), "--", *command], check=True)
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    raw = json.loads(prefix.with_suffix(".json").read_text())
    assert guard["status"] == raw["status"] == "completed" and guard["exit_code"] == 0
    assert guard["command"] == command and guard["gpu_index"] == 7
    assert guard["gpu_uuid"] == "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
    assert raw["plugin_sha256"] == proof["provider_sha256"]
    assert raw["graph_sha256"] == parent["graph_sha256"] == sha(model / "model.onnx")
    assert raw["config_sha256"] == parent["config_sha256"]
    assert raw["provider"] == "webgpu" and raw["chunk"] == args.chunk
    assert raw["webgpu_max_pending_dispatches_override"] == args.pending
    assert raw["webgpu_validation_mode_override"] is None
    controls = json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())
    references = [s for s in controls["samples"] if s["context"] == args.context]
    assert len(raw["samples"]) == len(references) == 4
    for index, sample in enumerate(raw["samples"]):
        reference = references[1] if args.phase == "windows" else references[index]
        assert sample["prompt_sha256"] == reference["prompt_sha256"]
        assert len(sample["output_token_ids"]) == (288 if args.phase == "windows" else 256)
        assert sample["output_token_ids"][:256] == reference["output_token_ids"]
        assert all(v["major_page_faults"] == 0 for v in sample["process_resource_usage"].values())
        if args.phase == "windows":
            assert sample["same_prompt_output_matches"]
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    result = verify_windows(raw, args.context) if args.phase == "windows" else raw["summary"][f"{args.context}:configured"]
    report = dict(status="original_sequences_qualified", context=args.context, phase=args.phase,
        dense_mode=None if args.phase == "windows" else args.mode, chunk=args.chunk,
        pending_dispatches=args.pending, zero_major_faults=True,
        original_output_prefixes_match=True, provider_sha256=proof["provider_sha256"],
        graph_sha256=parent["graph_sha256"], cache_preflight=cache,
        result=result, quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "context", "phase", "result")}), flush=True)


if __name__ == "__main__":
    main()
