"""Time a CPU pooler after matched CPU or precompiled NPU vision encoders."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import run_acc
import run_winml_ep

from drift_investigation.vision.benchmark_encoder_latency import (
    _parse_masks,
    _parse_named_paths,
    summarize,
    timed_runs,
)


class SplitPipeline:
    def __init__(self, encoder: Any, pooler: Any, feature_input_name: str, forwarded_inputs: list[str]):
        self.encoder = encoder
        self.pooler = pooler
        self.feature_input_name = feature_input_name
        self.forwarded_inputs = forwarded_inputs
        self.stages: list[tuple[float, float]] = []

    def run(self, _output_names: Any, inputs: dict[str, np.ndarray]):
        start = time.perf_counter_ns()
        features = self.encoder.run(None, inputs)[0]
        after_encoder = time.perf_counter_ns()
        pooler_inputs = {self.feature_input_name: features}
        pooler_inputs.update({name: inputs[name] for name in self.forwarded_inputs})
        outputs = self.pooler.run(None, pooler_inputs)
        after_pooler = time.perf_counter_ns()
        self.stages.append(
            ((after_encoder - start) / 1_000_000, (after_pooler - after_encoder) / 1_000_000)
        )
        return outputs


def compare_pipeline(
    ort: Any,
    device: Any,
    pooler: Any,
    source: Path,
    compiled: Path,
    input_path: Path,
    mask_name: str | None,
    feature_input_name: str,
    forwarded_inputs: list[str],
    warmups: int,
    iterations: int,
    provider_options: dict[str, str] | None = None,
) -> dict[str, Any]:
    cpu_options = ort.SessionOptions()
    cpu_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cpu_encoder = ort.InferenceSession(str(source), sess_options=cpu_options, providers=["CPUExecutionProvider"])
    inputs, _ = run_acc.load_inputs(cpu_encoder, input_path, mask_name)

    npu_options = ort.SessionOptions()
    npu_options.add_provider_for_devices([device], provider_options or {})
    npu_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    npu_encoder = ort.InferenceSession(str(compiled), sess_options=npu_options)
    if [value.name for value in cpu_encoder.get_outputs()] != [value.name for value in npu_encoder.get_outputs()]:
        raise RuntimeError(f"encoder outputs differ: {source}, {compiled}")
    missing = [name for name in forwarded_inputs if name not in inputs]
    if missing:
        raise RuntimeError(f"pooler forwarded inputs are absent from encoder inputs: {missing}")

    cpu_pipeline = SplitPipeline(cpu_encoder, pooler, feature_input_name, forwarded_inputs)
    npu_pipeline = SplitPipeline(npu_encoder, pooler, feature_input_name, forwarded_inputs)
    cpu_out, cpu_samples = timed_runs(cpu_pipeline, inputs, warmups, iterations)
    npu_out, npu_samples = timed_runs(npu_pipeline, inputs, warmups, iterations)
    if len(cpu_out) != 1 or len(npu_out) != 1 or cpu_out[0].shape != npu_out[0].shape:
        raise RuntimeError(f"pipeline output count or shape differs for {source}")
    metrics = run_acc.measure_float_output(cpu_out[0], npu_out[0], 0, 0)
    cpu_stats, npu_stats = summarize(cpu_samples), summarize(npu_samples)
    result = {
        "encoder": str(source),
        "compiled_encoder": str(compiled),
        "ep_context_nodes": run_acc.context_node_count(compiled),
        "output_shape": list(cpu_out[0].shape),
        "cpu": cpu_stats,
        "npu_encoder_plus_cpu_pooler": npu_stats,
        "cpu_encoder_stage": summarize([stage[0] for stage in cpu_pipeline.stages[warmups:]]),
        "cpu_pooler_stage": summarize([stage[1] for stage in cpu_pipeline.stages[warmups:]]),
        "npu_encoder_stage": summarize([stage[0] for stage in npu_pipeline.stages[warmups:]]),
        "npu_cpu_pooler_stage": summarize([stage[1] for stage in npu_pipeline.stages[warmups:]]),
        "median_speedup_cpu_over_npu_pipeline": cpu_stats["median_ms"] / npu_stats["median_ms"],
        "cpu_npu_mean_abs_error": metrics["mean_abs_error"],
        "cpu_npu_cosine": metrics["cosine_similarity"],
    }
    del cpu_encoder, npu_encoder
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", nargs=3, metavar=("NAME", "CPU_MODEL", "COMPILED_MODEL"),
                        required=True, help="Named source/precompiled encoder pair (repeatable).")
    parser.add_argument("--input-set", action="append", nargs=2, metavar=("NAME", "NPZ"), required=True,
                        help="Named input NPZ archive (repeatable).")
    parser.add_argument("--valid-mask", action="append", default=[], metavar="INPUT_SET=ARRAY_NAME",
                        help="Mask array excluded from session inputs (repeatable).")
    parser.add_argument("--pooler", type=Path, required=True, help="CPU pooler/projector ONNX model.")
    parser.add_argument("--feature-input-name", default="vision_features",
                        help="Pooler input receiving the encoder's sole output.")
    parser.add_argument("--forward-input", action="append",
                        help="Encoder input also forwarded to the pooler (repeatable; default: pixel_position_ids).")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--pooler-threads", type=int, default=0)
    parser.add_argument("--provider", choices=run_winml_ep.NPU_PROVIDERS, default="openvino")
    parser.add_argument("--provider-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.warmups < 1 or args.iterations < 3:
        raise ValueError("require at least one warmup and three measured runs")
    if args.pooler_threads < 0:
        raise ValueError("--pooler-threads must be non-negative")

    variants = _parse_named_paths(args.variant, "variant")
    input_pairs = _parse_named_paths([[name, path, path] for name, path in args.input_set], "input-set")
    inputs = {name: paths[0] for name, paths in input_pairs.items()}
    masks = _parse_masks(args.valid_mask, set(inputs))
    provider_options = run_winml_ep.parse_provider_options(args.provider_option)
    forwarded_inputs = args.forward_input or ["pixel_position_ids"]
    pooler_path = args.pooler.resolve()

    if not pooler_path.is_file():
        raise RuntimeError(f"CPU pooler model missing: {pooler_path}")
    for source, compiled in variants.values():
        if not source.is_file() or not compiled.is_file() or run_acc.context_node_count(compiled) != 1:
            raise RuntimeError(f"source or single-EPContext compiled model missing: {source}, {compiled}")
    for input_path in inputs.values():
        if not input_path.is_file():
            raise RuntimeError(f"input archive missing: {input_path}")
    config = {
        "warmups": args.warmups,
        "iterations": args.iterations,
        "pooler": str(pooler_path),
        "pooler_sha256": run_acc.file_sha256(pooler_path),
        "pooler_threads": args.pooler_threads,
        "provider": run_winml_ep.PROVIDER_NAMES[args.provider],
        "provider_options": provider_options,
        "cpu_fallback": False,
        "feature_input_name": args.feature_input_name,
        "forwarded_inputs": forwarded_inputs,
        "models": {
            name: {"source": run_acc.file_sha256(paths[0]), "compiled": run_acc.file_sha256(paths[1])}
            for name, paths in variants.items()
        },
        "inputs": {name: run_acc.file_sha256(path) for name, path in inputs.items()},
        "valid_masks": masks,
    }
    report_path = args.report.resolve()
    if report_path.exists():
        if not args.resume:
            raise ValueError(f"report already exists; use --resume: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["config"] != config:
            raise ValueError("existing pipeline benchmark configuration differs")
    else:
        if args.resume:
            raise ValueError(f"cannot resume missing report: {report_path}")
        report = {"config": config, "results": {}}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    provider_name = run_winml_ep.PROVIDER_NAMES[args.provider]
    ort = run_winml_ep.register_provider(provider_name)
    device = run_winml_ep.find_npu_device(ort, provider_name)
    pooler_options = ort.SessionOptions()
    if args.pooler_threads:
        pooler_options.intra_op_num_threads = args.pooler_threads
    pooler = ort.InferenceSession(str(pooler_path), sess_options=pooler_options, providers=["CPUExecutionProvider"])
    for input_name, input_path in inputs.items():
        report["results"].setdefault(input_name, {})
        for variant_name, (source, compiled) in variants.items():
            if variant_name in report["results"][input_name]:
                print(f"Skipping completed {input_name}/{variant_name}", flush=True)
                continue
            print(f"Benchmarking full pipeline {input_name}/{variant_name}", flush=True)
            result = compare_pipeline(
                ort, device, pooler, source, compiled, input_path, masks.get(input_name),
                args.feature_input_name, forwarded_inputs, args.warmups, args.iterations, provider_options,
            )
            report["results"][input_name][variant_name] = result
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(
                f"  CPU {result['cpu']['median_ms']:.1f} ms, NPU+CPU pooler "
                f"{result['npu_encoder_plus_cpu_pooler']['median_ms']:.1f} ms, "
                f"speedup {result['median_speedup_cpu_over_npu_pipeline']:.2f}x",
                flush=True,
            )
    print(f"Saved split-pipeline latency comparison: {report_path}")


if __name__ == "__main__":
    main()
