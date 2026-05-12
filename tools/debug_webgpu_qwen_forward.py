from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ort = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time Qwen ONNX forward passes with the WebGPU EP using onnxruntime only."
    )
    parser.add_argument(
        "--ort-source",
        choices=["auto", "local", "installed"],
        default="auto",
        help="Which onnxruntime Python package to import.",
    )
    parser.add_argument(
        "--ort-package-root",
        type=Path,
        default=None,
        help="Optional package root containing onnxruntime/ and onnxruntime_pybind11_state.pyd.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(r"d:\models\Qwen3-1.7B"),
        help="Path to the model directory containing genai_config.json and model.onnx.",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=16,
        help="Synthetic prefill prompt length.",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=1,
        help="Number of single-token decode iterations to run after prefill.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup iterations for each stage.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Timed iterations for each stage.",
    )
    parser.add_argument(
        "--verbose-io",
        action="store_true",
        help="Print model input/output metadata.",
    )
    parser.add_argument(
        "--provider",
        choices=["webgpu", "cpu"],
        default="webgpu",
        help="Execution provider to use for the session.",
    )
    parser.add_argument(
        "--opt-level",
        choices=["disable", "basic", "extended", "all"],
        default="all",
        help="ONNX Runtime graph optimization level for session creation.",
    )
    parser.add_argument(
        "--dump-optimized-model",
        type=Path,
        default=None,
        help="Optional output path for the optimized model produced during session creation.",
    )
    parser.add_argument(
        "--disable-optimizer",
        action="append",
        default=[],
        help="Optimizer name to disable during session creation. Can be provided multiple times.",
    )
    parser.add_argument(
        "--matmulnbits-silu-fusion",
        choices=["default", "on", "off"],
        default="default",
        help="Override session.enable_matmulnbits_silu_fusion for benchmarking.",
    )
    parser.add_argument(
        "--compare-matmulnbits-silu-fusion",
        action="store_true",
        help="Run the benchmark twice with session.enable_matmulnbits_silu_fusion forced off and on.",
    )
    parser.add_argument(
        "--log-severity",
        type=int,
        default=2,
        help="ORT log severity level: 0=verbose, 1=info, 2=warning, 3=error, 4=fatal.",
    )
    parser.add_argument(
        "--log-verbosity",
        type=int,
        default=0,
        help="ORT verbose logging level used when --log-severity is 0.",
    )
    parser.add_argument(
        "--profile-shaders",
        action="store_true",
        help="Enable ORT profiling and summarize WebGPU shader dispatch times from the emitted profile JSON.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Optional directory for ORT profile JSON output. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--profile-top-k",
        type=int,
        default=20,
        help="Number of aggregated shader entries to print from the profile summary.",
    )
    parser.add_argument(
        "--profile-name-filter",
        type=str,
        default=None,
        help="Optional case-insensitive substring filter applied to shader event names before aggregation.",
    )
    parser.add_argument(
        "--profile-stages-separately",
        action="store_true",
        help="Create separate profiled sessions for prefill and decode so shader summaries are stage-specific.",
    )
    return parser.parse_args()


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_local_ort_package_root() -> Path | None:
    candidate = get_repo_root() / "WGP4" / "Release" / "Release"
    if (candidate / "onnxruntime" / "__init__.py").exists() and (candidate / "onnxruntime_pybind11_state.pyd").exists():
        return candidate

    return None


def resolve_ort_package_root(args: argparse.Namespace) -> Path | None:
    if args.ort_source == "installed":
        return None

    if args.ort_package_root is not None:
        return args.ort_package_root.resolve()

    local_root = find_local_ort_package_root()
    if args.ort_source == "local" and local_root is None:
        raise RuntimeError("--ort-source local was requested, but no local ORT package root was found.")

    return local_root


def import_onnxruntime(package_root: Path | None) -> Any:
    if package_root is not None:
        sys.path.insert(0, str(package_root))

    import onnxruntime as ort_module

    return ort_module


def resolve_matmulnbits_silu_fusion_config(mode: str) -> str | None:
    if mode == "default":
        return None

    return "1" if mode == "on" else "0"


def load_config(model_dir: Path) -> dict[str, Any]:
    with (model_dir / "genai_config.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def ort_type_to_numpy(ort_type: str) -> np.dtype:
    mapping = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint64)": np.uint64,
        "tensor(uint32)": np.uint32,
        "tensor(uint16)": np.uint16,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    if ort_type not in mapping:
        raise ValueError(f"Unsupported ORT tensor type: {ort_type}")
    return mapping[ort_type]


def resolve_dim(dim: Any, *, batch_size: int, seq_len: int, past_seq_len: int, num_kv_heads: int, head_size: int) -> int:
    if isinstance(dim, int):
        return dim

    dim_text = "" if dim is None else str(dim).lower()
    if "batch" in dim_text:
        return batch_size
    if "num_key_value_heads" in dim_text or ("head" in dim_text and "num" in dim_text):
        return num_kv_heads
    if dim_text in {"sequence_length", "input_sequence_length", "decoder_sequence_length"}:
        return seq_len
    if "past_sequence_length" in dim_text or "past_seq" in dim_text or "kv_sequence_length" in dim_text:
        return past_seq_len
    if "total_sequence_length" in dim_text or "all_sequence_length" in dim_text:
        return past_seq_len + seq_len
    if "head_size" in dim_text or ("head" in dim_text and "size" in dim_text):
        return head_size
    if "sequence" in dim_text or "seq" in dim_text:
        return past_seq_len + seq_len

    return 1


def make_zero_past_tensor(
    input_meta: ort.NodeArg,
    *,
    batch_size: int,
    past_seq_len: int,
    num_kv_heads: int,
    head_size: int,
) -> np.ndarray:
    shape = [
        resolve_dim(
            dim,
            batch_size=batch_size,
            seq_len=0,
            past_seq_len=past_seq_len,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        for dim in input_meta.shape
    ]
    return np.zeros(shape, dtype=ort_type_to_numpy(input_meta.type))


def build_token_ids(batch_size: int, seq_len: int, vocab_size: int) -> np.ndarray:
    token_ids = (np.arange(seq_len, dtype=np.int64) + 1) % max(vocab_size - 1, 1)
    token_ids = np.maximum(token_ids, 1)
    return np.tile(token_ids.reshape(1, seq_len), (batch_size, 1))


def build_attention_mask(batch_size: int, total_seq_len: int, input_meta: ort.NodeArg) -> np.ndarray:
    dtype = ort_type_to_numpy(input_meta.type)
    return np.ones((batch_size, total_seq_len), dtype=dtype)


def build_prefill_feeds(
    session: ort.InferenceSession,
    config: dict[str, Any],
    prompt_len: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    decoder = config["model"]["decoder"]
    input_map = decoder["inputs"]
    vocab_size = int(config["model"]["vocab_size"])
    num_layers = int(decoder["num_hidden_layers"])
    num_kv_heads = int(decoder["num_key_value_heads"])
    head_size = int(decoder["head_size"])
    inputs_by_name = {meta.name: meta for meta in session.get_inputs()}

    feeds: dict[str, np.ndarray] = {}
    input_ids_name = input_map["input_ids"]
    feeds[input_ids_name] = build_token_ids(batch_size, prompt_len, vocab_size)

    attention_mask_name = input_map.get("attention_mask")
    if attention_mask_name and attention_mask_name in inputs_by_name:
        feeds[attention_mask_name] = build_attention_mask(batch_size, prompt_len, inputs_by_name[attention_mask_name])

    past_key_format = input_map.get("past_key_names")
    past_value_format = input_map.get("past_value_names")
    if past_key_format and past_value_format:
        for layer_idx in range(num_layers):
            key_name = past_key_format % layer_idx
            value_name = past_value_format % layer_idx
            if key_name in inputs_by_name:
                feeds[key_name] = make_zero_past_tensor(
                    inputs_by_name[key_name],
                    batch_size=batch_size,
                    past_seq_len=0,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                )
            if value_name in inputs_by_name:
                feeds[value_name] = make_zero_past_tensor(
                    inputs_by_name[value_name],
                    batch_size=batch_size,
                    past_seq_len=0,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                )

    return feeds


def build_decode_feeds(
    session: ort.InferenceSession,
    config: dict[str, Any],
    *,
    batch_size: int,
    total_seq_len: int,
    next_token_id: int,
    past_outputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    decoder = config["model"]["decoder"]
    input_map = decoder["inputs"]
    inputs_by_name = {meta.name: meta for meta in session.get_inputs()}
    feeds: dict[str, np.ndarray] = {}

    feeds[input_map["input_ids"]] = np.full((batch_size, 1), next_token_id, dtype=np.int64)

    attention_mask_name = input_map.get("attention_mask")
    if attention_mask_name and attention_mask_name in inputs_by_name:
        feeds[attention_mask_name] = build_attention_mask(batch_size, total_seq_len, inputs_by_name[attention_mask_name])

    past_key_format = input_map.get("past_key_names")
    past_value_format = input_map.get("past_value_names")
    if past_key_format and past_value_format:
        num_layers = int(decoder["num_hidden_layers"])
        for layer_idx in range(num_layers):
            feeds[past_key_format % layer_idx] = past_outputs[past_key_format % layer_idx]
            feeds[past_value_format % layer_idx] = past_outputs[past_value_format % layer_idx]

    return feeds


def collect_present_outputs(config: dict[str, Any], session: ort.InferenceSession, outputs: list[np.ndarray]) -> dict[str, np.ndarray]:
    decoder = config["model"]["decoder"]
    output_map = decoder["outputs"]
    num_layers = int(decoder["num_hidden_layers"])
    output_names = [meta.name for meta in session.get_outputs()]
    output_dict = dict(zip(output_names, outputs))
    present: dict[str, np.ndarray] = {}

    for layer_idx in range(num_layers):
        key_name = output_map["present_key_names"] % layer_idx
        value_name = output_map["present_value_names"] % layer_idx
        if key_name in output_dict:
            present[decoder["inputs"]["past_key_names"] % layer_idx] = output_dict[key_name]
        if value_name in output_dict:
            present[decoder["inputs"]["past_value_names"] % layer_idx] = output_dict[value_name]

    return present


def timed_run(session: ort.InferenceSession, feeds: dict[str, np.ndarray], warmup: int, runs: int) -> tuple[list[np.ndarray], list[float]]:
    outputs: list[np.ndarray] | None = None
    for _ in range(warmup):
        outputs = session.run(None, feeds)

    times_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        outputs = session.run(None, feeds)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    if outputs is None:
        raise RuntimeError("Session did not produce outputs.")

    return outputs, times_ms


def print_io(session: ort.InferenceSession) -> None:
    print("Inputs:")
    for meta in session.get_inputs():
        print(f"  {meta.name}: shape={meta.shape}, type={meta.type}")
    print("Outputs:")
    for meta in session.get_outputs():
        print(f"  {meta.name}: shape={meta.shape}, type={meta.type}")


def summarize(stage: str, times_ms: list[float]) -> None:
    mean_ms = float(np.mean(times_ms))
    median_ms = float(np.median(times_ms))
    std_ms = float(np.std(times_ms))
    print(f"{stage}: mean={mean_ms:.3f} ms median={median_ms:.3f} ms std={std_ms:.3f} ms runs={len(times_ms)}")


def summarize_stats(times_ms: list[float]) -> dict[str, float | int]:
    return {
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "runs": len(times_ms),
    }


def summarize_with_label(label: str, stage: str, times_ms: list[float]) -> None:
    stats = summarize_stats(times_ms)
    print(
        f"{label} {stage}: mean={stats['mean_ms']:.3f} ms median={stats['median_ms']:.3f} ms std={stats['std_ms']:.3f} ms runs={stats['runs']}"
    )


def sanitize_profile_label(label: str | None) -> str:
    if not label:
        return "default"

    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)


def resolve_profile_prefix(args: argparse.Namespace, label: str | None) -> str:
    profile_dir = args.profile_dir.resolve() if args.profile_dir is not None else Path.cwd()
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_name = f"qwen_webgpu_{sanitize_profile_label(label)}"
    return str(profile_dir / profile_name)


def load_profile_events(profile_path: Path) -> list[dict[str, Any]]:
    with profile_path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)

    if not isinstance(loaded, list):
        raise RuntimeError(f"Unexpected ORT profile payload in {profile_path}.")

    return [event for event in loaded if isinstance(event, dict)]


def summarize_shader_profile(
    profile_path: Path,
    *,
    top_k: int,
    name_filter: str | None,
    invocation_count: int | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    events = load_profile_events(profile_path)
    normalized_filter = name_filter.lower() if name_filter else None
    shader_events: list[dict[str, Any]] = []
    for event in events:
        if event.get("cat") != "Api":
            continue

        name = str(event.get("name", ""))
        if "&" not in name:
            continue

        if normalized_filter is not None and normalized_filter not in name.lower():
            continue

        shader_events.append(event)

    per_shader: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {"total_us": 0.0, "count": 0, "max_us": 0.0, "args": {}}
    )
    total_shader_us = 0.0
    for event in shader_events:
        name = str(event.get("name", ""))
        duration_us = float(event.get("dur", 0.0))
        total_shader_us += duration_us

        aggregated = per_shader[name]
        aggregated["total_us"] += duration_us
        aggregated["count"] += 1
        aggregated["max_us"] = max(float(aggregated["max_us"]), duration_us)
        if not aggregated["args"]:
            event_args = event.get("args", {})
            aggregated["args"] = event_args if isinstance(event_args, dict) else {}

    top_entries: list[dict[str, Any]] = []
    for name, aggregated in per_shader.items():
        mean_us = aggregated["total_us"] / aggregated["count"] if aggregated["count"] else 0.0
        top_entries.append(
            {
                "name": name,
                "total_us": aggregated["total_us"],
                "total_ms": aggregated["total_us"] / 1000.0,
                "count": aggregated["count"],
                "mean_us": mean_us,
                "mean_ms": mean_us / 1000.0,
                "max_us": aggregated["max_us"],
                "max_ms": aggregated["max_us"] / 1000.0,
                "args": aggregated["args"],
            }
        )

    top_entries.sort(key=lambda entry: float(entry["total_us"]), reverse=True)
    summary = {
        "path": str(profile_path),
        "event_count": len(shader_events),
        "total_shader_us": total_shader_us,
        "total_shader_ms": total_shader_us / 1000.0,
        "invocation_count": invocation_count,
        "avg_shader_ms_per_invocation": (total_shader_us / 1000.0 / invocation_count)
        if invocation_count and invocation_count > 0
        else None,
        "top_entries": top_entries[: max(top_k, 0)],
        "all_entries": top_entries,
        "label": label,
        "name_filter": name_filter,
    }

    prefix = label or "profile"
    filter_suffix = "" if not name_filter else f" filter={name_filter!r}"
    invocation_suffix = (
        ""
        if summary["avg_shader_ms_per_invocation"] is None
        else f" avg_per_invocation={summary['avg_shader_ms_per_invocation']:.3f} ms"
    )
    print(
        f"{prefix} shader profile: total_gpu={summary['total_shader_ms']:.3f} ms events={summary['event_count']} file={profile_path}{filter_suffix}{invocation_suffix}"
    )
    if not top_entries:
        print(f"{prefix} shader profile: no WebGPU shader events matched.")
        return summary

    for entry in summary["top_entries"]:
        print(
            f"  {prefix} shader: total={entry['total_ms']:.3f} ms mean={entry['mean_ms']:.3f} ms count={entry['count']} name={entry['name']}"
        )

    return summary


def compare_shader_profiles(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    title: str | None = None,
    before_label: str,
    after_label: str,
    top_k: int,
) -> None:
    before_total_ms = float(before["total_shader_ms"])
    after_total_ms = float(after["total_shader_ms"])
    if before_total_ms == 0.0:
        ratio_text = "n/a"
    else:
        ratio_text = f"{after_total_ms / before_total_ms:.3f}x"

    print(f"\n{title or 'Shader comparison'}:")
    print(
        f"  total_gpu: {before_label}={before_total_ms:.3f} ms {after_label}={after_total_ms:.3f} ms delta={after_total_ms - before_total_ms:.3f} ms ratio={ratio_text}"
    )
    before_avg = before.get("avg_shader_ms_per_invocation")
    after_avg = after.get("avg_shader_ms_per_invocation")
    if before_avg is not None and after_avg is not None:
        avg_ratio_text = "n/a" if before_avg == 0.0 else f"{after_avg / before_avg:.3f}x"
        print(
            f"  avg_per_invocation: {before_label}={before_avg:.3f} ms {after_label}={after_avg:.3f} ms delta={after_avg - before_avg:.3f} ms ratio={avg_ratio_text}"
        )

    before_entries = {entry["name"]: entry for entry in before["all_entries"]}
    after_entries = {entry["name"]: entry for entry in after["all_entries"]}
    deltas: list[dict[str, Any]] = []
    for name in sorted(set(before_entries) | set(after_entries)):
        before_entry = before_entries.get(name)
        after_entry = after_entries.get(name)
        before_ms = float(before_entry["total_ms"]) if before_entry else 0.0
        after_ms = float(after_entry["total_ms"]) if after_entry else 0.0
        deltas.append(
            {
                "name": name,
                "before_ms": before_ms,
                "after_ms": after_ms,
                "delta_ms": after_ms - before_ms,
                "before_count": int(before_entry["count"]) if before_entry else 0,
                "after_count": int(after_entry["count"]) if after_entry else 0,
            }
        )

    deltas.sort(key=lambda entry: abs(float(entry["delta_ms"])), reverse=True)
    for entry in deltas[: max(top_k, 0)]:
        print(
            f"  shader delta: delta={entry['delta_ms']:.3f} ms {before_label}={entry['before_ms']:.3f} ms ({entry['before_count']}) {after_label}={entry['after_ms']:.3f} ms ({entry['after_count']}) name={entry['name']}"
        )


def resolve_webgpu_provider(available: list[str]) -> str:
    for provider_name in ("WebGpuExecutionProvider", "WebGPUExecutionProvider"):
        if provider_name in available:
            return provider_name

    raise RuntimeError(f"WebGPU execution provider is not available. Providers: {available}")


def resolve_provider(available: list[str], provider_key: str) -> str:
    if provider_key == "cpu":
        if "CPUExecutionProvider" in available:
            return "CPUExecutionProvider"
        raise RuntimeError(f"CPU execution provider is not available. Providers: {available}")

    return resolve_webgpu_provider(available)


def resolve_optimization_level(opt_level: str) -> ort.GraphOptimizationLevel:
    mapping = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    return mapping[opt_level]


def resolve_dump_optimized_model_path(base_path: Path | None, label: str | None) -> Path | None:
    if base_path is None or label is None:
        return base_path

    safe_label = label.replace("=", "_").replace(" ", "_")
    return base_path.with_name(f"{base_path.stem}_{safe_label}{base_path.suffix}")


def _iter_graph_nodes(graph: Any) -> Any:
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.type == attr.GRAPH:
                yield from _iter_graph_nodes(attr.g)
            elif attr.type == attr.GRAPHS:
                for subgraph in attr.graphs:
                    yield from _iter_graph_nodes(subgraph)


def summarize_raw_mlp_patterns(model_path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("[debug] unable to import onnx; skipping raw model MLP pattern summary", flush=True)
        return

    if not model_path.exists():
        print(f"[debug] model file not found for raw pattern summary: {model_path}", flush=True)
        return

    model = onnx.load(str(model_path), load_external_data=False)
    consumers_by_input: dict[str, list[Any]] = collections.defaultdict(list)
    for node in _iter_graph_nodes(model.graph):
        for input_name in node.input:
            if input_name:
                consumers_by_input[input_name].append(node)

    def has_exact_mlp_pattern(matmul_nodes: list[Any]) -> bool:
        for gate_node in matmul_nodes:
            if not gate_node.output:
                continue

            gate_output = gate_node.output[0]
            sigmoid_nodes = [consumer for consumer in consumers_by_input.get(gate_output, []) if consumer.op_type == "Sigmoid"]
            silu_mul_nodes = [consumer for consumer in consumers_by_input.get(gate_output, []) if consumer.op_type == "Mul"]
            if not sigmoid_nodes or not silu_mul_nodes:
                continue

            up_nodes = [candidate for candidate in matmul_nodes if candidate is not gate_node and candidate.output]
            for sigmoid_node in sigmoid_nodes:
                sigmoid_output = sigmoid_node.output[0] if sigmoid_node.output else ""
                if not sigmoid_output:
                    continue

                matching_silu_mul_nodes = [
                    mul_node
                    for mul_node in silu_mul_nodes
                    if sigmoid_output in mul_node.input and gate_output in mul_node.input
                ]
                if not matching_silu_mul_nodes:
                    continue

                for silu_mul_node in matching_silu_mul_nodes:
                    silu_mul_output = silu_mul_node.output[0] if silu_mul_node.output else ""
                    if not silu_mul_output:
                        continue

                    for up_node in up_nodes:
                        up_output = up_node.output[0]
                        for final_mul_node in consumers_by_input.get(silu_mul_output, []):
                            if final_mul_node.op_type == "Mul" and up_output in final_mul_node.input:
                                return True

        return False

    counts = {
        "skip_two_matmuls": 0,
        "skip_exact_mlp": 0,
        "skip_exact_mlp_two_edges": 0,
        "simplified_two_matmuls": 0,
        "simplified_exact_mlp": 0,
        "simplified_exact_mlp_two_edges": 0,
    }

    for node in _iter_graph_nodes(model.graph):
        if node.op_type not in {"SkipSimplifiedLayerNormalization", "SimplifiedLayerNormalization"}:
            continue

        norm_outputs = [output_name for output_name in node.output if output_name]
        matmul_nodes = [
            consumer
            for output_name in norm_outputs
            for consumer in consumers_by_input.get(output_name, [])
            if consumer.op_type == "MatMulNBits"
        ]

        if len(matmul_nodes) < 2:
            continue

        if node.op_type == "SkipSimplifiedLayerNormalization":
            counts["skip_two_matmuls"] += 1
        else:
            counts["simplified_two_matmuls"] += 1

        if has_exact_mlp_pattern(matmul_nodes):
            total_output_edges = sum(len(consumers_by_input.get(output_name, [])) for output_name in norm_outputs)
            if node.op_type == "SkipSimplifiedLayerNormalization":
                counts["skip_exact_mlp"] += 1
                if total_output_edges == 2:
                    counts["skip_exact_mlp_two_edges"] += 1
            else:
                counts["simplified_exact_mlp"] += 1
                if total_output_edges == 2:
                    counts["simplified_exact_mlp_two_edges"] += 1

    print(
        "[debug] raw model MLP pattern summary: "
        f"skip_two_matmuls={counts['skip_two_matmuls']} skip_exact_mlp={counts['skip_exact_mlp']} "
        f"skip_exact_mlp_two_edges={counts['skip_exact_mlp_two_edges']} "
        f"simplified_two_matmuls={counts['simplified_two_matmuls']} simplified_exact_mlp={counts['simplified_exact_mlp']} "
        f"simplified_exact_mlp_two_edges={counts['simplified_exact_mlp_two_edges']}",
        flush=True,
    )


def summarize_fused_nodes(optimized_model_path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("[debug] unable to import onnx; skipping optimized model fusion summary", flush=True)
        return

    if not optimized_model_path.exists():
        print(f"[debug] optimized model file not found for fusion summary: {optimized_model_path}", flush=True)
        return

    model = onnx.load(str(optimized_model_path), load_external_data=False)
    qkv_fused_count = 0
    qkv_skip_fused_count = 0
    qkv_skip_passthrough_count = 0
    mlp_fused_count = 0
    mlp_plain_count = 0
    mlp_simplified_count = 0
    mlp_skip_count = 0
    mlp_skip_passthrough_count = 0
    gqa_total = 0
    gqa_with_qk_norm = 0

    for node in _iter_graph_nodes(model.graph):
        if node.op_type == "GroupQueryAttention":
            gqa_total += 1
            # GQA has q_norm_weight at input slot 14 and k_norm_weight at slot 15
            # (added by GroupQueryAttentionPreNormFusion).
            if len(node.input) > 15 and bool(node.input[14]) and bool(node.input[15]):
                gqa_with_qk_norm += 1
            continue

        if node.op_type == "MatMulNBitsQkv":
            qkv_fused_count += 1
            has_skip_input = len(node.input) > 1 and bool(node.input[1])
            if has_skip_input:
                qkv_skip_fused_count += 1
                if len(node.output) > 3 and bool(node.output[3]):
                    qkv_skip_passthrough_count += 1
            continue

        if node.op_type != "MatMulNBitsMlp":
            continue

        mlp_fused_count += 1
        has_skip_input = len(node.input) > 1 and bool(node.input[1])
        has_norm_scale = len(node.input) > 2 and bool(node.input[2])
        if has_skip_input:
            mlp_skip_count += 1
            if len(node.output) > 1 and bool(node.output[1]):
                mlp_skip_passthrough_count += 1
        elif has_norm_scale:
            mlp_simplified_count += 1
        else:
            mlp_plain_count += 1

    print(
        "[debug] optimized model fused-node summary: "
        f"qkv_total={qkv_fused_count} qkv_skip={qkv_skip_fused_count} "
        f"qkv_skip_passthrough={qkv_skip_passthrough_count} "
        f"qkv_simplified={qkv_fused_count - qkv_skip_fused_count} "
        f"mlp_total={mlp_fused_count} mlp_plain={mlp_plain_count} "
        f"mlp_simplified={mlp_simplified_count} mlp_skip={mlp_skip_count} "
        f"mlp_skip_passthrough={mlp_skip_passthrough_count} "
        f"gqa_total={gqa_total} gqa_with_qk_norm={gqa_with_qk_norm}",
        flush=True,
    )


def create_session(
    args: argparse.Namespace,
    model_path: Path,
    provider: str,
    fusion_config_value: str | None,
    *,
    label: str | None = None,
) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = resolve_optimization_level(args.opt_level)
    session_options.log_severity_level = args.log_severity
    session_options.log_verbosity_level = args.log_verbosity
    if args.profile_shaders:
        session_options.enable_profiling = True
        session_options.profile_file_prefix = resolve_profile_prefix(args, label)
    if fusion_config_value is not None:
        session_options.add_session_config_entry("session.enable_matmulnbits_silu_fusion", fusion_config_value)

    optimized_model_path = resolve_dump_optimized_model_path(args.dump_optimized_model, label)
    if optimized_model_path is not None:
        optimized_model_path.parent.mkdir(parents=True, exist_ok=True)
        session_options.optimized_model_filepath = str(optimized_model_path)
        print(f"[debug] optimized model output ({label or 'session'}): {optimized_model_path}", flush=True)

    print(f"[debug] graph optimization level: {args.opt_level}", flush=True)
    print(f"[debug] ORT log severity: {args.log_severity}", flush=True)
    print(f"[debug] ORT log verbosity: {args.log_verbosity}", flush=True)
    if args.disable_optimizer:
        print(f"[debug] disabled optimizers: {args.disable_optimizer}", flush=True)
    print(
        f"[debug] creating inference session{'' if label is None else f' ({label})'} with session.enable_matmulnbits_silu_fusion={fusion_config_value}",
        flush=True,
    )
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=[provider],
        disabled_optimizers=args.disable_optimizer,
    )
    print(f"[debug] session created{'' if label is None else f' ({label})'}", flush=True)
    if optimized_model_path is not None:
        summarize_fused_nodes(optimized_model_path)
    return session


def benchmark_prefill_stage(
    session: ort.InferenceSession,
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    batch_size = 1
    if args.verbose_io:
        print_io(session)

    print(f"[debug] building prefill feeds{'' if label is None else f' ({label})'}", flush=True)
    prefill_feeds = build_prefill_feeds(session, config, args.prompt_len, batch_size)
    print(f"[debug] running prefill{'' if label is None else f' ({label})'}", flush=True)
    prefill_outputs, prefill_times = timed_run(session, prefill_feeds, args.warmup, args.runs)
    present = collect_present_outputs(config, session, prefill_outputs)

    return {
        "feeds": prefill_feeds,
        "times": prefill_times,
        "present": present,
        "has_kv_cache": bool(present),
        "next_token_id": int(prefill_feeds[config["model"]["decoder"]["inputs"]["input_ids"]][0, -1]),
    }


def benchmark_decode_stage(
    session: ort.InferenceSession,
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    present: dict[str, np.ndarray],
    next_token_id: int,
    label: str | None = None,
) -> dict[str, Any]:
    decode_times: list[float] = []
    current_present = present
    total_seq_len = args.prompt_len + 1

    if current_present and args.decode_steps > 0:
        for step_idx in range(args.decode_steps):
            print(
                f"[debug] building decode feeds for step {step_idx}{'' if label is None else f' ({label})'}",
                flush=True,
            )
            decode_feeds = build_decode_feeds(
                session,
                config,
                batch_size=1,
                total_seq_len=total_seq_len,
                next_token_id=next_token_id,
                past_outputs=current_present,
            )
            print(
                f"[debug] running decode step {step_idx}{'' if label is None else f' ({label})'}",
                flush=True,
            )
            decode_outputs, step_times = timed_run(session, decode_feeds, args.warmup, args.runs)
            current_present = collect_present_outputs(config, session, decode_outputs)
            decode_times.extend(step_times)
            total_seq_len += 1

    return {"times": decode_times, "has_kv_cache": bool(current_present)}


def benchmark_forward_pass(
    session: ort.InferenceSession,
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    label: str | None = None,
) -> dict[str, list[float] | bool]:
    prefill_results = benchmark_prefill_stage(session, config, args, label=label)
    decode_results = benchmark_decode_stage(
        session,
        config,
        args,
        present=prefill_results["present"],
        next_token_id=int(prefill_results["next_token_id"]),
        label=label,
    )

    return {
        "prefill": list(prefill_results["times"]),
        "decode": list(decode_results["times"]),
        "has_kv_cache": bool(decode_results["has_kv_cache"]),
    }


def run_stage_split_profiled_benchmark(
    args: argparse.Namespace,
    model_path: Path,
    provider: str,
    config: dict[str, Any],
    fusion_config_value: str,
    *,
    label: str,
) -> tuple[dict[str, list[float] | bool], dict[str, dict[str, Any]]]:
    stage_results: dict[str, list[float] | bool] = {"prefill": [], "decode": [], "has_kv_cache": False}
    stage_profiles: dict[str, dict[str, Any]] = {}

    prefill_session = create_session(args, model_path, provider, fusion_config_value, label=f"{label}_prefill")
    print(f"Providers ({label} prefill): {prefill_session.get_providers()}")
    prefill_results = benchmark_prefill_stage(prefill_session, config, args, label=f"{label} prefill")
    stage_results["prefill"] = list(prefill_results["times"])
    stage_results["has_kv_cache"] = bool(prefill_results["has_kv_cache"])
    summarize_with_label(label, "prefill", list(prefill_results["times"]))
    prefill_profile_path = Path(prefill_session.end_profiling())
    stage_profiles["prefill"] = summarize_shader_profile(
        prefill_profile_path,
        top_k=args.profile_top_k,
        name_filter=args.profile_name_filter,
        invocation_count=args.warmup + args.runs,
        label=f"{label} prefill",
    )
    del prefill_session

    if prefill_results["has_kv_cache"] and args.decode_steps > 0:
        decode_session = create_session(args, model_path, provider, fusion_config_value, label=f"{label}_decode")
        print(f"Providers ({label} decode): {decode_session.get_providers()}")
        decode_results = benchmark_decode_stage(
            decode_session,
            config,
            args,
            present=dict(prefill_results["present"]),
            next_token_id=int(prefill_results["next_token_id"]),
            label=f"{label} decode",
        )
        stage_results["decode"] = list(decode_results["times"])
        stage_results["has_kv_cache"] = bool(decode_results["has_kv_cache"])
        summarize_with_label(label, "decode", list(decode_results["times"]))
        decode_profile_path = Path(decode_session.end_profiling())
        stage_profiles["decode"] = summarize_shader_profile(
            decode_profile_path,
            top_k=args.profile_top_k,
            name_filter=args.profile_name_filter,
            invocation_count=args.decode_steps * (args.warmup + args.runs),
            label=f"{label} decode",
        )
        del decode_session
    elif not prefill_results["has_kv_cache"]:
        print(f"{label}: no KV cache outputs found. Skipping decode timing.")
    else:
        print(f"{label}: decode timing skipped because --decode-steps <= 0.")

    return stage_results, stage_profiles


def print_comparison(results: dict[str, dict[str, list[float] | bool]]) -> None:
    print("\nFusion comparison:")
    for stage in ("prefill", "decode"):
        off_times = results["fusion=off"][stage]
        on_times = results["fusion=on"][stage]
        if not isinstance(off_times, list) or not isinstance(on_times, list):
            continue
        if not off_times or not on_times:
            continue

        off_stats = summarize_stats(off_times)
        on_stats = summarize_stats(on_times)
        delta_ms = float(on_stats["mean_ms"] - off_stats["mean_ms"])
        ratio = float(on_stats["mean_ms"] / off_stats["mean_ms"])
        print(
            f"  {stage}: off_mean={off_stats['mean_ms']:.3f} ms on_mean={on_stats['mean_ms']:.3f} ms delta={delta_ms:.3f} ms ratio={ratio:.3f}x"
        )


def main() -> None:
    global ort

    args = parse_args()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if args.compare_matmulnbits_silu_fusion and args.matmulnbits_silu_fusion != "default":
        raise RuntimeError(
            "--compare-matmulnbits-silu-fusion cannot be combined with --matmulnbits-silu-fusion on/off. Use the compare flag alone."
        )
    fusion_config_value = resolve_matmulnbits_silu_fusion_config(args.matmulnbits_silu_fusion)

    ort_package_root = resolve_ort_package_root(args)
    ort = import_onnxruntime(ort_package_root)

    print("[debug] loading config", flush=True)
    config = load_config(args.model_dir)
    model_path = args.model_dir / config["model"]["decoder"]["filename"]

    print(f"[debug] model path: {model_path}", flush=True)
    print(f"[debug] ort source: {args.ort_source}", flush=True)
    if ort_package_root is not None:
        print(f"[debug] ort package root: {ort_package_root}", flush=True)
    print(f"[debug] ort module path: {ort.__file__}", flush=True)
    print(f"[debug] ort version: {ort.__version__}", flush=True)
    available = ort.get_available_providers()
    print(f"[debug] available providers: {available}", flush=True)
    provider = resolve_provider(available, args.provider)
    print(f"[debug] selected provider: {provider}", flush=True)
    print(f"Model: {model_path}")
    summarize_raw_mlp_patterns(model_path)
    if args.compare_matmulnbits_silu_fusion:
        labels_and_configs = (("fusion=off", "0"), ("fusion=on", "1"))
        results: dict[str, dict[str, list[float] | bool]] = {}
        shader_profiles: dict[str, dict[str, Any]] = {}
        for label, config_value in labels_and_configs:
            if args.profile_shaders and args.profile_stages_separately:
                run_results, stage_profiles = run_stage_split_profiled_benchmark(
                    args,
                    model_path,
                    provider,
                    config,
                    config_value,
                    label=label,
                )
                results[label] = run_results
                shader_profiles[f"{label}:prefill"] = stage_profiles["prefill"]
                if "decode" in stage_profiles:
                    shader_profiles[f"{label}:decode"] = stage_profiles["decode"]
                continue

            session = create_session(args, model_path, provider, config_value, label=label)
            print(f"Providers ({label}): {session.get_providers()}")
            run_results = benchmark_forward_pass(session, config, args, label=label)
            results[label] = run_results
            summarize_with_label(label, "prefill", run_results["prefill"])
            decode_times = run_results["decode"]
            if decode_times:
                summarize_with_label(label, "decode", decode_times)
            elif not run_results["has_kv_cache"]:
                print(f"{label}: no KV cache outputs found. Skipping decode timing.")
            else:
                print(f"{label}: decode timing skipped because --decode-steps <= 0.")
            if args.profile_shaders:
                profile_path = Path(session.end_profiling())
                shader_profiles[label] = summarize_shader_profile(
                    profile_path,
                    top_k=args.profile_top_k,
                    name_filter=args.profile_name_filter,
                    label=label,
                )
            del session

        print_comparison(results)
        if args.profile_shaders:
            if args.profile_stages_separately:
                compare_shader_profiles(
                    shader_profiles["fusion=off:prefill"],
                    shader_profiles["fusion=on:prefill"],
                    title="Prefill shader comparison",
                    before_label="fusion=off prefill",
                    after_label="fusion=on prefill",
                    top_k=args.profile_top_k,
                )
                if "fusion=off:decode" in shader_profiles and "fusion=on:decode" in shader_profiles:
                    compare_shader_profiles(
                        shader_profiles["fusion=off:decode"],
                        shader_profiles["fusion=on:decode"],
                        title="Decode shader comparison",
                        before_label="fusion=off decode",
                        after_label="fusion=on decode",
                        top_k=args.profile_top_k,
                    )
            else:
                compare_shader_profiles(
                    shader_profiles["fusion=off"],
                    shader_profiles["fusion=on"],
                    before_label="fusion=off",
                    after_label="fusion=on",
                    top_k=args.profile_top_k,
                )
        return

    print(f"[debug] session.enable_matmulnbits_silu_fusion: {fusion_config_value}", flush=True)
    session = create_session(args, model_path, provider, fusion_config_value)
    print(f"Providers: {session.get_providers()}")
    run_results = benchmark_forward_pass(session, config, args)
    summarize("prefill", run_results["prefill"])
    decode_times = run_results["decode"]
    if decode_times:
        summarize("decode", decode_times)
    elif not run_results["has_kv_cache"]:
        print("No KV cache outputs found. Skipping decode timing.")
    else:
        print("Decode timing skipped because --decode-steps <= 0.")
    if args.profile_shaders:
        profile_path = Path(session.end_profiling())
        summarize_shader_profile(
            profile_path,
            top_k=args.profile_top_k,
            name_filter=args.profile_name_filter,
        )


if __name__ == "__main__":
    main()
