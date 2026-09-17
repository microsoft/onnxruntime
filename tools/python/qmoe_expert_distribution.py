#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import onnx

ROUTING_MARKER = "moe_routing "
ROUTING_TRUNCATED_MARKER = "moe_routing_truncated "
ROUTING_COMPLETE_MARKER = "moe_routing_complete "
PROMPT_START = re.compile(r"^\[qmoe_prompt_runner\] (\d+)/(\d+) prompt_start$")
PROMPT_END = re.compile(r"^\[qmoe_prompt_runner\] (\d+)/(\d+) prompt_end$")
LAYER_NUMBER = re.compile(r"/layers\.(\d+)/")
QMOE_EXPERT_WEIGHT_INPUT_INDICES = (2, 5)
MAX_PLOTTED_LAYERS = 9
MAX_EXTERNAL_DATA_VALUE = (1 << 63) - 1


def parse_args():
    parser = argparse.ArgumentParser(description="Compute expert selection distributions from an ORT QMoE routing log.")
    parser.add_argument("log", type=Path, help="Log containing 'moe_routing' JSON records.")
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        help="locodellm JSON used to label prompts (defaults to the log path with .json).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Output prefix (defaults to <log stem>-expert-distribution).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="ONNX model used to calculate QMoE expert sizes (auto-detected by default).",
    )
    return parser.parse_args()


def load_prompt_labels(path):
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark JSON not found: {path}")
    with path.open(encoding="utf-8") as stream:
        benchmark = json.load(stream)
    if not isinstance(benchmark, list) or not benchmark:
        raise ValueError(f"Benchmark JSON must contain a non-empty list: {path}")

    prompt_labels = {}
    for index, case in enumerate(benchmark, start=1):
        if not isinstance(case, dict) or case.get("prompt_index") != index or not isinstance(case.get("prompt"), str):
            raise ValueError(f"Benchmark JSON entry {index} has an invalid prompt index or prompt.")
        prompt_labels[index] = case["prompt"]
    return prompt_labels


def resolve_model_path(log_path, model_path):
    if model_path:
        candidate = model_path / "model.onnx" if model_path.is_dir() else model_path
        if not candidate.is_file():
            raise FileNotFoundError(f"ONNX model not found: {candidate}")
        return candidate

    model_root = log_path.parent.parent / "models" / "qwen"
    candidates = [
        directory / "model.onnx"
        for directory in model_root.iterdir()
        if directory.is_dir() and log_path.stem.startswith(directory.name) and (directory / "model.onnx").is_file()
    ]
    if not candidates:
        raise FileNotFoundError("Could not auto-detect model.onnx; specify it with --model.")
    return max(candidates, key=lambda path: len(path.parent.name))


def layer_sort_key(node_name):
    match = LAYER_NUMBER.search(node_name)
    return (int(match.group(1)), node_name) if match else (10**9, node_name)


def node_identity(event):
    return event["node_index"], event["node_type"], event["node_name"]


def node_sort_key(identity):
    node_index, node_type, node_name = identity
    return (*layer_sort_key(node_name), node_index, node_type)


def node_csv_fields(identity):
    node_index, node_type, node_name = identity
    return node_index, node_type, node_name


def node_display_name(identity):
    node_index, node_type, node_name = identity
    return node_name or f"{node_type}[{node_index}]"


def _validate_routing_event(event, line_number):
    if not isinstance(event, dict):
        raise ValueError(f"Line {line_number}: routing payload must be a JSON object.")
    if not isinstance(event.get("node_index"), int) or isinstance(event["node_index"], bool) or event["node_index"] < 0:
        raise ValueError(f"Line {line_number}: node_index must be a non-negative integer.")
    if not isinstance(event.get("node_type"), str) or not event["node_type"]:
        raise ValueError(f"Line {line_number}: node_type must be a non-empty string.")
    if not isinstance(event.get("node_name"), str):
        raise ValueError(f"Line {line_number}: node_name must be a string.")
    for field in ("num_rows", "top_k"):
        value = event.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Line {line_number}: {field} must be a positive integer.")

    expected = event["num_rows"] * event["top_k"]
    for field in ("expert_ids", "router_weights"):
        values = event.get(field)
        if not isinstance(values, list) or len(values) != expected:
            actual = len(values) if isinstance(values, list) else "non-list"
            raise ValueError(f"Line {line_number}: expected {expected} {field}, got {actual}.")
    if any(not isinstance(expert_id, int) or isinstance(expert_id, bool) for expert_id in event["expert_ids"]):
        raise ValueError(f"Line {line_number}: expert_ids must contain integers.")
    if any(
        not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight)
        for weight in event["router_weights"]
    ):
        raise ValueError(f"Line {line_number}: router_weights must contain finite numbers.")


def parse_routing_trace(log_path):
    active_prompt = None
    completed_prompts = 0
    expected_prompts = None
    routing_events = []
    completion = None

    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            if ROUTING_TRUNCATED_MARKER in line:
                raise ValueError(f"Incomplete routing trace at line {line_number}: {line.strip()}")

            stripped_line = line.strip()
            prompt_start = PROMPT_START.fullmatch(stripped_line)
            if prompt_start:
                prompt_index, prompt_total = map(int, prompt_start.groups())
                if completion is not None:
                    raise ValueError(f"Prompt started after the completion footer at line {line_number}.")
                if active_prompt is not None:
                    raise ValueError(
                        f"Prompt {prompt_index} started before prompt {active_prompt} ended at line {line_number}."
                    )
                if prompt_total <= 0 or (expected_prompts is not None and prompt_total != expected_prompts):
                    raise ValueError(f"Inconsistent prompt total at line {line_number}: {prompt_total}.")
                if prompt_index != completed_prompts + 1:
                    raise ValueError(
                        f"Expected prompt {completed_prompts + 1}, got {prompt_index} at line {line_number}."
                    )
                expected_prompts = prompt_total
                active_prompt = prompt_index
                continue

            prompt_end = PROMPT_END.fullmatch(stripped_line)
            if prompt_end:
                prompt_index, prompt_total = map(int, prompt_end.groups())
                if active_prompt != prompt_index or prompt_total != expected_prompts:
                    raise ValueError(f"Unexpected prompt_end marker at line {line_number}: {stripped_line}")
                active_prompt = None
                completed_prompts += 1
                continue

            completion_position = line.find(ROUTING_COMPLETE_MARKER)
            if completion_position >= 0:
                if completion is not None:
                    raise ValueError(f"Duplicate routing completion footer at line {line_number}.")
                if active_prompt is not None:
                    raise ValueError(f"Routing completion footer precedes prompt_end at line {line_number}.")
                try:
                    completion = json.loads(line[completion_position + len(ROUTING_COMPLETE_MARKER) :])
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid routing completion footer at line {line_number}: {exc}") from exc
                continue

            marker_position = line.find(ROUTING_MARKER)
            if marker_position < 0:
                continue
            if active_prompt is None:
                raise ValueError(f"Routing event at line {line_number} is outside an active prompt.")
            if completion is not None:
                raise ValueError(f"Routing event follows the completion footer at line {line_number}.")

            payload = line[marker_position + len(ROUTING_MARKER) :].strip()
            try:
                event = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid routing JSON at line {line_number}: {exc}") from exc
            _validate_routing_event(event, line_number)
            routing_events.append((active_prompt, event))

    if active_prompt is not None:
        raise ValueError(f"Incomplete routing trace: prompt {active_prompt} has no prompt_end marker.")
    if expected_prompts is None:
        raise ValueError("Incomplete routing trace: no prompt_start marker.")
    if completed_prompts != expected_prompts:
        raise ValueError(f"Incomplete routing trace: completed {completed_prompts} of {expected_prompts} prompts.")
    if not isinstance(completion, dict):
        raise ValueError("Incomplete routing trace: missing moe_routing_complete footer.")
    if any(
        not isinstance(completion.get(field), int) or isinstance(completion[field], bool)
        for field in ("prompts", "prompt_runs", "routing_records")
    ):
        raise ValueError("Routing completion footer counts must be integers.")

    expected_completion = {
        "prompts": expected_prompts,
        "prompt_runs": completed_prompts,
        "routing_records": len(routing_events),
    }
    if completion != expected_completion:
        raise ValueError(f"Routing completion footer mismatch: expected {expected_completion}, got {completion}.")
    return routing_events, completion


def iter_routing_events(log_path):
    routing_events, _ = parse_routing_trace(log_path)
    yield from routing_events


def read_distributions(log_path, num_experts):
    by_prompt_qmoe = defaultdict(Counter)
    by_qmoe = defaultdict(Counter)
    global_counts = Counter()
    event_count = 0

    for prompt_index, event in iter_routing_events(log_path):
        identity = node_identity(event)
        expert_ids = event["expert_ids"]
        for expert_id in expert_ids:
            if not 0 <= expert_id < num_experts:
                raise ValueError(f"Trace contains expert ID {expert_id}, but the model has {num_experts} experts.")
        counts = Counter(expert_ids)
        by_prompt_qmoe[(prompt_index, identity)].update(counts)
        by_qmoe[identity].update(counts)
        global_counts.update(counts)
        event_count += 1

    if event_count == 0:
        raise ValueError(f"No '{ROUTING_MARKER.strip()}' records found in {log_path}.")

    return by_prompt_qmoe, by_qmoe, global_counts, event_count


def distribution_rows(counts, num_experts):
    total = counts.total()
    for expert_id in range(num_experts):
        count = counts[expert_id]
        yield expert_id, count, count / total if total else 0.0


def write_prompt_qmoe_csv(path, distributions, prompt_labels, num_experts):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "prompt_index",
                "prompt",
                "node_index",
                "node_type",
                "node_name",
                "expert_id",
                "count",
                "selection_share",
            ]
        )
        keys = sorted(distributions, key=lambda key: (key[0], node_sort_key(key[1])))
        for prompt_index, identity in keys:
            for expert_id, count, share in distribution_rows(distributions[(prompt_index, identity)], num_experts):
                writer.writerow(
                    [
                        prompt_index,
                        prompt_labels.get(prompt_index, ""),
                        *node_csv_fields(identity),
                        expert_id,
                        count,
                        share,
                    ]
                )


def write_qmoe_csv(path, distributions, num_experts):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node_index", "node_type", "node_name", "expert_id", "count", "selection_share"])
        for identity in sorted(distributions, key=node_sort_key):
            for expert_id, count, share in distribution_rows(distributions[identity], num_experts):
                writer.writerow([*node_csv_fields(identity), expert_id, count, share])


def write_qmoe_pivot_csv(path, distributions, num_experts):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node_index", "node_type", "node_name", *range(num_experts)])
        for identity in sorted(distributions, key=node_sort_key):
            writer.writerow(
                [*node_csv_fields(identity), *(distributions[identity][expert_id] for expert_id in range(num_experts))]
            )


def rank_experts_by_frequency(distributions, num_experts):
    return {
        identity: sorted(
            range(num_experts),
            key=lambda expert_id: (-counts[expert_id], expert_id),
        )
        for identity, counts in distributions.items()
    }


def expert_rank_positions(expert_ids, ranked_expert_ids):
    rank_by_expert_id = {expert_id: rank for rank, expert_id in enumerate(ranked_expert_ids)}
    return [rank_by_expert_id[expert_id] for expert_id in expert_ids]


def inference_expert_ids(event):
    top_k = event["top_k"]
    return event["expert_ids"][-top_k:]


def expert_rank_threshold_counts(positions, num_experts):
    return [sum(position >= threshold for position in positions) for threshold in range(num_experts)]


def write_qmoe_ranked_experts_csv(path, rankings):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node_index", "node_type", "node_name", "expert_ids_by_decreasing_frequency"])
        for identity in sorted(rankings, key=node_sort_key):
            writer.writerow([*node_csv_fields(identity), json.dumps(rankings[identity], separators=(",", ":"))])


def write_inference_expert_ranks_csv(path, log_path, rankings, prompt_labels, num_experts):
    inference_indexes = Counter()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "prompt_index",
                "prompt",
                "inference_index",
                "node_index",
                "node_type",
                "node_name",
                "num_rows",
                "top_k",
                "selected_expert_ids",
                "expert_rank_positions_0_based",
                "max_expert_rank_position",
                *(f"experts_rank_ge_{threshold}" for threshold in range(num_experts)),
            ]
        )
        for prompt_index, event in iter_routing_events(log_path):
            identity = node_identity(event)
            key = (prompt_index, identity)
            inference_indexes[key] += 1
            expert_ids = inference_expert_ids(event)
            positions = expert_rank_positions(expert_ids, rankings[identity])
            threshold_counts = expert_rank_threshold_counts(positions, num_experts)
            writer.writerow(
                [
                    prompt_index,
                    prompt_labels.get(prompt_index, ""),
                    inference_indexes[key],
                    *node_csv_fields(identity),
                    event["num_rows"],
                    event["top_k"],
                    json.dumps(expert_ids, separators=(",", ":")),
                    json.dumps(positions, separators=(",", ":")),
                    max(positions),
                    *threshold_counts,
                ]
            )


def aggregate_rank_thresholds_by_qmoe(log_path, rankings, num_experts):
    inference_counts = Counter()
    threshold_totals = defaultdict(lambda: [0] * num_experts)
    for _, event in iter_routing_events(log_path):
        identity = node_identity(event)
        expert_ids = inference_expert_ids(event)
        positions = expert_rank_positions(expert_ids, rankings[identity])
        inference_counts[identity] += 1
        for index, count in enumerate(expert_rank_threshold_counts(positions, num_experts)):
            threshold_totals[identity][index] += count
    return inference_counts, threshold_totals


def load_qmoe_model_metadata(model_path):
    model = onnx.load(model_path, load_external_data=False)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    qmoe_nodes = {
        (node_index, node.op_type, node.name): node
        for node_index, node in enumerate(model.graph.node)
        if node.op_type == "QMoE"
    }
    expert_counts = {
        initializer.dims[0]
        for node in qmoe_nodes.values()
        for input_index in QMOE_EXPERT_WEIGHT_INPUT_INDICES
        if input_index < len(node.input)
        for input_name in (node.input[input_index],)
        if (initializer := initializers.get(input_name)) is not None and initializer.dims
    }
    if len(expert_counts) != 1:
        raise ValueError(f"Expected one QMoE expert count in the model, got {sorted(expert_counts)}.")
    return initializers, qmoe_nodes, expert_counts.pop()


def _external_data_integer(external_data, key, initializer_name):
    raw_value = external_data.get(key)
    if raw_value is None:
        return 0
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid external_data.{key} for initializer {initializer_name}: {raw_value!r}.") from exc
    if value < 0 or value > MAX_EXTERNAL_DATA_VALUE:
        raise ValueError(f"Invalid external_data.{key} for initializer {initializer_name}: {raw_value!r}.")
    return value


def calculate_qmoe_expert_bytes(initializers, qmoe_nodes, node_identities, num_experts, model_path=None):
    expert_bytes = {}
    for identity in node_identities:
        node = qmoe_nodes.get(identity)
        if node is None:
            raise ValueError(f"QMoE node from log not found in model: {node_display_name(identity)}")

        total_bytes = 0
        for input_name in node.input:
            initializer = initializers.get(input_name)
            if initializer is None or not initializer.dims:
                continue
            if initializer.dims[0] != num_experts:
                continue
            external_data = {entry.key: entry.value for entry in initializer.external_data}
            expected_tensor_bytes = initializer_byte_count(initializer)
            declared_length = _external_data_integer(external_data, "length", input_name)
            if declared_length and declared_length != expected_tensor_bytes:
                raise ValueError(
                    f"External data length for initializer {input_name} is {declared_length}, "
                    f"expected {expected_tensor_bytes}."
                )
            tensor_bytes = declared_length or expected_tensor_bytes

            location = external_data.get("location")
            if model_path is not None and location:
                external_path = Path(model_path).parent / location
                if external_path.is_file():
                    offset = _external_data_integer(external_data, "offset", input_name)
                    if offset + tensor_bytes > external_path.stat().st_size:
                        raise ValueError(
                            f"External data range for initializer {input_name} exceeds {external_path}: "
                            f"offset {offset}, length {tensor_bytes}, file size {external_path.stat().st_size}."
                        )
            if tensor_bytes % num_experts:
                raise ValueError(f"Initializer size is not divisible by {num_experts}: {input_name}")
            total_bytes += tensor_bytes // num_experts
        if total_bytes == 0:
            raise ValueError(f"No expert initializers found for QMoE node: {node_display_name(identity)}")
        expert_bytes[identity] = total_bytes
    return expert_bytes


def initializer_byte_count(initializer):
    packed_types = {
        onnx.TensorProto.INT4,
        onnx.TensorProto.UINT4,
        onnx.TensorProto.FLOAT4E2M1,
    }
    element_count = math.prod(initializer.dims)
    if initializer.data_type in packed_types:
        return (element_count + 1) // 2
    return element_count * onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type).itemsize


def write_qmoe_rank_threshold_totals_csv(path, inference_counts, threshold_totals, expert_bytes, num_experts):
    total_inferences = sum(inference_counts.values())
    column_totals = [sum(counts[threshold] for counts in threshold_totals.values()) for threshold in range(num_experts)]
    total_values = [total_inferences, *column_totals]
    maximum = max(total_values)

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "node_index",
                "node_type",
                "node_name",
                "inference_count",
                *(f"experts_rank_ge_{threshold}" for threshold in range(num_experts)),
            ]
        )
        for identity in sorted(threshold_totals, key=node_sort_key):
            writer.writerow(
                [
                    *node_csv_fields(identity),
                    inference_counts[identity],
                    *threshold_totals[identity],
                ]
            )
        writer.writerow(["", "", "TOTAL", *total_values])
        writer.writerow(
            [
                "",
                "",
                "TOTAL_NORMALIZED",
                *(value / maximum for value in total_values),
            ]
        )
        bytes_per_rank = sum(expert_bytes.values())
        writer.writerow(
            [
                "",
                "",
                "QMOE_EXPERT_BYTES",
                0,
                *(rank * bytes_per_rank for rank in range(num_experts)),
            ]
        )
        maximum_expert_bytes = num_experts * bytes_per_rank
        writer.writerow(
            [
                "",
                "",
                "QMOE_EXPERT_BYTES_COMPLEMENT",
                0,
                *((num_experts - rank) * bytes_per_rank for rank in range(num_experts)),
            ]
        )
        writer.writerow(
            [
                "",
                "",
                "QMOE_EXPERT_BYTES_COMPLEMENT_NORMALIZED",
                0.0,
                *(
                    ((num_experts - rank) * bytes_per_rank) / maximum_expert_bytes if maximum_expert_bytes else 0.0
                    for rank in range(num_experts)
                ),
            ]
        )
    return (
        [value / maximum for value in column_totals],
        [
            ((num_experts - rank) * bytes_per_rank) / maximum_expert_bytes if maximum_expert_bytes else 0.0
            for rank in range(num_experts)
        ],
    )


def write_normalized_comparison_plot(path, total_normalized, expert_bytes_complement_normalized):
    import matplotlib.pyplot as plt  # noqa: PLC0415 - Only plotting requires matplotlib.

    ranks = range(len(total_normalized))
    figure, axes = plt.subplots(figsize=(10, 6))
    total_line = axes.plot(ranks, total_normalized, label="TOTAL_NORMALIZED", linewidth=2)[0]
    bytes_line = axes.plot(
        ranks,
        expert_bytes_complement_normalized,
        label="QMOE_EXPERT_BYTES_COMPLEMENT_NORMALIZED",
        linewidth=2,
    )[0]
    for rank in (rank for rank in (64, 128, 192) if rank < len(total_normalized)):
        for values, line, offset in (
            (total_normalized, total_line, (8, 10)),
            (expert_bytes_complement_normalized, bytes_line, (8, -18)),
        ):
            value = values[rank]
            axes.scatter(rank, value, color=line.get_color(), zorder=3)
            axes.annotate(
                f"({rank}, {value:.3f})",
                (rank, value),
                xytext=offset,
                textcoords="offset points",
                color=line.get_color(),
                fontsize=9,
            )
    axes.set_xlabel("Expert rank threshold (0-based)")
    axes.set_ylabel("Normalized value")
    axes.set_xlim(0, len(total_normalized) - 1)
    axes.set_ylim(0, 1.02)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_selected_layers_rank_plot(path, threshold_totals):
    import matplotlib.pyplot as plt  # noqa: PLC0415 - Only plotting requires matplotlib.

    identities = sorted(threshold_totals, key=node_sort_key)
    if len(identities) > MAX_PLOTTED_LAYERS:
        identities = [
            identities[round(index * (len(identities) - 1) / (MAX_PLOTTED_LAYERS - 1))]
            for index in range(MAX_PLOTTED_LAYERS)
        ]

    figure, axes = plt.subplots(figsize=(10, 6))
    for identity in identities:
        values = threshold_totals[identity]
        maximum = max(values)
        normalized = [value / maximum for value in values]
        node_name = node_display_name(identity)
        match = LAYER_NUMBER.search(node_name)
        label = f"Layer {match.group(1)}" if match else node_name
        axes.plot(
            range(len(values)),
            normalized,
            label=label,
            linewidth=1.8,
        )

    axes.set_xlabel("Expert rank threshold (0-based)")
    axes.set_ylabel("Normalized experts with rank >= threshold")
    axes.set_xlim(0, len(next(iter(threshold_totals.values()))) - 1)
    axes.set_ylim(0, 1.02)
    axes.grid(True, alpha=0.3)
    axes.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_global_csv(path, counts, num_experts):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["expert_id", "count", "selection_share"])
        writer.writerows(distribution_rows(counts, num_experts))


def main():
    args = parse_args()
    benchmark_json = args.benchmark_json or args.log.with_suffix(".json")
    output_prefix = args.output_prefix or args.log.with_name(f"{args.log.stem}-expert-distribution")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    _, completion = parse_routing_trace(args.log)
    prompt_labels = load_prompt_labels(benchmark_json)
    if len(prompt_labels) != completion["prompts"]:
        raise ValueError(
            f"Benchmark JSON contains {len(prompt_labels)} prompts, "
            f"but the routing trace completed {completion['prompts']} prompts."
        )
    model_path = resolve_model_path(args.log, args.model)
    initializers, qmoe_nodes, num_experts = load_qmoe_model_metadata(model_path)
    by_prompt_qmoe, by_qmoe, global_counts, event_count = read_distributions(args.log, num_experts)
    expert_bytes = calculate_qmoe_expert_bytes(
        initializers, qmoe_nodes, by_qmoe.keys(), num_experts, model_path=model_path
    )

    prompt_qmoe_path = Path(f"{output_prefix}-by-prompt-qmoe.csv")
    qmoe_path = Path(f"{output_prefix}-by-qmoe.csv")
    qmoe_pivot_path = Path(f"{output_prefix}-by-qmoe-pivot.csv")
    qmoe_ranked_path = Path(f"{output_prefix}-by-qmoe-ranked-experts.csv")
    inference_ranks_path = Path(f"{output_prefix}-by-inference-qmoe-expert-ranks.csv")
    qmoe_rank_thresholds_path = Path(f"{output_prefix}-by-qmoe-rank-threshold-totals.csv")
    normalized_plot_path = Path(f"{output_prefix}-normalized-total-vs-expert-bytes.png")
    selected_layers_plot_path = Path(f"{output_prefix}-selected-layers-expert-ranks.png")
    global_path = Path(f"{output_prefix}-global.csv")
    rankings = rank_experts_by_frequency(by_qmoe, num_experts)
    write_prompt_qmoe_csv(prompt_qmoe_path, by_prompt_qmoe, prompt_labels, num_experts)
    write_qmoe_csv(qmoe_path, by_qmoe, num_experts)
    write_qmoe_pivot_csv(qmoe_pivot_path, by_qmoe, num_experts)
    write_qmoe_ranked_experts_csv(qmoe_ranked_path, rankings)
    write_inference_expert_ranks_csv(inference_ranks_path, args.log, rankings, prompt_labels, num_experts)
    inference_counts, threshold_totals = aggregate_rank_thresholds_by_qmoe(args.log, rankings, num_experts)
    total_normalized, expert_bytes_complement_normalized = write_qmoe_rank_threshold_totals_csv(
        qmoe_rank_thresholds_path,
        inference_counts,
        threshold_totals,
        expert_bytes,
        num_experts,
    )
    write_normalized_comparison_plot(
        normalized_plot_path,
        total_normalized,
        expert_bytes_complement_normalized,
    )
    write_selected_layers_rank_plot(selected_layers_plot_path, threshold_totals)
    write_global_csv(global_path, global_counts, num_experts)

    print(f"routing events: {event_count}")
    print(f"prompts: {len({key[0] for key in by_prompt_qmoe})}")
    print(f"QMoE nodes: {len(by_qmoe)}")
    print(f"experts: {num_experts}")
    print(f"expert selections: {global_counts.total()}")
    print(f"model: {model_path}")
    print(f"QMoE bytes per expert rank: {sum(expert_bytes.values())}")
    print(prompt_qmoe_path)
    print(qmoe_path)
    print(qmoe_pivot_path)
    print(qmoe_ranked_path)
    print(inference_ranks_path)
    print(qmoe_rank_thresholds_path)
    print(normalized_plot_path)
    print(selected_layers_plot_path)
    print(global_path)


if __name__ == "__main__":
    main()
