#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import onnx

ROUTING_MARKER = "moe_routing "
PROMPT_PROGRESS = re.compile(r"(\d+)/(\d+)")
LAYER_NUMBER = re.compile(r"/layers\.(\d+)/")
PLOTTED_LAYER_INDICES = (0, 5, 10, 15, 20, 25, 30, 35, 39)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute expert selection distributions from an ORT QMoE routing log."
    )
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
        return {}
    with path.open(encoding="utf-8") as stream:
        benchmark = json.load(stream)
    return {index: case["prompt"] for index, case in enumerate(benchmark, start=1)}


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
        if directory.is_dir()
        and log_path.stem.startswith(directory.name)
        and (directory / "model.onnx").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-detect model.onnx; specify it with --model."
        )
    return max(candidates, key=lambda path: len(path.parent.name))


def layer_sort_key(node_name):
    match = LAYER_NUMBER.search(node_name)
    return (int(match.group(1)), node_name) if match else (10**9, node_name)


def iter_routing_events(log_path):
    prompt_index = None

    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            progress = PROMPT_PROGRESS.search(line)
            if progress:
                prompt_index = int(progress.group(1))

            marker_position = line.find(ROUTING_MARKER)
            if marker_position < 0:
                continue
            if prompt_index is None:
                raise ValueError(
                    f"Routing event at line {line_number} precedes the first prompt marker."
                )

            payload = line[marker_position + len(ROUTING_MARKER) :].strip()
            try:
                event = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid routing JSON at line {line_number}: {exc}") from exc

            expert_ids = event["expert_ids"]
            expected = event["num_rows"] * event["top_k"]
            if len(expert_ids) != expected:
                raise ValueError(
                    f"Line {line_number}: expected {expected} expert IDs, got {len(expert_ids)}."
                )

            yield prompt_index, event


def read_distributions(log_path, num_experts):
    by_prompt_qmoe = defaultdict(Counter)
    by_qmoe = defaultdict(Counter)
    global_counts = Counter()
    event_count = 0
    max_expert_id = -1

    for prompt_index, event in iter_routing_events(log_path):
        node_name = event["node_name"]
        expert_ids = event["expert_ids"]
        counts = Counter(expert_ids)
        by_prompt_qmoe[(prompt_index, node_name)].update(counts)
        by_qmoe[node_name].update(counts)
        global_counts.update(counts)
        event_count += 1
        if expert_ids:
            max_expert_id = max(max_expert_id, *expert_ids)

    if event_count == 0:
        raise ValueError(f"No '{ROUTING_MARKER.strip()}' records found in {log_path}.")
    if max_expert_id >= num_experts:
        raise ValueError(
            f"Trace contains expert ID {max_expert_id}, but the model has "
            f"{num_experts} experts."
        )

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
            ["prompt_index", "prompt", "qmoe", "expert_id", "count", "selection_share"]
        )
        keys = sorted(distributions, key=lambda key: (key[0], layer_sort_key(key[1])))
        for prompt_index, node_name in keys:
            for expert_id, count, share in distribution_rows(
                distributions[(prompt_index, node_name)], num_experts
            ):
                writer.writerow(
                    [
                        prompt_index,
                        prompt_labels.get(prompt_index, ""),
                        node_name,
                        expert_id,
                        count,
                        share,
                    ]
                )


def write_qmoe_csv(path, distributions, num_experts):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["qmoe", "expert_id", "count", "selection_share"])
        for node_name in sorted(distributions, key=layer_sort_key):
            for expert_id, count, share in distribution_rows(
                distributions[node_name], num_experts
            ):
                writer.writerow([node_name, expert_id, count, share])


def write_qmoe_pivot_csv(path, distributions, num_experts):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["qmoe", *range(num_experts)])
        for node_name in sorted(distributions, key=layer_sort_key):
            writer.writerow(
                [node_name, *(distributions[node_name][expert_id] for expert_id in range(num_experts))]
            )


def rank_experts_by_frequency(distributions, num_experts):
    return {
        node_name: sorted(
            range(num_experts),
            key=lambda expert_id: (-counts[expert_id], expert_id),
        )
        for node_name, counts in distributions.items()
    }


def expert_rank_positions(expert_ids, ranked_expert_ids):
    rank_by_expert_id = {
        expert_id: rank
        for rank, expert_id in enumerate(ranked_expert_ids)
    }
    return [rank_by_expert_id[expert_id] for expert_id in expert_ids]


def inference_expert_ids(event):
    top_k = event["top_k"]
    return event["expert_ids"][-top_k:]


def expert_rank_threshold_counts(positions, num_experts):
    return [
        sum(position >= threshold for position in positions)
        for threshold in range(num_experts)
    ]


def write_qmoe_ranked_experts_csv(path, rankings):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["qmoe", "expert_ids_by_decreasing_frequency"])
        for node_name in sorted(rankings, key=layer_sort_key):
            writer.writerow([node_name, json.dumps(rankings[node_name], separators=(",", ":"))])


def write_inference_expert_ranks_csv(
    path, log_path, rankings, prompt_labels, num_experts
):
    inference_indexes = Counter()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "prompt_index",
                "prompt",
                "inference_index",
                "qmoe",
                "num_rows",
                "top_k",
                "selected_expert_ids",
                "expert_rank_positions_0_based",
                "max_expert_rank_position",
                *(
                    f"experts_rank_ge_{threshold}"
                    for threshold in range(num_experts)
                ),
            ]
        )
        for prompt_index, event in iter_routing_events(log_path):
            node_name = event["node_name"]
            key = (prompt_index, node_name)
            inference_indexes[key] += 1
            expert_ids = inference_expert_ids(event)
            positions = expert_rank_positions(expert_ids, rankings[node_name])
            threshold_counts = expert_rank_threshold_counts(positions, num_experts)
            writer.writerow(
                [
                    prompt_index,
                    prompt_labels.get(prompt_index, ""),
                    inference_indexes[key],
                    node_name,
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
        node_name = event["node_name"]
        expert_ids = inference_expert_ids(event)
        positions = expert_rank_positions(expert_ids, rankings[node_name])
        inference_counts[node_name] += 1
        for index, count in enumerate(
            expert_rank_threshold_counts(positions, num_experts)
        ):
            threshold_totals[node_name][index] += count
    return inference_counts, threshold_totals


def load_qmoe_model_metadata(model_path):
    model = onnx.load(model_path, load_external_data=False)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    qmoe_nodes = {
        node.name: node for node in model.graph.node if node.op_type == "QMoE"
    }
    expert_counts = {
        initializer.dims[0]
        for node in qmoe_nodes.values()
        for input_name in node.input
        if (initializer := initializers.get(input_name)) is not None
        and initializer.dims
    }
    if len(expert_counts) != 1:
        raise ValueError(
            f"Expected one QMoE expert count in the model, got {sorted(expert_counts)}."
        )
    return initializers, qmoe_nodes, expert_counts.pop()


def calculate_qmoe_expert_bytes(
    initializers, qmoe_nodes, node_names, num_experts
):
    expert_bytes = {}
    for node_name in node_names:
        node = qmoe_nodes.get(node_name)
        if node is None:
            raise ValueError(f"QMoE node from log not found in model: {node_name}")

        total_bytes = 0
        for input_name in node.input:
            initializer = initializers.get(input_name)
            if initializer is None or not initializer.dims:
                continue
            if initializer.dims[0] != num_experts:
                continue
            external_data = {
                entry.key: entry.value for entry in initializer.external_data
            }
            if "length" not in external_data:
                raise ValueError(
                    f"Initializer size is not available in external data: {input_name}"
                )
            tensor_bytes = int(external_data["length"])
            if tensor_bytes % num_experts:
                raise ValueError(
                    f"Initializer size is not divisible by {num_experts}: {input_name}"
                )
            total_bytes += tensor_bytes // num_experts
        if total_bytes == 0:
            raise ValueError(f"No expert initializers found for QMoE node: {node_name}")
        expert_bytes[node_name] = total_bytes
    return expert_bytes


def write_qmoe_rank_threshold_totals_csv(
    path, inference_counts, threshold_totals, expert_bytes, num_experts
):
    total_inferences = sum(inference_counts.values())
    column_totals = [
        sum(counts[threshold] for counts in threshold_totals.values())
        for threshold in range(num_experts)
    ]
    total_values = [total_inferences, *column_totals]
    maximum = max(total_values)

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "qmoe",
                "inference_count",
                *(f"experts_rank_ge_{threshold}" for threshold in range(num_experts)),
            ]
        )
        for node_name in sorted(threshold_totals, key=layer_sort_key):
            writer.writerow(
                [
                    node_name,
                    inference_counts[node_name],
                    *threshold_totals[node_name],
                ]
            )
        writer.writerow(["TOTAL", *total_values])
        writer.writerow(
            [
                "TOTAL_NORMALIZED",
                *(value / maximum for value in total_values),
            ]
        )
        bytes_per_rank = sum(expert_bytes.values())
        writer.writerow(
            [
                "QMOE_EXPERT_BYTES",
                0,
                *(rank * bytes_per_rank for rank in range(num_experts)),
            ]
        )
        maximum_expert_bytes = (num_experts - 1) * bytes_per_rank
        writer.writerow(
            [
                "QMOE_EXPERT_BYTES_COMPLEMENT",
                0,
                *(
                    maximum_expert_bytes - rank * bytes_per_rank
                    for rank in range(num_experts)
                ),
            ]
        )
        writer.writerow(
            [
                "QMOE_EXPERT_BYTES_COMPLEMENT_NORMALIZED",
                0.0,
                *(
                    (maximum_expert_bytes - rank * bytes_per_rank)
                    / maximum_expert_bytes
                    if maximum_expert_bytes
                    else 0.0
                    for rank in range(num_experts)
                ),
            ]
        )
    return (
        [value / maximum for value in column_totals],
        [
            (maximum_expert_bytes - rank * bytes_per_rank)
            / maximum_expert_bytes
            if maximum_expert_bytes
            else 0.0
            for rank in range(num_experts)
        ],
    )


def write_normalized_comparison_plot(
    path, total_normalized, expert_bytes_complement_normalized
):
    ranks = range(len(total_normalized))
    figure, axes = plt.subplots(figsize=(10, 6))
    total_line = axes.plot(
        ranks, total_normalized, label="TOTAL_NORMALIZED", linewidth=2
    )[0]
    bytes_line = axes.plot(
        ranks,
        expert_bytes_complement_normalized,
        label="QMOE_EXPERT_BYTES_COMPLEMENT_NORMALIZED",
        linewidth=2,
    )[0]
    for rank in (64, 128, 192):
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
    nodes_by_layer = {}
    for node_name in threshold_totals:
        match = LAYER_NUMBER.search(node_name)
        if match:
            nodes_by_layer[int(match.group(1))] = node_name

    missing = set(PLOTTED_LAYER_INDICES) - nodes_by_layer.keys()
    if missing:
        raise ValueError(f"QMoE layers missing from routing log: {sorted(missing)}")

    figure, axes = plt.subplots(figsize=(10, 6))
    for layer_index in PLOTTED_LAYER_INDICES:
        values = threshold_totals[nodes_by_layer[layer_index]]
        maximum = max(values)
        normalized = [value / maximum for value in values]
        axes.plot(
            range(len(values)),
            normalized,
            label=f"Layer {layer_index}",
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
    output_prefix = args.output_prefix or args.log.with_name(
        f"{args.log.stem}-expert-distribution"
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    prompt_labels = load_prompt_labels(benchmark_json)
    model_path = resolve_model_path(args.log, args.model)
    initializers, qmoe_nodes, num_experts = load_qmoe_model_metadata(model_path)
    by_prompt_qmoe, by_qmoe, global_counts, event_count = read_distributions(
        args.log, num_experts
    )
    expert_bytes = calculate_qmoe_expert_bytes(
        initializers, qmoe_nodes, by_qmoe.keys(), num_experts
    )

    prompt_qmoe_path = Path(f"{output_prefix}-by-prompt-qmoe.csv")
    qmoe_path = Path(f"{output_prefix}-by-qmoe.csv")
    qmoe_pivot_path = Path(f"{output_prefix}-by-qmoe-pivot.csv")
    qmoe_ranked_path = Path(f"{output_prefix}-by-qmoe-ranked-experts.csv")
    inference_ranks_path = Path(
        f"{output_prefix}-by-inference-qmoe-expert-ranks.csv"
    )
    qmoe_rank_thresholds_path = Path(
        f"{output_prefix}-by-qmoe-rank-threshold-totals.csv"
    )
    normalized_plot_path = Path(
        f"{output_prefix}-normalized-total-vs-expert-bytes.png"
    )
    selected_layers_plot_path = Path(
        f"{output_prefix}-selected-layers-expert-ranks.png"
    )
    global_path = Path(f"{output_prefix}-global.csv")
    rankings = rank_experts_by_frequency(by_qmoe, num_experts)
    write_prompt_qmoe_csv(
        prompt_qmoe_path, by_prompt_qmoe, prompt_labels, num_experts
    )
    write_qmoe_csv(qmoe_path, by_qmoe, num_experts)
    write_qmoe_pivot_csv(qmoe_pivot_path, by_qmoe, num_experts)
    write_qmoe_ranked_experts_csv(qmoe_ranked_path, rankings)
    write_inference_expert_ranks_csv(
        inference_ranks_path, args.log, rankings, prompt_labels, num_experts
    )
    inference_counts, threshold_totals = aggregate_rank_thresholds_by_qmoe(
        args.log, rankings, num_experts
    )
    total_normalized, expert_bytes_complement_normalized = (
        write_qmoe_rank_threshold_totals_csv(
            qmoe_rank_thresholds_path,
            inference_counts,
            threshold_totals,
            expert_bytes,
            num_experts,
        )
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
