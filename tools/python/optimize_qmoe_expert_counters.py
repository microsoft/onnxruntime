#!/usr/bin/env python3
import argparse
import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from qmoe_expert_distribution import (
    CUMULATIVE_GLOBAL_POSITION_THRESHOLDS,
    LAYER_NUMBER,
    SELECTED_EXPERT_COUNT,
    generation_iteration,
    inspect_trace,
    node_identity,
    node_sort_key,
    parse_counter_trace,
)


@dataclass(frozen=True)
class ReplayEvent:
    node_index: int
    selected_experts: tuple[int, ...]
    prompt_index: int
    generation_iteration: int


@dataclass(frozen=True)
class OptimizationResult:
    alpha: float
    threshold: int
    selected_experts_at_or_above_threshold: int
    selected_expert_count: int
    moe_invocations_at_or_above_threshold: int
    moe_invocation_count: int
    impacted_invocations_by_node: tuple[int, ...]
    invocations_by_node: tuple[int, ...]
    selected_experts_below_threshold_by_node: tuple[int, ...]

    @property
    def cumulative_share(self):
        return self.selected_experts_at_or_above_threshold / self.selected_expert_count

    @property
    def cumulative_moe_share(self):
        return self.moe_invocations_at_or_above_threshold / self.moe_invocation_count


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay an ORT MoE counter trace and grid-search alpha to minimize the cumulative share of selected "
            "experts whose zero-based global position is at or above several thresholds."
        )
    )
    parser.add_argument("log", type=Path, help="Log containing 'moe_expert_counters' JSON records.")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Alpha values to evaluate.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(CUMULATIVE_GLOBAL_POSITION_THRESHOLDS),
        help="Zero-based global-position thresholds.",
    )
    parser.add_argument(
        "--layer-thresholds",
        type=int,
        nargs="+",
        default=[8000, 9000],
        help="Thresholds included in the per-layer impact distribution.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Output prefix for the CSV and PNG files (defaults to <log stem>-counter-optimization).",
    )
    return parser.parse_args()


def load_replay_trace(log_path):
    summary, identities, _, moe_count, total_expert_count = inspect_trace(log_path)
    if len(identities) != moe_count:
        raise ValueError(
            f"Optimization requires an event from every declared MoE node: "
            f"observed {len(identities)} of {moe_count} nodes."
        )
    if total_expert_count % moe_count:
        raise ValueError(
            f"Optimization requires the same expert count for every MoE node: "
            f"{total_expert_count} experts across {moe_count} nodes."
        )
    experts_per_node = total_expert_count // moe_count
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    events = []
    expert_offsets = {}

    def append_event(prompt_index, event):
        identity = node_identity(event)
        if "expert_offset" in event:
            offset = event["expert_offset"]
            if type(offset) is not int or not 0 <= offset < total_expert_count or offset % experts_per_node:
                raise ValueError(f"Invalid runtime expert offset for {identity}: {offset}.")
            if identity in expert_offsets and expert_offsets[identity] != offset:
                raise ValueError(f"Inconsistent runtime expert offsets for {identity}.")
            expert_offsets[identity] = offset
        selected_experts = tuple(selected_expert["expert_id"] for selected_expert in event["selected_experts"])
        if any(not 0 <= expert_id < experts_per_node for expert_id in selected_experts):
            raise ValueError(f"Trace contains an expert ID outside [0, {experts_per_node}): {selected_experts}.")
        events.append(
            ReplayEvent(identity_to_index[identity], selected_experts, prompt_index, generation_iteration(event))
        )

    parsed_summary = parse_counter_trace(log_path, append_event)
    if parsed_summary != summary:
        raise ValueError("Trace changed while it was being loaded.")
    if expert_offsets:
        if len(expert_offsets) != moe_count or sorted(expert_offsets.values()) != list(
            range(0, total_expert_count, experts_per_node)
        ):
            raise ValueError("Trace must contain a complete, non-overlapping set of runtime expert offsets.")
        runtime_identities = sorted(identities, key=expert_offsets.__getitem__)
    else:
        if any(identity[0] != "main" for identity in identities):
            raise ValueError("Replay of subgraphs requires counter records with runtime expert_offset metadata.")
        # Legacy main-graph kernels are registered by resolved node index, not layer name or execution order.
        runtime_identities = sorted(identities, key=lambda identity: identity[1])
    runtime_indices = {identity: index for index, identity in enumerate(runtime_identities)}
    events = [
        ReplayEvent(
            runtime_indices[identities[event.node_index]],
            event.selected_experts,
            event.prompt_index,
            event.generation_iteration,
        )
        for event in events
    ]
    return events, runtime_identities, moe_count, experts_per_node


def evaluate_alpha(
    events,
    moe_count,
    experts_per_node,
    thresholds,
    alpha,
    selected_expert_count=SELECTED_EXPERT_COUNT,
):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Alpha must be finite and in [0, 1].")
    scores = [[0.0] * experts_per_node for _ in range(moe_count)]
    rankings = [
        [(0.0, node_index * experts_per_node + expert_id) for expert_id in range(experts_per_node)]
        for node_index in range(moe_count)
    ]
    selected_at_or_above_threshold = dict.fromkeys(thresholds, 0)
    moe_at_or_above_threshold = dict.fromkeys(thresholds, 0)
    impacted_by_node = {threshold: [0] * moe_count for threshold in thresholds}
    selected_below_by_node = {threshold: [0] * moe_count for threshold in thresholds}
    invocations_by_node = [0] * moe_count
    selected_total = 0
    moe_invocation_total = 0
    for event in events:
        node_scores = scores[event.node_index]
        if event.generation_iteration > 0:
            if len(event.selected_experts) != selected_expert_count:
                raise ValueError(f"Decode event must contain {selected_expert_count} selected experts.")
            global_offset = event.node_index * experts_per_node
            global_positions = []
            for expert_id in event.selected_experts:
                key = (-node_scores[expert_id], global_offset + expert_id)
                global_position = sum(bisect.bisect_left(ranking, key) for ranking in rankings)
                global_positions.append(global_position)
                for threshold in thresholds:
                    selected_at_or_above_threshold[threshold] += global_position >= threshold
                    selected_below_by_node[threshold][event.node_index] += global_position < threshold
            maximum_global_position = max(global_positions)
            for threshold in thresholds:
                impacted = maximum_global_position >= threshold
                moe_at_or_above_threshold[threshold] += impacted
                impacted_by_node[threshold][event.node_index] += impacted
            selected_total += selected_expert_count
            moe_invocation_total += 1
            invocations_by_node[event.node_index] += 1

        node_scores = [alpha * score for score in node_scores]
        for expert_id in event.selected_experts:
            node_scores[expert_id] += 1.0 - alpha
        scores[event.node_index] = node_scores
        rankings[event.node_index] = sorted(
            (-score, event.node_index * experts_per_node + expert_id) for expert_id, score in enumerate(node_scores)
        )

    if selected_total == 0:
        raise ValueError("Trace contains no decode events.")
    return [
        OptimizationResult(
            alpha=alpha,
            threshold=threshold,
            selected_experts_at_or_above_threshold=selected_at_or_above_threshold[threshold],
            selected_expert_count=selected_total,
            moe_invocations_at_or_above_threshold=moe_at_or_above_threshold[threshold],
            moe_invocation_count=moe_invocation_total,
            impacted_invocations_by_node=tuple(impacted_by_node[threshold]),
            invocations_by_node=tuple(invocations_by_node),
            selected_experts_below_threshold_by_node=tuple(selected_below_by_node[threshold]),
        )
        for threshold in thresholds
    ]


def validate_alpha_candidates(alphas):
    if any(not math.isfinite(alpha) or not 0 <= alpha <= 1 for alpha in alphas):
        raise ValueError("Every alpha must be finite and satisfy 0 <= alpha <= 1.")
    return sorted(set(alphas))


def write_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "alpha",
                "threshold",
                "selected_experts_at_or_above_threshold",
                "selected_expert_count",
                "cumulative_share",
                "moe_invocations_at_or_above_threshold",
                "moe_invocation_count",
                "cumulative_moe_share",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.alpha,
                    result.threshold,
                    result.selected_experts_at_or_above_threshold,
                    result.selected_expert_count,
                    result.cumulative_share,
                    result.moe_invocations_at_or_above_threshold,
                    result.moe_invocation_count,
                    result.cumulative_moe_share,
                ]
            )


def write_plot(path, results, value, ylabel, maximum=None, integer_grid=False):
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import MultipleLocator, PercentFormatter  # noqa: PLC0415

    thresholds = sorted({result.threshold for result in results})
    figure, axis = plt.subplots(figsize=(10, 6))
    for threshold in thresholds:
        threshold_results = sorted(
            (result for result in results if result.threshold == threshold),
            key=lambda result: result.alpha,
        )
        axis.plot(
            [result.alpha for result in threshold_results],
            [value(result) for result in threshold_results],
            marker="o",
            linewidth=1.5,
            label=f"position >= {threshold}",
        )
    axis.set_xlabel("Alpha")
    axis.set_ylabel(ylabel)
    if maximum is None:
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    else:
        axis.set_ylim(0, maximum)
    if integer_grid:
        axis.yaxis.set_major_locator(MultipleLocator(1))
        axis.yaxis.set_minor_locator(MultipleLocator(0.5))
    axis.set_xlim(min(result.alpha for result in results), max(result.alpha for result in results))
    axis.set_ylim(bottom=0)
    axis.minorticks_on()
    axis.grid(True, which="major", alpha=0.35)
    axis.grid(True, which="minor", linestyle=":", alpha=0.15)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def layer_number(identity):
    match = LAYER_NUMBER.search(identity[3])
    return int(match.group(1)) if match else identity[1]


def write_layer_impacts(csv_path, plot_path, results, identities, thresholds):
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import PercentFormatter  # noqa: PLC0415

    best_results = {}
    for threshold in thresholds:
        best_results[threshold] = min(
            (result for result in results if result.threshold == threshold),
            key=lambda result: (result.cumulative_moe_share, result.alpha),
        )

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "threshold",
                "alpha",
                "layer_number",
                "impacted_invocations",
                "invocation_count",
                "impact_share",
            ]
        )
        for threshold, result in best_results.items():
            for node_index, identity in sorted(enumerate(identities), key=lambda item: node_sort_key(item[1])):
                impacted = result.impacted_invocations_by_node[node_index]
                invocation_count = result.invocations_by_node[node_index]
                writer.writerow(
                    [
                        threshold,
                        result.alpha,
                        layer_number(identity),
                        impacted,
                        invocation_count,
                        impacted / invocation_count,
                    ]
                )

    figure, axis = plt.subplots(figsize=(12, 6))
    for threshold, result in best_results.items():
        values = sorted(
            (
                layer_number(identity),
                result.impacted_invocations_by_node[node_index] / result.invocations_by_node[node_index],
            )
            for node_index, identity in enumerate(identities)
        )
        axis.plot(
            [layer for layer, _ in values],
            [share for _, share in values],
            marker="o",
            linewidth=1.2,
            label=f"threshold {threshold}, alpha {result.alpha:g}",
        )
    axis.set_xlabel("MoE layer number")
    axis.set_ylabel("Share of impacted invocations")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axis.set_ylim(0, 1)
    axis.minorticks_on()
    axis.grid(True, which="major", alpha=0.35)
    axis.grid(True, which="minor", linestyle=":", alpha=0.15)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)


def average_all_experts_below_threshold(events, moe_count, experts_per_node, alpha, thresholds):
    scores = [[0.0] * experts_per_node for _ in range(moe_count)]
    totals = {threshold: [0] * moe_count for threshold in thresholds}
    iteration_count = 0
    nodes_seen = set()
    current_iteration = None

    for event in events:
        if event.generation_iteration > 0:
            iteration = (event.prompt_index, event.generation_iteration)
            if iteration != current_iteration:
                if nodes_seen and len(nodes_seen) != moe_count:
                    raise ValueError("The trace contains an incomplete decode iteration.")
                current_iteration = iteration
                nodes_seen.clear()
                ranked_experts = sorted(
                    (
                        -score,
                        node_index * experts_per_node + expert_id,
                        node_index,
                    )
                    for node_index, node_scores in enumerate(scores)
                    for expert_id, score in enumerate(node_scores)
                )
                for threshold in thresholds:
                    for _, _, node_index in ranked_experts[:threshold]:
                        totals[threshold][node_index] += 1
                iteration_count += 1
            if event.node_index in nodes_seen:
                raise ValueError("A MoE node was evaluated twice before the decode iteration completed.")
            nodes_seen.add(event.node_index)

        node_scores = [alpha * score for score in scores[event.node_index]]
        for expert_id in event.selected_experts:
            node_scores[expert_id] += 1.0 - alpha
        scores[event.node_index] = node_scores

    if nodes_seen and len(nodes_seen) != moe_count:
        raise ValueError("The trace ends during an incomplete decode iteration.")
    if iteration_count == 0:
        raise ValueError("Trace contains no decode events.")
    return {threshold: [total / iteration_count for total in node_totals] for threshold, node_totals in totals.items()}


def write_layer_experts_below_threshold(
    csv_path,
    plot_path,
    results,
    identities,
    thresholds,
    events,
    moe_count,
    experts_per_node,
):
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import MultipleLocator  # noqa: PLC0415

    best_results = {}
    for threshold in thresholds:
        best_results[threshold] = min(
            (result for result in results if result.threshold == threshold),
            key=lambda result: (result.cumulative_moe_share, result.alpha),
        )
    averages = {}
    for threshold, result in best_results.items():
        averages[threshold] = average_all_experts_below_threshold(
            events,
            moe_count,
            experts_per_node,
            result.alpha,
            [threshold],
        )[threshold]

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "threshold",
                "alpha",
                "layer_number",
                "average_experts_below_threshold",
            ]
        )
        for threshold, result in best_results.items():
            for node_index, identity in sorted(enumerate(identities), key=lambda item: node_sort_key(item[1])):
                writer.writerow(
                    [
                        threshold,
                        result.alpha,
                        layer_number(identity),
                        averages[threshold][node_index],
                    ]
                )

    figure, axis = plt.subplots(figsize=(12, 6))
    for threshold, result in best_results.items():
        values = sorted(
            (
                layer_number(identity),
                averages[threshold][node_index],
            )
            for node_index, identity in enumerate(identities)
        )
        axis.plot(
            [layer for layer, _ in values],
            [average for _, average in values],
            marker="o",
            linewidth=1.2,
            label=f"threshold {threshold}, alpha {result.alpha:g}",
        )
    axis.set_xlabel("MoE layer number")
    axis.set_ylabel("Average experts below threshold")
    axis.set_ylim(0, experts_per_node)
    axis.yaxis.set_major_locator(MultipleLocator(10))
    axis.yaxis.set_minor_locator(MultipleLocator(5))
    axis.grid(True, which="major", alpha=0.35)
    axis.grid(True, which="minor", linestyle=":", alpha=0.15)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    events, identities, moe_count, experts_per_node = load_replay_trace(args.log)
    total_expert_count = moe_count * experts_per_node
    thresholds = sorted(set(args.thresholds))
    if any(threshold < 0 or threshold >= total_expert_count for threshold in thresholds):
        raise ValueError(f"Every threshold must be in [0, {total_expert_count}).")
    output_prefix = args.output_prefix or args.log.with_name(f"{args.log.stem}-counter-optimization")
    layer_thresholds = sorted(set(args.layer_thresholds))
    if any(threshold not in thresholds for threshold in layer_thresholds):
        raise ValueError("Every layer threshold must also be present in --thresholds.")
    csv_path = Path(f"{output_prefix}.csv")
    expert_plot_path = Path(f"{output_prefix}-expert-share.png")
    moe_plot_path = Path(f"{output_prefix}-moe-share.png")
    layer_csv_path = Path(f"{output_prefix}-layer-impacts.csv")
    layer_plot_path = Path(f"{output_prefix}-layer-impacts.png")
    layer_experts_csv_path = Path(f"{output_prefix}-layer-experts-below-threshold.csv")
    layer_experts_plot_path = Path(f"{output_prefix}-layer-experts-below-threshold.png")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for alpha in validate_alpha_candidates(args.alphas):
        print(f"evaluating alpha={alpha:.17g}", flush=True)
        alpha_results = evaluate_alpha(events, moe_count, experts_per_node, thresholds, alpha)
        results.extend(alpha_results)
        print(
            "  "
            + ", ".join(f"threshold {result.threshold}: {result.cumulative_share:.12g}" for result in alpha_results),
            flush=True,
        )

    write_results(csv_path, results)
    write_plot(
        expert_plot_path,
        results,
        lambda result: result.cumulative_share,
        "Cumulative share among selected experts",
    )
    write_plot(
        moe_plot_path,
        results,
        lambda result: result.cumulative_moe_share * moe_count,
        "Cumulative average number of MoE nodes",
        maximum=moe_count,
        integer_grid=True,
    )
    write_layer_impacts(layer_csv_path, layer_plot_path, results, identities, layer_thresholds)
    write_layer_experts_below_threshold(
        layer_experts_csv_path,
        layer_experts_plot_path,
        results,
        identities,
        layer_thresholds,
        events,
        moe_count,
        experts_per_node,
    )
    print(f"evaluated alpha values: {len({result.alpha for result in results})}")
    for threshold in thresholds:
        best = min(
            (result for result in results if result.threshold == threshold),
            key=lambda result: (result.cumulative_share, result.alpha),
        )
        print(f"threshold {threshold}: alpha={best.alpha:.17g}, cumulative share={best.cumulative_share:.12g}")
    print(csv_path)
    print(expert_plot_path)
    print(moe_plot_path)
    print(layer_csv_path)
    print(layer_plot_path)
    print(layer_experts_csv_path)
    print(layer_experts_plot_path)


if __name__ == "__main__":
    main()
