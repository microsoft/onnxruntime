#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qmoe_expert_distribution import (
    LAYER_NUMBER,
    SELECTED_EXPERT_COUNT,
    generation_iteration,
    inspect_trace,
    node_display_name,
    node_identity,
    node_sort_key,
    parse_counter_trace,
)

PLOT_EXTREME_COUNT = 1000


@dataclass(frozen=True)
class ActivationTrace:
    identities: tuple[tuple[str, int, str, str], ...]
    prompt_activations: tuple[np.ndarray, ...]
    expert_count: int
    selected_expert_rows: tuple[tuple[int, int, int, int], ...] = ()


@dataclass(frozen=True)
class CorrelationMatrix:
    relation: str
    current_identity: tuple[str, int, str, str]
    context_identity: tuple[str, int, str, str]
    values: np.ndarray
    current_activation_counts: np.ndarray
    context_activation_counts: np.ndarray
    samples: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute binary expert-selection correlations for each QMoE against the same QMoE at the previous "
            "decode token and against the QMoE in the previous layer at the same token."
        )
    )
    parser.add_argument("log", type=Path, help="Log containing 'moe_expert_counters' JSON records.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Output prefix (defaults to <log stem>-expert-activation-correlations).",
    )
    parser.add_argument(
        "--pairs-per-tail",
        "--top-pairs",
        dest="pairs_per_tail",
        type=int,
        default=20,
        help="Number of highest and lowest correlations to write per matrix to the summary CSV.",
    )
    parser.add_argument(
        "--min-activations",
        type=int,
        default=10,
        help="Minimum activation count for both experts in pairs written to the summary CSV.",
    )
    return parser.parse_args()


def load_activation_trace(log_path, selected_expert_count=SELECTED_EXPERT_COUNT):
    summary, identities, _, moe_count, total_expert_count = inspect_trace(log_path)
    if len(identities) != moe_count:
        raise ValueError(
            f"Correlation analysis requires an event from every declared MoE node: "
            f"observed {len(identities)} of {moe_count} nodes."
        )
    if total_expert_count % moe_count:
        raise ValueError(
            f"Correlation analysis requires the same expert count for every MoE node: "
            f"{total_expert_count} experts across {moe_count} nodes."
        )

    expert_count = total_expert_count // moe_count
    identities = tuple(sorted(identities, key=node_sort_key))
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    selected_by_prompt = defaultdict(lambda: [{} for _ in identities])
    decode_iterations = [0] * len(identities)
    selected_expert_rows = []

    def append_event(prompt_index, event):
        selected_experts = event["selected_experts"]
        iteration = generation_iteration(event)
        if iteration == 0:
            return
        if len(selected_experts) != selected_expert_count:
            raise ValueError(f"Decode event must contain {selected_expert_count} selected experts.")
        node_index = identity_to_index[node_identity(event)]
        expert_ids = tuple(selected_expert["expert_id"] for selected_expert in selected_experts)
        if any(not 0 <= expert_id < expert_count for expert_id in expert_ids):
            raise ValueError(f"Trace contains an expert ID outside [0, {expert_count}): {expert_ids}.")
        if iteration in selected_by_prompt[prompt_index][node_index]:
            raise ValueError(f"Duplicate MoE node in prompt {prompt_index}, decode iteration {iteration}.")
        selected_by_prompt[prompt_index][node_index][iteration] = expert_ids
        decode_iterations[node_index] += 1
        selected_expert_rows.extend(
            (
                decode_iterations[node_index],
                node_index,
                selected_expert["expert_id"],
                selected_expert["global_position"],
            )
            for selected_expert in selected_experts
        )

    parsed_summary = parse_counter_trace(log_path, append_event)
    if parsed_summary != summary:
        raise ValueError("Trace changed while it was being loaded.")

    prompt_activations = []
    for prompt_index in range(1, summary.prompts + 1):
        selections_by_node = selected_by_prompt[prompt_index]
        iterations = [sorted(selections) for selections in selections_by_node]
        if not iterations or not iterations[0] or any(indexes != iterations[0] for indexes in iterations):
            counts = ", ".join(
                f"{node_display_name(identity)}={indexes}"
                for identity, indexes in zip(identities, iterations, strict=True)
            )
            raise ValueError(f"Prompt {prompt_index} has unaligned decode invocations across MoE nodes: {counts}.")

        activations = np.zeros((len(iterations[0]), moe_count, expert_count), dtype=np.uint8)
        for node_index, selections in enumerate(selections_by_node):
            for token_index, iteration in enumerate(iterations[0]):
                activations[token_index, node_index, list(selections[iteration])] = 1
        prompt_activations.append(activations)

    return ActivationTrace(
        identities,
        tuple(prompt_activations),
        expert_count,
        tuple(selected_expert_rows),
    )


def binary_activation_correlation(current, context):
    if current.ndim != 2 or context.ndim != 2 or current.shape[0] != context.shape[0]:
        raise ValueError("Activation matrices must be two-dimensional and have the same sample count.")
    if current.shape[0] == 0:
        raise ValueError("At least one activation sample is required.")

    current = current.astype(np.float64, copy=False)
    context = context.astype(np.float64, copy=False)
    sample_count = current.shape[0]
    current_counts = current.sum(axis=0)
    context_counts = context.sum(axis=0)
    joint_counts = current.T @ context
    numerator = sample_count * joint_counts - current_counts[:, None] * context_counts[None, :]
    denominator = np.sqrt(
        current_counts[:, None]
        * (sample_count - current_counts[:, None])
        * context_counts[None, :]
        * (sample_count - context_counts[None, :])
    )
    correlations = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=correlations, where=denominator != 0)
    return correlations, current_counts.astype(np.int64), context_counts.astype(np.int64)


def _layer_numbers(identities):
    layers = {}
    for node_index, identity in enumerate(identities):
        match = LAYER_NUMBER.search(identity[3])
        if match is None:
            raise ValueError(f"Cannot determine the layer number from MoE node name: {identity[3]!r}.")
        key = identity[0], int(match.group(1))
        if key in layers:
            raise ValueError(f"Multiple MoE nodes were found for graph scope {key[0]!r}, layer {key[1]}.")
        layers[key] = node_index
    return layers


def compute_activation_correlations(trace):
    same_qmoe_previous_token = []
    for node_index, identity in enumerate(trace.identities):
        current = np.concatenate(
            [activations[1:, node_index, :] for activations in trace.prompt_activations],
            axis=0,
        )
        previous = np.concatenate(
            [activations[:-1, node_index, :] for activations in trace.prompt_activations],
            axis=0,
        )
        values, current_counts, context_counts = binary_activation_correlation(current, previous)
        same_qmoe_previous_token.append(
            CorrelationMatrix(
                "same_qmoe_previous_token",
                identity,
                identity,
                values,
                current_counts,
                context_counts,
                current.shape[0],
            )
        )

    previous_qmoe_same_token = []
    layers = _layer_numbers(trace.identities)
    for (graph_scope, layer), current_node_index in sorted(layers.items()):
        previous_node_index = layers.get((graph_scope, layer - 1))
        if previous_node_index is None:
            continue
        current = np.concatenate(
            [activations[:, current_node_index, :] for activations in trace.prompt_activations],
            axis=0,
        )
        previous = np.concatenate(
            [activations[:, previous_node_index, :] for activations in trace.prompt_activations],
            axis=0,
        )
        values, current_counts, context_counts = binary_activation_correlation(current, previous)
        previous_qmoe_same_token.append(
            CorrelationMatrix(
                "previous_qmoe_same_token",
                trace.identities[current_node_index],
                trace.identities[previous_node_index],
                values,
                current_counts,
                context_counts,
                current.shape[0],
            )
        )

    return tuple(same_qmoe_previous_token), tuple(previous_qmoe_same_token)


def write_correlation_archive(path, same_qmoe_previous_token, previous_qmoe_same_token):
    path.parent.mkdir(parents=True, exist_ok=True)
    expert_count = same_qmoe_previous_token[0].values.shape[0]
    previous_values = (
        np.stack([correlation.values.astype(np.float32) for correlation in previous_qmoe_same_token])
        if previous_qmoe_same_token
        else np.empty((0, expert_count, expert_count), dtype=np.float32)
    )
    previous_current_counts = (
        np.stack([correlation.current_activation_counts for correlation in previous_qmoe_same_token])
        if previous_qmoe_same_token
        else np.empty((0, expert_count), dtype=np.int64)
    )
    previous_context_counts = (
        np.stack([correlation.context_activation_counts for correlation in previous_qmoe_same_token])
        if previous_qmoe_same_token
        else np.empty((0, expert_count), dtype=np.int64)
    )
    np.savez_compressed(
        path,
        same_qmoe_previous_token=np.stack(
            [correlation.values.astype(np.float32) for correlation in same_qmoe_previous_token]
        ),
        same_qmoe_current_activation_counts=np.stack(
            [correlation.current_activation_counts for correlation in same_qmoe_previous_token]
        ),
        same_qmoe_context_activation_counts=np.stack(
            [correlation.context_activation_counts for correlation in same_qmoe_previous_token]
        ),
        same_qmoe_node_names=np.array([correlation.current_identity[3] for correlation in same_qmoe_previous_token]),
        same_qmoe_samples=np.array([correlation.samples for correlation in same_qmoe_previous_token]),
        previous_qmoe_same_token=previous_values,
        previous_qmoe_current_activation_counts=previous_current_counts,
        previous_qmoe_context_activation_counts=previous_context_counts,
        previous_qmoe_current_node_names=np.array(
            [correlation.current_identity[3] for correlation in previous_qmoe_same_token]
        ),
        previous_qmoe_context_node_names=np.array(
            [correlation.context_identity[3] for correlation in previous_qmoe_same_token]
        ),
        previous_qmoe_samples=np.array([correlation.samples for correlation in previous_qmoe_same_token]),
    )


def _correlation_tails(correlation, pairs_per_tail, min_activations):
    supported = (
        (correlation.current_activation_counts[:, None] >= min_activations)
        & (correlation.context_activation_counts[None, :] >= min_activations)
        & np.isfinite(correlation.values)
    )
    finite_indexes = np.flatnonzero(supported)
    if pairs_per_tail <= 0 or finite_indexes.size == 0:
        return []
    pairs_per_tail = min(pairs_per_tail, finite_indexes.size)
    flat_values = correlation.values.ravel()
    ordered_indexes = finite_indexes[np.argsort(flat_values[finite_indexes], kind="stable")]
    lowest = [
        ("lowest", rank, *np.unravel_index(index, correlation.values.shape), flat_values[index])
        for rank, index in enumerate(ordered_indexes[:pairs_per_tail], start=1)
    ]
    highest = [
        ("highest", rank, *np.unravel_index(index, correlation.values.shape), flat_values[index])
        for rank, index in enumerate(reversed(ordered_indexes[-pairs_per_tail:]), start=1)
    ]
    return highest + lowest


def _positive_and_negative_extremes(values, count):
    finite_values = values[np.isfinite(values)]
    positive = np.sort(finite_values[finite_values > 0])[-count:][::-1]
    negative = np.sort(finite_values[finite_values < 0])[:count]
    return positive, negative


def write_correlation_summary(path, correlations, pairs_per_tail, min_activations):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "relation",
                "tail",
                "rank",
                "current_graph_scope",
                "current_node_index",
                "current_node_name",
                "context_graph_scope",
                "context_node_index",
                "context_node_name",
                "current_expert_id",
                "context_expert_id",
                "correlation",
                "absolute_correlation",
                "current_activation_count",
                "context_activation_count",
                "samples",
            ]
        )
        for correlation in correlations:
            for tail, rank, current_expert, context_expert, value in _correlation_tails(
                correlation, pairs_per_tail, min_activations
            ):
                writer.writerow(
                    [
                        correlation.relation,
                        tail,
                        rank,
                        correlation.current_identity[0],
                        correlation.current_identity[1],
                        correlation.current_identity[3],
                        correlation.context_identity[0],
                        correlation.context_identity[1],
                        correlation.context_identity[3],
                        current_expert,
                        context_expert,
                        value,
                        abs(value),
                        correlation.current_activation_counts[current_expert],
                        correlation.context_activation_counts[context_expert],
                        correlation.samples,
                    ]
                )


def write_selected_experts(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["iteration", "qmoe_number", "expert_number", "position"])
        writer.writerows(rows)


def write_correlation_plots(
    path,
    same_qmoe_previous_token,
    previous_qmoe_same_token,
):
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: PLC0415

    previous_by_current = {correlation.current_identity: correlation for correlation in previous_qmoe_same_token}

    with PdfPages(path) as pdf:
        for same_correlation in same_qmoe_previous_token:
            previous_correlation = previous_by_current.get(same_correlation.current_identity)
            panels = [(same_correlation, "Same QMoE, previous token")]
            if previous_correlation is not None:
                panels.append((previous_correlation, "Previous-layer QMoE, same token"))

            for correlation, title in panels:
                figure, heatmap_axis = plt.subplots(figsize=(12, 10))
                image = heatmap_axis.imshow(
                    correlation.values,
                    cmap="coolwarm",
                    interpolation="nearest",
                    vmin=-1,
                    vmax=1,
                    origin="lower",
                )
                heatmap_axis.set_title(
                    f"{node_display_name(same_correlation.current_identity)}\n{title} — {correlation.samples} samples"
                )
                heatmap_axis.set_xlabel("Context expert ID")
                heatmap_axis.set_ylabel("Current expert ID")
                figure.colorbar(image, ax=heatmap_axis, label="Pearson correlation")
                figure.tight_layout()
                pdf.savefig(figure)
                plt.close(figure)

            figure, extremes_axis = plt.subplots(figsize=(14, 8))
            colors = ("tab:blue", "tab:orange")
            for index, (correlation, title) in enumerate(panels):
                color = colors[index]
                positive, negative = _positive_and_negative_extremes(correlation.values, PLOT_EXTREME_COUNT)
                extremes_axis.plot(
                    np.arange(1, positive.size + 1),
                    positive,
                    color=color,
                    linewidth=1,
                    label=f"{title} — positive",
                )
                extremes_axis.plot(
                    np.arange(1, negative.size + 1),
                    negative,
                    color=color,
                    linestyle="--",
                    linewidth=1,
                    label=f"{title} — negative",
                )
            extremes_axis.axhline(0, color="black", linewidth=0.6)
            extremes_axis.set_xlim(1, PLOT_EXTREME_COUNT)
            extremes_axis.set_ylim(-1, 1)
            extremes_axis.set_xlabel("Rank")
            extremes_axis.set_ylabel("Pearson correlation")
            extremes_axis.set_title(
                f"{node_display_name(same_correlation.current_identity)}\n"
                f"{PLOT_EXTREME_COUNT} strongest positive and negative correlations"
            )
            extremes_axis.grid(True, alpha=0.3)
            extremes_axis.legend()
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)

            figure, sorted_axis = plt.subplots(figsize=(14, 8))
            for correlation, title in panels:
                values = np.sort(correlation.values[np.isfinite(correlation.values)])
                sorted_axis.plot(
                    np.arange(1, values.size + 1),
                    values,
                    linewidth=0.8,
                    label=f"{title} ({values.size:,} finite values)",
                )
            sorted_axis.axhline(0, color="black", linewidth=0.6)
            sorted_axis.set_ylim(-1, 1)
            sorted_axis.set_xlabel("Correlation rank")
            sorted_axis.set_ylabel("Pearson correlation")
            sorted_axis.set_title(
                f"{node_display_name(same_correlation.current_identity)}\n"
                "All finite expert-pair correlations in ascending order"
            )
            sorted_axis.grid(True, alpha=0.3)
            sorted_axis.legend()
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)


def main():
    args = parse_args()
    if args.pairs_per_tail < 0:
        raise ValueError("--pairs-per-tail must be non-negative.")
    if args.min_activations < 1:
        raise ValueError("--min-activations must be positive.")
    output_prefix = args.output_prefix or args.log.with_name(f"{args.log.stem}-expert-activation-correlations")
    trace = load_activation_trace(args.log)
    same_qmoe_previous_token, previous_qmoe_same_token = compute_activation_correlations(trace)
    archive_path = Path(f"{output_prefix}.npz")
    summary_path = Path(f"{output_prefix}.csv")
    selected_experts_path = Path(f"{output_prefix}-selected-experts.csv")
    plot_path = Path(f"{output_prefix}.pdf")
    write_correlation_archive(archive_path, same_qmoe_previous_token, previous_qmoe_same_token)
    write_correlation_summary(
        summary_path,
        (*same_qmoe_previous_token, *previous_qmoe_same_token),
        args.pairs_per_tail,
        args.min_activations,
    )
    write_selected_experts(selected_experts_path, trace.selected_expert_rows)
    write_correlation_plots(
        plot_path,
        same_qmoe_previous_token,
        previous_qmoe_same_token,
    )

    print(f"prompts: {len(trace.prompt_activations)}")
    print(f"MoE nodes: {len(trace.identities)}")
    print(f"experts per node: {trace.expert_count}")
    print(f"same-QMoE previous-token samples: {same_qmoe_previous_token[0].samples}")
    if previous_qmoe_same_token:
        print(f"previous-QMoE same-token samples: {previous_qmoe_same_token[0].samples}")
    print(archive_path)
    print(summary_path)
    print(selected_experts_path)
    print(plot_path)


if __name__ == "__main__":
    main()
