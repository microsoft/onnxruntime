#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

COUNTER_MARKER = "moe_expert_counters "
COUNTER_COMPLETE_MARKER = "moe_expert_counters_complete "
COUNTER_TRUNCATED_MARKER = "moe_expert_counters_truncated "
PROMPT_START = re.compile(r"^\[qmoe_prompt_runner\] (\d+)/(\d+) prompt_start$")
PROMPT_END = re.compile(r"^\[qmoe_prompt_runner\] (\d+)/(\d+) prompt_end$")
LAYER_NUMBER = re.compile(r"/layers\.(\d+)/")
SELECTED_EXPERT_COUNT = 8
GLOBAL_POSITION_THRESHOLDS = (5120, 8000)
CUMULATIVE_GLOBAL_POSITION_THRESHOLDS = (5120, 6000, 7000, 8000, 9000, 10000)


@dataclass(frozen=True)
class TraceSummary:
    prompts: int
    counter_records: int


@dataclass(frozen=True)
class TimelineRow:
    prompt_index: int
    time_index: int
    qmoe_time_index: int
    identity: tuple[str, int, str, str]
    minimum_score: float
    global_position: int
    expert_counts_by_global_position_threshold: tuple[int, ...]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the minimum score and maximum global position among the eight selected experts over time from "
            "an ORT MoE counter trace. Scores and positions are captured before the counter update. Global positions "
            "rank all model experts by score, with ties ordered by their fixed KernelPilot global expert index."
        )
    )
    parser.add_argument("log", type=Path, help="Log containing 'moe_expert_counters' JSON records.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Output prefix (defaults to <log stem>-expert-timeline).",
    )
    return parser.parse_args()


def layer_sort_key(node_name):
    match = LAYER_NUMBER.search(node_name)
    return (int(match.group(1)), node_name) if match else (10**9, node_name)


def node_identity(event):
    return event["graph_scope"], event["node_index"], event["node_type"], event["node_name"]


def node_sort_key(identity):
    graph_scope, node_index, node_type, node_name = identity
    return (graph_scope, *layer_sort_key(node_name), node_index, node_type)


def node_display_name(identity):
    graph_scope, node_index, node_type, node_name = identity
    return f"{graph_scope}:{node_name or f'{node_type}[{node_index}]'}"


def _validate_counter_event(event, line_number):
    if not isinstance(event, dict):
        raise ValueError(f"Line {line_number}: counter payload must be a JSON object.")
    if not isinstance(event.get("request_id"), str):
        raise ValueError(f"Line {line_number}: request_id must be a string.")
    if not isinstance(event.get("graph_scope"), str) or not event["graph_scope"]:
        raise ValueError(f"Line {line_number}: graph_scope must be a non-empty string.")
    if not isinstance(event.get("node_index"), int) or isinstance(event["node_index"], bool) or event["node_index"] < 0:
        raise ValueError(f"Line {line_number}: node_index must be a non-negative integer.")
    if not isinstance(event.get("node_type"), str) or not event["node_type"]:
        raise ValueError(f"Line {line_number}: node_type must be a non-empty string.")
    if not isinstance(event.get("node_name"), str):
        raise ValueError(f"Line {line_number}: node_name must be a string.")
    for count_name in ("moe_count", "total_expert_count"):
        count = event.get(count_name)
        if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count <= 0):
            raise ValueError(f"Line {line_number}: {count_name} must be a positive integer.")
    selected_experts = event.get("selected_experts")
    if not isinstance(selected_experts, list) or not selected_experts:
        raise ValueError(f"Line {line_number}: selected_experts must be a non-empty list.")
    for selected_expert in selected_experts:
        if not isinstance(selected_expert, dict):
            raise ValueError(f"Line {line_number}: selected_experts must contain objects.")
        expert_id = selected_expert.get("expert_id")
        if not isinstance(expert_id, int) or isinstance(expert_id, bool) or expert_id < 0:
            raise ValueError(f"Line {line_number}: selected expert IDs must be non-negative integers.")
        score = selected_expert.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) or score < 0:
            raise ValueError(f"Line {line_number}: selected expert scores must be finite non-negative numbers.")
        for rank_name in ("selected_rank", "node_rank", "global_position"):
            rank = selected_expert.get(rank_name)
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
                raise ValueError(f"Line {line_number}: {rank_name} must be a non-negative integer.")
        if selected_expert.get("device") not in ("CPU", "CUDA"):
            raise ValueError(f"Line {line_number}: selected expert device must be 'CPU' or 'CUDA'.")

    expert_ids = [selected_expert["expert_id"] for selected_expert in selected_experts]
    if len(expert_ids) != len(set(expert_ids)):
        raise ValueError(f"Line {line_number}: selected_experts must not contain duplicates.")
    global_positions = [selected_expert["global_position"] for selected_expert in selected_experts]
    if len(global_positions) != len(set(global_positions)):
        raise ValueError(f"Line {line_number}: selected expert global_position values must be unique.")
    selected_ranks = [selected_expert["selected_rank"] for selected_expert in selected_experts]
    if sorted(selected_ranks) != list(range(len(selected_experts))):
        raise ValueError(f"Line {line_number}: selected_rank values must be consecutive from zero.")
    expected_order = sorted(
        selected_experts,
        key=lambda selected_expert: (-selected_expert["score"], selected_expert["expert_id"]),
    )
    if [selected_expert["expert_id"] for selected_expert in expected_order] != [
        selected_expert["expert_id"]
        for selected_expert in sorted(selected_experts, key=lambda item: item["selected_rank"])
    ]:
        raise ValueError(
            f"Line {line_number}: selected_rank must rank scores by decreasing value and expert ID for ties."
        )


def parse_counter_trace(log_path, on_event=None):
    active_prompt = None
    completed_prompts = 0
    expected_prompts = None
    counter_event_count = 0
    completion = None

    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            if COUNTER_TRUNCATED_MARKER in line:
                raise ValueError(f"Incomplete counter trace: counter logging was truncated at line {line_number}.")

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

            completion_position = line.find(COUNTER_COMPLETE_MARKER)
            if completion_position >= 0:
                if completion is not None:
                    raise ValueError(f"Duplicate counter completion footer at line {line_number}.")
                if active_prompt is not None:
                    raise ValueError(f"Counter completion footer precedes prompt_end at line {line_number}.")
                try:
                    completion = json.loads(line[completion_position + len(COUNTER_COMPLETE_MARKER) :])
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid counter completion footer at line {line_number}: {exc}") from exc
                continue

            marker_position = line.find(COUNTER_MARKER)
            if marker_position < 0:
                continue
            if active_prompt is None:
                raise ValueError(f"Counter event at line {line_number} is outside an active prompt.")
            if completion is not None:
                raise ValueError(f"Counter event follows the completion footer at line {line_number}.")
            try:
                event = json.loads(line[marker_position + len(COUNTER_MARKER) :])
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid counter JSON at line {line_number}: {exc}") from exc
            _validate_counter_event(event, line_number)
            if on_event is not None:
                on_event(active_prompt, event)
            counter_event_count += 1

    if active_prompt is not None:
        raise ValueError(f"Incomplete counter trace: prompt {active_prompt} has no prompt_end marker.")
    if expected_prompts is None:
        raise ValueError("Incomplete counter trace: no prompt_start marker.")
    if completed_prompts != expected_prompts:
        raise ValueError(f"Incomplete counter trace: completed {completed_prompts} of {expected_prompts} prompts.")
    if not isinstance(completion, dict):
        raise ValueError("Incomplete counter trace: missing moe_expert_counters_complete footer.")
    expected_completion = {
        "prompts": expected_prompts,
        "prompt_runs": completed_prompts,
        "counter_records": counter_event_count,
    }
    if completion != expected_completion:
        raise ValueError(f"Counter completion footer mismatch: expected {expected_completion}, got {completion}.")
    return TraceSummary(expected_prompts, counter_event_count)


def inspect_trace(log_path):
    identities = set()
    selection_counts = Counter()
    moe_counts = set()
    total_expert_counts = set()
    maximum_global_position = -1

    def inspect_event(_, event):
        nonlocal maximum_global_position
        identities.add(node_identity(event))
        selection_counts[len(event["selected_experts"])] += 1
        if "moe_count" in event:
            moe_counts.add(event["moe_count"])
        if "total_expert_count" in event:
            total_expert_counts.add(event["total_expert_count"])
        maximum_global_position = max(
            maximum_global_position,
            *(selected_expert["global_position"] for selected_expert in event["selected_experts"]),
        )

    summary = parse_counter_trace(log_path, inspect_event)
    if not identities:
        raise ValueError(f"No '{COUNTER_MARKER.strip()}' records found in {log_path}.")
    if len(moe_counts) > 1 or len(total_expert_counts) > 1:
        raise ValueError("Trace contains inconsistent MoE or total expert counts.")
    moe_count = moe_counts.pop() if moe_counts else len(identities)
    total_expert_count = total_expert_counts.pop() if total_expert_counts else maximum_global_position + 1
    return (
        summary,
        sorted(identities, key=node_sort_key),
        selection_counts,
        moe_count,
        total_expert_count,
    )


def build_timeline(
    log_path,
    summary,
    selected_expert_count=SELECTED_EXPERT_COUNT,
):
    qmoe_time_indexes = Counter()
    rows = []
    time_index = 0

    def append_row(prompt_index, event):
        nonlocal time_index
        identity = node_identity(event)
        selected_experts = event["selected_experts"]
        if len(selected_experts) != selected_expert_count:
            return
        minimum_score = min(selected_expert["score"] for selected_expert in selected_experts)
        global_position = max(selected_expert["global_position"] for selected_expert in selected_experts)
        expert_counts_by_global_position_threshold = tuple(
            sum(selected_expert["global_position"] >= threshold for selected_expert in selected_experts)
            for threshold in CUMULATIVE_GLOBAL_POSITION_THRESHOLDS
        )
        qmoe_time_indexes[identity] += 1
        rows.append(
            TimelineRow(
                prompt_index=prompt_index,
                time_index=time_index,
                qmoe_time_index=qmoe_time_indexes[identity],
                identity=identity,
                minimum_score=minimum_score,
                global_position=global_position,
                expert_counts_by_global_position_threshold=expert_counts_by_global_position_threshold,
            )
        )
        time_index += 1

    parsed_summary = parse_counter_trace(log_path, append_row)
    if parsed_summary != summary:
        raise ValueError("Trace changed while it was being analyzed.")
    return rows


def write_timeline_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "prompt_index",
                "time_index",
                "qmoe_time_index",
                "graph_scope",
                "node_index",
                "node_type",
                "node_name",
                "minimum_score",
                "global_position_0_based",
                *(f"experts_global_position_ge_{threshold}" for threshold in GLOBAL_POSITION_THRESHOLDS),
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.prompt_index,
                    row.time_index,
                    row.qmoe_time_index,
                    *row.identity,
                    row.minimum_score,
                    row.global_position,
                    *(
                        row.expert_counts_by_global_position_threshold[
                            CUMULATIVE_GLOBAL_POSITION_THRESHOLDS.index(threshold)
                        ]
                        for threshold in GLOBAL_POSITION_THRESHOLDS
                    ),
                ]
            )


def aggregate_threshold_counts(rows, thresholds=GLOBAL_POSITION_THRESHOLDS):
    totals = {threshold: Counter() for threshold in thresholds}
    for row in rows:
        for threshold, count in zip(
            CUMULATIVE_GLOBAL_POSITION_THRESHOLDS,
            row.expert_counts_by_global_position_threshold,
            strict=True,
        ):
            if threshold in totals:
                totals[threshold][row.qmoe_time_index] += count
    return {threshold: sorted(counts.items()) for threshold, counts in totals.items()}


def cumulative_threshold_counts(rows):
    aggregates = aggregate_threshold_counts(rows, CUMULATIVE_GLOBAL_POSITION_THRESHOLDS)
    cumulative = {}
    for threshold, values in aggregates.items():
        running_total = 0
        cumulative_values = []
        for iteration, count in values:
            running_total += count
            cumulative_values.append((iteration, running_total))
        cumulative[threshold] = cumulative_values
    return cumulative


def cumulative_threshold_fractions(rows):
    cumulative = cumulative_threshold_counts(rows)
    selected_experts_by_iteration = Counter()
    for row in rows:
        selected_experts_by_iteration[row.qmoe_time_index] += SELECTED_EXPERT_COUNT

    cumulative_selected_experts = {}
    running_total = 0
    for iteration, count in sorted(selected_experts_by_iteration.items()):
        running_total += count
        cumulative_selected_experts[iteration] = running_total

    return {
        threshold: [(iteration, count / cumulative_selected_experts[iteration]) for iteration, count in values]
        for threshold, values in cumulative.items()
    }


def write_timeline_plots(path, rows, identities, moe_count, total_expert_count):
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: PLC0415
    from matplotlib.ticker import PercentFormatter  # noqa: PLC0415

    rows_by_identity = defaultdict(list)
    for row in rows:
        rows_by_identity[row.identity].append(row)

    with PdfPages(path) as pdf:
        for identity in identities:
            node_rows = rows_by_identity[identity]
            if not node_rows:
                continue
            x = [row.qmoe_time_index for row in node_rows]
            figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
            axes[0].plot(
                x,
                [row.minimum_score for row in node_rows],
                color="black",
                linewidth=1,
            )
            axes[0].set_ylabel("Minimum score")
            axes[1].plot(x, [row.global_position for row in node_rows], linewidth=0.9)
            axes[1].set_ylabel("Global position before update")
            axes[2].plot(
                x,
                [
                    row.expert_counts_by_global_position_threshold[
                        CUMULATIVE_GLOBAL_POSITION_THRESHOLDS.index(GLOBAL_POSITION_THRESHOLDS[0])
                    ]
                    for row in node_rows
                ],
                linewidth=0.9,
            )
            axes[2].set_ylabel(f"Experts with position >= {GLOBAL_POSITION_THRESHOLDS[0]}")
            axes[2].set_ylim(-0.25, SELECTED_EXPERT_COUNT + 0.25)
            axes[2].set_xlabel("Decode invocation")
            for axis in axes:
                axis.grid(True, alpha=0.3)

            previous_prompt = node_rows[0].prompt_index
            for row in node_rows[1:]:
                if row.prompt_index != previous_prompt:
                    for axis in axes:
                        axis.axvline(row.qmoe_time_index, color="gray", linewidth=0.6, alpha=0.5)
                    previous_prompt = row.prompt_index
            figure.suptitle(node_display_name(identity))
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)

        aggregate = aggregate_threshold_counts(rows)
        prompt_starts = {}
        for row in rows:
            prompt_starts[row.prompt_index] = min(
                row.qmoe_time_index,
                prompt_starts.get(row.prompt_index, row.qmoe_time_index),
            )
        for logarithmic in (False, True):
            figure, axis = plt.subplots(figsize=(13, 5))
            for threshold, values in aggregate.items():
                axis.plot(
                    [iteration for iteration, _ in values],
                    [total for _, total in values],
                    linewidth=1,
                    label=f"position >= {threshold}",
                )
            axis.set_xlabel("Decode iteration")
            axis.set_ylabel("Selected expert count across all MoE nodes")
            scale = "logarithmic" if logarithmic else "linear"
            axis.set_title(f"{moe_count} MoE nodes, {total_expert_count} experts total — {scale} scale")
            if logarithmic:
                axis.set_yscale("log", nonpositive="clip")
            axis.grid(True, alpha=0.3)
            axis.legend()
            for prompt_index, iteration in sorted(prompt_starts.items())[1:]:
                axis.axvline(iteration, color="gray", linewidth=0.8, alpha=0.6)
                axis.text(
                    iteration,
                    1,
                    f"Prompt {prompt_index}",
                    rotation=90,
                    verticalalignment="top",
                    horizontalalignment="right",
                    transform=axis.get_xaxis_transform(),
                    fontsize=8,
                    color="gray",
                )
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)

        cumulative = cumulative_threshold_counts(rows)
        for logarithmic in (False, True):
            figure, axis = plt.subplots(figsize=(13, 5))
            for threshold, values in cumulative.items():
                axis.plot(
                    [iteration for iteration, _ in values],
                    [total for _, total in values],
                    linewidth=1,
                    label=f"position >= {threshold}",
                )
            axis.set_xlabel("Decode iteration")
            axis.set_ylabel("Cumulative selected expert count")
            scale = "logarithmic" if logarithmic else "linear"
            axis.set_title(f"{moe_count} MoE nodes, {total_expert_count} experts total — cumulative, {scale} scale")
            if logarithmic:
                axis.set_yscale("log", nonpositive="clip")
            axis.grid(True, alpha=0.3)
            axis.legend()
            for prompt_index, iteration in sorted(prompt_starts.items())[1:]:
                axis.axvline(iteration, color="gray", linewidth=0.8, alpha=0.6)
                axis.text(
                    iteration,
                    1,
                    f"Prompt {prompt_index}",
                    rotation=90,
                    verticalalignment="top",
                    horizontalalignment="right",
                    transform=axis.get_xaxis_transform(),
                    fontsize=8,
                    color="gray",
                )
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)

        cumulative_fractions = cumulative_threshold_fractions(rows)
        for logarithmic in (False, True):
            figure, axis = plt.subplots(figsize=(13, 5))
            for threshold, values in cumulative_fractions.items():
                axis.plot(
                    [iteration for iteration, _ in values],
                    [fraction for _, fraction in values],
                    linewidth=1,
                    label=f"position >= {threshold}",
                )
            axis.set_xlabel("Decode iteration")
            axis.set_ylabel("Cumulative share of selected experts")
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            scale = "logarithmic" if logarithmic else "linear"
            axis.set_title(
                f"{moe_count} MoE nodes, {total_expert_count} experts total — cumulative share, {scale} scale"
            )
            if logarithmic:
                axis.set_yscale("log", nonpositive="clip")
            else:
                axis.set_ylim(bottom=0)
            axis.grid(True, alpha=0.3)
            axis.legend()
            for prompt_index, iteration in sorted(prompt_starts.items())[1:]:
                axis.axvline(iteration, color="gray", linewidth=0.8, alpha=0.6)
                axis.text(
                    iteration,
                    1,
                    f"Prompt {prompt_index}",
                    rotation=90,
                    verticalalignment="top",
                    horizontalalignment="right",
                    transform=axis.get_xaxis_transform(),
                    fontsize=8,
                    color="gray",
                )
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)

        first_prompt_indexes = sorted(prompt_starts)[:2]
        first_prompt_rows = [row for row in rows if row.prompt_index in first_prompt_indexes]
        first_prompt_aggregate = aggregate_threshold_counts(first_prompt_rows)
        figure, axis = plt.subplots(figsize=(13, 5))
        for threshold, values in first_prompt_aggregate.items():
            axis.plot(
                [iteration for iteration, _ in values],
                [total for _, total in values],
                linewidth=1,
                label=f"position >= {threshold}",
            )
        axis.set_xlabel("Decode iteration")
        axis.set_ylabel("Selected expert count across all MoE nodes")
        axis.set_title(
            f"{moe_count} MoE nodes, {total_expert_count} experts total — first 2 prompts, logarithmic scale"
        )
        axis.set_yscale("log", nonpositive="clip")
        axis.grid(True, alpha=0.3)
        axis.legend()
        for prompt_index in first_prompt_indexes[1:]:
            iteration = prompt_starts[prompt_index]
            axis.axvline(iteration, color="gray", linewidth=0.8, alpha=0.6)
            axis.text(
                iteration,
                1,
                f"Prompt {prompt_index}",
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
                transform=axis.get_xaxis_transform(),
                fontsize=8,
                color="gray",
            )
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)


def main():
    args = parse_args()
    output_prefix = args.output_prefix or args.log.with_name(f"{args.log.stem}-expert-timeline")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary, identities, selection_counts, moe_count, total_expert_count = inspect_trace(args.log)
    rows = build_timeline(args.log, summary)
    timeline_path = Path(f"{output_prefix}.csv")
    plot_path = Path(f"{output_prefix}.pdf")
    write_timeline_csv(timeline_path, rows)
    write_timeline_plots(plot_path, rows, identities, moe_count, total_expert_count)

    print(f"counter events: {summary.counter_records}")
    print(f"prompts: {summary.prompts}")
    print(f"MoE nodes: {moe_count}")
    print(f"selected experts per event: {dict(sorted(selection_counts.items()))}")
    print(f"timeline rows ({SELECTED_EXPERT_COUNT} selected experts): {len(rows)}")
    print(timeline_path)
    print(plot_path)


if __name__ == "__main__":
    main()
