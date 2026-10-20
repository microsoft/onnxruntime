"""Compare delayed, learned Markov-state replacements with a frozen EMA set."""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np


def validate_matrix(matrix):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("The transition matrix must be square with at least two states.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("Transition probabilities must be finite and non-negative.")
    if not np.allclose(matrix.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("Each transition row must sum to one.")
    if np.any(np.diag(matrix) >= 1):
        raise ValueError("Absorbing states are not supported.")


def update_selection(selected, probabilities, max_changes):
    outgoing = sorted(selected, key=lambda state: (probabilities[state], state))
    incoming = sorted(
        (state for state in range(len(probabilities)) if state not in selected),
        key=lambda state: (-probabilities[state], state),
    )
    updated = set(selected)
    changes = 0
    for old, new in zip(outgoing[:max_changes], incoming[:max_changes], strict=False):
        if probabilities[new] <= probabilities[old]:
            break
        updated.remove(old)
        updated.add(new)
        changes += 1
    return frozenset(updated), changes


def iter_selections(sequence, dimension, capacity, max_changes, alpha, warmup, delay, transition_matrix=None, beta=0.9):
    probabilities = np.full(dimension, 1 / dimension)
    observed = np.zeros(dimension, dtype=bool)
    estimated = np.zeros((dimension, dimension))
    pending = deque()
    for t in range(len(sequence) - 1):
        state = sequence[t]
        if t > 0:
            previous = sequence[t - 1]
            if not observed[previous]:
                estimated[previous] = probabilities
                observed[previous] = True
            estimated[previous] *= beta
            estimated[previous, state] += 1 - beta
            probabilities *= alpha
            probabilities[state] += 1 - alpha
        if t < warmup:
            continue
        if t == warmup:
            frozen = frozenset(sorted(range(dimension), key=lambda index: (-probabilities[index], index))[:capacity])
            selected = frozen
            pending.extend((frozen, 0) for _ in range(delay - 1))
        if transition_matrix is None:
            estimated[~observed] = probabilities
            transition = estimated
        else:
            transition = transition_matrix
        scores = transition[state]
        if delay == 2:
            scores = scores @ transition
        # New decisions build on the last committed set, not the currently active one.
        base = pending[-1][0] if pending else selected
        pending.append(update_selection(base, scores, max_changes))
        selected, swaps = pending.popleft()
        yield frozen, selected, swaps


def compare_sequence(
    matrix, sequence, capacity, max_changes, alpha, warmup, delay=2, transition_model="estimated", beta=0.9
):
    frozen_hits = 0
    greedy_hits = 0
    changes = 0
    frozen_coverage = 0.0
    greedy_coverage = 0.0
    selections = iter_selections(
        sequence,
        len(matrix),
        capacity,
        max_changes,
        alpha,
        warmup,
        delay,
        transition_matrix=matrix if transition_model == "known" else None,
        beta=beta,
    )
    for t, (frozen, selected, swaps) in enumerate(selections, start=warmup):
        conditional = matrix[sequence[t]]
        frozen_coverage += sum(conditional[state] for state in sorted(frozen))
        greedy_coverage += sum(conditional[state] for state in sorted(selected))
        next_state = sequence[t + 1]
        frozen_hits += int(next_state in frozen)
        greedy_hits += int(next_state in selected)
        changes += swaps
    steps = len(sequence) - warmup - 1
    return {
        "initial_set": sorted(frozen),
        "final_greedy_set": sorted(selected),
        "steps": steps,
        "frozen_hits": frozen_hits,
        "greedy_hits": greedy_hits,
        "frozen_hit_rate": frozen_hits / steps,
        "greedy_hit_rate": greedy_hits / steps,
        "frozen_expected_hit_rate": float(frozen_coverage / steps),
        "greedy_expected_hit_rate": float(greedy_coverage / steps),
        "changes": changes,
        "changes_per_step": changes / steps,
    }


def generate_sequence(matrix, length, rng):
    sequence = np.empty(length, dtype=np.int64)
    sequence[0] = rng.integers(len(matrix))
    for t in range(1, length):
        sequence[t] = rng.choice(len(matrix), p=matrix[sequence[t - 1]])
    return sequence


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=256, help="State count; ignored with --matrix.")
    parser.add_argument("--matrix", type=Path, help="JSON transition matrix, as a list of rows.")
    parser.add_argument("--capacity", type=int, default=8, help="Number n of states retained.")
    parser.add_argument("--max-changes", type=int, default=2, help="Maximum k replacements per observation.")
    parser.add_argument("--delay", type=int, choices=(1, 2), default=2, help="Decision-to-application delay.")
    parser.add_argument("--transition-model", choices=("estimated", "known"), default="estimated")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument(
        "--beta", type=float, default=0.9, help="History weight per observed departure in a transition row."
    )
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.matrix is None and args.dimension < 2:
        parser.error("--dimension must be at least two.")
    if args.capacity < 1:
        parser.error("--capacity must be positive.")
    if not 0 <= args.max_changes < args.capacity:
        parser.error("--max-changes must satisfy 0 <= k < capacity.")
    if not np.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error("--alpha must be finite and in [0, 1].")
    if not np.isfinite(args.beta) or not 0 <= args.beta <= 1:
        parser.error("--beta must be finite and in [0, 1].")
    if args.warmup < 0 or args.steps < 1 or args.repeats < 1:
        parser.error("--warmup must be non-negative; --steps and --repeats must be positive.")
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    if args.matrix is not None and args.output is not None and args.matrix.resolve() == args.output.resolve():
        parser.error("--matrix and --output must refer to different files.")
    return args


def main():
    args = parse_args()
    seeds = np.random.SeedSequence(args.seed).spawn(args.repeats + 1)
    if args.matrix is None:
        matrix = np.random.default_rng(seeds[0]).dirichlet(np.full(args.dimension, 0.2), size=args.dimension)
    else:
        matrix = np.asarray(json.loads(args.matrix.read_text(encoding="utf-8")), dtype=float)
    validate_matrix(matrix)
    if args.capacity >= len(matrix):
        raise ValueError("Capacity must be strictly smaller than the state count.")
    trials = []
    for seed in seeds[1:]:
        sequence = generate_sequence(matrix, args.warmup + args.steps + 1, np.random.default_rng(seed))
        trials.append(
            compare_sequence(
                matrix,
                sequence,
                args.capacity,
                args.max_changes,
                args.alpha,
                args.warmup,
                args.delay,
                args.transition_model,
                args.beta,
            )
        )
    total_steps = args.steps * args.repeats
    summary = {
        "frozen_hit_rate": sum(trial["frozen_hits"] for trial in trials) / total_steps,
        "greedy_hit_rate": sum(trial["greedy_hits"] for trial in trials) / total_steps,
        "frozen_expected_hit_rate": float(np.mean([trial["frozen_expected_hit_rate"] for trial in trials])),
        "greedy_expected_hit_rate": float(np.mean([trial["greedy_expected_hit_rate"] for trial in trials])),
        "changes_per_step": sum(trial["changes"] for trial in trials) / total_steps,
        "paired_hit_rate_differences": [trial["greedy_hit_rate"] - trial["frozen_hit_rate"] for trial in trials],
    }
    result = {
        "configuration": {
            "dimension": len(matrix),
            "capacity": args.capacity,
            "max_changes": args.max_changes,
            "delay": args.delay,
            "transition_model": args.transition_model,
            "alpha": args.alpha,
            "beta": args.beta,
            "warmup": args.warmup,
            "steps": args.steps,
            "repeats": args.repeats,
            "seed": args.seed,
            "matrix_source": str(args.matrix) if args.matrix is not None else "Dirichlet(0.2 per state)",
        },
        "transition_matrix": matrix.tolist(),
        "summary": summary,
        "trials": trials,
    }
    print("Strategy         Observed coverage   Expected coverage   Applied replacements/step")
    print(f"Frozen EMA       {summary['frozen_hit_rate']:17.4%}   {summary['frozen_expected_hit_rate']:17.4%}   0")
    label = "Learned-T greedy" if args.transition_model == "estimated" else "Known-T greedy"
    print(
        f"{label:16s} {summary['greedy_hit_rate']:17.4%}   {summary['greedy_expected_hit_rate']:17.4%}"
        f"   {summary['changes_per_step']:.4f}"
    )
    print(f"Same paths; decisions applied after {args.delay} steps. No transfer costs are modeled.")
    print("The true T is used for generation and coverage evaluation; selection uses " + args.transition_model + " T.")
    if args.transition_model == "estimated":
        print(f"Transition rows use an EMA with beta={args.beta}, updated only on departures from that state.")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
