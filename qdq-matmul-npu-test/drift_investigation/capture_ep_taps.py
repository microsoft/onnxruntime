#!/usr/bin/env python3
"""Capture full-model NPU intermediates only if tapping preserves the original output."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_acc import context_node_count, load_inputs
from run_winml_ep import NPU_PROVIDERS, PROVIDER_NAMES, find_npu_device, parse_provider_options, register_provider


def select_taps(
    outputs: dict[str, np.ndarray],
    baseline: np.ndarray,
    full_output: str,
    input_values: dict[str, str],
    context_value: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    missing = set(input_values.values()) | {context_value, full_output}
    missing -= outputs.keys()
    if missing:
        raise ValueError(f"compiled model does not expose the requested outputs: {sorted(missing)}")
    if not np.array_equal(outputs[full_output], baseline):
        raise RuntimeError(f"tapping changed EP output {full_output!r}; intermediate comparisons are not valid")
    return {name: outputs[value] for name, value in input_values.items()}, outputs[context_value]


def parse_input_values(raw_values: list[str]) -> dict[str, str]:
    values = {}
    for raw in raw_values:
        name, sep, value = raw.partition("=")
        if not sep or not name or not value or name in values:
            raise ValueError(f"invalid or repeated --input-value {raw!r}; expected NAME=GRAPH_VALUE")
        values[name] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiled_model", type=Path, help="Compiled, probed EPContext ONNX model.")
    parser.add_argument("--provider", choices=NPU_PROVIDERS, required=True)
    parser.add_argument("--provider-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--inputs", type=Path, required=True, help="Full-model input .npz.")
    parser.add_argument("--valid-mask", help="Extra boolean array in --inputs, not a model input.")
    parser.add_argument("--baseline-arrays", type=Path, required=True, help="Unprobed run_acc output arrays.")
    parser.add_argument("--baseline-index", type=int, default=0, help="Original graph output index; default: 0.")
    parser.add_argument("--full-output", required=True, help="Unprobed graph output name to compare bitwise.")
    parser.add_argument(
        "--input-value", action="append", required=True, metavar="NAME=GRAPH_VALUE",
        help="Map an island input to a tapped full-model output (repeatable).",
    )
    parser.add_argument("--context-value", required=True, help="Tapped attention output to save for comparison.")
    parser.add_argument("--island-inputs", type=Path, required=True, help="New island input .npz.")
    parser.add_argument("--context-output", type=Path, required=True, help="New .npz with a context array.")
    args = parser.parse_args()
    input_values = parse_input_values(args.input_value)
    compiled = args.compiled_model.resolve()
    island_inputs = args.island_inputs.resolve()
    context_output = args.context_output.resolve()
    if not compiled.is_file() or not args.inputs.is_file() or not args.baseline_arrays.is_file():
        raise ValueError("compiled model, original inputs, and baseline output archive must exist")
    if len({compiled, island_inputs, context_output}) != 3 or island_inputs.exists() or context_output.exists():
        raise ValueError("output archives must be new files and distinct from the compiled model")
    if args.baseline_index < 0:
        raise ValueError("--baseline-index must be non-negative")
    context_node_count(compiled)

    provider_name = PROVIDER_NAMES[args.provider]
    ort = register_provider(provider_name)
    options = ort.SessionOptions()
    options.add_provider_for_devices(
        [find_npu_device(ort, provider_name)], parse_provider_options(args.provider_option)
    )
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(str(compiled), sess_options=options)
    inputs, _ = load_inputs(session, args.inputs, args.valid_mask)
    outputs = dict(zip([value.name for value in session.get_outputs()], session.run(None, inputs)))
    with np.load(args.baseline_arrays, allow_pickle=False) as archive:
        baseline = archive[f"ep_{args.baseline_index}"]
    tapped_inputs, context = select_taps(outputs, baseline, args.full_output, input_values, args.context_value)
    np.savez_compressed(island_inputs, **tapped_inputs)
    np.savez_compressed(context_output, attention_context=context)
    print(f"EP output is unchanged; saved {island_inputs} and {context_output}")


if __name__ == "__main__":
    main()
