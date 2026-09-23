"""Replace recognized learned tensors in a reduced Gemma vision model with deterministic values."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def _parse_expected(entries: list[str]) -> dict[str, int]:
    expected: dict[str, int] = {}
    for entry in entries:
        suffix, separator, count_text = entry.partition("=")
        if not separator or not suffix or suffix in expected:
            raise ValueError(f"invalid or repeated --expected-count value {entry!r}")
        count = int(count_text)
        if count < 0:
            raise ValueError("expected counts must be non-negative")
        expected[suffix] = count
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Reduced source ONNX model.")
    parser.add_argument("--output", type=Path, required=True, help="New synthetic ONNX model path.")
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--weight-scale", type=float, default=0.02)
    parser.add_argument(
        "--expected-count",
        action="append",
        default=[],
        metavar="SUFFIX=COUNT",
        help="Require an exact replacement count for a tensor-name suffix (repeatable).",
    )
    parser.add_argument(
        "--allow-unclassified-external",
        action="store_true",
        help=(
            "Leave unrecognized external tensors unchanged. By default they are rejected to avoid leaking learned data."
        ),
    )
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    data_output = output.with_name(output.name + ".data")
    if not source.is_file() or output.exists() or data_output.exists() or source == output:
        raise ValueError("source must exist and synthetic output files must be new")
    if not np.isfinite(args.weight_scale) or args.weight_scale <= 0:
        raise ValueError("--weight-scale must be finite and positive")
    expected = _parse_expected(args.expected_count)

    model = onnx.load(source, load_external_data=True)
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    rng = np.random.default_rng(args.seed)
    replaced: list[str] = []
    for tensor in model.graph.initializer:
        name = tensor.name
        if name.endswith(".qweight"):
            data = numpy_helper.to_array(tensor)
            values = np.asarray(rng.integers(-3, 4, size=data.shape), dtype=data.dtype)
        elif name.endswith(".qzeros"):
            data = numpy_helper.to_array(tensor)
            values = np.zeros(data.shape, dtype=data.dtype)
        elif name.endswith(".scales"):
            data = numpy_helper.to_array(tensor)
            values = np.full(data.shape, args.weight_scale, dtype=np.float32)
        elif name.endswith("position_embedding_table_quantized"):
            data = numpy_helper.to_array(tensor)
            offsets = rng.integers(-1024, 1025, size=data.shape)
            values = np.asarray(32768 + offsets, dtype=np.uint16)
            base = name.removesuffix("_quantized")
            for suffix, replacement in (
                ("_scale", np.asarray(1 / 32768, dtype=np.float32)),
                ("_zero_point", np.asarray(32768, dtype=np.uint16)),
            ):
                companion = base + suffix
                if companion not in tensors:
                    raise ValueError(f"missing companion initializer {companion}")
                tensors[companion].CopyFrom(numpy_helper.from_array(replacement, companion))
        elif name.endswith(".weight_quantized"):
            data = numpy_helper.to_array(tensor)
            values = np.asarray(np.rint(rng.uniform(0.9, 1.1, size=data.shape) * 16384), dtype=np.uint16)
            base = name.removesuffix("_quantized")
            for suffix, replacement in (
                ("_scale", np.asarray(1 / 16384, dtype=np.float32)),
                ("_zero_point", np.asarray(0, dtype=np.uint16)),
            ):
                companion = base + suffix
                if companion not in tensors:
                    raise ValueError(f"missing companion initializer {companion}")
                tensors[companion].CopyFrom(numpy_helper.from_array(replacement, companion))
        elif tensor.data_location == onnx.TensorProto.EXTERNAL and not (
            name.endswith(".cos_cache_quantized") or name.endswith(".sin_cache_quantized")
        ):
            if args.allow_unclassified_external:
                continue
            raise ValueError(f"unclassified external tensor could contain learned weights: {name}")
        else:
            continue
        tensor.CopyFrom(numpy_helper.from_array(values, name))
        replaced.append(name)

    for suffix, count in expected.items():
        actual = sum(name.endswith(suffix) for name in replaced)
        if actual != count:
            raise ValueError(f"expected {count} synthetic {suffix} tensors, replaced {actual}")
    if not replaced:
        raise ValueError("no recognized learned tensors were replaced")
    model.metadata_props.add(key="synthetic_seed", value=str(args.seed))
    model.metadata_props.add(key="synthetic_learned_weights_replaced", value="true")
    model.metadata_props.add(key="original_activation_qdq_calibration_retained", value="true")

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output.name + ".data",
        size_threshold=1024,
    )
    print(f"Replaced {len(replaced)} learned tensors with seed {args.seed}; saved {output}")
    external_bytes = data_output.stat().st_size if data_output.exists() else 0
    print(f"Model bytes {output.stat().st_size}; external bytes {external_bytes}")


if __name__ == "__main__":
    main()
