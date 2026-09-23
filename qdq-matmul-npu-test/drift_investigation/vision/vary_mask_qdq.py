"""Change mask QDQ parameters while optionally proving CPU outputs remain bitwise identical."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


DEFAULT_SCALE_NAME = "v_vision_encoder.encoder.Where_31_scale"
DEFAULT_ZERO_POINT_NAME = "v_vision_encoder.encoder.Where_31_zero_point"


def _load_session_inputs(session, path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [value.name for value in session.get_inputs() if value.name not in archive]
        if missing:
            raise ValueError(f"{path}: missing model inputs {missing}")
        return {value.name: archive[value.name] for value in session.get_inputs()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source mask-QDQ/Add/Softmax ONNX model.")
    parser.add_argument("--output", type=Path, required=True, help="New ONNX model path.")
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--zero-point", type=int, required=True)
    parser.add_argument("--scale-initializer", default=DEFAULT_SCALE_NAME)
    parser.add_argument("--zero-point-initializer", default=DEFAULT_ZERO_POINT_NAME)
    parser.add_argument(
        "--preserve-value",
        type=float,
        action="append",
        default=None,
        help="Require exact float32 QDQ representation (repeatable; default: -100 and 0).",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        action="append",
        default=[],
        help="NPZ used for a CPU bitwise-equivalence check (repeatable).",
    )
    parser.add_argument(
        "--skip-cpu-check",
        action="store_true",
        help="Write the variant without running ONNX Runtime CPU equivalence checks.",
    )
    args = parser.parse_args()

    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_file() or output.exists() or source == output:
        raise ValueError("source must exist and output must be a new model")
    if not 0 <= args.zero_point <= 65535:
        raise ValueError("zero point must be representable as uint16")
    scale = np.float32(args.scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    preserve_values = args.preserve_value if args.preserve_value is not None else [-100.0, 0.0]
    for value in preserve_values:
        quantized = np.clip(np.rint(value / float(scale) + args.zero_point), 0, 65535)
        reconstructed = np.float32((quantized - args.zero_point) * scale)
        if float(reconstructed) != value:
            raise ValueError(f"{value} is not exactly representable by this mask QDQ")
    if not args.skip_cpu_check and not args.inputs:
        raise ValueError("provide at least one --inputs archive or explicitly use --skip-cpu-check")

    model = onnx.load(source, load_external_data=False)
    replacements = {
        args.scale_initializer: np.asarray(scale, dtype=np.float32),
        args.zero_point_initializer: np.asarray(args.zero_point, dtype=np.uint16),
    }
    for initializer in model.graph.initializer:
        if initializer.name in replacements:
            initializer.CopyFrom(
                numpy_helper.from_array(replacements.pop(initializer.name), initializer.name)
            )
    if replacements:
        raise ValueError(f"missing mask QDQ initializers: {list(replacements)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)

    if not args.skip_cpu_check:
        import onnxruntime as ort

        cpu_original = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
        cpu_changed = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        if [value.name for value in cpu_original.get_inputs()] != [
            value.name for value in cpu_changed.get_inputs()
        ]:
            raise RuntimeError("modified model input names differ from the source")
        for path in args.inputs:
            inputs = _load_session_inputs(cpu_original, path.resolve())
            before = cpu_original.run(None, inputs)
            after = cpu_changed.run(None, inputs)
            if len(before) != len(after) or any(
                not np.array_equal(before_value, after_value)
                for before_value, after_value in zip(before, after, strict=True)
            ):
                raise RuntimeError(f"mask QDQ change altered CPU output on {path}; variant retained at {output}")
            print(f"CPU outputs are bitwise identical for {path}")
    print(f"Saved {output} with mask QDQ scale={scale}, zero_point={args.zero_point}")


if __name__ == "__main__":
    main()
