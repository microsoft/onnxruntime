# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate synthetic MatMulNBits models and inputs for estimation experiments."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


@dataclass(frozen=True)
class Case:
    a_shape: tuple[int, ...]
    n: int
    bits: int = 4
    block_size: int = 32
    dtype: str = "float32"
    bias: bool = False
    zero_points: bool = False
    accuracy_level: int = 0
    cohort: str = "calibration"

    def __post_init__(self):
        if not isinstance(self.a_shape, tuple) or not self.a_shape:
            raise ValueError("a_shape must be a nonempty list of positive integer dimensions")
        for name, values in (("a_shape", self.a_shape), ("n", (self.n,)), ("block_size", (self.block_size,))):
            if any(type(value) is not int or value <= 0 for value in values):
                raise ValueError(f"{name} must contain positive integers")
        if type(self.bits) is not int or self.bits not in (2, 4, 8):
            raise ValueError("bits must be 2, 4 or 8")
        if self.block_size < 16 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a power of two >= 16")
        if self.dtype not in ("float32", "float16"):
            raise ValueError("dtype must be float32 or float16")
        if type(self.bias) is not bool or type(self.zero_points) is not bool:
            raise ValueError("bias and zero_points must be booleans")
        if type(self.accuracy_level) is not int or self.accuracy_level not in range(5):
            raise ValueError("accuracy_level must be an integer from 0 to 4")
        if self.cohort not in ("calibration", "held_out"):
            raise ValueError("cohort must be calibration or held_out")

    @property
    def k(self) -> int:
        return self.a_shape[-1]

    @property
    def m(self) -> int:
        return math.prod(self.a_shape[:-1])

    def identity(self) -> str:
        settings = asdict(self)
        del settings["cohort"]
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("ascii")).hexdigest()


def default_cases() -> list[Case]:
    """Return a small INT4 pilot, not an exhaustive route-coverage sweep."""
    cases = [
        Case((m, 64), 32, bias=bias, zero_points=zero_points)
        for m in (1, 128)
        for bias in (False, True)
        for zero_points in (False, True)
    ]
    cases.extend(
        [
            Case((64,), 32),
            Case((2, 3, 64), 32, bias=True),
            Case((4, 63), 31, cohort="held_out"),
            Case((4, 65), 33, zero_points=True, cohort="held_out"),
            Case((8, 128), 64, dtype="float16"),
            Case((32, 128), 64, accuracy_level=4),
        ]
    )
    return cases


def load_cases(path: Path | None) -> list[Case]:
    if path is None:
        return default_cases()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"cases"}:
        raise ValueError('configuration must be an object with exactly one key: "cases"')
    entries = config["cases"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("cases must be a nonempty list")
    cases = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("a_shape"), list):
            raise ValueError(f"case {index}: expected an object with a_shape as a list")
        settings = {**entry, "a_shape": tuple(entry["a_shape"])}
        try:
            cases.append(Case(**settings))
        except (TypeError, ValueError) as error:
            raise ValueError(f"case {index}: {error}") from error
    return cases


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack the last axis low-order bits first, padding the final byte with zero."""
    if bits not in (2, 4, 8) or codes.dtype != np.uint8 or codes.ndim == 0:
        raise ValueError("packing requires uint8 codes with rank >= 1 and bits in {2, 4, 8}")
    if np.any(codes >= 1 << bits):
        raise ValueError("quantized code exceeds bit width")
    values_per_byte = 8 // bits
    padding = (-codes.shape[-1]) % values_per_byte
    padded = np.pad(codes, [(0, 0)] * (codes.ndim - 1) + [(0, padding)])
    groups = padded.reshape(*codes.shape[:-1], -1, values_per_byte)
    packed = np.zeros(groups.shape[:-1], dtype=np.uint8)
    for lane in range(values_per_byte):
        packed |= groups[..., lane] << (lane * bits)
    return packed


def make_case(case: Case, seed: int) -> tuple[onnx.ModelProto, np.ndarray]:
    """Sample quantized codes directly; no ORT quantizer or runtime is required."""
    rng = np.random.default_rng(seed)
    dtype = np.dtype(case.dtype)
    blocks = (case.k + case.block_size - 1) // case.block_size
    codes = rng.integers(0, 1 << case.bits, (case.n, blocks, case.block_size), dtype=np.uint8)
    zeros = (
        rng.integers(0, 1 << case.bits, (case.n, blocks), dtype=np.uint8)
        if case.zero_points
        else np.full((case.n, blocks), 1 << (case.bits - 1), dtype=np.uint8)
    )
    # Encode padded K positions as dequantized zero, including asymmetric blocks.
    remainder = case.k % case.block_size
    if remainder:
        codes[:, -1, remainder:] = zeros[:, -1, None]
    scales = np.exp2(rng.integers(-5, -1, (case.n, blocks))).astype(dtype)
    initializers = [
        numpy_helper.from_array(pack_codes(codes, case.bits), "B"),
        numpy_helper.from_array(scales, "scales"),
    ]
    node_inputs = ["A", "B", "scales"]
    if case.zero_points:
        node_inputs.append("zero_points")
        initializers.append(numpy_helper.from_array(pack_codes(zeros, case.bits), "zero_points"))
    if case.bias:
        node_inputs.extend([""] * (5 - len(node_inputs)))
        node_inputs.append("bias")
        bias = rng.uniform(-0.25, 0.25, case.n).astype(dtype)
        initializers.append(numpy_helper.from_array(bias, "bias"))
    float_type = TensorProto.FLOAT if case.dtype == "float32" else TensorProto.FLOAT16
    node = helper.make_node(
        "MatMulNBits",
        node_inputs,
        ["Y"],
        name="MatMulNBits_0",
        domain="com.microsoft",
        K=case.k,
        N=case.n,
        bits=case.bits,
        block_size=case.block_size,
        accuracy_level=case.accuracy_level,
    )
    graph = helper.make_graph(
        [node],
        "matmul_nbits_estimation_case",
        [helper.make_tensor_value_info("A", float_type, case.a_shape)],
        [helper.make_tensor_value_info("Y", float_type, (*case.a_shape[:-1], case.n))],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
        producer_name="generate_matmul_nbits_cases",
    )
    onnx.checker.check_model(model)
    return model, rng.uniform(-0.5, 0.5, case.a_shape).astype(dtype)


def generate_cases(cases: list[Case], output: Path, seed: int) -> dict:
    """Write a new bundle; publish its manifest only after every case succeeds."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not cases:
        raise ValueError("at least one case is required")
    identities = [case.identity() for case in cases]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate case settings, including cases assigned to different cohorts")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for case, identity in zip(cases, identities, strict=True):
        case_id = f"matmul_nbits_{identity}"
        case_seed = int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode("ascii")).digest()[:8], "little")
        directory = output / case_id
        directory.mkdir()
        model, activation = make_case(case, case_seed)
        onnx.save_model(model, directory / "model.onnx")
        np.savez(directory / "inputs.npz", A=activation)
        records.append(
            {
                "case_id": case_id,
                "settings": asdict(case),
                "seed": str(case_seed),
                "model": f"{case_id}/model.onnx",
                "inputs": f"{case_id}/inputs.npz",
                "input_names": ["A"],
                "output_names": ["Y"],
                "output_shape": [*case.a_shape[:-1], case.n],
                "expected_invocations_per_run": 1,
                "logical_reference": {
                    "convention": "matmul_nbits_logical_v1",
                    "m": str(case.m),
                    "n": str(case.n),
                    "k": str(case.k),
                    "core_dense_equivalent_ops": str(2 * case.m * case.n * case.k),
                    "bias_float_ops": str(case.m * case.n if case.bias else 0),
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "generator": "generate_matmul_nbits_cases",
        "seed": str(seed),
        "numpy_version": np.__version__,
        "onnx_version": onnx.__version__,
        "weight_generation": "synthetic_unsigned_codes_with_positive_power_of_two_scales",
        "packing": "standard_columnwise_low_bits_first; weight_prepacked=0",
        "route_policy": "actual EP, route and effective mode must be verified by the runner",
        "reference_scope": "logical_core_and_bias_only; not executed or whole_operator_fp_counts",
        "cases": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON object containing an explicit cases list")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are rejected")
    parser.add_argument("--seed", type=int, default=123, help="Nonnegative base seed (default: 123)")
    args = parser.parse_args()
    try:
        manifest = generate_cases(load_cases(args.config), args.output, args.seed)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Case generation failed: {error}\n")
    print(f"Generated {len(manifest['cases'])} cases: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
