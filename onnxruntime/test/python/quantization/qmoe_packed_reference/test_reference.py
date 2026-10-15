"""CPU-only P0 oracles; tested draft semantic revision 0a15c0c.

No draft operator/opset dependency and no production schema or kernel claims.
All golden bytes and numerical results are manually specified literals.
"""

import json
import os
import runpy
import subprocess
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, checker, load_from_string
from onnx.reference import ReferenceEvaluator
from onnx_reference import build_int3_decode_model, build_unpack_model
from reference import (
    SIZE_LIMIT,
    checked_shape,
    decode_int3,
    pack,
    quantize_int3_rtn,
    row_bytes,
    unpack,
    validate_canonical_padding,
    validate_int3,
)

GOLDEN = json.loads(Path(__file__).with_name("golden_vectors.json").read_text())


def test_external_proposal_interoperability():
    checkout = os.environ.get("ONNX_PACK_PROPOSAL_CHECKOUT")
    if not checkout:
        pytest.skip("set ONNX_PACK_PROPOSAL_CHECKOUT to the pinned proposal checkout")
    revision = "0a15c0c82254ffabf2f5675005cc5cfc8086d9b8"
    actual = subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip()
    assert actual == revision, "external reference must use the recorded proposal revision"
    root = Path(checkout) / "onnx/reference/ops"
    proposal_pack = runpy.run_path(str(root / "op_pack.py"))["Pack"]
    proposal_unpack = runpy.run_path(str(root / "op_unpack.py"))["Unpack"]
    rng = np.random.default_rng(32657)
    for bits in range(1, 9):
        for shape in ((0,), (2, 0), (0, 3), (3,), (2, 3, 9), (2, 2, 65), (1, 1, 257)):
            codes = rng.integers(0, 1 << bits, shape, dtype=np.uint8)
            expected = proposal_pack._run(None, codes, bits=bits)[0]
            np.testing.assert_array_equal(pack(codes, bits), expected)
            decoded = proposal_unpack._run(None, expected, np.array(shape[-1], dtype=np.int64), bits=bits)[0]
            np.testing.assert_array_equal(unpack(expected, bits, shape[-1]), decoded)


@pytest.mark.parametrize("vector", GOLDEN["packing"], ids=lambda vector: vector["name"])
def test_golden_packing(vector):
    codes = np.array(vector["codes"], dtype=np.uint8)
    expected = np.frombuffer(bytes.fromhex(vector["hex"]), dtype=np.uint8)
    np.testing.assert_array_equal(pack(codes, vector["bits"]), expected)
    np.testing.assert_array_equal(unpack(expected, vector["bits"], codes.size), codes)
    validate_canonical_padding(expected, vector["bits"], codes.size)


def test_generic_unpack_ignores_padding():
    packed = np.array([0xE0, 0xFF], dtype=np.uint8)
    np.testing.assert_array_equal(unpack(packed, 3, 3), [0, 4, 7])
    with pytest.raises(ValueError, match="padding"):
        validate_canonical_padding(packed, 3, 3)


def scalar_pack(codes, bits):
    """Independent oracle: scatter individual bits, never assemble code words."""
    result = np.zeros((*codes.shape[:-1], (codes.shape[-1] * bits + 7) // 8), dtype=np.uint8)
    for row in np.ndindex(codes.shape[:-1]):
        for index, code in enumerate(codes[row]):
            for offset in range(bits):
                position = index * bits + offset
                if int(code) & (1 << offset):
                    result[(*row, position // 8)] |= np.uint8(1 << (position % 8))
    return result


def scalar_unpack(packed, bits, count):
    """Independent oracle: collect individual bits into each unsigned code."""
    result = np.zeros((*packed.shape[:-1], count), dtype=np.uint8)
    for row in np.ndindex(packed.shape[:-1]):
        for position in range(count * bits):
            bit = (int(packed[(*row, position // 8)]) // (2 ** (position % 8))) % 2
            result[(*row, position // bits)] += np.uint8(bit * (2 ** (position % bits)))
    return result


@pytest.mark.parametrize("bits", range(1, 9))
def test_randomized_independent_oracles(bits):
    rng = np.random.default_rng(1701 + bits)
    for shape in [
        (0,),
        (0, 3),
        (2, 0),
        (0, 2, 9),
        (2, 0, 3),
        (2, 3, 0),
        (1,),
        (3,),
        (9,),
        (2, 3, 7),
        (2, 3, 33),
        (2, 2, 129),
    ]:
        codes = rng.integers(0, 1 << bits, size=shape, dtype=np.uint8)
        expected = scalar_pack(codes, bits)
        np.testing.assert_array_equal(pack(codes, bits), expected)
        np.testing.assert_array_equal(unpack(expected, bits, shape[-1]), codes)
        np.testing.assert_array_equal(scalar_unpack(expected, bits, shape[-1]), codes)
        arbitrary = rng.integers(0, 256, size=expected.shape, dtype=np.uint8)
        np.testing.assert_array_equal(unpack(arbitrary, bits, shape[-1]), scalar_unpack(arbitrary, bits, shape[-1]))


@pytest.mark.parametrize("bits", [3, 5, 6])
def test_every_code_at_every_alignment(bits):
    for prefix in range(8):
        codes = np.concatenate([np.zeros(prefix, dtype=np.uint8), np.arange(1 << bits, dtype=np.uint8)])
        np.testing.assert_array_equal(pack(codes, bits), scalar_pack(codes, bits))
        np.testing.assert_array_equal(unpack(scalar_pack(codes, bits), bits, codes.size), codes)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_profile_literal_rows(dtype, block_size):
    fixture = GOLDEN["rows"]
    packed = np.array(fixture["packed"], dtype=np.uint8)
    scales = np.array(fixture["scales"], dtype=dtype)
    zeros = np.array(fixture["zeros"], dtype=np.uint8)
    np.testing.assert_array_equal(pack(np.array(fixture["codes"], dtype=np.uint8), 3), packed)
    for explicit in (None, zeros):
        result = decode_int3(packed, scales, 3, block_size, explicit)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, fixture["decoded"])


def test_bfloat16_profile():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    test_profile_literal_rows(ml_dtypes.bfloat16, 32)


@pytest.mark.parametrize("vector", GOLDEN["zero_point_tails"])
def test_explicit_zeros_count_is_blocks(vector):
    blocks = vector["blocks"]
    count = (blocks - 1) * 32 + 1
    codes = np.full((2, 2, count), 5, dtype=np.uint8)
    scales = np.broadcast_to(np.arange(1, blocks + 1, dtype=np.float32), (2, 2, blocks)).copy()
    zeros = np.broadcast_to(
        np.frombuffer(bytes.fromhex(vector["hex"]), dtype=np.uint8), (2, 2, len(bytes.fromhex(vector["hex"])))
    ).copy()
    np.testing.assert_array_equal(scalar_unpack(zeros, 3, blocks), np.full((2, 2, blocks), 4))
    expected = np.array([1 + index // 32 for index in range(count)], dtype=np.float32)
    for explicit in (None, zeros):
        np.testing.assert_array_equal(
            decode_int3(scalar_pack(codes, 3), scales, count, 32, explicit), np.broadcast_to(expected, codes.shape)
        )


def test_rtn_literal_nearest_even():
    fixture = GOLDEN["rtn"]
    result = quantize_int3_rtn(np.array(fixture["weights"], dtype=np.float32), 32)
    np.testing.assert_array_equal(
        result.packed, np.frombuffer(bytes.fromhex(fixture["hex"]), dtype=np.uint8)[None, None]
    )
    np.testing.assert_array_equal(unpack(result.packed, 3, 8), fixture["codes"])
    np.testing.assert_array_equal(result.scales, fixture["scales"])
    np.testing.assert_array_equal(result.max_abs_error, fixture["max_abs_error"])
    np.testing.assert_array_equal(decode_int3(result.packed, result.scales, 8, 32), fixture["decoded"])
    assert not np.any(result.underflow_blocks)


def test_rtn_rounded_scale_before_codes():
    weights = np.array([[[3.0009, 0.5001, -4.0012]]], dtype=np.float64)
    result = quantize_int3_rtn(weights, 32, np.float16)
    np.testing.assert_array_equal(result.scales, [[[1]]])
    np.testing.assert_array_equal(unpack(result.packed, 3, 3), [[[7, 5, 0]]])
    assert result.scales.dtype == np.float16


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_rtn_zeros_underflow_and_block_tail(dtype):
    weights = np.zeros((2, 1, 65), dtype=np.float64)
    weights[0, 0, 32] = 3
    weights[0, 0, 64] = 1e-50
    result = quantize_int3_rtn(weights, 32, dtype)
    np.testing.assert_array_equal(result.scales, [[[0, 1, 0]], [[0, 0, 0]]])
    np.testing.assert_array_equal(result.underflow_blocks, [[[False, False, True]], [[False, False, False]]])
    assert result.max_abs_error[0, 0, 2] == 1e-50
    codes = unpack(result.packed, 3, 65)
    assert codes[0, 0, 32] == 7
    codes[0, 0, 32] = 4
    assert np.all(codes == 4)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_rtn_float64_subnormal_underflow(dtype):
    smallest = np.nextafter(np.float64(0), np.float64(1))
    weights = np.array([[[smallest]], [[-smallest]], [[0]]], dtype=np.float64)
    result = quantize_int3_rtn(weights, 32, dtype)
    np.testing.assert_array_equal(result.underflow_blocks, [[[True]], [[True]], [[False]]])
    np.testing.assert_array_equal(result.scales, np.zeros((3, 1, 1)))
    np.testing.assert_array_equal(unpack(result.packed, 3, 1), np.full((3, 1, 1), 4))
    np.testing.assert_array_equal(result.max_abs_error, np.abs(weights))


@pytest.mark.parametrize("shape", [(0, 2, 3), (2, 0, 3), (2, 3, 0)])
def test_empty_profile_and_rtn(shape):
    result = quantize_int3_rtn(np.zeros(shape, dtype=np.float32), 32)
    decoded = decode_int3(result.packed, result.scales, shape[-1], 32)
    assert decoded.shape == shape
    zeros = scalar_pack(np.full(result.scales.shape, 4, dtype=np.uint8), 3)
    np.testing.assert_array_equal(decode_int3(result.packed, result.scales, shape[-1], 32, zeros), decoded)


@pytest.mark.parametrize("bad", [True, np.bool_(False), 3.0, "3", None, np.array(3)])
def test_strict_width_count_types(bad):
    for operation in (
        lambda: pack(np.array([0], np.uint8), bad),
        lambda: unpack(np.array([0], np.uint8), bad, 1),
        lambda: unpack(np.array([0], np.uint8), 3, bad),
        lambda: row_bytes(bad, 3),
        lambda: validate_canonical_padding(np.array([0], np.uint8), 3, bad),
    ):
        with pytest.raises(TypeError):
            operation()


@pytest.mark.parametrize("bits", [-1, 0, 9, 100])
def test_invalid_width(bits):
    with pytest.raises(ValueError):
        pack(np.array([0], np.uint8), bits)


def test_size_validation_without_allocating():
    assert row_bytes(SIZE_LIMIT, 8) == SIZE_LIMIT
    assert row_bytes(SIZE_LIMIT, 3) == (SIZE_LIMIT // 8) * 3 + ((SIZE_LIMIT % 8) * 3 + 7) // 8
    for shape in [(SIZE_LIMIT + 1,), (SIZE_LIMIT, 2), (0, SIZE_LIMIT, 2)]:
        with pytest.raises(OverflowError):
            checked_shape(shape)
    with pytest.raises(OverflowError):
        checked_shape((SIZE_LIMIT,), 4)
    with pytest.raises(OverflowError):
        row_bytes(SIZE_LIMIT + 1, 3)
    with pytest.raises(OverflowError):
        unpack(np.empty((2, 0), dtype=np.uint8), 3, SIZE_LIMIT)
    with pytest.raises(ValueError):
        row_bytes(-1, 3)
    with pytest.raises(TypeError):
        checked_shape((True,))
    with pytest.raises(ValueError):
        checked_shape((-1,))
    assert row_bytes(np.int64(3), np.int64(3)) == 2


@pytest.mark.parametrize("bad", [np.array(0, np.uint8), np.array([0], np.int8), np.array([0.0]), [0]])
def test_invalid_array(bad):
    for operation in (lambda: pack(bad, 3), lambda: unpack(bad, 3, 1), lambda: validate_canonical_padding(bad, 3, 1)):
        with pytest.raises((TypeError, ValueError)):
            operation()


def test_invalid_codes_and_exact_byte_counts():
    with pytest.raises(ValueError, match="outside"):
        pack(np.array([8], np.uint8), 3)
    for length in (0, 1, 3):
        for operation in (unpack, validate_canonical_padding):
            with pytest.raises(ValueError, match="exact byte count"):
                operation(np.zeros(length, np.uint8), 3, 3)


@pytest.mark.parametrize(
    "change",
    [
        "weight_pad",
        "weight_rank",
        "scale_rank",
        "scale_shape",
        "scale_dtype",
        "negative",
        "nan",
        "inf",
        "zero_pad",
        "zero_code",
        "zero_shape",
        "block",
    ],
)
def test_profile_rejections(change):
    packed = np.array([[[0xE0, 1]]], np.uint8)
    scales = np.array([[[1]]], np.float32)
    zeros = np.array([[[4]]], np.uint8)
    block_size = 32
    if change == "weight_pad":
        packed[..., -1] |= 128
    elif change == "weight_rank":
        packed = packed[0]
    elif change == "scale_rank":
        scales = scales[0]
    elif change == "scale_shape":
        scales = np.ones((1, 1, 2), np.float32)
    elif change == "scale_dtype":
        scales = scales.astype(np.float64)
    elif change in ("negative", "nan", "inf"):
        scales[...] = {"negative": -1, "nan": np.nan, "inf": np.inf}[change]
    elif change == "zero_pad":
        zeros[...] = 132
    elif change == "zero_code":
        zeros[...] = 3
    elif change == "zero_shape":
        zeros = np.zeros((1, 1, 2), np.uint8)
    elif change == "block":
        block_size = 16
    with pytest.raises((TypeError, ValueError)):
        validate_int3(packed, scales, 3, block_size, zeros)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rtn_nonfinite_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        quantize_int3_rtn(np.array([[[bad]]]), 32)


def test_rtn_invalid_inputs():
    for weights, block, dtype in [
        (np.ones((1, 1, 3), np.int32), 32, np.float32),
        (np.ones((1, 3), np.float32), 32, np.float32),
        (np.ones((1, 1, 3)), True, np.float32),
        (np.ones((1, 1, 3)), 32, np.float64),
        (np.full((1, 1, 3), 1e10), 32, np.float16),
    ]:
        with pytest.raises((TypeError, ValueError)):
            quantize_int3_rtn(weights, block, dtype)


def test_tiny_moe_gate_up_and_fc2():
    fixture = GOLDEN["tiny_moe"]
    decoded = []
    for name, count in (("fc1", 3), ("fc2", 2)):
        packed = np.array(fixture[f"{name}_packed"], np.uint8)
        np.testing.assert_array_equal(scalar_pack(np.array(fixture[f"{name}_codes"], np.uint8), 3), packed)
        weight = decode_int3(packed, np.array(fixture[f"{name}_scales"], np.float32), count, 32)
        np.testing.assert_array_equal(weight, fixture[f"{name}_dense"])
        decoded.append(weight.astype(np.float64))
    inputs = np.array(fixture["input"], np.float64)
    actual = np.zeros_like(inputs)
    independent = np.zeros_like(inputs)
    dense1 = fixture["fc1_dense"]
    dense2 = fixture["fc2_dense"]
    for token, activation in enumerate(inputs):
        for expert, probability in zip(fixture["experts"][token], fixture["route_weights"][token], strict=True):
            fc1 = decoded[0][expert] @ activation
            gate, up = fc1[0::2], fc1[1::2]
            hidden = (gate / (1 + np.exp(-gate))) * up
            actual[token] += probability * (decoded[1][expert] @ hidden)
            scalar_hidden = []
            for intermediate in range(2):
                gate_value = sum(dense1[expert][2 * intermediate][index] * activation[index] for index in range(3))
                up_value = sum(dense1[expert][2 * intermediate + 1][index] * activation[index] for index in range(3))
                scalar_hidden.append(gate_value / (1 + np.exp(-gate_value)) * up_value)
            for output in range(3):
                independent[token, output] += probability * sum(
                    dense2[expert][output][index] * scalar_hidden[index] for index in range(2)
                )
    np.testing.assert_allclose(actual, independent, rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual, fixture["output"], rtol=0, atol=1e-14)


def evaluate(model, **feeds):
    checker.check_model(model, full_check=True)
    assert [(entry.domain, entry.version) for entry in model.opset_import] == [("", 18)]
    assert all(node.domain == "" and node.op_type not in ("Pack", "Unpack") for node in model.graph.node)
    restored = load_from_string(model.SerializeToString())
    return ReferenceEvaluator(restored).run(None, feeds)[0]


@pytest.mark.parametrize("vector", GOLDEN["packing"], ids=lambda vector: vector["name"])
def test_onnx_literal_unpack(vector):
    packed = np.frombuffer(bytes.fromhex(vector["hex"]), np.uint8)
    count = len(vector["codes"])
    graph = build_unpack_model(packed.shape, vector["bits"], count)
    actual = evaluate(graph, packed=packed)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, vector["codes"])
    np.testing.assert_array_equal(actual, scalar_unpack(packed, vector["bits"], count))


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize(
    "shape", [(0,), (3,), (2, 3), (0, 3), (2, 0), (2, 3, 0), (0, 2, 9), (2, 0, 9), (2, 3, 9), (2, 1, 2, 17)]
)
def test_onnx_unpack_shapes_tails_and_padding(bits, shape):
    rng = np.random.default_rng(825 + bits)
    packed_shape = (*shape[:-1], (shape[-1] * bits + 7) // 8)
    packed = rng.integers(0, 256, size=packed_shape, dtype=np.uint8)
    actual = evaluate(build_unpack_model(packed_shape, bits, shape[-1]), packed=packed)
    assert actual.shape == shape
    np.testing.assert_array_equal(actual, scalar_unpack(packed, bits, shape[-1]))
    np.testing.assert_array_equal(actual, unpack(packed, bits, shape[-1]))


@pytest.mark.parametrize("dtype,enum", [(np.float32, TensorProto.FLOAT), (np.float16, TensorProto.FLOAT16)])
@pytest.mark.parametrize("explicit", [False, True])
def test_onnx_literal_profile(dtype, enum, explicit):
    fixture = GOLDEN["rows"]
    packed = np.array(fixture["packed"], np.uint8)
    scales = np.array(fixture["scales"], dtype)
    feeds = {"packed": packed, "scales": scales}
    if explicit:
        feeds["zeros"] = np.array(fixture["zeros"], np.uint8)
    model = build_int3_decode_model(packed.shape, 3, 32, enum, explicit)
    actual = evaluate(model, **feeds)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, fixture["decoded"])
    np.testing.assert_array_equal(actual, decode_int3(packed, scales, 3, 32, feeds.get("zeros")))


@pytest.mark.parametrize("shape", [(2, 3, 65), (2, 2, 257), (0, 2, 65), (2, 0, 65), (2, 3, 0)])
@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("explicit", [False, True])
def test_onnx_block_decode_tails_and_empty(shape, block_size, explicit):
    rng = np.random.default_rng(934)
    codes = rng.integers(0, 8, size=shape, dtype=np.uint8)
    packed = scalar_pack(codes, 3)
    blocks = (shape[-1] + block_size - 1) // block_size
    scales = rng.integers(0, 8, size=(*shape[:-1], blocks)).astype(np.float32) / 4
    zeros = scalar_pack(np.full(scales.shape, 4, dtype=np.uint8), 3)
    feeds = {"packed": packed, "scales": scales}
    if explicit:
        feeds["zeros"] = zeros
    actual = evaluate(build_int3_decode_model(packed.shape, shape[-1], block_size, explicit_zeros=explicit), **feeds)
    expected = np.empty(shape, dtype=np.float32)
    for row in np.ndindex(shape[:-1]):
        for index in range(shape[-1]):
            expected[(*row, index)] = (int(codes[(*row, index)]) - 4) * float(scales[(*row, index // block_size)])
    assert actual.shape == shape
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        actual, decode_int3(packed, scales, shape[-1], block_size, zeros if explicit else None)
    )


def test_onnx_literal_zero_point_block_tails():
    for fixture in GOLDEN["zero_point_tails"]:
        blocks = fixture["blocks"]
        count = (blocks - 1) * 32 + 1
        packed = scalar_pack(np.zeros((2, 2, count), np.uint8), 3)
        zeros = np.broadcast_to(
            np.frombuffer(bytes.fromhex(fixture["hex"]), np.uint8), (2, 2, len(bytes.fromhex(fixture["hex"])))
        ).copy()
        scales = np.ones((2, 2, blocks), np.float32)
        model = build_int3_decode_model(packed.shape, count, 32, explicit_zeros=True)
        np.testing.assert_array_equal(
            evaluate(model, packed=packed, scales=scales, zeros=zeros), np.full((2, 2, count), -4, np.float32)
        )


def test_onnx_invalid_metadata():
    for shape, bits, count in [
        ((), 3, 1),
        ((1,), 3, 3),
        ((1,), True, 1),
        ((1,), 3, 1.0),
        ((1,), 3, -1),
        ((SIZE_LIMIT,), 8, SIZE_LIMIT),
    ]:
        with pytest.raises((TypeError, ValueError, OverflowError)):
            build_unpack_model(shape, bits, count)
    for shape, count, block in [((2,), 3, 32), ((1, 1, 2), 3, 16)]:
        with pytest.raises(ValueError):
            build_int3_decode_model(shape, count, block)
    with pytest.raises(TypeError):
        build_int3_decode_model((1, 1, 2), 3, 32, TensorProto.DOUBLE)


@pytest.mark.parametrize("bits", range(1, 9))
def test_noncontiguous_rows(bits):
    codes = (np.arange(120, dtype=np.uint8) & np.uint8((1 << bits) - 1)).reshape(2, 3, 20)[..., ::-2]
    codes = codes.transpose(1, 0, 2)
    expected = scalar_pack(codes, bits)
    np.testing.assert_array_equal(pack(codes, bits), expected)
    packed_view = expected.transpose(1, 0, 2)
    np.testing.assert_array_equal(unpack(packed_view, bits, codes.shape[-1]), codes.transpose(1, 0, 2))


def test_strict_scalar_and_shape_limits():
    for count in (-1, SIZE_LIMIT + 1, np.uint64(2**64 - 1)):
        for operation in (unpack, validate_canonical_padding):
            with pytest.raises((ValueError, OverflowError)):
                operation(np.empty((0,), np.uint8), 3, count)
    for width in (SIZE_LIMIT + 1, np.uint64(2**64 - 1)):
        with pytest.raises(OverflowError):
            pack(np.empty((0,), np.uint8), width)
    for dim in (1.0, "2", np.bool_(False)):
        with pytest.raises(TypeError):
            checked_shape((dim,))
    for block in (True, np.bool_(False), 32.0):
        with pytest.raises(TypeError):
            decode_int3(np.array([[[4]]], np.uint8), np.ones((1, 1, 1), np.float32), 1, block)


@pytest.mark.parametrize("bits", range(1, 8))
def test_code_out_of_range_for_each_width(bits):
    with pytest.raises(ValueError, match="outside"):
        pack(np.array([1 << bits], np.uint8), bits)


@pytest.mark.parametrize("corruption", ["crossing_code", "tail_code", "pad", "truncated", "dtype", "rank"])
def test_cross_byte_zero_point_rejections(corruption):
    packed = scalar_pack(np.full((1, 1, 257), 4, np.uint8), 3)
    scales = np.ones((1, 1, 9), np.float32)
    zeros = np.array([[[0x24, 0x49, 0x92, 0x04]]], np.uint8)
    if corruption == "crossing_code":
        zeros[..., 0] ^= 0x40
    elif corruption == "tail_code":
        zeros[..., -1] ^= 1
    elif corruption == "pad":
        zeros[..., -1] |= 128
    elif corruption == "truncated":
        zeros = zeros[..., :-1]
    elif corruption == "dtype":
        zeros = zeros.astype(np.int8)
    elif corruption == "rank":
        zeros = zeros[0]
    with pytest.raises((ValueError, TypeError)):
        decode_int3(packed, scales, 257, 32, zeros)


def test_onnx_and_rtn_bfloat16():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    dtype = ml_dtypes.bfloat16
    weights = np.array([[[3.003, 0.5001, -4.004]]], np.float64)
    result = quantize_int3_rtn(weights, 32, dtype)
    assert result.scales.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result.scales.astype(np.float32), [[[1]]])
    np.testing.assert_array_equal(unpack(result.packed, 3, 3), [[[7, 5, 0]]])
    zeros = np.array([[[4]]], np.uint8)
    for explicit in (False, True):
        model = build_int3_decode_model(result.packed.shape, 3, 32, TensorProto.BFLOAT16, explicit)
        feeds = {"packed": result.packed, "scales": result.scales}
        if explicit:
            feeds["zeros"] = zeros
        actual = evaluate(model, **feeds)
        np.testing.assert_array_equal(actual, [[[3, 1, -4]]])
        np.testing.assert_array_equal(actual, decode_int3(result.packed, result.scales, 3, 32, zeros))


def test_zero_leading_axis_short_circuits_logical_walk():
    codes = np.empty((0, 10**9), dtype=np.uint8)
    packed = pack(codes, 3)
    assert packed.shape == (0, 375000000)
    assert unpack(packed, 3, 10**9).shape == codes.shape
    profile = np.empty((0, 1, 10**9), dtype=np.float32)
    result = quantize_int3_rtn(profile, 32)
    assert decode_int3(result.packed, result.scales, 10**9, 32).shape == profile.shape
