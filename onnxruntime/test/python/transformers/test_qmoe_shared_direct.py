# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Direct CPU/CUDA execution of identical SM80 INT4/INT2 and row-local INT3 banks."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort

_ROW_PERMUTATION = np.array(
    [
        0,
        1,
        8,
        9,
        16,
        17,
        24,
        25,
        2,
        3,
        10,
        11,
        18,
        19,
        26,
        27,
        4,
        5,
        12,
        13,
        20,
        21,
        28,
        29,
        6,
        7,
        14,
        15,
        22,
        23,
        30,
        31,
        32,
        33,
        40,
        41,
        48,
        49,
        56,
        57,
        34,
        35,
        42,
        43,
        50,
        51,
        58,
        59,
        36,
        37,
        44,
        45,
        52,
        53,
        60,
        61,
        38,
        39,
        46,
        47,
        54,
        55,
        62,
        63,
    ]
)
_REGISTER_PERMUTATION = np.array([0, 2, 4, 6, 1, 3, 5, 7])


def _round(values, dtype):
    values = np.asarray(values, dtype=np.float32)
    if dtype == TensorProto.FLOAT16:
        return values.astype(np.float16).astype(np.float32)
    if dtype == TensorProto.BFLOAT16:
        bits = values.view(np.uint32)
        return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)
    return values


def _tensor(name, values, dtype):
    values = _round(values, dtype)
    if dtype == TensorProto.BFLOAT16:
        data = (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        return helper.make_tensor(name, dtype, values.shape, data, raw=True)
    return numpy_helper.from_array(values.astype(np.float16 if dtype == TensorProto.FLOAT16 else np.float32), name)


def _pack_sm80(codes):
    """Offline transpose/permutation/interleave, independent of runtime byte addressing."""
    experts, n, k = codes.shape
    permuted = codes.reshape(experts, n, k // 64, 64)[..., _ROW_PERMUTATION].reshape(experts, n, k)
    interleaved = (permuted.astype(np.int16) + 8).astype(np.uint8)
    interleaved = interleaved.reshape(experts, n, k // 8, 8)[..., _REGISTER_PERMUTATION]
    pairs = interleaved.reshape(experts, n, k // 2, 2)
    packed = pairs[..., 0] | (pairs[..., 1] << 4)
    return packed.reshape(experts, n // 4, 4, k // 64, 32).transpose(0, 1, 3, 2, 4).copy().reshape(experts, k, n // 2)


def _unpack_sm80(packed, n, k):
    experts = packed.shape[0]
    rows = packed.reshape(experts, n // 4, k // 64, 4, 32).transpose(0, 1, 3, 2, 4).reshape(experts, n, k // 2)
    pairs = np.stack((rows & 15, rows >> 4), axis=-1).reshape(experts, n, k // 8, 8)
    permuted = pairs[..., np.argsort(_REGISTER_PERMUTATION)].reshape(experts, n, k // 64, 64)
    return permuted[..., np.argsort(_ROW_PERMUTATION)].reshape(experts, n, k).astype(np.int16) - 8


def _pack_int2_sm80(codes):
    experts, n, k = codes.shape
    permutation = np.array([32 * (b & 1) + 8 * s + (b >> 1) for b in range(16) for s in range(4)])
    unsigned = (codes.astype(np.int16) + 2).astype(np.uint8)
    groups = unsigned.reshape(experts, n, k // 64, 64)[..., permutation].reshape(experts, n, k // 64, 16, 4)
    packed = np.bitwise_or.reduce(groups << np.arange(0, 8, 2, dtype=np.uint8), axis=-1)
    return packed.reshape(experts, n // 8, 8, k // 64, 16).transpose(0, 1, 3, 2, 4).copy().reshape(experts, k, n // 4)


def _unpack_int2_sm80(packed, n, k):
    experts = packed.shape[0]
    rows = packed.reshape(experts, n // 8, k // 64, 8, 16).transpose(0, 1, 3, 2, 4)
    groups = (rows[..., None] >> np.arange(0, 8, 2, dtype=np.uint8)) & 3
    permutation = np.array([32 * (b & 1) + 8 * s + (b >> 1) for b in range(16) for s in range(4)])
    return (
        groups.reshape(experts, n, k // 64, 64)[..., np.argsort(permutation)].reshape(experts, n, k).astype(np.int16)
        - 2
    )


def _pack_int3(codes):
    unsigned = (codes.astype(np.int16) + 4).astype(np.uint64).reshape(*codes.shape[:-1], -1, 8)
    words = np.bitwise_or.reduce(unsigned << np.arange(0, 24, 3, dtype=np.uint64), axis=-1)
    return (
        np.stack((words & 255, (words >> 8) & 255, words >> 16), axis=-1)
        .astype(np.uint8)
        .reshape(*codes.shape[:-1], codes.shape[-1] * 3 // 8)
    )


def _unpack_int3(packed, k):
    triples = packed.reshape(*packed.shape[:-1], -1, 3).astype(np.uint32)
    words = triples[..., 0] | (triples[..., 1] << 8) | (triples[..., 2] << 16)
    return ((words[..., None] >> np.arange(0, 24, 3, dtype=np.uint32)) & 7).reshape(*packed.shape[:-1], k).astype(
        np.int16
    ) - 4


def _fixture(bits, dtype, hidden=128, inter=64, experts=4, block=None, with_bias=False):
    rng = np.random.default_rng(452 + bits)
    block = block if block is not None else (32 if bits == 4 else 64)
    codes = [
        rng.integers(-(1 << (bits - 1)), 1 << (bits - 1), size=(experts, n, k), dtype=np.int16)
        for n, k in [(2 * inter, hidden), (hidden, inter)]
    ]
    scales = [
        _round(rng.uniform(0.01, 0.04, size=(experts, n, k // block)), dtype)
        for n, k in [(2 * inter, hidden), (hidden, inter)]
    ]
    pack = _pack_sm80 if bits == 4 else (_pack_int2_sm80 if bits == 2 else _pack_int3)
    packed = [pack(c) for c in codes]
    biases = [_round(rng.uniform(-0.05, 0.05, (experts, n)), dtype) if with_bias else None for n in (2 * inter, hidden)]
    return {
        "bits": bits,
        "block": block,
        "codes": codes,
        "scales": scales,
        "packed": packed,
        "dtype": dtype,
        "biases": biases,
    }


def _reference(fixture, x, router, top_k=2, normalize=True):
    dtype = fixture["dtype"]
    x, router = _round(x, dtype), _round(router, dtype)
    experts, hidden, inter = router.shape[1], x.shape[1], fixture["codes"][1].shape[-1]
    projections = [
        _round(codes * np.repeat(scales, fixture["block"], axis=-1), dtype).astype(np.float64)
        for codes, scales in zip(fixture["codes"], fixture["scales"], strict=True)
    ]
    result = np.zeros_like(x, dtype=np.float64)
    for row in range(x.shape[0]):
        selected = np.argsort(-router[row], kind="stable")[:top_k]
        probabilities = np.exp(router[row].astype(np.float64) - float(router[row].max()))
        denominator = probabilities[selected].sum() if normalize else probabilities.sum()
        for expert in selected:
            assert expert < experts
            fc1 = x[row].astype(np.float64) @ projections[0][expert].T
            if fixture["biases"][0] is not None:
                fc1 += fixture["biases"][0][expert]
            gate, up = fc1[0::2], fc1[1::2]
            intermediate = _round(gate / (1 + np.exp(-gate)) * up, dtype)
            assert intermediate.size == inter
            fc2 = intermediate.astype(np.float64) @ projections[1][expert].T
            if fixture["biases"][1] is not None:
                fc2 += fixture["biases"][1][expert]
            fc2 = _round(fc2, dtype)
            result[row] += fc2 * (probabilities[expert] / denominator)
    assert result.shape[1] == hidden
    return _round(result, dtype)


def _model(fixture, dynamic=False, canonical_shape=False, normalize=True, mixed_next=None, top_k=2):
    dtype = fixture["dtype"]
    experts, _, hidden = fixture["codes"][0].shape
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, hidden]),
        helper.make_tensor_value_info("router", TensorProto.FLOAT, [None, experts]),
    ]
    nodes = [
        helper.make_node("Cast", ["x"], ["typed_x"], to=dtype),
        helper.make_node("Cast", ["router"], ["typed_router"], to=dtype),
    ]
    initializers, feeds = [], {}

    def add_qmoe(current, name, source):
        bank_names = []
        bias_names = []
        for index, (source_weights, scales) in enumerate(zip(current["packed"], current["scales"], strict=True)):
            w_name, s_name = f"{name}_w{index}", f"{name}_s{index}"
            packed = source_weights
            if current["bits"] in (2, 4) and canonical_shape:
                n, k = current["codes"][index].shape[-2:]
                packed = packed.reshape(experts, n, k * current["bits"] // 8)
            if dynamic:
                inputs.extend(
                    [
                        helper.make_tensor_value_info(w_name, TensorProto.UINT8, packed.shape),
                        helper.make_tensor_value_info(s_name + "_float", TensorProto.FLOAT, scales.shape),
                    ]
                )
                feeds[w_name], feeds[s_name + "_float"] = packed, scales
                nodes.append(helper.make_node("Cast", [s_name + "_float"], [s_name], to=dtype))
            else:
                initializers.extend([numpy_helper.from_array(packed, w_name), _tensor(s_name, scales, dtype)])
            bank_names.extend([w_name, s_name])
            bias = current["biases"][index]
            bias_name = f"{name}_b{index}" if bias is not None else ""
            if bias is not None:
                initializers.append(_tensor(bias_name, bias, dtype))
            bias_names.append(bias_name)
        nodes.append(
            helper.make_node(
                "QMoE",
                [
                    source,
                    "typed_router",
                    bank_names[0],
                    bank_names[1],
                    bias_names[0],
                    bank_names[2],
                    bank_names[3],
                    bias_names[1],
                ],
                [name],
                domain="com.microsoft",
                name=name,
                k=top_k,
                activation_type="swiglu",
                swiglu_fusion=1,
                normalize_routing_weights=int(normalize),
                expert_weight_bits=current["bits"],
                weights_prepacked=int(current["bits"] in (2, 4)),
                block_size=current["block"],
            )
        )

    add_qmoe(fixture, "direct_qmoe", "typed_x")
    last = "direct_qmoe"
    if mixed_next is not None:
        add_qmoe(mixed_next, "direct_qmoe_next", last)
        last = "direct_qmoe_next"
    nodes.append(helper.make_node("Cast", [last], ["output"], to=TensorProto.FLOAT))
    graph = helper.make_graph(
        nodes,
        "shared_direct",
        inputs,
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, hidden])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    return model, feeds


class TestQMoESharedDirect(unittest.TestCase):
    def _providers(self):
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        if os.getenv("ORT_QMOE_REQUIRE_CUDA") == "1":
            self.assertIn("CUDAExecutionProvider", providers)
        return providers

    def _run(
        self,
        fixture,
        rows,
        dynamic=False,
        canonical_shape=False,
        normalize=True,
        mixed_next=None,
        top_k=2,
        providers=None,
    ):
        model, feeds = _model(fixture, dynamic, canonical_shape, normalize, mixed_next, top_k)
        hidden, experts = fixture["codes"][0].shape[-1], fixture["codes"][0].shape[0]
        rng = np.random.default_rng(938 + rows)
        feeds.update(
            x=rng.uniform(-0.5, 0.5, (rows, hidden)).astype(np.float32),
            router=rng.normal(size=(rows, experts)).astype(np.float32),
        )
        expected = _reference(fixture, feeds["x"], feeds["router"], top_k=top_k, normalize=normalize)
        if mixed_next is not None:
            expected = _reference(mixed_next, expected, feeds["router"], top_k=top_k, normalize=normalize)
        before = [hashlib.sha256(value.tobytes()).digest() for value in feeds.values()]
        for provider in self._providers() if providers is None else providers:
            if provider == "CUDAExecutionProvider" and fixture["dtype"] == TensorProto.FLOAT:
                continue
            for disable_prepacking in [False, True]:
                with self.subTest(
                    bits=fixture["bits"],
                    dtype=fixture["dtype"],
                    rows=rows,
                    dynamic=dynamic,
                    provider=provider,
                    disable_prepacking=disable_prepacking,
                ):
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 4
                    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    if disable_prepacking:
                        options.add_session_config_entry("session.disable_prepacking", "1")
                    if provider == "CUDAExecutionProvider":
                        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                    try:
                        session = ort.InferenceSession(model.SerializeToString(), options, providers=[provider])
                    except Exception as exc:
                        if (
                            provider == "CUDAExecutionProvider"
                            and os.getenv("ORT_QMOE_REQUIRE_CUDA") != "1"
                            and "Build configuration may be missing the corresponding kernel" in str(exc)
                        ):
                            continue
                        raise
                    output = session.run(None, feeds)[0]
                    np.testing.assert_allclose(output, expected, rtol=0.025, atol=0.0016)
        self.assertEqual(before, [hashlib.sha256(value.tobytes()).digest() for value in feeds.values()])

    def test_offline_layout_fixtures_roundtrip(self):
        for bits in [2, 3, 4]:
            fixture = _fixture(bits, TensorProto.FLOAT16)
            for codes, packed in zip(fixture["codes"], fixture["packed"], strict=True):
                decoded = (
                    _unpack_sm80(packed, *codes.shape[-2:])
                    if bits == 4
                    else _unpack_int2_sm80(packed, *codes.shape[-2:])
                    if bits == 2
                    else _unpack_int3(packed, codes.shape[-1])
                )
                np.testing.assert_array_equal(codes, decoded)

    def test_decode_and_prefill_direct(self):
        for bits in [3, 4]:
            for dtype in [TensorProto.FLOAT16, TensorProto.BFLOAT16]:
                fixture = _fixture(bits, dtype)
                for rows in [1, 4, 8, 33, 65]:
                    self._run(fixture, rows)

    def test_dynamic_banks_and_non_square_shapes(self):
        for hidden, inter in [(256, 64), (64, 128)]:
            for bits in [3, 4]:
                fixture = _fixture(bits, TensorProto.BFLOAT16, hidden, inter)
                self._run(fixture, 9, dynamic=True)
                self._run(fixture, 4, canonical_shape=True, normalize=False)

    def test_float32_cpu_prepacked_int4(self):
        self._run(_fixture(4, TensorProto.FLOAT), 9, dynamic=True)

    def test_cpu_prepacked_int2_shapes_blocks_dtypes_and_bias(self):
        for block in (32, 64, 128):
            for dtype in (TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.FLOAT):
                for canonical_shape in (False, True):
                    for with_bias in (False, True):
                        with self.subTest(
                            block=block, dtype=dtype, canonical_shape=canonical_shape, with_bias=with_bias
                        ):
                            fixture = _fixture(2, dtype, hidden=256, inter=128, block=block, with_bias=with_bias)
                            for rows in (1, 9):
                                self._run(
                                    fixture, rows, canonical_shape=canonical_shape, providers=["CPUExecutionProvider"]
                                )

    def test_mixed_int4_int3_graph(self):
        first, second = _fixture(4, TensorProto.BFLOAT16), _fixture(3, TensorProto.BFLOAT16)
        self._run(first, 9, mixed_next=second)

    def test_candidate_routing_envelope(self):
        for bits in [3, 4]:
            self._run(_fixture(bits, TensorProto.BFLOAT16, experts=512), 4, top_k=10)

    def test_cuda_profile_has_no_bank_conversion(self):
        if os.getenv("ORT_QMOE_REQUIRE_CUDA_PROFILE") != "1":
            self.skipTest("Set ORT_QMOE_REQUIRE_CUDA_PROFILE=1 with a CUPTI-enabled CUDA build")
        self.assertIn("CUDAExecutionProvider", ort.get_available_providers())
        for bits in [3, 4]:
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as directory:
                fixture = _fixture(bits, TensorProto.BFLOAT16)
                model, feeds = _model(fixture)
                feeds.update(
                    x=np.full((65, 128), 0.125, dtype=np.float32),
                    router=np.tile(np.arange(4, dtype=np.float32), (65, 1)),
                )
                options = ort.SessionOptions()
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(directory) / "direct")
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                session = ort.InferenceSession(model.SerializeToString(), options, providers=["CUDAExecutionProvider"])
                session.run(None, feeds)
                events = json.loads(Path(session.end_profiling()).read_text())
                kernels = [event for event in events if "block_x" in event.get("args", {})]
                names = [event["name"] for event in kernels]
                self.assertTrue(
                    any("FusedFc1" in name for name in names), f"No direct device kernels were observed: {names}"
                )
                for name in names:
                    self.assertFalse(
                        any(forbidden in name for forbidden in ["Repack", "Transpose", "Dequantize", "Unpack"]), name
                    )

    def test_candidate_expert_slices(self):
        model_path = os.getenv("ORT_QMOE_CANDIDATE_MODEL")
        if not model_path:
            self.skipTest("Set ORT_QMOE_CANDIDATE_MODEL to validate exact candidate bank slices")
        path = Path(model_path)
        graph = onnx.load(path, load_external_data=False).graph
        initializers = {value.name: value for value in graph.initializer}
        found = set()
        for node in graph.node:
            if node.op_type != "QMoE":
                continue
            attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            bits = attrs["expert_weight_bits"]
            if bits in found:
                continue
            found.add(bits)
            banks = []
            for slot in [2, 3, 5, 6]:
                value = initializers[node.input[slot]]
                external = {entry.key: entry.value for entry in value.external_data}
                stride = int(external["length"]) // value.dims[0]
                slices = []
                with (path.parent / external["location"]).open("rb") as stream:
                    for expert in [0, 31, 511]:
                        stream.seek(int(external.get("offset", 0)) + expert * stride)
                        payload = stream.read(stride)
                        self.assertEqual(len(payload), stride)
                        slices.append(payload)
                dtype = np.uint8 if slot in [2, 5] else np.uint16
                data = np.frombuffer(b"".join(slices), dtype=dtype).reshape(3, *value.dims[1:]).copy()
                if slot in [3, 6]:
                    self.assertEqual(value.data_type, TensorProto.BFLOAT16)
                    data = (data.astype(np.uint32) << 16).view(np.float32)
                banks.append(data)
            hidden, inter = banks[3].shape[1], banks[3].shape[2] * attrs["block_size"]
            decode = (lambda p, n, k: _unpack_sm80(p, n, k)) if bits == 4 else (lambda p, n, k: _unpack_int3(p, k))
            fixture = {
                "bits": bits,
                "dtype": TensorProto.BFLOAT16,
                "block": attrs["block_size"],
                "packed": [banks[0], banks[2]],
                "scales": [banks[1], banks[3]],
                "codes": [decode(banks[0], 2 * inter, hidden), decode(banks[2], hidden, inter)],
                "biases": [None, None],
            }
            self._run(fixture, 4)
        self.assertEqual(found, {3, 4})


if __name__ == "__main__":
    unittest.main()
