import os
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto, helper
from onnxruntime.capi import _pybind_state

import onnxruntime as ort


def int4_kernel_available():
    if os.getenv("ORT_PAGED_ATTENTION_TEST_RUNNER"):
        return True
    try:
        return any(
            kernel.op_name == "PagedAttention"
            and kernel.provider == "CUDAExecutionProvider"
            and "tensor(uint8)" in kernel.type_constraints.get("T_CACHE", [])
            for kernel in _pybind_state.get_all_opkernel_def()
        )
    except (ImportError, AttributeError):
        return False


def set_attribute(model, name, value):
    node = model.graph.node[0]
    retained = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend([*retained, helper.make_attribute(name, value)])


def replace_input(model, feeds, name, values):
    feeds[name] = values
    for value_info in model.graph.input:
        if value_info.name == name:
            value_info.CopyFrom(
                helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(values.dtype), values.shape)
            )
            return
    model.graph.input.append(
        helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(values.dtype), values.shape)
    )


def remove_input(model, feeds, name):
    feeds.pop(name)
    remaining = [value_info for value_info in model.graph.input if value_info.name != name]
    del model.graph.input[:]
    model.graph.input.extend(remaining)
    node = model.graph.node[0]
    for index, input_name in enumerate(node.input):
        if input_name == name:
            node.input[index] = ""


def hadamard_matrix(width):
    matrix = np.ones((1, 1), dtype=np.float32)
    while matrix.shape[0] < width:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / np.sqrt(np.float32(width))


def rotate(values):
    return values.astype(np.float32) @ hadamard_matrix(values.shape[-1])


def quantize(values, scale_dtype):
    scales = np.max(np.abs(values), axis=-1) / 7.0
    if scale_dtype == np.float16:
        scales = np.where(scales == 0, 0, np.clip(scales, 2.0**-24, 65504))
    scales = scales.astype(scale_dtype)
    scaled = np.divide(values, scales[..., None], out=np.zeros_like(values), where=scales[..., None] != 0)
    signed = np.clip(np.rint(scaled), -8, 7).astype(np.int8)
    biased = (signed + 8).astype(np.uint8)
    packed = biased[..., ::2] | (biased[..., 1::2] << 4)
    return packed, scales


def unpack(packed, scales):
    values = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.float32)
    values[..., ::2] = (packed & 15).astype(np.float32) - 8
    values[..., 1::2] = (packed >> 4).astype(np.float32) - 8
    return values * scales[..., None]


def make_case(
    width=64,
    lengths=(1, 1),
    past=(19, 7),
    int4=True,
    qk_rotation=True,
    v_rotation=True,
    scale_dtype=np.float16,
    packed=False,
    skip=False,
    sink=False,
    softcap=0.0,
    window=-1,
    activation_dtype=np.float16,
    heads=4,
    kv_heads=2,
    block_size=16,
):
    rng = np.random.default_rng(1234)
    batch = len(lengths)
    tokens = sum(lengths)
    max_blocks = max((old + new + block_size - 1) // block_size for old, new in zip(past, lengths, strict=True))
    num_blocks = batch * max_blocks + 1
    block_table = rng.permutation(num_blocks - 1).astype(np.int32).reshape(batch, max_blocks)
    cumulative = np.array([0, *np.cumsum(lengths)], dtype=np.int32)
    query = rng.normal(0, 0.2, (tokens, heads, width)).astype(activation_dtype)
    key = rng.normal(0, 0.4, (tokens, kv_heads, width)).astype(activation_dtype)
    value = rng.normal(0, 0.6, (tokens, kv_heads, width)).astype(activation_dtype)
    if tokens:
        key[0, 0] = 0
        value[0, 0] = 0
    cache_inputs = {}
    expected_cache = {}
    logical_cache = {}
    slots = []
    for sequence, (old, new) in enumerate(zip(past, lengths, strict=True)):
        for offset in range(new):
            position = old + offset
            slots.append(block_table[sequence, position // block_size] * block_size + position % block_size)
    slots = np.array(slots, dtype=np.int32)
    if skip and tokens:
        slots[-1] = -1
    for name, current, rotation in (("key", key, qk_rotation), ("value", value, v_rotation)):
        dense = rng.normal(0, 0.5, (num_blocks, block_size, kv_heads, width)).astype(np.float16).astype(np.float32)
        dense[-1] = 0
        stored = rotate(dense) if rotation else dense
        if int4:
            cache, scales = quantize(stored, scale_dtype)
            cache_inputs[f"{name}_scale_cache"] = scales.copy()
        else:
            cache = stored.astype(np.float16)
        cache_inputs[f"{name}_cache"] = cache.copy()
        transformed = rotate(current) if rotation else current.astype(np.float32)
        for token, slot in enumerate(slots):
            if slot < 0:
                continue
            page, offset = divmod(int(slot), block_size)
            if int4:
                cache[page, offset], scales[page, offset] = quantize(transformed[token], scale_dtype)
            else:
                cache[page, offset] = transformed[token].astype(np.float16)
        expected_cache[f"{name}_cache_out"] = cache
        if int4:
            expected_cache[f"{name}_scale_cache_out"] = scales
            logical_cache[name] = unpack(cache, scales)
        else:
            logical_cache[name] = cache.astype(np.float32)

    feeds = {
        "query": query.reshape(tokens, heads * width),
        "key": key.reshape(tokens, kv_heads * width),
        "value": value.reshape(tokens, kv_heads * width),
        **cache_inputs,
        "cumulative_sequence_length": cumulative,
        "past_seqlens": np.array(past, dtype=np.int32),
        "block_table": block_table,
        "slot_mapping": slots,
        "attention_metadata": np.array([max(lengths), max(np.array(past) + lengths), 1], dtype=np.int32),
    }
    if sink:
        feeds["head_sink"] = np.linspace(-0.5, 0.5, heads).astype(np.float16)
    if packed:
        feeds["query"] = np.concatenate([feeds["query"], feeds.pop("key"), feeds.pop("value")], axis=1)
    inputs = [
        "query",
        "" if packed else "key",
        "" if packed else "value",
        "key_cache",
        "value_cache",
        "cumulative_sequence_length",
        "past_seqlens",
        "block_table",
        "",
        "",
        "slot_mapping",
        "head_sink" if sink else "",
        "",
        "",
        "",
        "",
        "attention_metadata",
        "key_scale_cache" if int4 else "",
        "value_scale_cache" if int4 else "",
    ]
    output_info = [("output", activation_dtype, (tokens, heads * width))]
    output_info.extend((name, values.dtype, values.shape) for name, values in expected_cache.items())
    output_order = ["output", "key_cache_out", "value_cache_out"]
    if int4:
        output_order += ["key_scale_cache_out", "value_scale_cache_out"]
    output_info.sort(key=lambda info: output_order.index(info[0]))
    attributes = {
        "num_heads": heads,
        "kv_num_heads": kv_heads,
        "qk_rotation": "HADAMARD" if qk_rotation else "NONE",
        "v_rotation": "HADAMARD" if v_rotation else "NONE",
        "k_cache_dtype": "int4" if int4 else "",
        "v_cache_dtype": "int4" if int4 else "",
        "k_quant_type": "PER_TOKEN" if int4 else "NONE",
        "v_quant_type": "PER_TOKEN" if int4 else "NONE",
        "softcap": softcap,
        "local_window_size": window,
    }
    node = helper.make_node("PagedAttention", inputs, output_order, domain="com.microsoft", **attributes)
    graph = helper.make_graph(
        [node],
        "int4_paged_attention",
        [
            helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(values.dtype), values.shape)
            for name, values in feeds.items()
        ],
        [
            helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), shape)
            for name, dtype, shape in output_info
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    transformed_query = (rotate(query).astype(activation_dtype) if qk_rotation else query).astype(np.float32)
    expected_output = np.zeros_like(query, dtype=np.float32)
    for sequence, (old, new) in enumerate(zip(past, lengths, strict=True)):
        for offset in range(new):
            token = cumulative[sequence] + offset
            end = old + offset + 1
            begin = max(0, end - window) if window > 0 else 0
            positions = np.arange(begin, end)
            pages = block_table[sequence, positions // block_size]
            for head in range(heads):
                kv_head = head // (heads // kv_heads)
                keys = logical_cache["key"][pages, positions % block_size, kv_head]
                values = logical_cache["value"][pages, positions % block_size, kv_head]
                logits = keys @ transformed_query[token, head] / np.sqrt(width)
                if softcap:
                    logits = softcap * np.tanh(logits / softcap)
                maximum = max(np.max(logits), float(feeds["head_sink"][head]) if sink else -np.inf)
                probabilities = np.exp(logits - maximum)
                denominator = probabilities.sum() + (np.exp(float(feeds["head_sink"][head]) - maximum) if sink else 0)
                expected_output[token, head] = probabilities @ values / denominator
    expected_output = expected_output.astype(activation_dtype)
    if v_rotation:
        expected_output = rotate(expected_output).astype(activation_dtype)
    return model, feeds, {"output": expected_output.reshape(tokens, heads * width), **expected_cache}


def run_case(model, feeds, steps=1, updates=None, cuda_graph=False):
    updates = updates or {}
    runner = os.getenv("ORT_PAGED_ATTENTION_TEST_RUNNER")
    if runner:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            onnx.save(model, directory / "model.onnx")
            for name, values in feeds.items():
                values.tofile(directory / f"{name}.bin")
            for step, values in updates.items():
                for name, array in values.items():
                    array.tofile(directory / f"{name}.{step}.bin")
            result = subprocess.run(
                [runner, str(directory), str(steps), str(int(cuda_graph))], capture_output=True, text=True, check=False
            )
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            results = []
            for step in range(steps):
                outputs = {}
                for output in model.graph.output:
                    tensor = output.type.tensor_type
                    shape = [dimension.dim_value for dimension in tensor.shape.dim]
                    dtype = helper.tensor_dtype_to_np_dtype(tensor.elem_type)
                    outputs[output.name] = np.fromfile(directory / f"{output.name}.{step}.bin", dtype=dtype).reshape(
                        shape
                    )
                results.append(outputs)
            return results
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(),
        options,
        providers=[("CUDAExecutionProvider", {"enable_cuda_graph": int(cuda_graph)})],
    )
    binding = session.io_binding()

    def storage(array):
        if array.dtype == np.dtype(ml_dtypes.bfloat16):
            return array.view(np.uint16)
        if array.dtype == np.dtype(ml_dtypes.float8_e4m3fn):
            return array.view(np.uint8)
        return array

    values = {
        name: ort.OrtValue.ortvalue_from_numpy(storage(array), "cpu" if name == "attention_metadata" else "cuda", 0)
        for name, array in feeds.items()
    }
    for input_info in model.graph.input:
        name = input_info.name
        binding.bind_input(
            name,
            "cpu" if name == "attention_metadata" else "cuda",
            0,
            input_info.type.tensor_type.elem_type,
            feeds[name].shape,
            values[name].data_ptr(),
        )
    outputs = {}
    for output in model.graph.output:
        tensor = output.type.tensor_type
        shape = [dimension.dim_value for dimension in tensor.shape.dim]
        if output.name.endswith("_out") and output.name[:-4] in values:
            outputs[output.name] = values[output.name[:-4]]
        else:
            dtype = helper.tensor_dtype_to_np_dtype(tensor.elem_type)
            outputs[output.name] = ort.OrtValue.ortvalue_from_numpy(storage(np.zeros(shape, dtype=dtype)), "cuda", 0)
        binding.bind_output(output.name, "cuda", 0, tensor.elem_type, shape, outputs[output.name].data_ptr())
    results = []
    for step in range(steps):
        for name, array in updates.get(step, {}).items():
            values[name].update_inplace(storage(array))
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        results.append(
            {
                output.name: outputs[output.name]
                .numpy()
                .view(helper.tensor_dtype_to_np_dtype(output.type.tensor_type.elem_type))
                .copy()
                for output in model.graph.output
            }
        )
    return results


def per_channel_int4_case(**kwargs):
    """Both cache sides PER_CHANNEL, which forbids rotation and replaces the scale caches."""
    model, feeds, _ = make_case(qk_rotation=False, v_rotation=False, **kwargs)
    node = model.graph.node[0]
    kv_heads, width = kwargs["kv_heads"], kwargs["width"]
    for side, index, cache_name in (("k", 14, "key_scale_cache"), ("v", 15, "value_scale_cache")):
        scale = np.linspace(0.02, 0.15, kv_heads * width, dtype=np.float32).reshape(kv_heads, 1, width)
        remove_input(model, feeds, cache_name)
        replace_input(model, feeds, f"{side}_scale", scale)
        node.input[index] = f"{side}_scale"
        set_attribute(model, f"{side}_quant_type", "PER_CHANNEL")
    del node.output[3:]
    del model.graph.output[3:]
    return model, feeds


@unittest.skipUnless(int4_kernel_available(), "Requires CUDA PagedAttention built with USE_INT4_KV_CACHE")
class TestPagedAttentionInt4(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"ORT_ENABLE_XQA": "0"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def check_case(self, **kwargs):
        model, feeds, expected = make_case(**kwargs)
        actual = run_case(model, feeds)[0]
        for name, reference in expected.items():
            if name == "output":
                tolerance = 6e-3 if str(reference.dtype) == "bfloat16" else 8e-4
                np.testing.assert_allclose(
                    actual[name].astype(np.float32), reference.astype(np.float32), atol=tolerance, rtol=5e-3
                )
            elif "scale" in name:
                np.testing.assert_allclose(actual[name], reference, atol=1e-6, rtol=1e-3)
            elif reference.dtype == np.float16:
                np.testing.assert_allclose(actual[name], reference, atol=1e-6, rtol=1e-3)
            else:
                np.testing.assert_array_equal(actual[name], reference)
        return actual

    def test_hadamard_rotation_is_output_neutral(self):
        for width in (16, 32, 64, 128, 256):
            with self.subTest(width=width):
                rotated = self.check_case(width=width, int4=False)
                plain = self.check_case(width=width, int4=False, qk_rotation=False, v_rotation=False)
                np.testing.assert_allclose(rotated["output"], plain["output"], atol=8e-4, rtol=5e-3)

    def test_int4_decode_pack_and_scale_cache(self):
        for width in (16, 32, 64, 128, 256):
            for dtype in (np.float16, np.float32):
                with self.subTest(width=width, scale_dtype=dtype):
                    self.check_case(width=width, scale_dtype=dtype)

    def test_int4_packed_qkv_and_skipped_slot(self):
        self.check_case(width=128, lengths=(3, 0, 2), past=(15, 7, 31), packed=True, skip=True)

    def test_int4_prefill(self):
        self.check_case(width=128, lengths=(65, 33), past=(0, 0))

    def test_int4_chunked_prefill(self):
        self.check_case(width=128, lengths=(33, 17), past=(23, 7))

    def test_int4_speculative_decode(self):
        self.check_case(width=256, lengths=(8, 3), past=(31, 7), sink=True, softcap=2.0, window=23)
        self.check_case(width=128, lengths=(8, 3), past=(257, 7))
        self.check_case(width=256, lengths=(8, 3), past=(31, 7), heads=24, kv_heads=4)

    def test_int4_splitkv_and_derived_slots(self):
        model, feeds, expected = make_case(width=128, past=(513, 0))
        remove_input(model, feeds, "slot_mapping")
        actual = run_case(model, feeds)[0]
        for name, reference in expected.items():
            if name == "output":
                np.testing.assert_allclose(actual[name], reference, atol=8e-4, rtol=5e-3)
            else:
                np.testing.assert_array_equal(actual[name], reference)

    def test_int4_xqa_decode(self):
        with patch.dict(os.environ, {"ORT_ENABLE_XQA": "1"}):
            for block_size in (128, 256):
                for window in (-1, 129):
                    with self.subTest(block_size=block_size, window=window):
                        self.check_case(
                            width=256,
                            heads=24,
                            kv_heads=4,
                            past=(513, 138),
                            block_size=block_size,
                            window=window,
                            sink=True,
                        )

    def test_int4_xqa_unsupported_scales_fall_back(self):
        with patch.dict(os.environ, {"ORT_ENABLE_XQA": "1"}):
            self.check_case(width=256, heads=24, kv_heads=4, past=(513, 138), block_size=256, scale_dtype=np.float32)

    def test_int4_xqa_speculative_decode(self):
        with patch.dict(os.environ, {"ORT_ENABLE_XQA": "1"}):
            for lengths in ((2, 1), (8, 3), (0, 8)):
                for window in (-1, 129):
                    with self.subTest(lengths=lengths, window=window):
                        self.check_case(
                            width=256,
                            heads=24,
                            kv_heads=4,
                            past=(513, 138),
                            lengths=lengths,
                            block_size=256,
                            window=window,
                            sink=True,
                        )

    def test_int4_xqa_cuda_graph_replay(self):
        with patch.dict(os.environ, {"ORT_ENABLE_XQA": "1"}):
            model, feeds, _ = make_case(
                width=256, heads=24, kv_heads=4, block_size=256, lengths=(8, 3), past=(513, 138)
            )
            changed = {"key": feeds["key"] * np.float16(3), "value": feeds["value"] * np.float16(0.25)}
            actual = run_case(model, feeds, steps=3, updates={1: changed}, cuda_graph=True)
            reference = run_case(model, {**feeds, **changed})[0]
            self.assertFalse(np.array_equal(actual[0]["value_scale_cache_out"], actual[1]["value_scale_cache_out"]))
            for name in reference:
                np.testing.assert_array_equal(actual[1][name], actual[2][name])
                np.testing.assert_allclose(actual[1][name], reference[name], atol=8e-4, rtol=5e-3)

    def test_int4_without_rotation_and_k_only(self):
        self.check_case(qk_rotation=False, v_rotation=False)
        self.check_case(v_rotation=False)

    def test_int4_cuda_graph_replay(self):
        model, feeds, _ = make_case(width=128)
        changed = {"key": feeds["key"] * np.float16(3), "value": feeds["value"] * np.float16(0.25)}
        actual = run_case(model, feeds, steps=3, updates={1: changed}, cuda_graph=True)
        reference = run_case(model, {**feeds, **changed})[0]
        self.assertFalse(np.array_equal(actual[0]["value_scale_cache_out"], actual[1]["value_scale_cache_out"]))
        for name in reference:
            np.testing.assert_array_equal(actual[1][name], actual[2][name])
            np.testing.assert_allclose(actual[1][name], reference[name], atol=8e-4, rtol=5e-3)

    def test_hadamard_follows_norm_and_partial_rope(self):
        for interleaved in (False, True):
            with self.subTest(interleaved=interleaved):
                model, feeds, _ = make_case(width=64, lengths=(5,), past=(0,), int4=False)
                reference_model, reference_feeds, _ = make_case(width=64, lengths=(5,), past=(0,), int4=False)
                width, rotary_width = 64, 32
                positions = np.arange(5, dtype=np.float32)
                angles = positions[:, None] * np.linspace(0.01, 0.4, rotary_width // 2, dtype=np.float32)
                cos = np.cos(angles).astype(np.float16)
                sin = np.sin(angles).astype(np.float16)
                for name, heads, input_index in (("query", 4, 12), ("key", 2, 13)):
                    weight_name = "q_norm_weight" if name == "query" else "k_norm_weight"
                    weight = np.linspace(0.8, 1.2, width, dtype=np.float16)
                    values = feeds[name].reshape(5, heads, width).astype(np.float32)
                    normalized = (
                        values / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + 1e-6) * weight
                    ).astype(np.float16)
                    channels = np.arange(rotary_width)
                    partner = channels ^ 1 if interleaved else (channels + rotary_width // 2) % rotary_width
                    cache_index = channels // 2 if interleaved else channels % (rotary_width // 2)
                    sign = np.where(channels % 2 == 0 if interleaved else channels < rotary_width // 2, -1, 1)
                    result = normalized.copy()
                    result[..., :rotary_width] = (
                        normalized[..., :rotary_width] * cos[:, None, cache_index]
                        + (normalized[..., partner] * sign.astype(np.float16)) * sin[:, None, cache_index]
                    )
                    reference_feeds[name] = result.reshape(5, heads * width)
                    replace_input(model, feeds, weight_name, weight)
                    model.graph.node[0].input[input_index] = weight_name
                for index, name, values in ((8, "cos_cache", cos), (9, "sin_cache", sin)):
                    replace_input(model, feeds, name, values)
                    model.graph.node[0].input[index] = name
                set_attribute(model, "do_rotary", 1)
                set_attribute(model, "rotary_interleaved", int(interleaved))
                actual = run_case(model, feeds)[0]
                reference = run_case(reference_model, reference_feeds)[0]
                for name in reference:
                    np.testing.assert_allclose(actual[name], reference[name], atol=8e-4, rtol=5e-3)

    def test_optional_scale_outputs(self):
        for output_count in (1, 3, 4, 5):
            with self.subTest(output_count=output_count):
                model, feeds, expected = make_case()
                del model.graph.node[0].output[output_count:]
                del model.graph.output[output_count:]
                actual = run_case(model, feeds)[0]
                for name in actual:
                    np.testing.assert_allclose(actual[name], expected[name], atol=8e-4, rtol=5e-3)

    def test_invalid_contracts(self):
        cases = []
        for side in ("key", "value"):
            prefix = "k" if side == "key" else "v"
            for dimension in range(3):
                cases.append((f"{side}_scale_dim_{dimension}", side, "shape", dimension, "Scale cache must have shape"))
            cases.extend(
                [
                    (f"{side}_missing_scale", side, "missing", None, "PER_TOKEN requires"),
                    (f"{side}_static_scale", side, "static", None, "PER_TOKEN requires"),
                    (f"{side}_ambiguous_uint8", side, "attribute", (f"{prefix}_cache_dtype", ""), "explicit int4"),
                    (
                        f"{side}_wrong_dtype",
                        side,
                        "attribute",
                        (f"{prefix}_cache_dtype", "float4e2m1"),
                        "explicit int4",
                    ),
                    (
                        f"{side}_wrong_mode",
                        side,
                        "attribute",
                        (f"{prefix}_quant_type", "PER_TENSOR"),
                        "only allowed with PER_TOKEN",
                    ),
                    (
                        f"{side}_rotation_channel",
                        side,
                        "attribute",
                        (f"{prefix}_quant_type", "PER_CHANNEL"),
                        "PER_CHANNEL",
                    ),
                ]
            )
        cases.extend(
            [
                ("invalid_rotation", "key", "attribute", ("qk_rotation", "BAD"), "must be NONE or HADAMARD"),
                ("invalid_v_rotation", "value", "attribute", ("v_rotation", "BAD"), "must be NONE or HADAMARD"),
                ("invalid_width", "key", "width", 24, "power-of-two"),
                ("too_small_width", "key", "width", 8, "power-of-two"),
                ("too_large_width", "key", "width", 512, "power-of-two"),
            ]
        )
        for label, side, operation, value, message in cases:
            with self.subTest(case=label):
                model, feeds, _ = make_case(
                    width=value if operation == "width" else 64,
                    qk_rotation=operation != "width",
                    v_rotation=operation != "width",
                )
                if operation == "width":
                    set_attribute(model, "qk_rotation", "HADAMARD")
                if operation == "attribute":
                    set_attribute(model, *value)
                elif operation == "shape":
                    name = f"{side}_scale_cache"
                    shape = list(feeds[name].shape)
                    shape[value] += 1
                    replace_input(model, feeds, name, np.ones(shape, dtype=np.float16))
                elif operation == "missing":
                    remove_input(model, feeds, f"{side}_scale_cache")
                elif operation == "static":
                    name, index = ("k_scale", 14) if side == "key" else ("v_scale", 15)
                    replace_input(model, feeds, name, np.ones(1, dtype=np.float32))
                    model.graph.node[0].input[index] = name
                del model.graph.node[0].output[1:]
                del model.graph.output[1:]
                with self.assertRaisesRegex(Exception, message):
                    run_case(model, feeds)

    def test_optional_scale_output_holes(self):
        for side, prefix, input_index, output_index in (("key", "k", 14, 3), ("value", "v", 15, 4)):
            with self.subTest(side=side):
                model, feeds, _ = make_case()
                remove_input(model, feeds, f"{side}_scale_cache")
                set_attribute(model, f"{prefix}_quant_type", "PER_TENSOR")
                replace_input(model, feeds, f"{prefix}_scale", np.array([0.125], dtype=np.float32))
                model.graph.node[0].input[input_index] = f"{prefix}_scale"
                missing_output = model.graph.node[0].output[output_index]
                model.graph.node[0].output[output_index] = ""
                del model.graph.output[output_index]
                actual = run_case(model, feeds)[0]
                reference_model = onnx.ModelProto.FromString(model.SerializeToString())
                del reference_model.graph.node[0].output[3:]
                del reference_model.graph.output[3:]
                reference = run_case(reference_model, feeds)[0]
                np.testing.assert_array_equal(actual["output"], reference["output"])
                model.graph.node[0].output[output_index] = missing_output
                model.graph.output.append(
                    helper.make_tensor_value_info(missing_output, TensorProto.FLOAT16, feeds[f"{side}_cache"].shape[:3])
                )
                with self.assertRaisesRegex(Exception, "Scale cache output requires"):
                    run_case(model, feeds)

    def test_invalid_packed_dimensions(self):
        for side in ("key", "value"):
            with self.subTest(side=side):
                model, feeds, _ = make_case()
                name = f"{side}_cache"
                replace_input(model, feeds, name, np.repeat(feeds[name], 2, axis=-1))
                del model.graph.node[0].output[1:]
                del model.graph.output[1:]
                with self.assertRaisesRegex(Exception, "dimension 3"):
                    run_case(model, feeds)

    def test_int4_bfloat16_activations(self):
        for width in (16, 32, 128, 256):
            with self.subTest(width=width):
                self.check_case(width=width, activation_dtype=ml_dtypes.bfloat16)

    def test_int4_known_packing_and_padding(self):
        model, feeds, _ = make_case(width=32, lengths=(1,), past=(0,), qk_rotation=False, v_rotation=False)
        pattern = np.array(
            [
                -9,
                -8,
                -7,
                -6,
                -5,
                -4,
                -3,
                -2,
                -1,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                -2.5,
                -1.5,
                -0.5,
                0.5,
                1.5,
                2.5,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=np.float16,
        )
        signed = np.clip(np.rint(pattern), -8, 7).astype(np.int8)
        biased = (signed + 8).astype(np.uint8)
        packed = biased[::2] | (biased[1::2] << 4)
        expected = {}
        slot = int(feeds["slot_mapping"][0])
        for side, prefix, index in (("key", "k", 14), ("value", "v", 15)):
            feeds[side][:] = np.tile(pattern, 2)
            feeds[f"{side}_cache"][:] = 0x88
            remove_input(model, feeds, f"{side}_scale_cache")
            replace_input(model, feeds, f"{prefix}_scale", np.ones(1, dtype=np.float32))
            model.graph.node[0].input[index] = f"{prefix}_scale"
            set_attribute(model, f"{prefix}_quant_type", "PER_TENSOR")
            cache = feeds[f"{side}_cache"].copy()
            cache.reshape(-1, 2, 16)[slot] = packed
            expected[f"{side}_cache_out"] = cache
        del model.graph.node[0].output[3:]
        del model.graph.output[3:]
        actual = run_case(model, feeds)[0]
        for name, values in expected.items():
            np.testing.assert_array_equal(actual[name], values)
        np.testing.assert_array_equal(actual["output"], np.tile(signed, 4).reshape(1, -1).astype(np.float16))

    def test_int4_k_per_token_v_per_channel(self):
        model, feeds, expected = make_case(width=32, lengths=(1,), past=(0,), v_rotation=False)
        scale = np.linspace(0.02, 0.15, 64, dtype=np.float32).reshape(2, 1, 32)
        remove_input(model, feeds, "value_scale_cache")
        replace_input(model, feeds, "v_scale", scale)
        model.graph.node[0].input[15] = "v_scale"
        set_attribute(model, "v_quant_type", "PER_CHANNEL")
        del model.graph.node[0].output[4:]
        del model.graph.output[4:]
        scaled = feeds["value"].reshape(2, 32).astype(np.float32) / scale.reshape(2, 32)
        quantized = np.clip(np.rint(scaled), -8, 7)
        decoded = quantized * scale.reshape(2, 32)
        actual = run_case(model, feeds)[0]
        np.testing.assert_allclose(actual["output"], np.repeat(decoded, 2, axis=0).reshape(1, -1), atol=5e-4, rtol=1e-3)
        np.testing.assert_array_equal(actual["key_cache_out"], expected["key_cache_out"])
        np.testing.assert_array_equal(actual["key_scale_cache_out"], expected["key_scale_cache_out"])

    def test_int4_per_channel_xqa_matches_portable(self):
        # A PER_CHANNEL scale is folded into Q and the output, so XQA has to agree with the
        # portable kernel. lengths (2, 1) also covers the speculative INT4 kernel.
        for lengths in ((1, 1), (2, 1)):
            with self.subTest(lengths=lengths):
                model, feeds = per_channel_int4_case(
                    width=256, heads=24, kv_heads=4, past=(513, 138), block_size=256, lengths=lengths
                )
                with patch.dict(os.environ, {"ORT_ENABLE_XQA": "0"}):
                    portable = run_case(model, feeds)[0]
                with patch.dict(os.environ, {"ORT_ENABLE_XQA": "1"}):
                    accelerated = run_case(model, feeds)[0]
                np.testing.assert_allclose(
                    accelerated["output"].astype(np.float32),
                    portable["output"].astype(np.float32),
                    atol=8e-4,
                    rtol=5e-3,
                )
                for name in ("key_cache_out", "value_cache_out"):
                    np.testing.assert_array_equal(accelerated[name], portable[name])

    def test_int4_scale_extremes(self):
        for dynamic in (False, True):
            for magnitude in (2.0**-24, 1e10):
                with self.subTest(dynamic=dynamic, magnitude=magnitude):
                    model, feeds, _ = make_case(
                        width=32,
                        lengths=(1,),
                        past=(0,),
                        qk_rotation=False,
                        v_rotation=False,
                        activation_dtype=ml_dtypes.bfloat16,
                    )
                    pattern = np.tile(np.array([-magnitude, magnitude], dtype=ml_dtypes.bfloat16), 32).reshape(1, 64)
                    for side, prefix, index in (("key", "k", 14), ("value", "v", 15)):
                        feeds[side][:] = pattern
                        if not dynamic:
                            remove_input(model, feeds, f"{side}_scale_cache")
                            replace_input(model, feeds, f"{prefix}_scale", np.array([1e-6], dtype=np.float32))
                            model.graph.node[0].input[index] = f"{prefix}_scale"
                            set_attribute(model, f"{prefix}_quant_type", "PER_TENSOR")
                    if not dynamic:
                        del model.graph.node[0].output[3:]
                        del model.graph.output[3:]
                    actual = run_case(model, feeds)[0]
                    raw = pattern.reshape(2, 32).astype(np.float32)
                    if dynamic:
                        packed, scales = quantize(raw, np.float16)
                    else:
                        signed = np.clip(np.rint(raw / np.float32(1e-6)), -8, 7).astype(np.int8)
                        biased = (signed + 8).astype(np.uint8)
                        packed = biased[:, ::2] | (biased[:, 1::2] << 4)
                    page, offset = divmod(int(feeds["slot_mapping"][0]), 16)
                    for side in ("key", "value"):
                        np.testing.assert_array_equal(actual[f"{side}_cache_out"][page, offset], packed)
                        if dynamic:
                            np.testing.assert_array_equal(actual[f"{side}_scale_cache_out"][page, offset], scales)

    def test_reject_scale_rank_dtype_and_latent_rotation(self):
        for operation, message in (
            ("rank", "Scale cache must have shape"),
            ("dtype", "invalid"),
            ("mixed_dtype", "bound to different types"),
            ("latent", "LATENT"),
        ):
            with self.subTest(operation=operation):
                model, feeds, _ = make_case()
                if operation == "rank":
                    replace_input(model, feeds, "key_scale_cache", feeds["key_scale_cache"].reshape(-1, 2))
                elif operation == "dtype":
                    replace_input(model, feeds, "key_scale_cache", feeds["key_scale_cache"].astype(np.int32))
                elif operation == "mixed_dtype":
                    replace_input(model, feeds, "key_scale_cache", feeds["key_scale_cache"].astype(np.float32))
                else:
                    model, feeds, _ = make_case(int4=False)
                    set_attribute(model, "kv_cache_layout", "LATENT")
                    remove_input(model, feeds, "value")
                    remove_input(model, feeds, "value_cache")
                    set_attribute(model, "kv_num_heads", 1)
                    replace_input(model, feeds, "key", feeds["key"][:, :64])
                    replace_input(model, feeds, "key_cache", feeds["key_cache"][:, :, :1, :].copy())
                del model.graph.node[0].output[1:]
                del model.graph.output[1:]
                with self.assertRaisesRegex(Exception, message):
                    run_case(model, feeds)

    def test_int8_fp8_cache_regression(self):
        for cache_dtype, qmax in ((np.int8, 127), (ml_dtypes.float8_e4m3fn, 448)):
            for mode in ("PER_TENSOR", "PER_CHANNEL", "PER_TOKEN"):
                for lengths in ((1, 1), (33, 17)):
                    with self.subTest(cache_dtype=cache_dtype, mode=mode, lengths=lengths):
                        rotated = mode == "PER_TOKEN"
                        model, feeds, _ = make_case(
                            width=128, lengths=lengths, int4=False, qk_rotation=rotated, v_rotation=rotated
                        )
                        reference_model = onnx.ModelProto.FromString(model.SerializeToString())
                        reference_feeds = {name: array.copy() for name, array in feeds.items()}
                        reference_feeds["slot_mapping"][:] = -1
                        expected_cache = {}
                        for side, prefix, index in (("key", "k", 14), ("value", "v", 15)):
                            cache_name = f"{side}_cache"
                            dense = feeds[cache_name].astype(np.float32)
                            current = feeds[side].reshape(-1, 2, 128).astype(np.float32)
                            if rotated:
                                current = rotate(current)
                            if mode == "PER_TOKEN":
                                scales = (np.max(np.abs(dense), axis=-1) / qmax).astype(np.float16)
                                current_scales = (np.max(np.abs(current), axis=-1) / qmax).astype(np.float16)
                                scale_name = f"{side}_scale_cache"
                                replace_input(model, feeds, scale_name, scales.copy())
                                model.graph.node[0].input[index + 3] = scale_name
                                model.graph.node[0].output.append(f"{scale_name}_out")
                                model.graph.output.append(
                                    helper.make_tensor_value_info(
                                        f"{scale_name}_out", TensorProto.FLOAT16, scales.shape
                                    )
                                )
                                divisor = scales[..., None]
                                current_divisor = current_scales[..., None]
                            else:
                                scale = (
                                    np.array([0.03125], dtype=np.float32)
                                    if mode == "PER_TENSOR"
                                    else np.linspace(0.02, 0.06, 256, dtype=np.float32).reshape(2, 1, 128)
                                )
                                scale_name = f"{prefix}_scale"
                                replace_input(model, feeds, scale_name, scale)
                                model.graph.node[0].input[index] = scale_name
                                divisor = scale.reshape(2, 128) if mode == "PER_CHANNEL" else scale
                                current_divisor = divisor

                            def encode(array, scale, cache_dtype=cache_dtype, qmax=qmax):
                                scaled = np.divide(array, scale, out=np.zeros_like(array), where=scale != 0)
                                if cache_dtype == np.int8:
                                    scaled = np.rint(scaled)
                                return np.clip(scaled, -qmax, qmax).astype(cache_dtype)

                            cache = encode(dense, divisor)
                            replace_input(model, feeds, cache_name, cache.copy())
                            cache_type = helper.np_dtype_to_tensor_dtype(np.dtype(cache_dtype))
                            for output in model.graph.output:
                                if output.name == f"{cache_name}_out":
                                    output.type.tensor_type.elem_type = cache_type
                            encoded_current = encode(current, current_divisor)
                            for token, slot in enumerate(feeds["slot_mapping"]):
                                page, offset = divmod(int(slot), 16)
                                cache[page, offset] = encoded_current[token]
                                if mode == "PER_TOKEN":
                                    scales[page, offset] = current_scales[token]
                            expected_cache[f"{cache_name}_out"] = cache
                            if mode == "PER_TOKEN":
                                expected_cache[f"{scale_name}_out"] = scales
                            reference_feeds[cache_name] = (cache.astype(np.float32) * divisor).astype(np.float16)
                            set_attribute(model, f"{prefix}_quant_type", mode)
                        actual = run_case(model, feeds)[0]
                        reference = run_case(reference_model, reference_feeds)[0]
                        np.testing.assert_allclose(actual["output"], reference["output"], atol=8e-4, rtol=5e-3)
                        for name, expected in expected_cache.items():
                            if "scale" in name:
                                np.testing.assert_allclose(actual[name], expected, atol=1e-6, rtol=1e-3)
                            elif cache_dtype == np.int8:
                                np.testing.assert_allclose(actual[name], expected, atol=1, rtol=0)
                            else:
                                lower = np.nextafter(expected, np.array(-448, dtype=cache_dtype)).astype(np.float32)
                                upper = np.nextafter(expected, np.array(448, dtype=cache_dtype)).astype(np.float32)
                                self.assertTrue(np.all(actual[name].astype(np.float32) >= lower))
                                self.assertTrue(np.all(actual[name].astype(np.float32) <= upper))


if __name__ == "__main__":
    unittest.main()
