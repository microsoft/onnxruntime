import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import ml_dtypes
import numpy as np
import onnx
import torch

import onnxruntime as ort
from onnxruntime.capi import _pybind_state

helper = onnx.helper


def has_sm80_cuda():
    return bool(os.getenv("ORT_PAGED_ATTENTION_TEST_RUNNER")) or (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    )


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


def static_scale(quant_type, kv_heads, width):
    """Schema scale shape per granularity: (1,) for PER_TENSOR, (kv_num_heads, 1, head_size) otherwise."""
    if quant_type == "PER_TENSOR":
        return np.array([0.2], dtype=np.float32)
    return np.linspace(0.05, 0.25, kv_heads * width, dtype=np.float32).reshape(kv_heads, 1, width)


def quantize(values, scale):
    """Signed INT4 codes in [-8, 7] stored biased by +8, two per byte, even channel in the low nibble."""
    values = values.astype(np.float32)
    scaled = np.divide(values, scale, out=np.zeros_like(values), where=scale != 0)
    biased = (np.clip(np.rint(scaled), -8, 7).astype(np.int8) + 8).astype(np.uint8)
    return biased[..., ::2] | (biased[..., 1::2] << 4)


def unpack(packed, scale):
    values = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.float32)
    values[..., ::2] = (packed & 15).astype(np.float32) - 8
    values[..., 1::2] = (packed >> 4).astype(np.float32) - 8
    return values * scale


def make_case(
    width=64,
    lengths=(1, 1),
    past=(19, 7),
    int4=True,
    quant_type="PER_CHANNEL",
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
    for name, prefix, current in (("key", "k", key), ("value", "v", value)):
        dense = rng.normal(0, 0.5, (num_blocks, block_size, kv_heads, width)).astype(np.float16).astype(np.float32)
        dense[-1] = 0
        broadcast = None
        if int4:
            scale = static_scale(quant_type, kv_heads, width)
            broadcast = scale.reshape(kv_heads, width) if quant_type == "PER_CHANNEL" else scale
            cache_inputs[f"{prefix}_scale"] = scale
            cache = quantize(dense, broadcast)
        else:
            cache = dense.astype(np.float16)
        cache_inputs[f"{name}_cache"] = cache.copy()
        for token, slot in enumerate(slots):
            if slot < 0:
                continue
            page, offset = divmod(int(slot), block_size)
            if int4:
                cache[page, offset] = quantize(current[token], broadcast)
            else:
                cache[page, offset] = current[token].astype(np.float16)
        expected_cache[f"{name}_cache_out"] = cache
        logical_cache[name] = unpack(cache, broadcast) if int4 else cache.astype(np.float32)

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
        "k_scale" if int4 else "",
        "v_scale" if int4 else "",
        "attention_metadata",
    ]
    output_info = [("output", activation_dtype, (tokens, heads * width))]
    output_info.extend((name, values.dtype, values.shape) for name, values in expected_cache.items())
    output_order = ["output", "key_cache_out", "value_cache_out"]
    output_info.sort(key=lambda info: output_order.index(info[0]))
    attributes = {
        "num_heads": heads,
        "kv_num_heads": kv_heads,
        "k_cache_dtype": "int4" if int4 else "",
        "v_cache_dtype": "int4" if int4 else "",
        "k_quant_type": quant_type if int4 else "NONE",
        "v_quant_type": quant_type if int4 else "NONE",
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
                logits = keys @ query[token, head].astype(np.float32) / np.sqrt(width)
                if softcap:
                    logits = softcap * np.tanh(logits / softcap)
                maximum = max(np.max(logits), float(feeds["head_sink"][head]) if sink else -np.inf)
                probabilities = np.exp(logits - maximum)
                denominator = probabilities.sum() + (np.exp(float(feeds["head_sink"][head]) - maximum) if sink else 0)
                expected_output[token, head] = probabilities @ values / denominator
    expected_output = expected_output.astype(activation_dtype)
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
            if os.getenv("ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO") == "1":
                print(result.stdout, end="")
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


def run_with_kernel(model, feeds, expected_kernel, **kwargs):
    sys.stdout.flush()
    saved_fd = os.dup(1)
    try:
        with tempfile.TemporaryFile() as captured:
            os.dup2(captured.fileno(), 1)
            try:
                with patch.dict(os.environ, {"ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO": "1"}):
                    results = run_case(model, feeds, **kwargs)
            finally:
                try:
                    sys.stdout.flush()
                finally:
                    os.dup2(saved_fd, 1)
            captured.seek(0)
            debug_output = captured.read().decode(errors="replace")
    finally:
        os.close(saved_fd)
    dispatches = [line for line in debug_output.splitlines() if "Operator=PagedAttention" in line]
    assert dispatches, f"Missing PagedAttention dispatch telemetry: {debug_output}"
    assert all(f"SdpaKernel={expected_kernel}" in line for line in dispatches), debug_output
    return results


class TestPagedAttentionInt4Helpers(unittest.TestCase):
    def test_dispatch_capture_accepts_xqa(self):
        result = [object()]

        def run(*args, **kwargs):
            self.assertEqual(os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"], "1")
            os.write(1, b"Operator=PagedAttention SdpaKernel=XQA\n")
            return result

        with patch(__name__ + ".run_case", side_effect=run):
            self.assertIs(run_with_kernel(None, None, "XQA"), result)

    def test_dispatch_capture_rejects_fallback_and_missing_telemetry(self):
        for telemetry in (
            b"Operator=PagedAttention SdpaKernel=DECODER_ATTENTION\n",
            b"",
            b"Operator=PagedAttention SdpaKernel=XQA\nOperator=PagedAttention SdpaKernel=DECODER_ATTENTION\n",
        ):
            with (
                self.subTest(telemetry=telemetry),
                patch(
                    __name__ + ".run_case",
                    side_effect=lambda *args, telemetry=telemetry, **kwargs: os.write(1, telemetry),
                ),
                self.assertRaises(AssertionError),
            ):
                run_with_kernel(None, None, "XQA")

    def test_dispatch_capture_restores_stdout_on_error(self):
        original_stdout = os.fstat(1)
        with patch.dict(os.environ, {"ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO": "0"}):
            with (
                patch(__name__ + ".run_case", side_effect=RuntimeError("kernel failure")),
                self.assertRaisesRegex(RuntimeError, "kernel failure"),
            ):
                run_with_kernel(None, None, "XQA")
            self.assertEqual(os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"], "0")
        restored_stdout = os.fstat(1)
        self.assertEqual(
            (restored_stdout.st_dev, restored_stdout.st_ino), (original_stdout.st_dev, original_stdout.st_ino)
        )


@unittest.skipUnless(int4_kernel_available(), "Requires CUDA PagedAttention built with USE_INT4_KV_CACHE")
class TestPagedAttentionInt4(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"ORT_ENABLE_XQA": "0"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def check_case(self, expected_kernel=None, **kwargs):
        model, feeds, expected = make_case(**kwargs)
        actual = (
            run_case(model, feeds) if expected_kernel is None else run_with_kernel(model, feeds, expected_kernel)
        )[0]
        for name, reference in expected.items():
            if name == "output":
                tolerance = 6e-3 if str(reference.dtype) == "bfloat16" else 8e-4
                np.testing.assert_allclose(
                    actual[name].astype(np.float32), reference.astype(np.float32), atol=tolerance, rtol=5e-3
                )
            elif reference.dtype == np.float16:
                np.testing.assert_allclose(actual[name], reference, atol=1e-6, rtol=1e-3)
            else:
                np.testing.assert_array_equal(actual[name], reference)
        return actual

    def test_int4_decode_pack(self):
        for width in (16, 32, 64, 128, 256):
            for quant_type in ("PER_CHANNEL", "PER_TENSOR"):
                with self.subTest(width=width, quant_type=quant_type):
                    self.check_case(width=width, quant_type=quant_type)

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

    @unittest.skipUnless(has_sm80_cuda(), "XQA requires an SM80 or newer GPU")
    def test_int4_xqa_decode(self):
        with patch.dict(os.environ, {"ORT_ENABLE_XQA": "1"}):
            for block_size in (128, 256):
                for window in (-1, 129):
                    with self.subTest(block_size=block_size, window=window):
                        self.check_case(
                            expected_kernel="XQA",
                            width=256,
                            heads=24,
                            kv_heads=4,
                            past=(513, 138),
                            block_size=block_size,
                            window=window,
                            sink=True,
                        )

    def test_int4_xqa_unsupported_scales_fall_back(self):
        # INT4 XQA only covers PER_CHANNEL scales, so PER_TENSOR must take the portable kernel.
        with patch.dict(os.environ, {"ORT_ENABLE_XQA": "1"}):
            self.check_case(
                expected_kernel="DECODER_ATTENTION",
                width=256,
                heads=24,
                kv_heads=4,
                past=(513, 138),
                block_size=256,
                quant_type="PER_TENSOR",
            )

    @unittest.skipUnless(has_sm80_cuda(), "XQA requires an SM80 or newer GPU")
    def test_int4_xqa_speculative_decode(self):
        with patch.dict(os.environ, {"ORT_ENABLE_XQA": "1"}):
            for lengths in ((2, 1), (8, 3), (0, 8)):
                for window in (-1, 129):
                    with self.subTest(lengths=lengths, window=window):
                        self.check_case(
                            expected_kernel="XQA",
                            width=256,
                            heads=24,
                            kv_heads=4,
                            past=(513, 138),
                            lengths=lengths,
                            block_size=256,
                            window=window,
                            sink=True,
                        )

    @unittest.skipUnless(has_sm80_cuda(), "XQA requires an SM80 or newer GPU")
    def test_int4_xqa_cuda_graph_replay(self):
        with patch.dict(os.environ, {"ORT_ENABLE_XQA": "1"}):
            model, feeds, _ = make_case(
                width=256, heads=24, kv_heads=4, block_size=256, lengths=(8, 3), past=(513, 138)
            )
            changed = {"key": feeds["key"] * np.float16(3), "value": feeds["value"] * np.float16(0.25)}
            actual = run_with_kernel(model, feeds, "XQA", steps=3, updates={1: changed}, cuda_graph=True)
            reference = run_with_kernel(model, {**feeds, **changed}, "XQA")[0]
            self.assertFalse(np.array_equal(actual[0]["value_cache_out"], actual[1]["value_cache_out"]))
            for name in reference:
                np.testing.assert_array_equal(actual[1][name], actual[2][name])
                np.testing.assert_allclose(actual[1][name], reference[name], atol=8e-4, rtol=5e-3)

    @unittest.skipUnless(has_sm80_cuda(), "Large-batch fallback requires an SM80 or newer GPU")
    def test_int4_speculative_decode_exceeds_grid_y_limit(self):
        batch_size = 8192
        model, feeds, expected = make_case(
            width=64, lengths=(8,), past=(0,), heads=1, kv_heads=1, quant_type="PER_TENSOR"
        )
        num_blocks, block_size = feeds["key_cache"].shape[:2]
        for name in ("query", "key", "value", "key_cache", "value_cache"):
            values = feeds[name]
            replace_input(model, feeds, name, np.tile(values, (batch_size, *([1] * (values.ndim - 1)))))
        replace_input(model, feeds, "past_seqlens", np.zeros(batch_size, dtype=np.int32))
        replace_input(model, feeds, "cumulative_sequence_length", np.arange(batch_size + 1, dtype=np.int32) * 8)
        block_offsets = np.arange(batch_size, dtype=np.int32)[:, None] * num_blocks
        replace_input(model, feeds, "block_table", feeds["block_table"] + block_offsets)
        replace_input(model, feeds, "slot_mapping", (feeds["slot_mapping"] + block_offsets * block_size).reshape(-1))
        for output in model.graph.output:
            reference = expected[output.name]
            reference = np.tile(reference, (batch_size, *([1] * (reference.ndim - 1))))
            expected[output.name] = reference
            output.CopyFrom(
                helper.make_tensor_value_info(
                    output.name, helper.np_dtype_to_tensor_dtype(reference.dtype), reference.shape
                )
            )
        self.assertEqual(feeds["query"].shape[0], 65536)
        actual = run_case(model, feeds)[0]
        np.testing.assert_allclose(actual["output"], expected["output"], atol=8e-4, rtol=5e-3)
        for name in ("key_cache_out", "value_cache_out"):
            np.testing.assert_array_equal(actual[name], expected[name])

    def test_int4_cuda_graph_replay(self):
        model, feeds, _ = make_case(width=128)
        changed = {"key": feeds["key"] * np.float16(3), "value": feeds["value"] * np.float16(0.25)}
        actual = run_case(model, feeds, steps=3, updates={1: changed}, cuda_graph=True)
        reference = run_case(model, {**feeds, **changed})[0]
        self.assertFalse(np.array_equal(actual[0]["value_cache_out"], actual[1]["value_cache_out"]))
        for name in reference:
            np.testing.assert_array_equal(actual[1][name], actual[2][name])
            np.testing.assert_allclose(actual[1][name], reference[name], atol=8e-4, rtol=5e-3)

    def test_cache_write_follows_norm_and_partial_rope(self):
        for interleaved in (False, True):
            with self.subTest(interleaved=interleaved):
                model, feeds, _ = make_case(width=64, lengths=(5,), past=(0,))
                reference_model, reference_feeds, _ = make_case(width=64, lengths=(5,), past=(0,))
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
                    if name == "output":
                        np.testing.assert_allclose(actual[name], reference[name], atol=8e-4, rtol=5e-3)
                    else:
                        np.testing.assert_array_equal(actual[name], reference[name])

    def test_optional_cache_outputs(self):
        for output_count in (1, 3):
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
            cases.extend(
                [
                    (f"{side}_ambiguous_uint8", (f"{prefix}_cache_dtype", ""), "explicit int4"),
                    (f"{side}_wrong_dtype", (f"{prefix}_cache_dtype", "float4e2m1"), "explicit int4"),
                ]
            )
        for label, attribute, message in cases:
            with self.subTest(case=label):
                model, feeds, _ = make_case(width=64)
                set_attribute(model, *attribute)
                del model.graph.node[0].output[1:]
                del model.graph.output[1:]
                with self.assertRaisesRegex(Exception, message):
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
        model, feeds, _ = make_case(width=32, lengths=(1,), past=(0,), quant_type="PER_TENSOR")
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
        for side, prefix in (("key", "k"), ("value", "v")):
            feeds[side][:] = np.tile(pattern, 2)
            feeds[f"{side}_cache"][:] = 0x88
            replace_input(model, feeds, f"{prefix}_scale", np.ones(1, dtype=np.float32))
            cache = feeds[f"{side}_cache"].copy()
            cache.reshape(-1, 2, 16)[slot] = packed
            expected[f"{side}_cache_out"] = cache
        actual = run_case(model, feeds)[0]
        for name, values in expected.items():
            np.testing.assert_array_equal(actual[name], values)
        np.testing.assert_array_equal(actual["output"], np.tile(signed, 4).reshape(1, -1).astype(np.float16))

    @unittest.skipUnless(has_sm80_cuda(), "XQA requires an SM80 or newer GPU")
    def test_int4_per_channel_xqa_matches_portable(self):
        for lengths in ((1, 1), (2, 1)):
            with self.subTest(lengths=lengths):
                model, feeds, _ = make_case(
                    width=256, heads=24, kv_heads=4, past=(513, 138), block_size=256, lengths=lengths
                )
                # The reference arm keeps XQA enabled and opts out of per-channel folding only, so
                # this also pins that ORT_ENABLE_XQA_PER_CHANNEL_KV alone selects the portable kernel.
                with patch.dict(
                    os.environ,
                    {"ORT_ENABLE_XQA": "1", "ORT_ENABLE_XQA_PER_CHANNEL_KV": "0"},
                ):
                    portable = run_with_kernel(model, feeds, "DECODER_ATTENTION")[0]
                with patch.dict(os.environ, {"ORT_ENABLE_XQA": "1"}):
                    accelerated = run_with_kernel(model, feeds, "XQA")[0]
                np.testing.assert_allclose(
                    accelerated["output"].astype(np.float32),
                    portable["output"].astype(np.float32),
                    atol=8e-4,
                    rtol=5e-3,
                )
                for name in ("key_cache_out", "value_cache_out"):
                    np.testing.assert_array_equal(accelerated[name], portable[name])

    @unittest.skipUnless(has_sm80_cuda(), "XQA requires an SM80 or newer GPU")
    def test_int4_xqa_large_per_channel_k_scale_matches_portable(self):
        heads, kv_heads, width = 24, 4, 256
        model, feeds, _ = make_case(width=width, heads=heads, kv_heads=kv_heads, past=(513, 138), block_size=256)
        feeds["key"][:] = 0
        feeds["key_cache"][:] = 0x88  # two zero codes per byte

        query = np.abs(feeds["query"].reshape(-1, heads, width).astype(np.float32))
        k_scale = np.tile(np.linspace(0.5, 1.0, width, dtype=np.float32), (kv_heads, 1)).reshape(kv_heads, 1, width)
        k_scale *= np.float32(1.0e6 / (query * k_scale[:, 0, :].repeat(heads // kv_heads, axis=0)).max())
        replace_input(model, feeds, "k_scale", k_scale)
        self.assertGreater((query * k_scale[:, 0, :].repeat(heads // kv_heads, axis=0)).max(), np.finfo(np.float16).max)

        with patch.dict(os.environ, {"ORT_ENABLE_XQA": "0"}):
            portable = run_with_kernel(model, feeds, "DECODER_ATTENTION")[0]
        with patch.dict(os.environ, {"ORT_ENABLE_XQA": "1"}):
            accelerated = run_with_kernel(model, feeds, "XQA")[0]
        self.assertTrue(np.isfinite(accelerated["output"].astype(np.float32)).all())
        np.testing.assert_allclose(
            accelerated["output"].astype(np.float32), portable["output"].astype(np.float32), atol=8e-4, rtol=5e-3
        )

    def test_xqa_large_attention_scale_and_k_scale(self):
        # An attention scale above one together with a channel scale at FLT_MAX would make
        # attention_scale * normalizer overflow fp32 and every logit NaN. The normalizer exponent is
        # bounded to prevent that, and this table spans one binade so it stays on XQA.
        heads, width = 6, 256
        for cache_dtype in (np.uint8, np.int8, ml_dtypes.float8_e4m3fn):
            for length in (1, 3):
                with self.subTest(cache_dtype=cache_dtype, length=length):
                    model, feeds, _ = make_case(
                        width=width, heads=heads, kv_heads=1, block_size=128, lengths=(length,), past=(1,)
                    )
                    feeds["query"][:] = 0
                    feeds["query"].reshape(length, heads, width)[..., 0] = 0.25
                    feeds["slot_mapping"][:] = -1
                    scale = np.full((1, 1, width), np.finfo(np.float32).max, dtype=np.float32)
                    replace_input(model, feeds, "k_scale", scale)
                    replace_input(model, feeds, "v_scale", np.ones_like(scale))
                    set_attribute(model, "scale", 2.0)
                    page = int(feeds["block_table"][0, 0])
                    for side, prefix in (("key", "k"), ("value", "v")):
                        codes = np.zeros((*feeds[f"{side}_cache"].shape[:-1], width), dtype=np.float32)
                        if side == "key":
                            codes[page, 0, 0, 0] = 1
                        else:
                            codes[page, 0] = 1
                        if cache_dtype == np.uint8:
                            cache = quantize(codes, np.ones_like(scale[:, 0]))
                        else:
                            cache = codes.astype(cache_dtype)
                            set_attribute(model, f"{prefix}_cache_dtype", "")
                        replace_input(model, feeds, f"{side}_cache", cache)
                        output = next(info for info in model.graph.output if info.name == f"{side}_cache_out")
                        output.CopyFrom(
                            helper.make_tensor_value_info(
                                output.name, helper.np_dtype_to_tensor_dtype(cache.dtype), cache.shape
                            )
                        )
                    with patch.dict(os.environ, {"ORT_ENABLE_XQA": "1"}):
                        actual = run_with_kernel(model, feeds, "XQA")[0]
                    self.assertTrue(np.isfinite(actual["output"]).all())
                    np.testing.assert_allclose(
                        actual["output"],
                        np.repeat(np.ones(length)[:, None], heads * width, axis=1),
                        atol=8e-4,
                        rtol=5e-3,
                        equal_nan=False,
                    )

    def test_per_channel_scale_dynamic_range(self):
        # 1e8 between the smallest and largest channel is wider than folding into an fp16 query can
        # hold, so this pins the portable kernel reached through the per-channel opt-out.
        heads, width = 6, 256
        for cache_dtype in (np.uint8, np.int8, ml_dtypes.float8_e4m3fn):
            for length in (1, 3):
                for extreme in (False, True):
                    with self.subTest(cache_dtype=cache_dtype, length=length, extreme=extreme):
                        model, feeds, _ = make_case(
                            width=width, heads=heads, kv_heads=1, block_size=128, lengths=(length,), past=(1,)
                        )
                        feeds["query"][:] = 0
                        feeds["query"].reshape(length, heads, width)[..., 0] = 0.25 if extreme else 1
                        feeds["slot_mapping"][:] = -1
                        scale = np.ones((1, 1, width), dtype=np.float32)
                        scale[..., 1] = 1e8
                        if extreme:
                            scale[:] = np.finfo(np.float32).max
                        replace_input(model, feeds, "k_scale", scale)
                        replace_input(model, feeds, "v_scale", np.ones_like(scale))
                        set_attribute(model, "scale", 2.0 if extreme else 1.0)
                        page = int(feeds["block_table"][0, 0])
                        for side, prefix in (("key", "k"), ("value", "v")):
                            codes = np.zeros((*feeds[f"{side}_cache"].shape[:-1], width), dtype=np.float32)
                            if side == "key":
                                codes[page, 0, 0, 0] = 1
                            else:
                                codes[page, 0] = 1
                            if cache_dtype == np.uint8:
                                cache = quantize(codes, np.ones_like(scale[:, 0]))
                            else:
                                cache = codes.astype(cache_dtype)
                                set_attribute(model, f"{prefix}_cache_dtype", "")
                            replace_input(model, feeds, f"{side}_cache", cache)
                            output = next(info for info in model.graph.output if info.name == f"{side}_cache_out")
                            output.CopyFrom(
                                helper.make_tensor_value_info(
                                    output.name, helper.np_dtype_to_tensor_dtype(cache.dtype), cache.shape
                                )
                            )
                        changed = scale.copy()
                        if not extreme:
                            changed[..., 0] = 2
                        with patch.dict(
                            os.environ,
                            {"ORT_ENABLE_XQA": "1", "ORT_ENABLE_XQA_PER_CHANNEL_KV": "0"},
                        ):
                            results = run_with_kernel(
                                model,
                                feeds,
                                "DECODER_ATTENTION",
                                steps=3,
                                updates={1: {"k_scale": changed}},
                                cuda_graph=True,
                            )
                        for step, actual in enumerate(results):
                            self.assertTrue(np.isfinite(actual["output"]).all())
                            weight = np.exp(1.0 if step == 0 else 2.0)
                            expected = np.ones(length) if extreme else weight / (weight + np.arange(1, length + 1))
                            np.testing.assert_allclose(
                                actual["output"],
                                np.repeat(expected[:, None], heads * width, axis=1),
                                atol=8e-4,
                                rtol=5e-3,
                                equal_nan=False,
                            )
                            for side in ("key", "value"):
                                np.testing.assert_array_equal(actual[f"{side}_cache_out"], feeds[f"{side}_cache"])

    def test_per_channel_k_keeps_int8_xqa(self):
        # PER_CHANNEL K on an INT8 cache is XQA-eligible without this feature, so the normalized
        # fold has to keep it there rather than demoting an existing path to portable decode.
        for lengths in ((1, 1), (3, 1)):
            with self.subTest(lengths=lengths):
                model, feeds, _ = make_case(
                    width=256, heads=6, kv_heads=1, block_size=128, lengths=lengths, past=(129, 17)
                )
                for side, prefix in (("key", "k"), ("value", "v")):
                    cache = unpack(feeds[f"{side}_cache"], np.float32(1)).astype(np.int8)
                    replace_input(model, feeds, f"{side}_cache", cache)
                    set_attribute(model, f"{prefix}_cache_dtype", "")
                    output = next(info for info in model.graph.output if info.name == f"{side}_cache_out")
                    output.CopyFrom(helper.make_tensor_value_info(output.name, onnx.TensorProto.INT8, cache.shape))
                with patch.dict(
                    os.environ,
                    {"ORT_ENABLE_XQA": "1", "ORT_ENABLE_XQA_PER_CHANNEL_KV": "0"},
                ):
                    portable = run_with_kernel(model, feeds, "DECODER_ATTENTION")[0]
                with patch.dict(os.environ, {"ORT_ENABLE_XQA": "1"}):
                    accelerated = run_with_kernel(model, feeds, "XQA")[0]
                self.assertTrue(np.isfinite(accelerated["output"]).all())
                np.testing.assert_allclose(
                    accelerated["output"], portable["output"], atol=8e-4, rtol=5e-3, equal_nan=False
                )
                for side in ("key", "value"):
                    np.testing.assert_array_equal(accelerated[f"{side}_cache_out"], portable[f"{side}_cache_out"])

    def test_scalar_k_per_channel_v_keeps_int8_xqa(self):
        for lengths in ((1, 1), (3, 1)):
            with self.subTest(lengths=lengths):
                model, feeds, _ = make_case(
                    width=256, heads=6, kv_heads=1, block_size=128, lengths=lengths, past=(129, 17)
                )
                replace_input(model, feeds, "k_scale", np.array([0.125], dtype=np.float32))
                set_attribute(model, "k_quant_type", "PER_TENSOR")
                for side, prefix in (("key", "k"), ("value", "v")):
                    cache = unpack(feeds[f"{side}_cache"], np.float32(1)).astype(np.int8)
                    replace_input(model, feeds, f"{side}_cache", cache)
                    set_attribute(model, f"{prefix}_cache_dtype", "")
                    output = next(info for info in model.graph.output if info.name == f"{side}_cache_out")
                    output.CopyFrom(helper.make_tensor_value_info(output.name, onnx.TensorProto.INT8, cache.shape))
                with patch.dict(os.environ, {"ORT_ENABLE_XQA": "0"}):
                    portable = run_case(model, feeds)[0]
                with patch.dict(os.environ, {"ORT_ENABLE_XQA": "1"}):
                    accelerated = run_with_kernel(model, feeds, "XQA")[0]
                self.assertTrue(np.isfinite(accelerated["output"]).all())
                np.testing.assert_allclose(
                    accelerated["output"], portable["output"], atol=8e-4, rtol=5e-3, equal_nan=False
                )
                for side in ("key", "value"):
                    np.testing.assert_array_equal(accelerated[f"{side}_cache_out"], portable[f"{side}_cache_out"])

    def test_per_channel_scale_values_and_nonfinite_routing(self):
        # Zero, negative, subnormal and non-finite tables pin portable behaviour, reached through
        # the per-channel opt-out so the assertions describe one kernel.
        width = 256
        cases = {
            "all_zero": np.zeros(width, dtype=np.float32),
            "mixed_zero": np.tile(np.array([0, 1], dtype=np.float32), width // 2),
            "negative": np.full(width, -1, dtype=np.float32),
            "subnormal": np.full(width, np.nextafter(np.float32(0), np.float32(1)), dtype=np.float32),
            "nan": np.full(width, np.nan, dtype=np.float32),
            "infinity": np.full(width, np.inf, dtype=np.float32),
        }
        for cache_dtype in (np.uint8, np.int8, ml_dtypes.float8_e4m3fn):
            for label, channel_scale in cases.items():
                with self.subTest(cache_dtype=cache_dtype, scale=label):
                    model, feeds, _ = make_case(
                        width=width, heads=6, kv_heads=1, block_size=128, lengths=(1,), past=(0,)
                    )
                    feeds["query"][:] = 0
                    raw = np.tile(np.array([0, 1, -1, 0], dtype=np.float32), width // 4)
                    scale = channel_scale.reshape(1, 1, width)
                    replace_input(model, feeds, "k_scale", scale)
                    replace_input(model, feeds, "v_scale", np.ones_like(scale))
                    expected_cache = {}
                    for side, prefix in (("key", "k"), ("value", "v")):
                        feeds[side][:] = raw
                        cache = np.zeros((*feeds[f"{side}_cache"].shape[:-1], width), dtype=np.float32)
                        if cache_dtype == np.uint8:
                            cache = quantize(cache, np.ones(width, dtype=np.float32))
                        else:
                            cache = cache.astype(cache_dtype)
                            set_attribute(model, f"{prefix}_cache_dtype", "")
                        replace_input(model, feeds, f"{side}_cache", cache)
                        output = next(info for info in model.graph.output if info.name == f"{side}_cache_out")
                        output.CopyFrom(
                            helper.make_tensor_value_info(
                                output.name, helper.np_dtype_to_tensor_dtype(cache.dtype), cache.shape
                            )
                        )
                        if np.isfinite(scale).all():
                            divisor = channel_scale if side == "key" else np.ones(width, dtype=np.float32)
                            with np.errstate(over="ignore"):
                                scaled = np.divide(raw, divisor, out=np.zeros_like(raw), where=divisor != 0)
                            if cache_dtype == np.uint8:
                                expected_cache[side] = quantize(
                                    np.clip(scaled, -8, 7), np.ones(width, dtype=np.float32)
                                )
                            else:
                                lower, upper = (-128, 127) if cache_dtype == np.int8 else (-448, 448)
                                expected_cache[side] = np.clip(np.rint(scaled), lower, upper).astype(cache_dtype)
                    with patch.dict(
                        os.environ,
                        {"ORT_ENABLE_XQA": "1", "ORT_ENABLE_XQA_PER_CHANNEL_KV": "0"},
                    ):
                        actual = run_with_kernel(model, feeds, "DECODER_ATTENTION")[0]
                    if np.isfinite(scale).all():
                        self.assertTrue(np.isfinite(actual["output"]).all())
                        np.testing.assert_allclose(actual["output"], np.tile(raw, 6).reshape(1, -1), equal_nan=False)
                        page, offset = divmod(int(feeds["slot_mapping"][0]), 128)
                        for side, expected in expected_cache.items():
                            np.testing.assert_array_equal(actual[f"{side}_cache_out"][page, offset, 0], expected)

    def test_int4_scale_extremes(self):
        for magnitude in (2.0**-24, 1e10):
            with self.subTest(magnitude=magnitude):
                model, feeds, _ = make_case(
                    width=32,
                    lengths=(1,),
                    past=(0,),
                    quant_type="PER_TENSOR",
                    activation_dtype=ml_dtypes.bfloat16,
                )
                pattern = np.tile(np.array([-magnitude, magnitude], dtype=ml_dtypes.bfloat16), 32).reshape(1, 64)
                for side, prefix in (("key", "k"), ("value", "v")):
                    feeds[side][:] = pattern
                    replace_input(model, feeds, f"{prefix}_scale", np.array([1e-6], dtype=np.float32))
                actual = run_case(model, feeds)[0]
                raw = pattern.reshape(2, 32).astype(np.float32)
                biased = (np.clip(np.rint(raw / np.float32(1e-6)), -8, 7).astype(np.int8) + 8).astype(np.uint8)
                packed = biased[:, ::2] | (biased[:, 1::2] << 4)
                page, offset = divmod(int(feeds["slot_mapping"][0]), 16)
                for side in ("key", "value"):
                    np.testing.assert_array_equal(actual[f"{side}_cache_out"][page, offset], packed)

    def test_reject_latent_int4(self):
        model, feeds, _ = make_case(width=64, lengths=(1,), past=(0,), heads=4, kv_heads=1)
        set_attribute(model, "kv_cache_layout", "LATENT")
        set_attribute(model, "v_quant_type", "NONE")
        set_attribute(model, "v_cache_dtype", "")
        remove_input(model, feeds, "value")
        remove_input(model, feeds, "value_cache")
        remove_input(model, feeds, "v_scale")
        del model.graph.node[0].output[1:]
        del model.graph.output[1:]
        with self.assertRaisesRegex(Exception, "LATENT"):
            run_case(model, feeds)

    def test_int8_fp8_cache_regression(self):
        for cache_dtype, qmax in ((np.int8, 127), (ml_dtypes.float8_e4m3fn, 448)):
            for mode in ("PER_TENSOR", "PER_CHANNEL"):
                for lengths in ((1, 1), (33, 17)):
                    with self.subTest(cache_dtype=cache_dtype, mode=mode, lengths=lengths):
                        model, feeds, _ = make_case(width=128, lengths=lengths, int4=False)
                        reference_model = onnx.ModelProto.FromString(model.SerializeToString())
                        reference_feeds = {name: array.copy() for name, array in feeds.items()}
                        reference_feeds["slot_mapping"][:] = -1
                        expected_cache = {}
                        for side, prefix, index in (("key", "k", 14), ("value", "v", 15)):
                            cache_name = f"{side}_cache"
                            dense = feeds[cache_name].astype(np.float32)
                            current = feeds[side].reshape(-1, 2, 128).astype(np.float32)
                            scale = (
                                np.array([0.03125], dtype=np.float32)
                                if mode == "PER_TENSOR"
                                else np.linspace(0.02, 0.06, 256, dtype=np.float32).reshape(2, 1, 128)
                            )
                            scale_name = f"{prefix}_scale"
                            replace_input(model, feeds, scale_name, scale)
                            model.graph.node[0].input[index] = scale_name
                            divisor = scale.reshape(2, 128) if mode == "PER_CHANNEL" else scale

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
                            encoded_current = encode(current, divisor)
                            for token, slot in enumerate(feeds["slot_mapping"]):
                                page, offset = divmod(int(slot), 16)
                                cache[page, offset] = encoded_current[token]
                            expected_cache[f"{cache_name}_out"] = cache
                            reference_feeds[cache_name] = (cache.astype(np.float32) * divisor).astype(np.float16)
                            set_attribute(model, f"{prefix}_quant_type", mode)
                        actual = run_case(model, feeds)[0]
                        reference = run_case(reference_model, reference_feeds)[0]
                        np.testing.assert_allclose(actual["output"], reference["output"], atol=8e-4, rtol=5e-3)
                        for name, expected in expected_cache.items():
                            if cache_dtype == np.int8:
                                np.testing.assert_allclose(actual[name], expected, atol=1, rtol=0)
                            else:
                                lower = np.nextafter(expected, np.array(-448, dtype=cache_dtype)).astype(np.float32)
                                upper = np.nextafter(expected, np.array(448, dtype=cache_dtype)).astype(np.float32)
                                self.assertTrue(np.all(actual[name].astype(np.float32) >= lower))
                                self.assertTrue(np.all(actual[name].astype(np.float32) <= upper))


if __name__ == "__main__":
    unittest.main()
