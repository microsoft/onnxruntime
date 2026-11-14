# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
"""Parity tests for com.microsoft.MultiHeadLatentAttention on the CPU execution provider.

The op is checked against a float64 numpy reference, and against GroupQueryAttention with kv_num_heads=1,
value=key and past_value=past_key followed by a slice of channels [v_offset, v_offset + v_head_size) of every
output head, which is how absorbed MLA is expressed with GroupQueryAttention.
"""

import unittest
from dataclasses import dataclass

import numpy
from onnx import TensorProto, helper

from onnxruntime import InferenceSession, OrtValue, SessionOptions
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument


@dataclass(frozen=True)
class Shape:
    name: str
    num_heads: int
    head_size: int
    v_head_size: int
    v_offset: int
    scale: float


# DeepSeek-V2-Lite and DeepSeek-V3: 512 latent channels then 64 RoPE channels, scale 1/sqrt(128 + 64).
DEEPSEEK_V2_LITE = Shape("deepseek_v2_lite", 16, 576, 512, 0, 1 / numpy.sqrt(192))
DEEPSEEK_V3 = Shape("deepseek_v3", 128, 576, 512, 0, 1 / numpy.sqrt(192))
MID_ROW_VALUE = Shape("mid_row_value", 8, 96, 64, 16, 0.125)
ODD_SIZES = Shape("odd_sizes", 3, 37, 21, 11, 0.2)
SHAPES = [DEEPSEEK_V2_LITE, DEEPSEEK_V3, MID_ROW_VALUE, ODD_SIZES]

TYPES = {"fp32": (numpy.float32, TensorProto.FLOAT, 2e-5), "fp16": (numpy.float16, TensorProto.FLOAT16, 2e-3)}


def make_session(nodes, inputs, outputs, threads=0):
    graph = helper.make_graph(nodes, "graph", inputs, outputs)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    options = SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    return InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def mla_session(shape: Shape, ort_type, has_past=True, has_present=True, threads=0):
    h, d, dv = shape.num_heads, shape.head_size, shape.v_head_size
    vi = helper.make_tensor_value_info
    inputs = [vi("query", ort_type, ["batch", "seq", h * d]), vi("key", ort_type, ["batch", "seq", d])]
    if has_past:
        inputs.append(vi("past_kv", ort_type, ["batch", 1, "past_len", d]))
    inputs += [vi("seqlens_k", TensorProto.INT32, ["batch"]), vi("total_sequence_length", TensorProto.INT32, [1])]
    outputs = [vi("output", ort_type, ["batch", "seq", h * dv])]
    if has_present:
        outputs.append(vi("present_kv", ort_type, ["batch", 1, "present_len", d]))
    node = helper.make_node(
        "MultiHeadLatentAttention",
        ["query", "key", "past_kv" if has_past else "", "seqlens_k", "total_sequence_length"],
        [o.name for o in outputs],
        domain="com.microsoft",
        num_heads=h,
        v_head_size=dv,
        v_offset=shape.v_offset,
        scale=float(shape.scale),
    )
    return make_session([node], inputs, outputs, threads)


def gqa_session(shape: Shape, ort_type, has_past=True):
    """GroupQueryAttention with one KV head and value = key: absorbed MLA without this op."""
    h, d = shape.num_heads, shape.head_size
    vi = helper.make_tensor_value_info
    inputs = [vi("query", ort_type, ["batch", "seq", h * d]), vi("key", ort_type, ["batch", "seq", d])]
    if has_past:
        inputs.append(vi("past_kv", ort_type, ["batch", 1, "past_len", d]))
    inputs += [vi("seqlens_k", TensorProto.INT32, ["batch"]), vi("total_sequence_length", TensorProto.INT32, [1])]
    outputs = [
        vi("output", ort_type, ["batch", "seq", h * d]),
        vi("present_key", ort_type, ["batch", 1, "present_len", d]),
        vi("present_value", ort_type, ["batch", 1, "present_len", d]),
    ]
    past = "past_kv" if has_past else ""
    node = helper.make_node(
        "GroupQueryAttention",
        ["query", "key", "key", past, past, "seqlens_k", "total_sequence_length"],
        ["output", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=h,
        kv_num_heads=1,
        scale=float(shape.scale),
        do_rotary=0,
    )
    return make_session([node], inputs, outputs)


def mla_reference(shape: Shape, query, key, past_kv, seqlens_k, total_sequence_length):
    """float64 reference with GroupQueryAttention's cache rules.

    Returns (output, present, written): output is NaN for padding tokens of a first prompt (unspecified), present has
    max(total_sequence_length, past length) rows, and written marks the rows of present the op defines.
    """
    h, d, dv, off = shape.num_heads, shape.head_size, shape.v_head_size, shape.v_offset
    batch, seq = key.shape[0], key.shape[1]
    q = query.astype(numpy.float64).reshape(batch, seq, h, d)
    past_len = 0 if past_kv is None else past_kv.shape[2]
    present_len = max(total_sequence_length, past_len)
    present = numpy.zeros((batch, 1, present_len, d))
    written = numpy.zeros((batch, present_len), dtype=bool)
    output = numpy.full((batch, seq, h, dv), numpy.nan)
    first_prompt = seq == total_sequence_length
    for b in range(batch):
        total = int(seqlens_k[b]) + 1
        past = 0 if first_prompt else total - seq
        if past:
            present[b, 0, :past] = past_kv[b, 0, :past]
        present[b, 0, past : past + seq] = key[b]
        written[b, : past + seq] = True
        rows = present[b, 0]
        for i in range(seq):
            if first_prompt and i >= total:
                continue
            limit = min(past + i + 1, total)
            scores = q[b, i] @ rows[:limit].T * shape.scale
            p = numpy.exp(scores - scores.max(axis=-1, keepdims=True))
            output[b, i] = (p / p.sum(axis=-1, keepdims=True)) @ rows[:limit, off : off + dv]
    return output.reshape(batch, seq, h * dv), present, written


def random_inputs(shape: Shape, dtype, batch, seq, past_len, seed):
    rng = numpy.random.default_rng(seed)
    query = (0.5 * rng.standard_normal((batch, seq, shape.num_heads * shape.head_size))).astype(dtype)
    key = rng.standard_normal((batch, seq, shape.head_size)).astype(dtype)
    past_kv = rng.standard_normal((batch, 1, past_len, shape.head_size)).astype(dtype)
    return query, key, past_kv


def run_shared(session, feeds, cache):
    """Runs with past_kv and present_kv bound to one buffer, `cache`, which is updated in place."""
    cache_value = OrtValue.ortvalue_from_numpy(cache)
    binding = session.io_binding()
    for name, value in feeds.items():
        binding.bind_cpu_input(name, value)
    binding.bind_ortvalue_input("past_kv", cache_value)
    binding.bind_output("output")
    binding.bind_ortvalue_output("present_kv", cache_value)
    session.run_with_iobinding(binding)
    return binding.copy_outputs_to_cpu()[0]


class TestMultiHeadLatentAttentionCPU(unittest.TestCase):
    def assert_close(self, actual, expected, tol, msg=""):
        real = ~numpy.isnan(expected)
        scale = max(numpy.abs(expected[real]).max(), 1e-3)
        err = numpy.abs(actual.astype(numpy.float64)[real] - expected[real]).max() / scale
        self.assertLess(err, tol, f"{msg}: relative error {err:.2e}")

    def assert_present(self, present, expected, written, msg=""):
        self.assertEqual(present.shape, expected.shape, msg)
        for b in range(written.shape[0]):
            rows = written[b]
            numpy.testing.assert_array_equal(
                present[b, 0, rows], expected[b, 0, rows].astype(present.dtype), err_msg=f"{msg}: present_kv[{b}]"
            )

    def check_reference(self, shape, type_name, batch, seq, past_len, seqlens_k, total, threads=0, shared=False):
        """Runs one call and compares output and present_kv with the float64 reference."""
        dtype, ort_type, tol = TYPES[type_name]
        msg = f"{shape.name} {type_name} B={batch} S={seq} L={past_len} seqlens_k={seqlens_k} total={total}"
        query, key, past_kv = random_inputs(shape, dtype, batch, seq, past_len, seed=batch * 1000 + seq + past_len)
        seqlens = numpy.array(seqlens_k, numpy.int32)
        feeds = {
            "query": query,
            "key": key,
            "seqlens_k": seqlens,
            "total_sequence_length": numpy.array([total], numpy.int32),
        }
        expected, present, written = mla_reference(shape, query, key, past_kv, seqlens, total)
        session = mla_session(shape, ort_type, threads=threads)
        if shared:
            cache = past_kv.copy()
            output = run_shared(session, feeds, cache)
            self.assert_present(cache, present, written, msg)
            # Rows after the written ones keep their contents.
            for b in range(batch):
                end = int(written[b].sum())
                numpy.testing.assert_array_equal(cache[b, 0, end:], past_kv[b, 0, end:], err_msg=msg)
        else:
            output, present_kv = session.run(None, dict(feeds, past_kv=past_kv))
            self.assert_present(present_kv, present, written, msg)
        self.assert_close(output, expected, tol, msg)

    def test_reference_decode(self):
        for shape in SHAPES:
            for type_name in TYPES:
                with self.subTest(shape=shape.name, type=type_name):
                    # Preallocated cache, ragged batch.
                    self.check_reference(shape, type_name, 3, 1, 150, [99, 5, 149], 150)
                    # Growing cache.
                    self.check_reference(shape, type_name, 2, 1, 70, [70, 33], 71)

    def test_reference_prefill(self):
        for shape in SHAPES:
            for type_name in TYPES:
                with self.subTest(shape=shape.name, type=type_name):
                    seq = 4 if shape.num_heads > 64 else 33
                    # First prompt into a preallocated cache, one entry right-padded.
                    self.check_reference(shape, type_name, 2, seq, 64, [seq - 1, seq // 2], seq, shared=True)
                    # A later chunk of a long prompt, growing and preallocated caches.
                    self.check_reference(shape, type_name, 1, seq, 100, [100 + seq - 1], 100 + seq)
                    self.check_reference(shape, type_name, 1, seq, 256, [100 + seq - 1], 256, shared=True)

    def test_reference_first_prompt_without_past(self):
        for shape in [MID_ROW_VALUE, ODD_SIZES]:
            for type_name, (dtype, ort_type, tol) in TYPES.items():
                with self.subTest(shape=shape.name, type=type_name):
                    query, key, _ = random_inputs(shape, dtype, 2, 9, 0, seed=7)
                    seqlens = numpy.array([8, 4], numpy.int32)
                    feeds = {"query": query, "key": key, "seqlens_k": seqlens}
                    feeds["total_sequence_length"] = numpy.array([9], numpy.int32)
                    expected, present, written = mla_reference(shape, query, key, None, seqlens, 9)
                    for has_present in (True, False):
                        session = mla_session(shape, ort_type, has_past=False, has_present=has_present)
                        results = session.run(None, feeds)
                        self.assert_close(results[0], expected, tol, f"{shape.name} {type_name}")
                        if has_present:
                            self.assert_present(results[1], present, written, f"{shape.name} {type_name}")

    def test_long_context(self):
        # 8K cached rows: decode splits the cache across threads and merges the partial softmax states.
        for type_name in TYPES:
            with self.subTest(type=type_name):
                self.check_reference(DEEPSEEK_V2_LITE, type_name, 1, 1, 8192, [8191], 8192, threads=4, shared=True)
                self.check_reference(DEEPSEEK_V2_LITE, type_name, 2, 1, 8192, [8191, 3000], 8192, threads=4)
                self.check_reference(DEEPSEEK_V2_LITE, type_name, 1, 16, 8192, [8000], 8192, threads=4, shared=True)

    def test_generation_steps(self):
        """A right-padded first prompt, then decode steps, all in one shared preallocated buffer."""
        shape = MID_ROW_VALUE
        for type_name, (dtype, ort_type, tol) in TYPES.items():
            with self.subTest(type=type_name):
                session = mla_session(shape, ort_type, threads=2)
                rng = numpy.random.default_rng(11)
                capacity, prompt = 48, 10
                lengths = numpy.array([prompt, 6], numpy.int32)
                cache = numpy.zeros((2, 1, capacity, shape.head_size), dtype)
                ref_cache = cache.astype(numpy.float64)
                for step in range(6):
                    seq = prompt if step == 0 else 1
                    total = prompt if step == 0 else int(lengths.max())
                    query = (0.5 * rng.standard_normal((2, seq, shape.num_heads * shape.head_size))).astype(dtype)
                    key = rng.standard_normal((2, seq, shape.head_size)).astype(dtype)
                    seqlens = lengths - 1
                    feeds = {"query": query, "key": key, "seqlens_k": seqlens}
                    feeds["total_sequence_length"] = numpy.array([total], numpy.int32)
                    expected, ref_present, _ = mla_reference(shape, query, key, ref_cache, seqlens, total)
                    output = run_shared(session, feeds, cache)
                    self.assert_close(output, expected, tol, f"{type_name} step {step}")
                    ref_cache = ref_present
                    lengths = lengths + 1

    def test_gqa_oracle(self):
        """Matches GroupQueryAttention(kv_num_heads=1, value=key) sliced to V in every cache mode."""
        # (label, batch, seq, past_len, filled past rows per entry or None for absent past, real new rows, total)
        cases = [
            ("first prompt, preallocated", 2, 8, 64, [0, 0], [8, 8], 8),
            ("first prompt, right-padded", 3, 8, 64, [0, 0, 0], [8, 5, 1], 8),
            ("first prompt, empty past", 1, 8, 0, [0], [8], 8),
            ("first prompt, no past", 2, 8, None, [0, 0], [8, 3], 8),
            ("decode, growing", 2, 1, 40, [40, 17], [1, 1], 41),
            ("chunk, growing", 1, 8, 40, [40], [8], 48),
            ("decode, preallocated", 2, 1, 64, [40, 63 - 1], [1, 1], 64),
            ("chunk, preallocated", 1, 8, 64, [40], [8], 64),
        ]
        for shape in [DEEPSEEK_V2_LITE, MID_ROW_VALUE]:
            for type_name, (dtype, ort_type, tol) in TYPES.items():
                # GroupQueryAttention's fp16 path runs fp16 GEMMs on CPUs with fp16 arithmetic (ARM64), so it is
                # only accurate to a few 1e-3 there. Accuracy is checked against the float64 reference above.
                oracle_tol = tol if type_name == "fp32" else 1e-2
                for label, batch, seq, past_len, past_rows, new_rows, total in cases:
                    with self.subTest(shape=shape.name, type=type_name, case=label):
                        has_past = past_len is not None
                        query, key, past_kv = random_inputs(
                            shape, dtype, batch, seq, past_len or 0, seed=seq * 7 + batch
                        )
                        seqlens = numpy.array(
                            [p + n - 1 for p, n in zip(past_rows, new_rows, strict=True)], numpy.int32
                        )
                        feeds = {"query": query, "key": key, "seqlens_k": seqlens}
                        feeds["total_sequence_length"] = numpy.array([total], numpy.int32)
                        if has_past:
                            feeds["past_kv"] = past_kv
                        gqa_out, gqa_present, _ = gqa_session(shape, ort_type, has_past).run(None, feeds)
                        out, present = mla_session(shape, ort_type, has_past).run(None, feeds)

                        h, d, dv, off = shape.num_heads, shape.head_size, shape.v_head_size, shape.v_offset
                        expected = gqa_out.reshape(batch, seq, h, d)[..., off : off + dv].astype(numpy.float64)
                        if seq == total:  # padding tokens of a first prompt are unspecified
                            for b in range(batch):
                                expected[b, new_rows[b] :] = numpy.nan
                        self.assert_close(out.reshape(batch, seq, h, dv), expected, oracle_tol, label)
                        self.assertEqual(present.shape, gqa_present.shape, label)
                        for b in range(batch):
                            rows = past_rows[b] + seq if seq != total else seq
                            numpy.testing.assert_array_equal(present[b, 0, :rows], gqa_present[b, 0, :rows], label)

    def test_errors_match_gqa(self):
        """Lengths GroupQueryAttention rejects are rejected too."""
        shape = MID_ROW_VALUE
        # (label, seq, past_len, seqlens_k, total, expected message)
        cases = [
            ("full cache", 1, 32, [32], 32, "out of range"),
            ("seqlens_k too small", 4, 32, [2], 32, "too small for sequence_length"),
            ("growing cache not grown by sequence_length", 1, 32, [32], 40, "total_sequence_length must equal"),
            ("total_sequence_length less than sequence_length", 4, 32, [3], 3, "must be at least sequence_length"),
        ]
        for label, seq, past_len, seqlens_k, total, message in cases:
            with self.subTest(case=label):
                query, key, past_kv = random_inputs(shape, numpy.float32, 1, seq, past_len, seed=3)
                feeds = {
                    "query": query,
                    "key": key,
                    "past_kv": past_kv,
                    "seqlens_k": numpy.array(seqlens_k, numpy.int32),
                    "total_sequence_length": numpy.array([total], numpy.int32),
                }
                with self.assertRaisesRegex(InvalidArgument, message):
                    mla_session(shape, TensorProto.FLOAT).run(None, feeds)
                with self.assertRaises(InvalidArgument):
                    gqa_session(shape, TensorProto.FLOAT).run(None, feeds)


if __name__ == "__main__":
    unittest.main()
