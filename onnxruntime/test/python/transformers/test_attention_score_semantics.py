# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Numerical and interface invariants for Python Attention/MHA fusion."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from bert_model_generator import create_bert_attention
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

import onnxruntime as ort

source = Path(__file__).resolve().parents[3] / "python/tools/transformers"
if source.is_dir():
    sys.path.insert(0, str(source))
if importlib.util.find_spec("fusion_attention") is not None:
    from fusion_attention import AttentionMask, FusionAttention
    from fusion_options import AttentionMaskFormat
    from onnx_model import OnnxModel
else:
    from onnxruntime.transformers.fusion_attention import AttentionMask, FusionAttention
    from onnxruntime.transformers.fusion_options import AttentionMaskFormat
    from onnxruntime.transformers.onnx_model import OnnxModel


def native(model, feeds, all_outputs=False):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    outputs = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"]).run(None, feeds)
    return outputs if all_outputs else outputs[0]


def make(case, batch=2, sequence=2, mask_value=-10000):
    """Build a valid graph and independently evaluate its attention equation."""
    model = create_bert_attention(input_hidden_size=4, num_heads=2, pruned_qk_hidden_size=4, pruned_v_hidden_size=4)
    model.ir_version = 8
    del model.opset_import[:]
    model.opset_import.extend([helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)])
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "epsion":
                attr.name = "epsilon"
    for inp in model.graph.input:
        dims = [batch, sequence] if inp.name == "input_mask" else [batch, sequence, 4]
        inp.CopyFrom(helper.make_tensor_value_info(inp.name, inp.type.tensor_type.elem_type, dims))
    del model.graph.output[:]
    model.graph.output.append(helper.make_tensor_value_info("reshape_qkv_out", TensorProto.FLOAT, [batch, sequence, 4]))
    rng = np.random.default_rng(20261017)
    for tensor in model.graph.initializer:
        if tensor.name.startswith("matmul_") or tensor.name.startswith("add_"):
            tensor.CopyFrom(
                numpy_helper.from_array(rng.normal(0, 0.5, tuple(tensor.dims)).astype(np.float32), tensor.name)
            )
    feeds = {
        "input_1": rng.normal(size=(batch, sequence, 4)).astype(np.float32),
        "input_2": rng.normal(size=(batch, sequence, 4)).astype(np.float32),
        "input_mask": np.ones((batch, sequence), np.int64),
    }
    feeds["input_mask"][-1, -1] = 0
    weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    next(t for t in model.graph.initializer if t.name == "mul_weight").CopyFrom(
        numpy_helper.from_array(np.array([mask_value], np.float32), "mul_weight")
    )
    if case in ["scale", "runtime_scale", "no_mask_scale"]:
        if case in ["scale", "no_mask_scale"]:
            t = next(t for t in model.graph.initializer if t.name == "div_weight")
            t.CopyFrom(numpy_helper.from_array(np.array([2], np.float32), "div_weight"))
        else:
            model.graph.input.append(helper.make_tensor_value_info("div_weight", TensorProto.FLOAT, [1]))
            feeds["div_weight"] = np.array([2], np.float32)
    axis = (
        int(case[-1])
        if case in ("softmax_axis0", "softmax_axis1", "softmax_axis2")
        else (-1 if case == "negative_axis" else 3)
    )
    perms = {
        "transpose_q": [0, 2, 1, 3],
        "transpose_k": [0, 2, 3, 1],
        "transpose_v": [0, 2, 1, 3],
        "transpose_qkv": [0, 2, 1, 3],
    }
    if case in perms:
        perms[case] = [0, 1, 2, 3]
    for node in model.graph.node:
        if node.name == "softmax_qk":
            next(a for a in node.attribute if a.name == "axis").i = axis
        if node.name in perms:
            next(a for a in node.attribute if a.name == "perm").ints[:] = perms[node.name]
    if case == "default_axis":
        node = next(n for n in model.graph.node if n.name == "softmax_qk")
        del node.attribute[:]
    if case == "double_k_transpose":
        node = next(n for n in model.graph.node if n.name == "transpose_k")
        next(a for a in node.attribute if a.name == "perm").ints[:] = [0, 2, 1, 3]
        node.output[0] = "k_intermediate"
        model.graph.node.insert(
            list(model.graph.node).index(node) + 1,
            helper.make_node("Transpose", ["k_intermediate"], ["transpose_k_out"], "k_second", perm=[0, 1, 3, 2]),
        )
    score_node = next(n for n in model.graph.node if n.name == "div_qk")
    multiplier = None
    if case in (
        "multiply",
        "multiply_mask_first",
        "negative_scale",
        "zero_scale",
        "zero_scale_direct",
        "multiply_direct",
        "reversed_div",
    ):
        multiplier = {
            "multiply": 0.5,
            "multiply_mask_first": 0.5,
            "negative_scale": -0.5,
            "zero_scale": 0.0,
            "zero_scale_direct": 0.0,
            "multiply_direct": 0.5,
            "reversed_div": 2.0,
        }[case]
        tensor = next(t for t in model.graph.initializer if t.name == "div_weight")
        tensor.CopyFrom(numpy_helper.from_array(np.array([multiplier], np.float32), "div_weight"))
        if case == "reversed_div":
            score_node.input.reverse()
        else:
            score_node.op_type = "Mul"
            if not case.endswith("_direct"):
                score_node.input.reverse()  # Commuted scalar multiplication is valid.
    if case == "multiply_mask_first":
        next(n for n in model.graph.node if n.name == "add_qk").input.reverse()
    if case in ("no_mask", "no_mask_scale"):
        next(n for n in model.graph.node if n.name == "softmax_qk").input[0] = "div_qk_out"
        # Keep the public input interface free of an unused mask input.
        model.graph.input.remove(next(v for v in model.graph.input if v.name == "input_mask"))
    if case == "query_scale":
        model.graph.initializer.append(numpy_helper.from_array(np.array([2], np.float32), "query_divisor"))
        qk = next(n for n in model.graph.node if n.name == "matmul_qk")
        qk.input[0] = "scaled_query"
        model.graph.node.insert(
            list(model.graph.node).index(qk),
            helper.make_node("Div", ["transpose_q_out", "query_divisor"], ["scaled_query"]),
        )
    sdpa = case in ("sdpa", "sdpa_asymmetric")
    if sdpa:
        qk = next(n for n in model.graph.node if n.name == "matmul_qk")
        new_nodes = []
        for index, branch in enumerate("qk"):
            value = 0.5 if case == "sdpa" else (0.25 if branch == "q" else 1.0)
            model.graph.initializer.append(
                numpy_helper.from_array(np.array(value, np.float32), branch + "_scale_squared")
            )
            new_nodes.extend(
                [
                    helper.make_node("Sqrt", [branch + "_scale_squared"], [branch + "_scale"]),
                    helper.make_node("Mul", ["transpose_" + branch + "_out", branch + "_scale"], ["scaled_" + branch]),
                ]
            )
            qk.input[index] = "scaled_" + branch
        position = list(model.graph.node).index(qk)
        for offset, new_node in enumerate(new_nodes):
            model.graph.node.insert(position + offset, new_node)
        next(n for n in model.graph.node if n.name == "add_qk").input[0] = "matmul_qk_out"
        model.graph.node.remove(score_node)
    x = feeds["input_1"] + feeds["input_2"]
    centered = x - x.mean(axis=-1, keepdims=True)
    normalized = (
        centered
        / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + np.float32(1e-5))
        * weights["layer_norm_weight"]
        + weights["layer_norm_bias"]
    )
    q, k, v = [
        (normalized @ weights["matmul_" + p + "_weight"] + weights["add_" + p + "_weight"])
        .reshape(batch, sequence, 2, 2)
        .transpose(perms["transpose_" + p])
        for p in "qkv"
    ]
    if case == "query_scale":
        q = q / np.float32(2)
    if sdpa:
        q = q * np.sqrt(np.float32(0.5 if case == "sdpa" else 0.25))
        k = k * np.sqrt(np.float32(0.5 if case == "sdpa" else 1.0))
    scores = q @ k
    if multiplier is not None:
        scores = np.float32(multiplier) / scores if case == "reversed_div" else scores * np.float32(multiplier)
    elif not sdpa:
        scores = scores / (
            np.float32(2) if case in ["scale", "runtime_scale", "no_mask_scale"] else weights["div_weight"]
        )
    if case not in ("no_mask", "no_mask_scale"):
        scores += (1 - feeds["input_mask"][:, None, None, :]).astype(np.float32) * np.float32(mask_value)
    else:
        # Remove only the mask branch; retain the post-attention fusion anchor.
        for node in list(model.graph.node):
            if node.name in ("unsqueeze0", "unsqueeze1", "cast", "sub", "mul_mask", "add_qk"):
                model.graph.node.remove(node)
        del feeds["input_mask"]
    probabilities = np.exp(scores - scores.max(axis=axis, keepdims=True))
    probabilities /= probabilities.sum(axis=axis, keepdims=True)
    expected = (probabilities @ v).transpose(perms["transpose_qkv"]).reshape(batch, sequence, 4)
    return model, feeds, expected


def optimize(model, use_mha):
    wrapped = OnnxModel(model)
    mask = AttentionMask(wrapped)
    mask.set_mask_format(AttentionMaskFormat.AttentionMask)
    FusionAttention(wrapped, 4, 2, attention_mask=mask, use_multi_head_attention=use_mha).apply()
    wrapped.topological_sort()
    onnx.checker.check_model(wrapped.model)
    return wrapped


@pytest.mark.parametrize("use_mha", [False, True])
@pytest.mark.parametrize(
    "case,expected_fused",
    [
        ("control", 1),
        ("scale", 1),
        ("runtime_scale", 0),
        ("softmax_axis0", 0),
        ("softmax_axis1", 0),
        ("softmax_axis2", 0),
        ("transpose_q", 0),
        ("transpose_k", 0),
        ("transpose_v", 0),
        ("transpose_qkv", 0),
        ("negative_axis", 1),
        ("default_axis", 1),
        ("double_k_transpose", 1),
        ("multiply", 1),
        ("multiply_direct", 1),
        ("multiply_mask_first", 1),
        ("negative_scale", 1),
        ("zero_scale", 0),
        ("zero_scale_direct", 0),
        ("reversed_div", 0),
        ("query_scale", 1),
        ("sdpa", 1),
        ("sdpa_asymmetric", 1),
        ("no_mask", 1),
        ("no_mask_scale", 1),
    ],
)
def test_attention_score_and_layout_semantics(case, expected_fused, use_mha):
    model, feeds, expected = make(case)
    onnx.checker.check_model(model)
    np.testing.assert_allclose(native(model, feeds), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ReferenceEvaluator(model).run(None, feeds)[0], expected, rtol=1e-5, atol=1e-5)
    wrapped = optimize(model, use_mha)
    assert {v.name for v in wrapped.model.graph.input} == set(feeds)
    np.testing.assert_allclose(native(wrapped.model, feeds), expected, rtol=1e-5, atol=1e-5)
    assert len(wrapped.get_nodes_by_op_type("MultiHeadAttention" if use_mha else "Attention")) == expected_fused


@pytest.mark.parametrize("use_mha", [False, True])
@pytest.mark.parametrize("batch,sequence", [(1, 3), (3, 5)])
def test_nondegenerate_dimensions_still_fuse(batch, sequence, use_mha):
    model, feeds, expected = make("control", batch=batch, sequence=sequence)
    onnx.checker.check_model(model)
    np.testing.assert_allclose(native(model, feeds), expected, rtol=1e-5, atol=1e-5)
    wrapped = optimize(model, use_mha)
    np.testing.assert_allclose(native(wrapped.model, feeds), expected, rtol=1e-5, atol=1e-5)
    assert len(wrapped.get_nodes_by_op_type("MultiHeadAttention" if use_mha else "Attention")) == 1


def prefix_graph(graph, prefix, shared=()):
    def rename(name):
        return name if name in shared else prefix + name

    for node in graph.node:
        node.name = prefix + node.name
        for i, name in enumerate(node.input):
            node.input[i] = rename(name)
        for i, name in enumerate(node.output):
            node.output[i] = rename(name)
    for tensor in graph.initializer:
        tensor.name = rename(tensor.name)
    for value in list(graph.input) + list(graph.output):
        value.name = rename(value.name)
    graph.name = prefix + graph.name


@pytest.mark.parametrize("use_mha", [False, True])
def test_mask_filter_is_local_to_each_fusion(use_mha):
    models = []
    feeds = {}
    expected = []
    for prefix, mask_value in [("first_", -2), ("second_", -10000)]:
        model, inputs, reference = make("control", mask_value=mask_value)
        prefix_graph(model.graph, prefix)
        models.append(model)
        feeds.update({prefix + k: v for k, v in inputs.items()})
        expected.append(reference)
    model = models[0]
    for field in ["node", "initializer", "input", "output"]:
        getattr(model.graph, field).extend(getattr(models[1].graph, field))
    onnx.checker.check_model(model)
    for actual, reference in zip(native(model, feeds, True), expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
    wrapped = optimize(model, use_mha)
    for actual, reference in zip(native(wrapped.model, feeds, True), expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
    assert len(wrapped.get_nodes_by_op_type("MultiHeadAttention" if use_mha else "Attention")) == 2


@pytest.mark.parametrize("use_mha", [False, True])
@pytest.mark.parametrize("condition", [False, True])
def test_sibling_scale_bindings_are_not_interchangeable(use_mha, condition):
    models = []
    feeds = {}
    inputs = []
    expected = []
    for prefix, case in [("then_", "scale"), ("else_", "control")]:
        model, values, reference = make(case)
        prefix_graph(model.graph, prefix, shared=["div_weight"])
        inputs.extend(model.graph.input)
        del model.graph.input[:]
        models.append(model)
        feeds.update({prefix + k: v for k, v in values.items()})
        expected.append(reference)
    inputs.append(helper.make_tensor_value_info("condition", TensorProto.BOOL, []))
    feeds["condition"] = np.array(condition)
    graph = helper.make_graph(
        [helper.make_node("If", ["condition"], ["Y"], then_branch=models[0].graph, else_branch=models[1].graph)],
        "sibling_attention_scopes",
        inputs,
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2, 4])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)], ir_version=8
    )
    onnx.checker.check_model(model)
    reference = expected[0 if condition else 1]
    np.testing.assert_allclose(native(model, feeds), reference, rtol=1e-5, atol=1e-5)
    wrapped = optimize(model, use_mha)
    np.testing.assert_allclose(native(wrapped.model, feeds), reference, rtol=1e-5, atol=1e-5)
    assert not wrapped.get_nodes_by_op_type("MultiHeadAttention" if use_mha else "Attention")
