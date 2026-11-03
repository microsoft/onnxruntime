from copy import deepcopy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.inliner import inline_local_functions
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
import pytest

from fuse_gdn_prep import DOMAIN, OP, definition, fuse


class CausalConvWithState(OpRun):
    op_domain = DOMAIN

    def _run(self, X, W, Bias, State, activation=None, ndim=None, state_window=None):
        length = X.shape[-1]
        virtual = np.concatenate([State, X], axis=-1)
        out = np.zeros_like(X)
        for j in range(4):
            product = (virtual[..., j:j + length] * W[:, 0, j][None, :, None]).astype(np.float16)
            out = (out + product).astype(np.float16)
        out = (out + Bias[None, :, None]).astype(np.float16)
        out = (out / (np.float16(1) + np.exp(-out).astype(np.float16)).astype(np.float16)).astype(np.float16)
        return (out, virtual[..., -3:])


def fixture(batch=1, length=2):
    values = [
        ("Projection", TensorProto.FLOAT16, [batch, length, 16384]),
        ("AB", TensorProto.FLOAT16, [batch, length, 96]),
        ("W", TensorProto.FLOAT16, [10240, 1, 4]),
        ("Bias", TensorProto.FLOAT16, [10240]),
        ("State", TensorProto.FLOAT16, [batch, 10240, 3]),
    ]
    outputs = [
        ("Conv", TensorProto.FLOAT16, [batch, length, 10240]),
        ("Z", TensorProto.FLOAT16, [batch, length, 6144]),
        ("A32", TensorProto.FLOAT, [batch, length, 48]),
        ("B32", TensorProto.FLOAT, [batch, length, 48]),
        ("Present", TensorProto.FLOAT16, [batch, 10240, 3]),
    ]
    infos = [helper.make_tensor_value_info(*v) for v in values]
    out = [helper.make_tensor_value_info(*v) for v in outputs]
    model = helper.make_model(helper.make_graph([
        helper.make_node(OP, [v[0] for v in values], [v[0] for v in outputs], domain=DOMAIN),
    ], "gdn-prep", infos, out), functions=[definition(22)],
        opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid(DOMAIN, 1)])
    return model


def original_fixture(batch=1, length=2):
    model = fixture(batch, length)
    graph = inline_local_functions(model)
    # Add a harmless consumer with the exact packed GDN input contract used by the matcher.
    graph.graph.node.append(helper.make_node("GatedDeltaNet",
        ["Conv", "", "", "", "A32", "B32"], ["Ignored"], domain=DOMAIN))
    for node in graph.graph.node:
        if node.op_type == "CausalConvWithState":
            prefix = node.input[0]
            weight = node.input[1]
            break
    before = next(n for n in graph.graph.node if prefix in n.output)
    split = next(n for n in graph.graph.node if before.input[0] in n.output)
    names = {v.name for v in graph.graph.value_info}
    for name in [split.output[0], before.output[0]]:
        if name not in names:
            graph.graph.value_info.append(helper.make_tensor_value_info(name, TensorProto.FLOAT16, None))
    return graph


@pytest.mark.parametrize("batch,length", [(1, 1), (2, 2), (1, 4), (1, 0)])
def test_exact_function_expansion(batch, length):
    model = fixture(batch, length)
    original = inline_local_functions(model)
    onnx.checker.check_model(model)
    onnx.checker.check_model(original)
    rng = np.random.default_rng(42)
    feeds = {v.name: rng.uniform(-.2, .2, [d.dim_value for d in v.type.tensor_type.shape.dim]).astype(np.float16)
             for v in model.graph.input}
    projection, ab = feeds["Projection"], feeds["AB"]
    qkv = projection[..., :10240].transpose(0, 2, 1)
    conv, present = CausalConvWithState._run(
        None, qkv, feeds["W"], feeds["Bias"], feeds["State"],
        activation="silu", ndim=1, state_window=0)
    expected = [conv.transpose(0, 2, 1), projection[..., 10240:],
                ab[..., :48].astype(np.float32), ab[..., 48:].astype(np.float32), present]
    actual = ReferenceEvaluator(inline_local_functions(model), new_ops=[CausalConvWithState]).run(None, feeds)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_actual_chain_rewrite_and_original_unchanged():
    model = original_fixture()
    before = model.SerializeToString()
    fused, count = fuse(model)
    assert count == 1
    assert model.SerializeToString() == before
    assert sum(n.op_type == OP for n in fused.graph.node) == 1
    assert not any(n.op_type in ("Transpose", "Cast", "Split", "CausalConvWithState") for n in fused.graph.node)


@pytest.mark.parametrize("case", ["shared", "exposed", "activation", "channels", "width", "cast", "explicit_qkv"])
def test_unsupported_chains_not_claimed(case):
    model = original_fixture()
    conv = next(n for n in model.graph.node if n.op_type == "CausalConvWithState")
    before = next(n for n in model.graph.node if conv.input[0] in n.output)
    if case == "shared":
        model.graph.node.append(helper.make_node("Identity", [before.output[0]], ["Extra"]))
    elif case == "exposed":
        model.graph.output.append(helper.make_tensor_value_info(before.output[0], TensorProto.FLOAT16, None))
    elif case == "activation":
        next(a for a in conv.attribute if a.name == "activation").s = b"none"
    elif case == "channels":
        conv.attribute.append(helper.make_attribute("channels_last", 1))
    elif case == "width":
        model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value = 100
    elif case == "cast":
        next(n for n in model.graph.node if n.op_type == "Cast").attribute[0].i = TensorProto.FLOAT16
    elif case == "explicit_qkv":
        model.graph.node[-1].input[1] = "Key"
    with pytest.raises(ValueError, match="No eligible"):
        fuse(model)


def test_repeat_definition_rejected():
    model, _ = fuse(original_fixture())
    with pytest.raises(ValueError, match="already defined"):
        fuse(model)
