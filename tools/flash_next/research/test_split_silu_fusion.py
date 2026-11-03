from copy import deepcopy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.inliner import inline_local_functions
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
import pytest

from fuse_split_silu import DOMAIN, OP, fuse


class ScaledSiLU(OpRun):
    op_domain = DOMAIN

    def _run(self, X, alpha=None):
        z = (X.astype(np.float32) * alpha).astype(X.dtype)
        f = z.astype(np.float32)
        e = np.exp(-np.abs(f))
        sigmoid = np.where(f >= 0, 1 / (1 + e), e / (1 + e)).astype(X.dtype)
        return ((z.astype(np.float32) * sigmoid.astype(np.float32)).astype(X.dtype),)


def fixture(rows=2, prefix=320, suffix=4, alpha=0.25, dtype=TensorProto.FLOAT16):
    vi = lambda n, s: helper.make_tensor_value_info(n, dtype, s)
    shape = [1, rows, prefix + suffix]
    prefix_shape = [1, rows, prefix]
    suffix_shape = [1, rows, suffix]
    return helper.make_model(helper.make_graph([
        helper.make_node("Split", ["X", "Sizes"], ["P", "Tail"], axis=-1),
        helper.make_node("ScaledSiLU", ["P"], ["Y"], domain=DOMAIN, alpha=alpha),
    ], "split-silu", [vi("X", shape)], [vi("Y", prefix_shape), vi("Tail", suffix_shape)],
        [numpy_helper.from_array(np.array([prefix, suffix], np.int64), "Sizes")],
        value_info=[vi("P", prefix_shape)]),
        opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid(DOMAIN, 1)])


@pytest.mark.parametrize("rows,prefix,suffix,alpha", [(1, 320, 4, .25), (16, 320, 4, .25),
                                                    (2, 7, 3, 1.), (1, 1, 1, -.5)])
def test_exact_expanded_reference(rows, prefix, suffix, alpha):
    model = fixture(rows, prefix, suffix, alpha)
    before = model.SerializeToString()
    fused, count = fuse(model)
    assert count == 1 and len(fused.graph.node) == 1
    assert model.SerializeToString() == before
    onnx.checker.check_model(fused)
    x = np.random.default_rng(42).uniform(-20, 20, size=(1, rows, prefix + suffix)).astype(np.float16)
    expected = ReferenceEvaluator(model, new_ops=[ScaledSiLU]).run(None, {"X": x})
    actual = ReferenceEvaluator(inline_local_functions(fused), new_ops=[ScaledSiLU]).run(None, {"X": x})
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a.view(np.uint16), b.view(np.uint16))


@pytest.mark.parametrize("case", ["shared", "exposed", "float32", "axis", "scale", "sizes"])
def test_unsupported_patterns_rejected(case):
    model = fixture(dtype=TensorProto.FLOAT if case == "float32" else TensorProto.FLOAT16)
    if case == "shared":
        model.graph.node.append(helper.make_node("Identity", ["P"], ["Extra"]))
    elif case == "exposed":
        model.graph.output.append(deepcopy(model.graph.value_info[0]))
    elif case == "axis":
        model.graph.node[0].attribute[0].i = 1
    elif case == "scale":
        model.graph.node[1].input.append("Scale")
    elif case == "sizes":
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.array([0, 324], np.int64), "Sizes"))
    with pytest.raises(ValueError, match="No eligible"):
        fuse(model)


def test_repeat_rejected():
    model, _ = fuse(fixture())
    with pytest.raises(ValueError, match="already defined"):
        fuse(model)


def test_constant_sizes_preserved():
    model = fixture()
    sizes = deepcopy(model.graph.initializer[0])
    del model.graph.initializer[:]
    nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend([helper.make_node("Constant", [], ["Sizes"], value=sizes), *nodes])
    fused, count = fuse(model)
    assert count == 1 and fused.graph.node[0] == model.graph.node[0]
    onnx.checker.check_model(fused)
