from copy import deepcopy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.inliner import inline_local_functions
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
import pytest

from fuse_hyper_connections import DOMAIN, POST, function_definitions
from fuse_shared_post import OP, fuse


class GatedAdd(OpRun):
    op_domain = DOMAIN

    def _run(self, X, Y, Gate):
        return ((X + (Y * Gate).astype(np.float16)).astype(np.float16),)


class HyperConnectionPostMix(OpRun):
    op_domain = DOMAIN

    def _run(self, X, Z, Gate, num_branches=None):
        hidden = Z.shape[-1]
        streams = X.reshape(*Z.shape[:-1], num_branches, hidden).astype(np.float32)
        value = streams + Gate.astype(np.float32)[..., :, None] * Z.astype(np.float32)[..., None, :]
        return (value.astype(np.float16).reshape(X.shape),)


def fixture(rows=2, hidden=7, branches=4):
    shapes = {
        "X": [1, rows, branches * hidden], "Expert": [1, rows, hidden],
        "G": [1, rows, branches], "Shared": [1, rows, hidden], "SharedGate": [1, rows, 1],
        "SharedSigmoid": [1, rows, 1], "Branch": [1, rows, hidden],
        "Y": [1, rows, branches * hidden],
    }
    inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT16, shapes[n])
              for n in ("X", "Expert", "G", "Shared", "SharedGate")]
    infos = [helper.make_tensor_value_info(n, TensorProto.FLOAT16, shapes[n])
             for n in ("SharedSigmoid", "Branch")]
    model = helper.make_model(helper.make_graph([
        helper.make_node("Sigmoid", ["SharedGate"], ["SharedSigmoid"]),
        helper.make_node("GatedAdd", ["Expert", "Shared", "SharedSigmoid"], ["Branch"], domain=DOMAIN),
        helper.make_node(POST, ["X", "Branch", "G", "D", "S"], ["Y"], domain=DOMAIN,
                         num_branches=branches),
    ], "shared-post", inputs, [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, shapes["Y"])],
        initializer=[numpy_helper.from_array(np.array(v, np.float16), n) for n, v in (("D", 4), ("S", 2))],
        value_info=infos), functions=[function_definitions(22)[POST]],
        opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid(DOMAIN, 1)])
    return model


@pytest.mark.parametrize("rows,hidden,branches", [(1, 1, 1), (2, 7, 4), (1, 2560, 4), (16, 64, 4), (0, 7, 4)])
def test_reference_body_exact(rows, hidden, branches):
    original = fixture(rows, hidden, branches)
    before = original.SerializeToString()
    fused, count = fuse(original)
    assert count == 1 and original.SerializeToString() == before
    onnx.checker.check_model(fused)
    rng = np.random.default_rng(42)
    feeds = {v.name: rng.uniform(-5, 5, [d.dim_value for d in v.type.tensor_type.shape.dim]).astype(np.float16)
             for v in original.graph.input}
    expected = ReferenceEvaluator(inline_local_functions(original),
                                  new_ops=[GatedAdd, HyperConnectionPostMix]).run(None, feeds)
    actual = ReferenceEvaluator(inline_local_functions(fused),
                                new_ops=[GatedAdd, HyperConnectionPostMix]).run(None, feeds)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert [v.SerializeToString() for v in original.graph.initializer] == [
        v.SerializeToString() for v in fused.graph.initializer]


@pytest.mark.parametrize("case", ["shared_add", "shared_sigmoid", "exposed_add", "exposed_sigmoid",
                                 "dtype", "unknown_dtype", "sigmoid_attribute", "add_attribute"])
def test_unsupported_chain_rejected(case):
    model = fixture()
    if case in ("shared_add", "shared_sigmoid"):
        value = "Branch" if case == "shared_add" else "SharedSigmoid"
        model.graph.node.append(helper.make_node("Identity", [value], ["Extra"]))
    elif case in ("exposed_add", "exposed_sigmoid"):
        value = "Branch" if case == "exposed_add" else "SharedSigmoid"
        model.graph.output.append(deepcopy(next(v for v in model.graph.value_info if v.name == value)))
    elif case == "dtype":
        model.graph.input[-1].type.tensor_type.elem_type = TensorProto.FLOAT
    elif case == "unknown_dtype":
        model.graph.input[-1].type.Clear()
    elif case == "sigmoid_attribute":
        model.graph.node[0].attribute.append(helper.make_attribute("invalid", 1))
    elif case == "add_attribute":
        model.graph.node[1].attribute.append(helper.make_attribute("invalid", 1))
    with pytest.raises(ValueError, match="No eligible"):
        fuse(model)


def test_already_defined_rejected():
    model, _ = fuse(fixture())
    with pytest.raises(ValueError, match="already defined"):
        fuse(model)


def test_changed_reference_function_rejected():
    model = fixture()
    model.functions[0].node[0].op_type = "Mul"
    with pytest.raises(ValueError, match="exact qualified"):
        fuse(model)
