from copy import deepcopy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.inliner import inline_local_functions
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
import pytest

from fuse_hyper_connections import DOMAIN, POST, PRE, fuse


class HyperConnectionPreMix(OpRun):
    op_domain = DOMAIN

    def _run(self, X, G, num_branches=None, reduction_scale=None):
        hidden = X.shape[-1] // num_branches
        x = X.reshape(*X.shape[:-1], num_branches, hidden).astype(np.float32)
        g = G.reshape(x.shape).astype(np.float32)
        out = np.zeros(x.shape[:-2] + (hidden,), dtype=np.float32)
        for c in range(num_branches):
            out += x[..., c, :] * g[..., c, :]
        return ((out * reduction_scale).astype(X.dtype),)


class HyperConnectionPostMix(OpRun):
    op_domain = DOMAIN

    def _run(self, X, Z, G, num_branches=None):
        hidden = X.shape[-1] // num_branches
        x = X.reshape(*X.shape[:-1], num_branches, hidden).astype(np.float32)
        g = G.astype(np.float32)[..., :, None]
        y = x + g * Z.astype(np.float32)[..., None, :]
        return (y.reshape(X.shape).astype(X.dtype),)


def fixture(dtype=TensorProto.FLOAT16):
    info = lambda name, shape: helper.make_tensor_value_info(name, dtype, shape)
    inputs = [info("X", [2, 3, 16]), info("P", [2, 3, 16]),
              info("G", [2, 3, 4]), info("Z", [2, 3, 4])]
    nodes = [
        helper.make_node("Sigmoid", ["P"], ["PG"], name="pre_sigmoid"),
        helper.make_node("HyperConnectionPreMix", ["X", "PG"], ["A"], domain=DOMAIN,
                         name="premix", num_branches=4, reduction_scale=0.25),
        helper.make_node("Div", ["G", "D"], ["DG"], name="div"),
        helper.make_node("Sigmoid", ["DG"], ["SG"], name="post_sigmoid"),
        helper.make_node("Mul", ["SG", "S"], ["MG"], name="mul"),
        helper.make_node("HyperConnectionPostMix", ["X", "Z", "MG"], ["Y"], domain=DOMAIN,
                         name="postmix", num_branches=4),
    ]
    ndtype = np.float16 if dtype == TensorProto.FLOAT16 else np.float32
    initializers = [numpy_helper.from_array(np.array(value, dtype=ndtype), name)
                    for name, value in (("D", 4), ("S", 2))]
    graph = helper.make_graph(nodes, "fusion", inputs, [info("A", [2, 3, 4]), info("Y", [2, 3, 16])],
                              initializer=initializers,
                              value_info=[info("PG", [2, 3, 16]),
                                          *[info(n, [2, 3, 4]) for n in ("DG", "SG", "MG")]])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22),
                                                  helper.make_opsetid(DOMAIN, 1)])


def test_exact_reference_function_bodies_and_original_unchanged():
    original = fixture()
    before = original.SerializeToString()
    fused, counts = fuse(original)
    assert original.SerializeToString() == before
    assert counts == {PRE: 1, POST: 1}
    assert len(fused.graph.node) == 2
    assert [i.SerializeToString() for i in fused.graph.initializer] == [
        i.SerializeToString() for i in original.graph.initializer
    ]
    onnx.checker.check_model(fused)
    rng = np.random.default_rng(42)
    feeds = {v.name: rng.normal(size=[d.dim_value for d in v.type.tensor_type.shape.dim]).astype(np.float16)
             for v in original.graph.input}
    ops = [HyperConnectionPreMix, HyperConnectionPostMix]
    expected = ReferenceEvaluator(original, new_ops=ops).run(None, feeds)
    expanded = inline_local_functions(fused)
    assert len(expanded.graph.node) == len(original.graph.node)
    actual = ReferenceEvaluator(expanded, new_ops=ops).run(None, feeds)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a.view(np.uint16), b.view(np.uint16))


@pytest.mark.parametrize("exposed", ["PG", "DG", "SG", "MG"])
def test_exposed_intermediate_not_removed(exposed):
    model = fixture()
    model.graph.output.append(deepcopy(next(v for v in model.graph.value_info if v.name == exposed)))
    fused, counts = fuse(model)
    assert counts[PRE if exposed == "PG" else POST] == 0
    assert any(exposed in n.output for n in fused.graph.node)


@pytest.mark.parametrize("shared", ["PG", "DG", "SG", "MG"])
def test_shared_intermediate_not_removed(shared):
    model = fixture()
    model.graph.node.append(helper.make_node("Identity", [shared], ["Shared"]))
    fused, counts = fuse(model)
    assert counts[PRE if shared == "PG" else POST] == 0


@pytest.mark.parametrize("scalar", ["D", "S"])
def test_nonscalar_broadcast_not_claimed(scalar):
    model = fixture()
    tensor = next(v for v in model.graph.initializer if v.name == scalar)
    tensor.dims.append(1)
    _, counts = fuse(model)
    assert counts[POST] == 0


def test_non_fp16_and_missing_metadata_not_claimed():
    with pytest.raises(ValueError, match="No eligible"):
        fuse(fixture(TensorProto.FLOAT))
    model = fixture()
    del model.graph.value_info[:]
    with pytest.raises(ValueError, match="No eligible"):
        fuse(model)


def test_repeated_fusion_rejected():
    model, _ = fuse(fixture())
    with pytest.raises(ValueError, match="already defines"):
        fuse(model)


def test_scalar_constant_nodes_remain_unmodified():
    model = fixture()
    constants = [
        helper.make_node("Constant", [], [v.name], value=v)
        for v in model.graph.initializer
    ]
    del model.graph.initializer[:]
    old_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend([*constants, *old_nodes])
    model.graph.value_info.extend(
        helper.make_tensor_value_info(n.output[0], TensorProto.FLOAT16, []) for n in constants
    )
    fused, counts = fuse(model)
    assert counts == {PRE: 1, POST: 1}
    assert [n.SerializeToString() for n in fused.graph.node[:2]] == [
        n.SerializeToString() for n in constants
    ]
    onnx.checker.check_model(fused)
