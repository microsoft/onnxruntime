from copy import deepcopy

import onnx
import pytest
from onnx import TensorProto, helper

from fuse_quickgelu_mul import OP, fuse


def fixture(shape=(1, 1, 1024)):
    inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT16, shape) for n in ("X", "Up")]
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, shape)
    graph = helper.make_graph([
        helper.make_node("Sigmoid", ["X"], ["S"]),
        helper.make_node("Mul", ["X", "S"], ["A"]),
        helper.make_node("Mul", ["A", "Up"], ["Y"]),
    ], "silu_mul", inputs, [output], value_info=[
        helper.make_tensor_value_info(n, TensorProto.FLOAT16, shape) for n in ("S", "A")])
    return helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)])


@pytest.mark.parametrize("shape", [(1,), (1, 7), (0, 7), (1, 16, 1024)])
@pytest.mark.parametrize("swap_activation", [False, True])
@pytest.mark.parametrize("swap_final", [False, True])
def test_supported(shape, swap_activation, swap_final):
    original = fixture(shape)
    if swap_activation:
        original.graph.node[1].input.reverse()
    if swap_final:
        original.graph.node[2].input.reverse()
    before = original.SerializeToString()
    model, count = fuse(original)
    assert count == 1 and len(model.graph.node) == 1 and model.graph.node[0].op_type == OP
    assert list(model.graph.node[0].input) == ["X", "Up"]
    assert original.SerializeToString() == before
    assert [n.op_type for n in model.functions[-1].node] == ["QuickGelu", "Mul"]
    onnx.checker.check_model(model)
    with pytest.raises(ValueError, match="Already fused"):
        fuse(model)


@pytest.mark.parametrize("name", ["S", "A"])
def test_shared_intermediate(name):
    model = fixture()
    model.graph.node.append(helper.make_node("Identity", [name], ["Other"]))
    with pytest.raises(ValueError, match="No matching"):
        fuse(model)


@pytest.mark.parametrize("name", ["S", "A"])
def test_exposed_intermediate(name):
    model = fixture()
    model.graph.output.append(deepcopy(next(v for v in model.graph.value_info if v.name == name)))
    with pytest.raises(ValueError, match="No matching"):
        fuse(model)


@pytest.mark.parametrize("name", ["X", "Up", "Y", "S", "A"])
def test_f32_rejected(name):
    model = fixture()
    values = [*model.graph.input, *model.graph.output, *model.graph.value_info]
    next(v for v in values if v.name == name).type.tensor_type.elem_type = TensorProto.FLOAT
    with pytest.raises(ValueError, match="No matching"):
        fuse(model)


def test_broadcast_rejected():
    model = fixture()
    model.graph.input[1].type.tensor_type.shape.dim[-1].dim_value = 1
    with pytest.raises(ValueError, match="No matching"):
        fuse(model)


def test_wrong_sigmoid_source_rejected():
    model = fixture()
    model.graph.node[0].input[0] = "Up"
    with pytest.raises(ValueError, match="No matching"):
        fuse(model)
