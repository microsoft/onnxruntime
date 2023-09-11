# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import os
import sys
import numpy as np
from onnx import load, AttributeProto, FunctionProto, TensorProto
from onnx.helper import (
    make_node,
    make_function,
    make_tensor,
    make_opsetid,
)
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.op_cast import Cast_19 as Cast
from onnx.reference.ops.op_dequantize_linear import DequantizeLinear
from onnx.reference.op_run import to_array_extended

from onnxruntime import InferenceSession, SessionOptions


def make_dynamic_quantize_linear_function_proto(domain: str, opset: int, to: int | None = None) -> FunctionProto:
    """
    Creates the FunctionProto for function `DynamicQuantizeLinear`
    doing a quantization to float 8.

    :param domain: local domain name
    :param opset: opset to use to define the function
    :param to: if None, the function has an attribute,
        otherwise, it is replaced by the given value
    :return: FunctionProto

    The function takes 1 input and returns 3 outputs like
    operator `DynamicQuantizeLinear
    <https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html>`_.
    It has one attribute *to* which specified the quantized type.
    """
    normalization_values = list(
        {
            TensorProto.FLOAT8E4M3FN: 100.057724,
            TensorProto.FLOAT8E4M3FNUZ: 54.26635,
            TensorProto.FLOAT8E5M2: 9535.286,
            TensorProto.FLOAT8E5M2FNUZ: 9403.499,
        }.items()
    )

    if to is None:
        cast = make_node("Cast", ["zerof"], ["Zeropoint"])
        att = AttributeProto()
        att.name = "to"
        att.ref_attr_name = "to"
        att.type = AttributeProto.INT
        cast.attribute.append(att)

        cst = make_node("Constant", [], ["vto"])
        att = AttributeProto()
        att.name = "value_int"
        att.ref_attr_name = "to"
        att.type = AttributeProto.INT
        cst.attribute.append(att)
    else:
        cast = make_node("Cast", ["zerof"], ["Zeropoint"], to=to)
        cst = make_node("Constant", [], ["vto"], value_int=to)

    nodes = [
        make_node(
            "Constant",
            [],
            ["zerof"],
            value=make_tensor("zerof", TensorProto.FLOAT, [], [0]),
        ),
        make_node(
            "Constant",
            [],
            ["newshape"],
            value=make_tensor("newshape", TensorProto.INT64, [1], [-1]),
        ),
        make_node("CastLike", ["zerof", "x"], ["zero"]),
        cast,
        make_node("IsNaN", ["x"], ["nanxp"]),
        make_node("Not", ["nanxp"], ["nanx"]),
        make_node("CastLike", ["nanx", "x"], ["nanxc"]),
        make_node("Where", ["nanx", "x", "zero"], ["xf"]),
        make_node("Mul", ["xf", "xf"], ["xsquare"]),
        make_node("ReduceSum", ["xsquare"], ["Num"], keepdims=0),
        make_node("ReduceSum", ["nanxc"], ["Den"], keepdims=0),
        make_node("Div", ["Num", "Den"], ["Dev"]),
        make_node("Sqrt", ["Dev"], ["Scale"]),
        cst,
        make_node("Reshape", ["vto", "newshape"], ["vtotensor"]),
        make_node(
            "LabelEncoder",
            ["vtotensor"],
            ["stdftensor"],
            keys_int64s=[v[0] for v in normalization_values],
            values_floats=[v[1] for v in normalization_values],
            domain="ai.onnx.ml",
        ),
        make_node("ReduceSum", ["stdftensor"], ["stdf"], keepdims=0),
        make_node("CastLike", ["stdf", "Scale"], ["std"]),
        make_node("Div", ["Scale", "std"], ["ScaleScaled"]),
        make_node("QuantizeLinear", ["x", "ScaleScaled", "Zeropoint"], ["y"]),
    ]
    return make_function(
        domain,
        "DynamicQuantizeLinear",
        ["x"],
        ["y", "ScaleScaled", "Zeropoint"],
        nodes,
        opset_imports=[make_opsetid("", opset), make_opsetid("ai.onnx.ml", 2)],
        attributes=["to"],
    )


class TestOnnxToolsGraph(unittest.TestCase):
    def test_basic_all(self):
        if sys.platform.startswith("win"):
            shared_library = "custom_op_local_function.dll"
        elif sys.platform.startswith("darwin"):
            shared_library = "libcustom_op_local_function.dylib"
        else:
            shared_library = "./libcustom_op_local_function.so"
        if not os.path.exists(shared_library):
            raise FileNotFoundError(f"Unable to find '{shared_library}'")

        filename = "custom_ops_type_inference_fails_0.onnx"

        with open(os.path.join(os.path.dirname(__file__), filename), "rb") as f:
            onxo = load(f)

        sess_opts = SessionOptions()
        sess_opts.register_custom_ops_library(shared_library)

        dynql = ReferenceEvaluator(make_dynamic_quantize_linear_function_proto(domain="qtest", opset=18))
        atts = dict(to=TensorProto.FLOAT8E4M3FN)

        def dynamic_qdq_linear(x):
            qx, scale, _ = dynql.run(None, dict(x=x), attributes=atts)
            qdq = DequantizeLinear.eval(qx, scale)
            return qdq

        x = np.arange(2**3).reshape((2,) * 3).astype(np.float32)

        cst = Cast.eval(to_array_extended(onxo.graph.initializer[0]), to=TensorProto.FLOAT)
        qx, qc = dynamic_qdq_linear(x), dynamic_qdq_linear(cst)
        expected = qx @ qc

        sess = InferenceSession(
            onxo.SerializeToString(),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )
        got = sess.run(None, dict(X=x))[0]
        self.assertEqualArray(expected, got, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
