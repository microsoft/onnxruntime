# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import onnx
from onnx import TensorProto, helper

if __name__ == "__main__":
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Mul", ["x", "y"], ["z"], name="mul")],
            "encrypted-context-mul",
            [helper.make_tensor_value_info(name, TensorProto.FLOAT, [3, 2]) for name in ("x", "y")],
            [helper.make_tensor_value_info("z", TensorProto.FLOAT, [3, 2])],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    onnx.save(model, Path(__file__).with_suffix(".onnx"))
