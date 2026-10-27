# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import onnx

if __name__ == "__main__":
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Mul", ["x", "y"], ["z"], name="mul")],
            "encrypted-context-mul",
            [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [3, 2]) for name in ("x", "y")],
            [onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [3, 2])],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    onnx.save(model, Path(__file__).with_suffix(".onnx"))
