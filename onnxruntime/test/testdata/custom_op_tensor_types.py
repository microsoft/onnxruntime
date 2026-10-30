# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# Licensed under the MIT License.

"""Generate models with newer tensor types on custom-op inputs and outputs."""

from pathlib import Path

from onnx import TensorProto, checker, helper

for name, tensor_type, ir_version in [
    ("e8m0", TensorProto.FLOAT8E8M0, 12),
    ("uint2", TensorProto.UINT2, 13),
    ("int2", TensorProto.INT2, 13),
]:
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Foo", ["X"], ["Y"], domain="test")],
            f"custom_op_{name}",
            [helper.make_tensor_value_info("X", tensor_type, [4])],
            [helper.make_tensor_value_info("Y", tensor_type, [4])],
        ),
        opset_imports=[helper.make_opsetid("", 25), helper.make_opsetid("test", 1)],
        ir_version=ir_version,
    )
    checker.check_model(model)
    Path(__file__).with_name(f"custom_op_{name}.onnx").write_bytes(model.SerializeToString())
