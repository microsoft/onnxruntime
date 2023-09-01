# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnx
import onnxscript
from onnx import numpy_helper
from onnxscript import opset17 as op


@onnxscript.script()
def build_model(x: onnxscript.FLOAT):
    y_scale = op.Constant(value_float=1.0)
    y_zero_point = op.Constant(value=numpy_helper.from_array(np.array(1, dtype=np.uint8)))
    return op.Add(x, y_scale), y_scale, y_zero_point


model: onnx.ModelProto = build_model.to_model_proto()
onnx.save(model, "scalar_const_not_share.onnx")
