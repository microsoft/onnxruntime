# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import numpy as np

# Expose available (onnx::* and protobuf::*) symbols from onnxruntime to resolve references in
# the custom ops shared library. Deepbind flag is required to avoid conflicts with other
# instances of onnx/protobuf libraries.
import onnxruntime

# Restore dlopen flags.
import orttraining_external_custom_ops

so = onnxruntime.SessionOptions()
sess = onnxruntime.InferenceSession("testdata/model.onnx", so)
input = np.random.rand(2, 2).astype(np.float32)
output = sess.run(None, {"input1": input})[0]
np.testing.assert_equal(input, output)
