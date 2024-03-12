# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import onnx
import onnxscript
from onnxscript.onnx_opset import opset18
from onnxscript.onnx_types import FLOAT, INT64

import onnxruntime


class TestAggressiveCpuFallback(unittest.TestCase):
    def test_cpu_fallback(self):
        @onnxscript.script(default_opset=opset18)
        def foo(x: FLOAT[12], w: FLOAT[6, 2], dim0: INT64[1], dim1: INT64[1]):
            # This should be computed by CPU but is placed
            # on CUDA (i.e., all inputs and outputs are GPU tensors).
            dim2 = dim1 + 1
            # Same as `dim2 = dim1 + 1`. Another GPU node.
            dim3 = dim2 - 1
            # Same as `dim2 = dim1 + 1`. Another GPU node.
            new_shape = opset18.Concat(dim0, dim3, axis=0)
            # A memcpy node will be inserted to copy GPU output
            # to CPU since Reshape's 2nd input is a CPU tensor
            # per schema definition.
            #
            # Use ORT_AGGRESSIVE_CPU_FALLBACK=1 to
            #  1. remove memcpy node.
            #  2. fallback all computation above this line to CPU.
            new_x = opset18.Reshape(x, new_shape)
            y = opset18.MatMul(new_x, w)
            return y

        model = foo.to_model_proto()

        session_options = onnxruntime.SessionOptions()
        session_options.optimized_model_filepath = "cpu_fallback_test.onnx"
        session_options.add_session_config_entry("session.reverse_traverse_cpu_fallback", "1")
        # This call should trigger GetCpuPreferredNodes and then GetShapeRelatedNodes
        # when environment variable ORT_AGGRESSIVE_CPU_FALLBACK=1 is set.
        # As a result, no memcopy node should be observed in optimized graph.
        #
        # See comments inside `foo`.
        onnxruntime.InferenceSession(
            path_or_bytes=model.SerializeToString(), sess_options=session_options, providers=["CUDAExecutionProvider"]
        )
        optimized = onnx.load("cpu_fallback_test.onnx")

        self.assertTrue(all(node.op_type != "MemcpyToHost" for node in optimized.graph.node))


if __name__ == "__main__":
    unittest.main()
