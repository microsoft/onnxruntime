# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import unittest

import torch
from functorch.compile import min_cut_rematerialization_partition
from torch._dynamo.backends.common import aot_autograd
from torch.onnx import register_custom_op_symbolic

import onnxruntime as onnxrt
from onnxruntime.training.torchdynamo import register_custom_op_in_dort
from onnxruntime.training.torchdynamo.ort_backend import ATEN2ATEN_DECOMP, OrtBackend


def onnx_custom_add(g, x, y):
    return g.op("test.customop::CustomOpOne", x, y, outputs=1)


class TestTorchDynamoOrtCustomOp(unittest.TestCase):
    """Containers of custom op lib test for TorchDynamo ORT (DORT) backend."""

    def setUp(self):
        # Make computation deterministic.
        torch.manual_seed(42)

    def test_DORT_custom_ops(self):
        torch._dynamo.reset()

        # register custom op in onnx
        register_custom_op_symbolic("aten::mul", onnx_custom_add, opset_version=14)

        # register custom op in dort
        register_custom_op_in_dort("test.customop::CustomOpOne")

        if sys.platform.startswith("win"):
            shared_library = "custom_op_library.dll"
            if not os.path.exists(shared_library):
                raise FileNotFoundError(f"Unable to find '{shared_library}'")

        elif sys.platform.startswith("darwin"):
            shared_library = "libcustom_op_library.dylib"
            if not os.path.exists(shared_library):
                raise FileNotFoundError(f"Unable to find '{shared_library}'")

        else:
            shared_library = "./libcustom_op_library.so"
            if not os.path.exists(shared_library):
                raise FileNotFoundError(f"Unable to find '{shared_library}'")

        session_options = onnxrt.SessionOptions()
        session_options.register_custom_ops_library(shared_library)

        ort_backend = OrtBackend(ep="CPUExecutionProvider", session_options=session_options)
        aot_ort = aot_autograd(
            fw_compiler=ort_backend, partition_fn=min_cut_rematerialization_partition, decompositions=ATEN2ATEN_DECOMP
        )

        def custom_add(tensor_x: torch.Tensor, tensor_y: torch.Tensor):
            return torch.mul(tensor_x, tensor_y)

        opt_add = torch._dynamo.optimize(aot_ort)(custom_add)

        tensor_x = torch.ones((64, 64), dtype=torch.float32)
        tensor_y = torch.ones((64, 64), dtype=torch.float32)

        # Baseline.
        result_ref = torch.add(tensor_x, tensor_y)
        # ORT result.
        result_ort = opt_add(tensor_x, tensor_y)

        torch.testing.assert_close(result_ref, result_ort)


if __name__ == "__main__":
    unittest.main()
