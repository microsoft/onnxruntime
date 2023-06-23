# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import unittest

import torch
import torch._dynamo
from functorch.compile import min_cut_rematerialization_partition
from torch._dynamo.backends.common import aot_autograd

import onnxruntime
import onnxscript
from onnxruntime.training.torchdynamo.ort_backend import (
    DORT_DECOMPOSITION_TABLE,
    OrtBackend,
    DEFAULT_ONNX_EXPORTER_OPTIONS,
)

# Dummy operator set to map aten::mul.Tensor to test.customop::CustomOpOne
# in ONNX model executed by DORT.
# Print the output of to_model_proto in ort_backend.py for the generated
# ONNX model.
custom_opset = onnxscript.values.Opset(domain="test.customop", version=1)


@onnxscript.script(custom_opset)
def custom_exporter_for_aten_add_Tensor(x, y):
    # This function represents an ONNX function. Register below
    # set this function as the FX-to-ONNX exporter of "aten::mul.Tensor".
    return custom_opset.CustomOpOne(x, y)


# Register custom_exporter_for_aten_add_Tensor as "aten::mul.Tensor"'s
# exporter.
# Use custom_exporter_for_aten_add_Tensor.to_function_proto() to investigate
# function representing "aten::mul.Tensor".
DEFAULT_ONNX_EXPORTER_OPTIONS.onnxfunction_dispatcher.onnx_registry.register(
    "aten::mul.Tensor",
    DEFAULT_ONNX_EXPORTER_OPTIONS.opset_version,
    custom_exporter_for_aten_add_Tensor,
    True,
)


class TestTorchDynamoOrtCustomOp(unittest.TestCase):
    """Containers of custom op lib test for TorchDynamo ORT (DORT) backend."""

    def setUp(self):
        # Make computation deterministic.
        torch.manual_seed(42)

    def test_DORT_custom_ops(self):
        torch._dynamo.reset()

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

        session_options = onnxruntime.SessionOptions()
        session_options.register_custom_ops_library(shared_library)

        ort_backend = OrtBackend(ep="CPUExecutionProvider", session_options=session_options)
        aot_ort = aot_autograd(
            fw_compiler=ort_backend,
            partition_fn=min_cut_rematerialization_partition,
            decompositions=DORT_DECOMPOSITION_TABLE,
        )

        def one_mul(tensor_x: torch.Tensor, tensor_y: torch.Tensor):
            return torch.mul(tensor_x, tensor_y)

        opt_add = torch._dynamo.optimize(aot_ort)(one_mul)

        tensor_x = torch.ones((64, 64), dtype=torch.float32)
        tensor_y = torch.ones((64, 64), dtype=torch.float32)

        for _ in range(5):
            result_ref = torch.add(tensor_x, tensor_y)
            result_ort = opt_add(tensor_x, tensor_y)
            torch.testing.assert_close(result_ref, result_ort)


if __name__ == "__main__":
    unittest.main()
