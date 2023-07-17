# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import unittest

import onnxscript
import torch
import torch._dynamo
from functorch.compile import min_cut_rematerialization_partition
from torch._dynamo.backends.common import aot_autograd
from torch.library import Library

import onnxruntime
from onnxruntime.training.torchdynamo.ort_backend import (
    _SUPPORT_DICT,
    DEFAULT_ONNX_EXPORTER_OPTIONS,
    DORT_DECOMPOSITION_TABLE,
    OrtBackend,
)

# Dummy operator set to map aten::mul.Tensor to test.customop::CustomOpOne
# in ONNX model executed by DORT.
# Print the output of to_model_proto in ort_backend.py for the generated
# ONNX model.
custom_opset = onnxscript.values.Opset(domain="test.customop", version=1)


# Exporter for torch.ops.aten.mul.Tensor.
@onnxscript.script(custom_opset)
def custom_exporter_for_aten_add_Tensor(x, y):
    # This function represents an ONNX function. Register below
    # set this function as the FX-to-ONNX exporter of "aten::mul.Tensor".
    return custom_opset.CustomOpOne(x, y)


# Register custom_exporter_for_aten_add_Tensor as "aten::mul.Tensor"'s
# exporter.
# Use custom_exporter_for_aten_add_Tensor.to_function_proto() to investigate
# function representing "aten::mul.Tensor".
DEFAULT_ONNX_EXPORTER_OPTIONS.onnxfunction_dispatcher.onnx_registry.register_custom_op(
    function=custom_exporter_for_aten_add_Tensor,
    namespace="aten",
    op_name="mul",
    overload="Tensor",
)


# Exporter for torch.ops.foo.bar.default.
@onnxscript.script(custom_opset)
def custom_exporter_for_foo_bar_default(x):
    # This function represents an ONNX function. Register below
    # set this function as the FX-to-ONNX exporter of "aten::mul.Tensor".
    return custom_opset.CustomOpOne(x, x)


# Ask exporter to map "torch.ops.foo.bar" to
# custom_exporter_for_foo_bar_default.
DEFAULT_ONNX_EXPORTER_OPTIONS.onnxfunction_dispatcher.onnx_registry.register_custom_op(
    function=custom_exporter_for_foo_bar_default,
    namespace="foo",
    op_name="bar",
)


class TestTorchDynamoOrtCustomOp(unittest.TestCase):
    """Containers of custom op lib test for TorchDynamo ORT (DORT) backend."""

    def setUp(self):
        # Make computation deterministic.
        torch.manual_seed(42)

    @staticmethod
    def search_for_custom_op_library_path():
        """Searches for the path of the custom op library file.

        The returned path may change depending on the platform of the CI.

        Returns:
            str: The path of the custom op library file.

        Raises:
            FileNotFoundError: If the custom op library file is not found
            in the expected location.
        """
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

        return shared_library

    @staticmethod
    def create_onnxruntime_session_options():
        """Creates an ONNXRuntime session options object.

        The returned option object is configured to enable custom
        operator's implementation visible in ONNXRuntime.

        Returns:
            onnxruntime.SessionOptions: An ONNXRuntime session options object.
        """
        custom_op_library_path = TestTorchDynamoOrtCustomOp.search_for_custom_op_library_path()
        session_options = onnxruntime.SessionOptions()
        session_options.register_custom_ops_library(custom_op_library_path)
        return session_options

    def test_DORT_custom_ops(self):
        torch._dynamo.reset()

        session_options = TestTorchDynamoOrtCustomOp.create_onnxruntime_session_options()

        ort_backend = OrtBackend(ep="CPUExecutionProvider", session_options=session_options)
        aot_ort = aot_autograd(
            fw_compiler=ort_backend,
            partition_fn=min_cut_rematerialization_partition,
            decompositions=DORT_DECOMPOSITION_TABLE,
        )

        def one_mul(tensor_x: torch.Tensor, tensor_y: torch.Tensor):
            return torch.mul(tensor_x, tensor_y)

        opt_mul = torch._dynamo.optimize(aot_ort)(one_mul)

        tensor_x = torch.ones((64, 64), dtype=torch.float32)
        tensor_y = torch.ones((64, 64), dtype=torch.float32)

        for _ in range(5):
            result_ref = torch.add(tensor_x, tensor_y)
            result_ort = opt_mul(tensor_x, tensor_y)
            torch.testing.assert_close(result_ref, result_ort)

    def test_dort_with_custom_torch_op_library(self):
        torch._dynamo.reset()

        foo_lib = Library("foo", "DEF")
        bar_name = foo_lib.define("bar(Tensor self) -> Tensor")

        def bar_impl(self: torch.Tensor) -> torch.Tensor:
            # foo::bar.default will be mapped to test.customop::CustomOpOne.
            # In ORT, test.customop::CustomOpOne is simply an Add for testing.
            return torch.add(self, self)

        foo_lib.impl(bar_name, bar_impl, "CompositeExplicitAutograd")

        # TODO(wechi): Redesign API to expose this better.
        _SUPPORT_DICT.add(torch.ops.foo.bar.default)

        session_options = TestTorchDynamoOrtCustomOp.create_onnxruntime_session_options()
        ort_backend = OrtBackend(ep="CPUExecutionProvider", session_options=session_options)
        aot_ort = aot_autograd(
            fw_compiler=ort_backend,
            partition_fn=min_cut_rematerialization_partition,
            decompositions=DORT_DECOMPOSITION_TABLE,
        )

        def one_foo(tensor_x: torch.Tensor):
            return torch.ops.foo.bar(tensor_x)

        opt_foo = torch._dynamo.optimize(aot_ort)(one_foo)

        for _ in range(5):
            x = torch.randn(3, 2, device="cpu")
            expected = torch.ops.foo.bar(x)
            actual = opt_foo(x)
            torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
