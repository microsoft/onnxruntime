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
from torch.onnx._internal.exporter import ExportOptions

import onnxruntime
from onnxruntime.training.torchdynamo.ort_backend import OrtBackend

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


# Exporter for torch.ops.foo.bar.default.
@onnxscript.script(custom_opset)
def custom_exporter_for_foo_bar_default(x):
    # This function represents an ONNX function. Register below
    # set this function as the FX-to-ONNX exporter of "aten::mul.Tensor".
    return custom_opset.CustomOpOne(x, x)


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

    def test_export_aten_mul_as_onnx_custom_op_and_run_ort(self):
        """A Custom Operator Test for DORT

        In this test, aten.mul.Tensor is exported to test.customop::CustomOpOne and
        executed by ORT.
        """
        torch._dynamo.reset()

        # Create executor of ONNX model.
        # We will register a custom exporter for aten.mul.Tensor
        # in the following step.
        ort_backend = OrtBackend(
            ep="CPUExecutionProvider",
            session_options=TestTorchDynamoOrtCustomOp.create_onnxruntime_session_options(),
            onnx_exporter_options=ExportOptions(dynamic_shapes=True),
        )
        # Register custom_exporter_for_aten_add_Tensor as "aten::mul.Tensor"'s
        # exporter.
        # Use custom_exporter_for_aten_add_Tensor.to_function_proto() to see
        # the sub-graph representing "aten::mul.Tensor".
        ort_backend.resolved_onnx_exporter_options.onnxfunction_dispatcher.onnx_registry.register_custom_op(
            function=custom_exporter_for_aten_add_Tensor,
            namespace="aten",
            op_name="mul",
            overload="Tensor",
        )

        # Wrap ORT executor as a Dynamo backend.
        aot_ort = aot_autograd(
            fw_compiler=ort_backend,
            partition_fn=min_cut_rematerialization_partition,
            decompositions=ort_backend.resolved_onnx_exporter_options.decomposition_table,
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

    def test_export_pytorch_custom_op_to_onnx_custom_op_and_run_ort(self):
        """A Custom Operator Test.

        In this test, torch.ops.foo.bar.default is exported to
        test.customop::CustomOpOne and executed by ORT.

        See test_export_aten_mul_as_onnx_custom_op_and_run_ort for mapping
        official PyTorch operator (e.g., aten.mul.Tensor) to ONNX custom operator.
        """
        torch._dynamo.reset()

        foo_lib = Library("foo", "DEF")
        bar_name = foo_lib.define("bar(Tensor self) -> Tensor")

        def bar_impl(self: torch.Tensor) -> torch.Tensor:
            # foo::bar.default will be mapped to test.customop::CustomOpOne.
            # In ORT, test.customop::CustomOpOne is simply an Add for testing.
            return torch.add(self, self)

        foo_lib.impl(bar_name, bar_impl, "CompositeExplicitAutograd")

        # Create executor of ONNX model.
        ort_backend = OrtBackend(
            ep="CPUExecutionProvider", session_options=TestTorchDynamoOrtCustomOp.create_onnxruntime_session_options()
        )
        # Allow torch.ops.foo.bar.default to be sent to DORT.
        # _support_dict tells Dynamo which ops to sent to DORT.
        ort_backend._supported_ops._support_dict.add(torch.ops.foo.bar.default)
        # Ask exporter to map "torch.ops.foo.bar" to
        # custom_exporter_for_foo_bar_default.
        # TODO(wechi): Redesign API to expose this better.
        ort_backend.resolved_onnx_exporter_options.onnxfunction_dispatcher.onnx_registry.register_custom_op(
            function=custom_exporter_for_foo_bar_default,
            namespace="foo",
            op_name="bar",
        )
        # Wrap ORT executor as a Dynamo backend.
        aot_ort = aot_autograd(
            fw_compiler=ort_backend,
            partition_fn=min_cut_rematerialization_partition,
            decompositions=ort_backend.resolved_onnx_exporter_options.decomposition_table,
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
