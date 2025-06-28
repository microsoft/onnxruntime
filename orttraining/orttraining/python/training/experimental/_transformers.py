import onnxscript
import torch
import torch._dynamo.backends.registry
from torch.onnx import ExportOptions
from torch.onnx import _OrtBackend as OrtBackend
from torch.onnx import _OrtBackendOptions as OrtBackendOptions

from onnxruntime.training.experimental._modeling_llama import (
    scaled_dot_product_efficient_attention,
    scaled_dot_product_efficient_attention_backward,
)
from onnxruntime.training.ortmodule.torch_cpp_extensions.cpu.aten_op_executor import load_aten_op_executor_cpp_extension

load_aten_op_executor_cpp_extension()

custom_opset = onnxscript.values.Opset(domain="com.microsoft", version=1)
aten_opset = onnxscript.values.Opset(domain="org.pytorch.aten", version=1)


def apply_onnx_rewritter(onnx_model):
    from onnxrewriter.optimizer import optimize
    from onnxrewriter.rewriter.transformers import rewrite

    onnx_model = optimize(
        onnx_model,
        num_iterations=2,
        onnx_shape_inference=False,
        function_aware_folding=True,
    )
    onnx_model = rewrite(onnx_model)
    return onnx_model


def make_onnxrt_transformer_backend(dynamic: bool = False):
    onnx_registry = torch.onnx.OnnxRegistry()
    onnx_registry.register_op(
        function=scaled_dot_product_efficient_attention,
        namespace="aten",
        op_name="_scaled_dot_product_efficient_attention",
        overload="default",
    )
    onnx_registry.register_op(
        function=scaled_dot_product_efficient_attention_backward,
        namespace="aten",
        op_name="_scaled_dot_product_efficient_attention_backward",
        overload="default",
    )

    custom_backend = OrtBackend(
        options=OrtBackendOptions(
            export_options=ExportOptions(
                dynamic_shapes=dynamic,
                onnx_registry=onnx_registry,
            ),
            use_aot_autograd=True,
            pre_ort_model_transforms=[
                apply_onnx_rewritter,
            ],
        )
    )

    custom_backend._supported_ops._support_dict["torch.ops.aten._scaled_dot_product_efficient_attention.default"] = None
    custom_backend._supported_ops._support_dict[
        "torch.ops.aten._scaled_dot_product_efficient_attention_backward.default"
    ] = None

    return custom_backend
