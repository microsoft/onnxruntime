from copy import deepcopy

import torch
from opgen.generator import MakeTorchFallback as MakeTorchFallback
from opgen.generator import ONNXOp as ONNXOp
from opgen.generator import ORTGen as ORTGen
from opgen.generator import SignatureOnly as SignatureOnly
from opgen.onnxops import *
from packaging import version

TORCH_API_CHANGE_VERSION = "1.11.1"

kMSDomain = "onnxruntime::kMSDomain"


class ReluGrad(ONNXOp):
    def __init__(self, dY, X):
        super().__init__(
            "ReluGrad",
            1,
            [{"at::kHalf", "at::kFloat", "at::kBFloat16"}, {"at::kHalf", "at::kFloat", "at::kBFloat16"}],
            dY,
            X,
        )
        self.domain = kMSDomain


class Gelu(ONNXOp):
    def __init__(self, X):
        super().__init__("Gelu", 1, [{"at::kHalf", "at::kFloat", "at::kBFloat16"}], X)
        self.domain = kMSDomain


class GeluGrad(ONNXOp):
    def __init__(self, dY, X):
        super().__init__(
            "GeluGrad",
            1,
            [{"at::kHalf", "at::kFloat", "at::kBFloat16"}, {"at::kHalf", "at::kFloat", "at::kBFloat16"}],
            dY,
            X,
        )
        self.domain = kMSDomain


ops = {}
type_promotion_ops = []

for binary_op, onnx_op in {
    "add": Add("self", Mul("alpha", "other")),
    "sub": Sub("self", Mul("alpha", "other")),
    "mul": Mul("self", "other"),
    "div": Div("self", "other"),
}.items():
    for dtype in ["Tensor", "Scalar"]:
        for variant in ["", "_"]:
            name = f"aten::{binary_op}{variant}.{dtype}"
            if name not in ops:
                ops[f"aten::{binary_op}{variant}.{dtype}"] = deepcopy(onnx_op)
                type_promotion_ops.append(f"aten::{binary_op}{variant}.{dtype}")

for unary_op in [
    "abs",
    "acos",
    "acosh",
    "asinh",
    "atanh",
    "asin",
    "atan",
    "ceil",
    "cos",
    "cosh",
    "erf",
    "exp",
    "floor",
    "isnan",
    "log",
    "reciprocal",
    "neg",
    "round",
    "relu",
    "selu",
    "sigmoid",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "nonzero",
    "sign",
    "hardsigmoid",
    "isinf",
    "det",
]:
    aten_name = f"aten::{unary_op}"
    onnx_op = onnx_ops[unary_op]("self")
    ops[aten_name] = onnx_op
    # produce the in-place variant as well for ops that support it
    if unary_op not in ["isnan", "nonzero", "min", "max", "isinf", "det"]:
        ops[f"{aten_name}_"] = onnx_op

hand_implemented = {
    "aten::empty.memory_format": SignatureOnly(),
    "aten::empty_strided": SignatureOnly(),
    "aten::zero_": SignatureOnly(),
    "aten::copy_": SignatureOnly(),
    "aten::_reshape_alias": SignatureOnly(),
    "aten::view": SignatureOnly(),
    "aten::_copy_from_and_resize": SignatureOnly(),
    "aten::as_strided": SignatureOnly(),
    # manually implement Slice using stride and offset.
    "aten::slice.Tensor": SignatureOnly(),
    "aten::addmm": Gemm("mat1", "mat2", "self", alpha="alpha", beta="beta"),
    "aten::add_.Tensor": SignatureOnly(),
    "aten::t": Transpose("self"),
    "aten::mm": MatMul("self", "mat2"),
    "aten::zeros_like": ConstantOfShape(
        Shape("self")
    ),  # the default constant is 0, so don't need to speicify attribute
    "aten::sum.dim_IntList": ReduceSum("self", "dim", keepdims="keepdim"),
    "aten::threshold_backward": ReluGrad("grad_output", "self"),
    "aten::fmod.Scalar": Mod("self", "other", fmod=1),
    "aten::fmod.Tensor": Mod("self", "other", fmod=1),
    "aten::softshrink": Shrink("self", bias="lambd", lambd="lambd"),  # yes, bias is set to 'lambd'
    "aten::hardshrink": Shrink("self", bias=0, lambd="lambd"),
    "aten::gelu": Gelu("self"),
    "aten::max": ReduceMax("self", keepdims=1),
    "aten::min": ReduceMin("self", keepdims=1),
    "aten::_cat": Concat("tensors", "dim"),
    "aten::fill_.Scalar": ConstantOfShape("self", value="value"),
    "aten::ne.Scalar": MakeTorchFallback(),
    "aten::ne.Scalar_out": MakeTorchFallback(),
    "aten::ne.Tensor_out": MakeTorchFallback(),
    "aten::eq.Tensor": MakeTorchFallback(),
    "aten::eq.Tensor_out": MakeTorchFallback(),
    "aten::bitwise_and.Tensor_out": MakeTorchFallback(),
    "aten::masked_select": MakeTorchFallback(),
    "aten::_local_scalar_dense": MakeTorchFallback(),
    "aten::gt.Scalar_out": MakeTorchFallback(),
    "aten::equal": MakeTorchFallback(),
    "aten::_softmax": Softmax("self", axis="dim"),
}

# Signature of gelu_backward was changed in this commit id 983ba5e585485ed61a0c0012ef6944f5685e3d97 and PR 61439
# This is done to make sure it is backward and future compatible
if version.parse(torch.__version__) < version.parse(TORCH_API_CHANGE_VERSION):
    hand_implemented["aten::gelu_backward"] = GeluGrad("grad", "self")
else:
    hand_implemented["aten::gelu_backward"] = GeluGrad("grad_output", "self")

ops = {**ops, **hand_implemented}
# TODO: this is a temporary allowlist for ops need type promotion
# Need to enhance the support for onnx type constrains to automatically
# resolve whether the op need type promotion.
# Will remove this list in the future.
type_promotion_ops = (*type_promotion_ops, "aten::gelu_backward")
