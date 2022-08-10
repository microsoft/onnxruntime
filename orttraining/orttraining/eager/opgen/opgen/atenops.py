from copy import deepcopy

import torch
from opgen.generator import MakeTorchFallback, ONNXOp, SignatureOnly
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
aten_output_type = {}

# the following op list is for ops that have a .out version. Often this is the only op needing to be implemented
# and the regular and inplace(_) version derive from the .out.
# Example: abs.out is the elementary op and abs and abs_ derive from that.
# However, that is not alway the case. Example: nonzero has .out and a normal version which is not derived.
# In this case nonzero will also be in the regular list.
unary_ops_with_out = [
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "ceil",
    "cos",
    "cosh",
    "erf",
    "exp",
    "floor",
    "hardsigmoid",
    "log",
    "neg",
    "reciprocal",
    "round",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
]

# the following ops have explicit inplace(_) needing to be implemented.
unary_ops_with_inplace = [
    "relu",
    "selu",
]

# the following ops have the regular version needing to be implemented.
# Note det, isinf, selu - are composite aten ops but have direct onnx ops.
unary_ops = [
    "det",
    "isinf",
    "isnan",
    "relu",
    "selu",
]

for unary_op in unary_ops_with_out:
    ops[f"aten::{unary_op}.out"] = onnx_ops[unary_op]("self")

for unary_op in unary_ops_with_inplace:
    ops[f"aten::{unary_op}_"] = onnx_ops[unary_op]("self")

for unary_op in unary_ops:
    ops[f"aten::{unary_op}"] = onnx_ops[unary_op]("self")

for binary_op, onnx_op in {
    "add": Add("self", Mul("alpha", "other")),
    "sub": Sub("self", Mul("alpha", "other")),
    "mul": Mul("self", "other"),
    "div": Div("self", "other"),
}.items():
    # for Tensor, binary_op.out is used by both binary_op and binary_op_, so we only generate .out
    # from testing and call stacks, it also apears scalar ops fall back to the (Tensor) binary_op.out,
    # so this is all we need.
    name = f"aten::{binary_op}.out"
    if name in ops:
        raise RuntimeError("Duplicate binary op found in op dictionary.")
    ops[f"aten::{binary_op}.out"] = deepcopy(onnx_op)
    type_promotion_ops.append(f"aten::{binary_op}.out")

# Notes on Onnx op mapping
#
# Equal - Onnx spec has the return as a bool tensor, but aten will keep the tensor
#         return type matching that of the "out" tensor if one is passed. To support this behavior
#         we will CAST the Equal result to match the "out" as seen in eq and ne below.
#
# ---------------------------

hand_implemented = {
    "aten::empty.memory_format": SignatureOnly(),
    "aten::empty_strided": SignatureOnly(),
    "aten::zero_": SignatureOnly(),
    "aten::copy_": SignatureOnly(),
    "aten::_reshape_alias": SignatureOnly(),
    "aten::view": SignatureOnly(),
    "aten::_copy_from_and_resize": SignatureOnly(),
    "aten::resize_": SignatureOnly(),
    "aten::as_strided": SignatureOnly(),
    # manually implement Slice using stride and offset.
    "aten::slice.Tensor": SignatureOnly(),
    "aten::addmm": Gemm("mat1", "mat2", "self", alpha="alpha", beta="beta"),
    "aten::t": Transpose("self"),
    # MatMul("self", "mat2"), fails since it resizes based on self but should be based on result shape of the mult
    "aten::mm.out": SignatureOnly(),
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
    "aten::max": ReduceMax("self", keepdims=0),
    "aten::min": ReduceMin("self", keepdims=0),
    "aten::cat.out": SignatureOnly(),
    "aten::fill_.Scalar": SignatureOnly(),
    "aten::ne.Scalar_out": Cast(Not(Equal("self", "other")), to="GetONNXTensorProtoDataType(out.scalar_type())"),
    "aten::ne.Tensor_out": Cast(Not(Equal("self", "other")), to="GetONNXTensorProtoDataType(out.scalar_type())"),
    "aten::eq.Tensor_out": Cast(Equal("self", "other"), to="GetONNXTensorProtoDataType(out.scalar_type())"),
    "aten::eq.Scalar_out": Cast(Equal("self", "other"), to="GetONNXTensorProtoDataType(out.scalar_type())"),
    "aten::bitwise_and.Tensor_out": And("self", "other"),  # This generates a fallback for all but Bool, as expected.
    "aten::masked_select": GatherND("self", Transpose(NonZero(Expand("mask", Shape("self"))))),
    "aten::_local_scalar_dense": MakeTorchFallback(),  # This function extracts a scalar value from
    #   a tensor with exactly one value; there's no need to try to do this on an ORT device.
    #   See CPU impl at pytorch/blob/master/aten/src/ATen/native/Scalar.cpp
    "aten::lt.Scalar_out": Cast(Less(A="self", B="other"), to="GetONNXTensorProtoDataType(out.scalar_type())"),
    "aten::lt.Tensor_out": Cast(Less(A="self", B="other"), to="GetONNXTensorProtoDataType(out.scalar_type())"),
    "aten::gt.Scalar_out": Cast(Greater(A="self", B="other"), to="GetONNXTensorProtoDataType(out.scalar_type())"),
    "aten::gt.Tensor_out": Cast(Greater(A="self", B="other"), to="GetONNXTensorProtoDataType(out.scalar_type())"),
    "aten::equal": SignatureOnly(),
    "aten::_softmax": Softmax("self", axis="dim"),
    "aten::argmax.out": SignatureOnly(),
    "aten::nonzero": Transpose(NonZero("self")),
    "aten::nonzero.out": SignatureOnly(),
    "aten::_log_softmax.out": SignatureOnly(),
    # NegativeLogLikelihoodLoss is not supported by the CPU Execution Provider so testing is not possible
    # Leaving nll_loss_forward.output set to fallback. https://github.com/microsoft/onnxruntime/blob/master/docs/OperatorKernels.md.
    "aten::nll_loss_forward.output": MakeTorchFallback(),
    "aten::nll_loss_backward.grad_input": MakeTorchFallback(),
    "aten::_log_softmax_backward_data.out": MakeTorchFallback(),
    "aten::squeeze.dim": Squeeze("self", "dim"),
    "aten::squeeze": SignatureOnly(),
    "aten::unsqueeze": Unsqueeze(data="self", axes="dim"),
}

# If the aten op expects a specific output type that differs from self
# add the op and type to aten_output_type
aten_output_type["aten::nonzero"] = "at::ScalarType::Long"

# Signature of gelu_backward was changed in this commit id 983ba5e585485ed61a0c0012ef6944f5685e3d97 and PR 61439
# This is done to make sure it is backward and future compatible
if version.parse(torch.__version__) < version.parse(TORCH_API_CHANGE_VERSION):
    hand_implemented["aten::gelu_backward"] = GeluGrad("grad", "self")
    hand_implemented["aten::_cat"] = Concat("tensors", "dim")
else:
    hand_implemented["aten::gelu_backward"] = GeluGrad("grad_output", "self")

ops = {**ops, **hand_implemented}
# TODO: this is a temporary allowlist for ops need type promotion
# Need to enhance the support for onnx type constrains to automatically
# resolve whether the op need type promotion.
# Will remove this list in the future.
type_promotion_ops.append("aten::gelu_backward")
type_promotion_ops.append("aten::gt.Tensor_out")
type_promotion_ops.append("aten::lt.Tensor_out")
type_promotion_ops.append("aten::gt.Scalar_out")
type_promotion_ops.append("aten::lt.Scalar_out")
type_promotion_ops.append("aten::ne.Tensor_out")
type_promotion_ops.append("aten::eq.Tensor_out")
type_promotion_ops.append("aten::ne.Scalar_out")
type_promotion_ops.append("aten::eq.Scalar_out")
