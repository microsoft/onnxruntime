// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../telum_kernel_common.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace telum {

/**
 * @brief Base class for unary activation functions
 */
template <typename OpFunc>
class UnaryActivation : public TelumKernel {
 public:
  explicit UnaryActivation(const OpKernelInfo& info) : TelumKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    ORT_RETURN_IF_NOT(X != nullptr, "Input is null");

    // Validate static shape
    ORT_RETURN_IF_ERROR(ValidateStaticShape(X->Shape()));

    // Allocate output with same shape as input
    Tensor* Y = context->Output(0, X->Shape());
    ORT_RETURN_IF_NOT(Y != nullptr, "Failed to allocate output tensor");

    // Determine layout
    zdnn_data_layouts layout = TensorConverter::GetLayoutForShape(X->Shape());

    // Convert tensors
    zdnn_ztensor zdnn_x, zdnn_y;
    ORT_RETURN_IF_ERROR(ConvertToZTensor(*X, zdnn_x, layout));
    ZTensorGuard guard_x(&zdnn_x);

    ORT_RETURN_IF_ERROR(InitZTensorForOutput(*Y, zdnn_y, layout));
    ZTensorGuard guard_y(&zdnn_y);

    // Execute operation
    zdnn_status status = OpFunc::Execute(&zdnn_x, &zdnn_y);
    ORT_RETURN_IF_ERROR(CheckStatus(status, OpFunc::Name()));

    // Convert result
    ORT_RETURN_IF_ERROR(ConvertFromZTensor(zdnn_y, *Y));

    return Status::OK();
  }
};

// Operation functors
struct ReluOp {
  static zdnn_status Execute(const zdnn_ztensor* x, zdnn_ztensor* y) {
    return zdnn_relu(x, nullptr, y);  // No clipping
  }
  static const char* Name() { return "zdnn_relu"; }
};

struct GeluOp {
  static zdnn_status Execute(const zdnn_ztensor* x, zdnn_ztensor* y) {
    return zdnn_gelu(x, y);
  }
  static const char* Name() { return "zdnn_gelu"; }
};

struct TanhOp {
  static zdnn_status Execute(const zdnn_ztensor* x, zdnn_ztensor* y) {
    return zdnn_tanh(x, y);
  }
  static const char* Name() { return "zdnn_tanh"; }
};

struct SigmoidOp {
  static zdnn_status Execute(const zdnn_ztensor* x, zdnn_ztensor* y) {
    return zdnn_sigmoid(x, y);
  }
  static const char* Name() { return "zdnn_sigmoid"; }
};

struct ExpOp {
  static zdnn_status Execute(const zdnn_ztensor* x, zdnn_ztensor* y) {
    return zdnn_exp(x, y);
  }
  static const char* Name() { return "zdnn_exp"; }
};

struct LogOp {
  static zdnn_status Execute(const zdnn_ztensor* x, zdnn_ztensor* y) {
    return zdnn_log(x, y);
  }
  static const char* Name() { return "zdnn_log"; }
};

struct SqrtOp {
  static zdnn_status Execute(const zdnn_ztensor* x, zdnn_ztensor* y) {
    return zdnn_sqrt(x, y);
  }
  static const char* Name() { return "zdnn_sqrt"; }
};

// Concrete kernel classes
using Relu = UnaryActivation<ReluOp>;
using Gelu = UnaryActivation<GeluOp>;
using Tanh = UnaryActivation<TanhOp>;
using Sigmoid = UnaryActivation<SigmoidOp>;
using Exp = UnaryActivation<ExpOp>;
using Log = UnaryActivation<LogOp>;
using Sqrt = UnaryActivation<SqrtOp>;

// Register Relu kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Relu,
    kOnnxDomain,
    6, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Relu);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Relu,
    kOnnxDomain,
    13, 13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Relu);

ONNX_OPERATOR_KERNEL_EX(
    Relu,
    kOnnxDomain,
    14,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Relu);

// Register Gelu kernel (custom domain)
ONNX_OPERATOR_KERNEL_EX(
    Gelu,
    kMSDomain,
    1,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Gelu);

// Register Tanh kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Tanh,
    kOnnxDomain,
    6, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Tanh);

ONNX_OPERATOR_KERNEL_EX(
    Tanh,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Tanh);

// Register Sigmoid kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sigmoid,
    kOnnxDomain,
    6, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Sigmoid);

ONNX_OPERATOR_KERNEL_EX(
    Sigmoid,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Sigmoid);

// Register Exp kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Exp,
    kOnnxDomain,
    6, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Exp);

ONNX_OPERATOR_KERNEL_EX(
    Exp,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Exp);

// Register Log kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Log,
    kOnnxDomain,
    6, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Log);

ONNX_OPERATOR_KERNEL_EX(
    Log,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Log);

// Register Sqrt kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sqrt,
    kOnnxDomain,
    6, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Sqrt);

ONNX_OPERATOR_KERNEL_EX(
    Sqrt,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Sqrt);

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
