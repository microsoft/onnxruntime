// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../telum_kernel_common.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace telum {

/**
 * @brief Base class for elementwise binary operations
 */
template <typename OpFunc>
class BinaryElementwise : public TelumKernel {
 public:
  explicit BinaryElementwise(const OpKernelInfo& info) : TelumKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const Tensor* A = context->Input<Tensor>(0);
    const Tensor* B = context->Input<Tensor>(1);

    ORT_RETURN_IF_NOT(A != nullptr, "Input A is null");
    ORT_RETURN_IF_NOT(B != nullptr, "Input B is null");

    // Validate static shapes
    ORT_RETURN_IF_ERROR(ValidateStaticShape(A->Shape()));
    ORT_RETURN_IF_ERROR(ValidateStaticShape(B->Shape()));

    // zDNN elementwise ops do not do broadcasting. For now we only support identical shapes.
    if (A->Shape() != B->Shape()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Telum EP: elementwise broadcasting is not implemented for ",
                             OpFunc::Name(), ". Shapes: ", A->Shape().ToString(),
                             " and ", B->Shape().ToString());
    }

    TensorShape output_shape = A->Shape();

    // Allocate output
    Tensor* Y = context->Output(0, output_shape);
    ORT_RETURN_IF_NOT(Y != nullptr, "Failed to allocate output tensor");

    if (output_shape.NumDimensions() > 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Telum EP: elementwise ops only support rank <= 4, got rank ",
                             output_shape.NumDimensions());
    }

    // Determine layout
    zdnn_data_layouts layout = TensorConverter::GetLayoutForShape(output_shape);

    // Convert tensors
    zdnn_ztensor zdnn_a, zdnn_b, zdnn_y;
    ORT_RETURN_IF_ERROR(ConvertToZTensor(*A, zdnn_a, layout));
    ZTensorGuard guard_a(&zdnn_a);

    ORT_RETURN_IF_ERROR(ConvertToZTensor(*B, zdnn_b, layout));
    ZTensorGuard guard_b(&zdnn_b);

    ORT_RETURN_IF_ERROR(InitZTensorForOutput(*Y, zdnn_y, layout));
    ZTensorGuard guard_y(&zdnn_y);

    // Execute operation
    zdnn_status status = OpFunc::Execute(&zdnn_a, &zdnn_b, &zdnn_y);
    ORT_RETURN_IF_ERROR(CheckStatus(status, OpFunc::Name()));

    // Convert result
    ORT_RETURN_IF_ERROR(ConvertFromZTensor(zdnn_y, *Y));

    return Status::OK();
  }

 private:
};

// Operation functors
struct AddOp {
  static zdnn_status Execute(const zdnn_ztensor* a, const zdnn_ztensor* b, zdnn_ztensor* y) {
    return zdnn_add(a, b, y);
  }
  static const char* Name() { return "zdnn_add"; }
};

struct SubOp {
  static zdnn_status Execute(const zdnn_ztensor* a, const zdnn_ztensor* b, zdnn_ztensor* y) {
    return zdnn_sub(a, b, y);
  }
  static const char* Name() { return "zdnn_sub"; }
};

struct MulOp {
  static zdnn_status Execute(const zdnn_ztensor* a, const zdnn_ztensor* b, zdnn_ztensor* y) {
    return zdnn_mul(a, b, y);
  }
  static const char* Name() { return "zdnn_mul"; }
};

struct DivOp {
  static zdnn_status Execute(const zdnn_ztensor* a, const zdnn_ztensor* b, zdnn_ztensor* y) {
    return zdnn_div(a, b, y);
  }
  static const char* Name() { return "zdnn_div"; }
};

struct MinOp {
  static zdnn_status Execute(const zdnn_ztensor* a, const zdnn_ztensor* b, zdnn_ztensor* y) {
    return zdnn_min(a, b, y);
  }
  static const char* Name() { return "zdnn_min"; }
};

struct MaxOp {
  static zdnn_status Execute(const zdnn_ztensor* a, const zdnn_ztensor* b, zdnn_ztensor* y) {
    return zdnn_max(a, b, y);
  }
  static const char* Name() { return "zdnn_max"; }
};

// Concrete kernel classes
using Add = BinaryElementwise<AddOp>;
using Sub = BinaryElementwise<SubOp>;
using Mul = BinaryElementwise<MulOp>;
using Div = BinaryElementwise<DivOp>;
using Min = BinaryElementwise<MinOp>;
using Max = BinaryElementwise<MaxOp>;

// Register Add kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Add,
    kOnnxDomain,
    7, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Add);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Add,
    kOnnxDomain,
    13, 13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Add);

ONNX_OPERATOR_KERNEL_EX(
    Add,
    kOnnxDomain,
    14,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Add);

// Register Sub kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sub,
    kOnnxDomain,
    7, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Sub);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sub,
    kOnnxDomain,
    13, 13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Sub);

ONNX_OPERATOR_KERNEL_EX(
    Sub,
    kOnnxDomain,
    14,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Sub);

// Register Mul kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Mul,
    kOnnxDomain,
    7, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Mul);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Mul,
    kOnnxDomain,
    13, 13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Mul);

ONNX_OPERATOR_KERNEL_EX(
    Mul,
    kOnnxDomain,
    14,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Mul);

// Register Div kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Div,
    kOnnxDomain,
    7, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Div);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Div,
    kOnnxDomain,
    13, 13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Div);

ONNX_OPERATOR_KERNEL_EX(
    Div,
    kOnnxDomain,
    14,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Div);

// Register Min kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Min,
    kOnnxDomain,
    8, 11,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Min);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Min,
    kOnnxDomain,
    12, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Min);

ONNX_OPERATOR_KERNEL_EX(
    Min,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Min);

// Register Max kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Max,
    kOnnxDomain,
    8, 11,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Max);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Max,
    kOnnxDomain,
    12, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Max);

ONNX_OPERATOR_KERNEL_EX(
    Max,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Max);

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
