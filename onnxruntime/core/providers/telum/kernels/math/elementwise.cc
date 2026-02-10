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

    // Compute output shape (with broadcasting)
    TensorShape output_shape = ComputeBroadcastOutputShape(A->Shape(), B->Shape());

    // Allocate output
    Tensor* Y = context->Output(0, output_shape);
    ORT_RETURN_IF_NOT(Y != nullptr, "Failed to allocate output tensor");

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
  TensorShape ComputeBroadcastOutputShape(const TensorShape& shape_a,
                                          const TensorShape& shape_b) const {
    const auto& dims_a = shape_a.GetDims();
    const auto& dims_b = shape_b.GetDims();

    size_t rank_a = dims_a.size();
    size_t rank_b = dims_b.size();
    size_t max_rank = std::max(rank_a, rank_b);

    std::vector<int64_t> output_dims(max_rank);

    for (size_t i = 0; i < max_rank; ++i) {
      int64_t dim_a = (i < rank_a) ? dims_a[rank_a - 1 - i] : 1;
      int64_t dim_b = (i < rank_b) ? dims_b[rank_b - 1 - i] : 1;

      if (dim_a == dim_b) {
        output_dims[max_rank - 1 - i] = dim_a;
      } else if (dim_a == 1) {
        output_dims[max_rank - 1 - i] = dim_b;
      } else if (dim_b == 1) {
        output_dims[max_rank - 1 - i] = dim_a;
      } else {
        ORT_THROW("Incompatible dimensions for broadcasting: ", dim_a, " and ", dim_b);
      }
    }

    return TensorShape(output_dims);
  }
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
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Add);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Add,
    kOnnxDomain,
    13, 13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Add);

ONNX_OPERATOR_KERNEL_EX(
    Add,
    kOnnxDomain,
    14,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Add);

// Register Sub kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sub,
    kOnnxDomain,
    7, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Sub);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Sub,
    kOnnxDomain,
    13, 13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Sub);

ONNX_OPERATOR_KERNEL_EX(
    Sub,
    kOnnxDomain,
    14,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Sub);

// Register Mul kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Mul,
    kOnnxDomain,
    7, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Mul);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Mul,
    kOnnxDomain,
    13, 13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Mul);

ONNX_OPERATOR_KERNEL_EX(
    Mul,
    kOnnxDomain,
    14,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Mul);

// Register Div kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Div,
    kOnnxDomain,
    7, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Div);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Div,
    kOnnxDomain,
    13, 13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Div);

ONNX_OPERATOR_KERNEL_EX(
    Div,
    kOnnxDomain,
    14,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Div);

// Register Min kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Min,
    kOnnxDomain,
    8, 11,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Min);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Min,
    kOnnxDomain,
    12, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Min);

ONNX_OPERATOR_KERNEL_EX(
    Min,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Min);

// Register Max kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Max,
    kOnnxDomain,
    8, 11,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Max);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Max,
    kOnnxDomain,
    12, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Max);

ONNX_OPERATOR_KERNEL_EX(
    Max,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Max);

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
