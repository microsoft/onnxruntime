// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../telum_kernel_common.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace telum {

/**
 * @brief Gemm kernel implementation for Telum EP
 *
 * Implements General Matrix Multiplication (GEMM): Y = alpha * A * B + beta * C
 * Uses zDNN's zdnn_matmul_op with bias addition for optimal performance.
 *
 * This is a critical operation for transformer models as it combines
 * matrix multiplication with bias addition in a single fused operation.
 */
class Gemm final : public TelumKernel {
 public:
  explicit Gemm(const OpKernelInfo& info) : TelumKernel(info) {
    // Get attributes
    int64_t temp;
    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = (temp != 0);

    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = (temp != 0);

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    // Get input tensors
    const Tensor* A = context->Input<Tensor>(0);
    const Tensor* B = context->Input<Tensor>(1);
    const Tensor* C = context->Input<Tensor>(2);  // Bias (optional)

    ORT_RETURN_IF_NOT(A != nullptr, "Input A is null");
    ORT_RETURN_IF_NOT(B != nullptr, "Input B is null");

    // Validate shapes are static
    ORT_RETURN_IF_ERROR(ValidateStaticShape(A->Shape()));
    ORT_RETURN_IF_ERROR(ValidateStaticShape(B->Shape()));
    if (C != nullptr) {
      ORT_RETURN_IF_ERROR(ValidateStaticShape(C->Shape()));
    }

    // Get shapes
    const auto& shape_A = A->Shape();
    const auto& shape_B = B->Shape();

    // Validate GEMM dimensions
    ORT_RETURN_IF_ERROR(ValidateGemmShapes(shape_A, shape_B, C));

    // Compute output shape
    TensorShape output_shape = ComputeOutputShape(shape_A, shape_B);

    // Allocate output tensor
    Tensor* Y = context->Output(0, output_shape);
    ORT_RETURN_IF_NOT(Y != nullptr, "Failed to allocate output tensor");

    // Handle transpose and scaling
    if (trans_A_ || trans_B_ || alpha_ != 1.0f || beta_ != 1.0f) {
      // For now, fall back to CPU for non-standard cases
      // TODO: Implement transpose support using zdnn_matmul_transpose_op
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                            "Telum EP: Gemm with transpose or scaling not yet implemented. "
                            "transA=", trans_A_, ", transB=", trans_B_,
                            ", alpha=", alpha_, ", beta=", beta_);
    }

    // Standard case: Y = A * B + C (no transpose, alpha=1, beta=1)
    zdnn_data_layouts layout = TensorConverter::GetLayoutForShape(shape_A);

    // Convert input tensors to zDNN format
    zdnn_ztensor zdnn_a, zdnn_b, zdnn_c, zdnn_y;
    ORT_RETURN_IF_ERROR(ConvertToZTensor(*A, zdnn_a, layout));
    ZTensorGuard guard_a(&zdnn_a);

    ORT_RETURN_IF_ERROR(ConvertToZTensor(*B, zdnn_b, layout));
    ZTensorGuard guard_b(&zdnn_b);

    // Convert bias if present
    zdnn_ztensor* bias_ptr = nullptr;
    ZTensorGuard guard_c(nullptr);
    if (C != nullptr) {
      ORT_RETURN_IF_ERROR(ConvertToZTensor(*C, zdnn_c, layout));
      guard_c = ZTensorGuard(&zdnn_c);
      bias_ptr = &zdnn_c;
    }

    ORT_RETURN_IF_ERROR(InitZTensorForOutput(*Y, zdnn_y, layout));
    ZTensorGuard guard_y(&zdnn_y);

    // Execute zDNN GEMM (MatMul with bias addition)
    zdnn_status status = zdnn_matmul_op(&zdnn_a, &zdnn_b, bias_ptr,
                                        MATMUL_OP_ADDITION, &zdnn_y);
    ORT_RETURN_IF_ERROR(CheckStatus(status, "zdnn_matmul_op (Gemm)"));

    // Convert result back to ORT tensor
    ORT_RETURN_IF_ERROR(ConvertFromZTensor(zdnn_y, *Y));

    return Status::OK();
  }

 private:
  bool trans_A_;
  bool trans_B_;
  float alpha_;
  float beta_;

  /**
   * @brief Validate GEMM dimensions
   */
  Status ValidateGemmShapes(const TensorShape& shape_A,
                            const TensorShape& shape_B,
                            const Tensor* C) const {
    const auto& dims_A = shape_A.GetDims();
    const auto& dims_B = shape_B.GetDims();

    // A and B must be 2D
    if (dims_A.size() != 2 || dims_B.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "Gemm requires 2D tensors. Got shapes: ",
                            shape_A.ToString(), " and ", shape_B.ToString());
    }

    // Check dimension compatibility
    int64_t M = trans_A_ ? dims_A[1] : dims_A[0];
    int64_t K_A = trans_A_ ? dims_A[0] : dims_A[1];
    int64_t K_B = trans_B_ ? dims_B[1] : dims_B[0];
    int64_t N = trans_B_ ? dims_B[0] : dims_B[1];

    if (K_A != K_B) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "Gemm inner dimensions must match. Got K_A=", K_A,
                            " and K_B=", K_B);
    }

    // Validate bias shape if present
    if (C != nullptr) {
      const auto& dims_C = C->Shape().GetDims();

      // Bias must be broadcastable to output shape [M, N]
      if (dims_C.size() == 1) {
        if (dims_C[0] != N && dims_C[0] != 1) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                "Gemm bias dimension mismatch. Expected ", N,
                                " or 1, got ", dims_C[0]);
        }
      } else if (dims_C.size() == 2) {
        if ((dims_C[0] != M && dims_C[0] != 1) ||
            (dims_C[1] != N && dims_C[1] != 1)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                "Gemm bias shape mismatch. Expected [", M, ",", N,
                                "] or broadcastable, got ", C->Shape().ToString());
        }
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "Gemm bias must be 1D or 2D, got rank ", dims_C.size());
      }
    }

    return Status::OK();
  }

  /**
   * @brief Compute output shape for GEMM
   */
  TensorShape ComputeOutputShape(const TensorShape& shape_A,
                                 const TensorShape& shape_B) const {
    const auto& dims_A = shape_A.GetDims();
    const auto& dims_B = shape_B.GetDims();

    int64_t M = trans_A_ ? dims_A[1] : dims_A[0];
    int64_t N = trans_B_ ? dims_B[0] : dims_B[1];

    return TensorShape({M, N});
  }
};

// Register the kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    7, 8,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Gemm);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    9, 10,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Gemm);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    11, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Gemm);

ONNX_OPERATOR_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    Gemm);

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
