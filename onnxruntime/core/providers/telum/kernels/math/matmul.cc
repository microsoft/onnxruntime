// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../telum_kernel_common.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace telum {

/**
 * @brief MatMul kernel implementation for Telum EP
 *
 * Implements matrix multiplication using zDNN's zdnn_matmul_op.
 * Supports 2D and higher-dimensional matrix multiplication with
 * static shapes only.
 */
class MatMul final : public TelumKernel {
 public:
  explicit MatMul(const OpKernelInfo& info) : TelumKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    // Get input tensors
    const Tensor* A = context->Input<Tensor>(0);
    const Tensor* B = context->Input<Tensor>(1);

    ORT_RETURN_IF_NOT(A != nullptr, "Input A is null");
    ORT_RETURN_IF_NOT(B != nullptr, "Input B is null");

    // Validate shapes are static
    ORT_RETURN_IF_ERROR(ValidateStaticShape(A->Shape()));
    ORT_RETURN_IF_ERROR(ValidateStaticShape(B->Shape()));

    // Get shapes
    const auto& shape_A = A->Shape();
    const auto& shape_B = B->Shape();

    // Validate matrix multiplication dimensions
    ORT_RETURN_IF_ERROR(ValidateMatMulShapes(shape_A, shape_B));

    // Compute output shape
    TensorShape output_shape = ComputeOutputShape(shape_A, shape_B);

    // Allocate output tensor
    Tensor* Y = context->Output(0, output_shape);
    ORT_RETURN_IF_NOT(Y != nullptr, "Failed to allocate output tensor");

    // Determine appropriate layout based on tensor rank
    zdnn_data_layouts layout = TensorConverter::GetLayoutForShape(shape_A);

    // Convert input tensors to zDNN format
    zdnn_ztensor zdnn_a, zdnn_b, zdnn_y;
    ORT_RETURN_IF_ERROR(ConvertToZTensor(*A, zdnn_a, layout));
    ZTensorGuard guard_a(&zdnn_a);

    ORT_RETURN_IF_ERROR(ConvertToZTensor(*B, zdnn_b, layout));
    ZTensorGuard guard_b(&zdnn_b);

    ORT_RETURN_IF_ERROR(InitZTensorForOutput(*Y, zdnn_y, layout));
    ZTensorGuard guard_y(&zdnn_y);

    // Execute zDNN matrix multiplication
    // Using MATMUL_OP_ADDITION with nullptr for bias (no bias addition)
    zdnn_status status = zdnn_matmul_op(&zdnn_a, &zdnn_b, nullptr,
                                        MATMUL_OP_ADDITION, &zdnn_y);
    ORT_RETURN_IF_ERROR(CheckStatus(status, "zdnn_matmul_op"));

    // Convert result back to ORT tensor
    ORT_RETURN_IF_ERROR(ConvertFromZTensor(zdnn_y, *Y));

    return Status::OK();
  }

 private:
  /**
   * @brief Validate that matrix multiplication dimensions are compatible
   *
   * @param shape_A Shape of first input
   * @param shape_B Shape of second input
   * @return Status indicating if shapes are valid
   */
  Status ValidateMatMulShapes(const TensorShape& shape_A,
                              const TensorShape& shape_B) const {
    const auto& dims_A = shape_A.GetDims();
    const auto& dims_B = shape_B.GetDims();

    // Both tensors must be at least 2D
    if (dims_A.size() < 2 || dims_B.size() < 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "MatMul requires at least 2D tensors. Got shapes: ",
                            shape_A.ToString(), " and ", shape_B.ToString());
    }

    // Inner dimensions must match: A[..., M, K] × B[..., K, N]
    int64_t K_A = dims_A[dims_A.size() - 1];
    int64_t K_B = dims_B[dims_B.size() - 2];

    if (K_A != K_B) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "MatMul inner dimensions must match. Got K_A=", K_A,
                            " and K_B=", K_B);
    }

    // For higher-dimensional tensors, batch dimensions must be compatible
    if (dims_A.size() > 2 || dims_B.size() > 2) {
      size_t batch_dims_A = dims_A.size() - 2;
      size_t batch_dims_B = dims_B.size() - 2;

      // Check batch dimension compatibility (broadcasting rules)
      size_t max_batch_dims = std::max(batch_dims_A, batch_dims_B);
      for (size_t i = 0; i < max_batch_dims; ++i) {
        int64_t dim_A = (i < batch_dims_A) ? dims_A[i] : 1;
        int64_t dim_B = (i < batch_dims_B) ? dims_B[i] : 1;

        if (dim_A != dim_B && dim_A != 1 && dim_B != 1) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                "MatMul batch dimensions are not compatible");
        }
      }
    }

    return Status::OK();
  }

  /**
   * @brief Compute output shape for matrix multiplication
   *
   * @param shape_A Shape of first input
   * @param shape_B Shape of second input
   * @return Output tensor shape
   */
  TensorShape ComputeOutputShape(const TensorShape& shape_A,
                                 const TensorShape& shape_B) const {
    const auto& dims_A = shape_A.GetDims();
    const auto& dims_B = shape_B.GetDims();

    std::vector<int64_t> output_dims;

    // Handle batch dimensions
    size_t batch_dims_A = dims_A.size() - 2;
    size_t batch_dims_B = dims_B.size() - 2;
    size_t max_batch_dims = std::max(batch_dims_A, batch_dims_B);

    for (size_t i = 0; i < max_batch_dims; ++i) {
      int64_t dim_A = (i < batch_dims_A) ? dims_A[i] : 1;
      int64_t dim_B = (i < batch_dims_B) ? dims_B[i] : 1;
      output_dims.push_back(std::max(dim_A, dim_B));
    }

    // Add matrix dimensions: M × N
    int64_t M = dims_A[dims_A.size() - 2];
    int64_t N = dims_B[dims_B.size() - 1];
    output_dims.push_back(M);
    output_dims.push_back(N);

    return TensorShape(output_dims);
  }
};

// Register the kernel
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    1, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    MatMul);

ONNX_OPERATOR_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>()}),
    MatMul);

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
