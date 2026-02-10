// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../telum_kernel_common.h"
#include "core/providers/common.h"

#include <optional>

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

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr /*alloc*/,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* /*prepacked_weights*/) override {
    is_packed = false;

    // Prepack RHS operand B (input index 1) when it is a constant initializer.
    if (input_idx != 1) {
      return Status::OK();
    }

    // We need A's (static) shape to compute the MatMul plan so we can pack B with the correct logical shape/layout.
    const auto& node_inputs = Node().InputDefs();
    if (node_inputs.size() < 2 || node_inputs[0] == nullptr || node_inputs[0]->Shape() == nullptr) {
      return Status::OK();
    }

    std::vector<int64_t> a_dims;
    for (const auto& dim : node_inputs[0]->Shape()->dim()) {
      if (!dim.has_dim_value()) return Status::OK();
      a_dims.push_back(dim.dim_value());
    }

    TensorShape output_shape;
    MatMulPlan plan;
    ORT_RETURN_IF_ERROR(CreatePlan(TensorShape(a_dims), tensor.Shape(), plan, output_shape));

    packed_b_guard_.reset();
    packed_b_shape_ = tensor.Shape();
    packed_plan_ = plan;
    packed_output_shape_ = output_shape;
    have_packed_plan_ = true;

    ORT_RETURN_IF_ERROR(TensorConverter::ConvertToZTensorWithShape(tensor, plan.logical_shape_b, packed_b_, plan.layout_b));
    packed_b_guard_.emplace(&packed_b_);

    is_packed = true;
    return Status::OK();
  }

  Status Compute(OpKernelContext* context) const override {
    // Get input tensors
    const Tensor* A = context->Input<Tensor>(0);
    const Tensor* B = packed_b_guard_.has_value() ? nullptr : context->Input<Tensor>(1);

    ORT_RETURN_IF_NOT(A != nullptr, "Input A is null");
    ORT_RETURN_IF_NOT(B != nullptr || packed_b_guard_.has_value(), "Input B is null");

    // Validate shapes are static
    ORT_RETURN_IF_ERROR(ValidateStaticShape(A->Shape()));
    if (B != nullptr) {
      ORT_RETURN_IF_ERROR(ValidateStaticShape(B->Shape()));
    } else {
      ORT_RETURN_IF_ERROR(ValidateStaticShape(packed_b_shape_));
    }

    const auto& shape_A = A->Shape();
    const auto& shape_B = (B != nullptr) ? B->Shape() : packed_b_shape_;

    // Validate matrix multiplication dimensions and compute output shape.
    TensorShape output_shape;
    MatMulPlan plan;
    if (packed_b_guard_.has_value() && have_packed_plan_) {
      plan = packed_plan_;
      output_shape = packed_output_shape_;
    } else {
      ORT_RETURN_IF_ERROR(CreatePlan(shape_A, shape_B, plan, output_shape));
    }

    // Allocate output tensor
    Tensor* Y = context->Output(0, output_shape);
    ORT_RETURN_IF_NOT(Y != nullptr, "Failed to allocate output tensor");

    // Convert input tensors to zDNN format (with optional logical shape).
    zdnn_ztensor zdnn_a, zdnn_b, zdnn_c, zdnn_y;
    ORT_RETURN_IF_ERROR(TensorConverter::ConvertToZTensorWithShape(*A, plan.logical_shape_a, zdnn_a, plan.layout_a));
    ZTensorGuard guard_a(&zdnn_a);

    const zdnn_ztensor* b_ztensor = nullptr;
    std::optional<ZTensorGuard> guard_b;
    if (packed_b_guard_.has_value()) {
      b_ztensor = &packed_b_;
    } else {
      ORT_RETURN_IF_ERROR(TensorConverter::ConvertToZTensorWithShape(*B, plan.logical_shape_b, zdnn_b, plan.layout_b));
      guard_b.emplace(&zdnn_b);
      b_ztensor = &zdnn_b;
    }

    ORT_RETURN_IF_ERROR(CreateZeroBiasZTensor(A->GetElementType(),
                                              plan.logical_shape_c,
                                              plan.layout_c,
                                              zdnn_c));
    ZTensorGuard guard_c(&zdnn_c);

    ORT_RETURN_IF_ERROR(TensorConverter::InitZTensorForOutputWithShape(*Y, plan.logical_shape_y, zdnn_y, plan.layout_y));
    ZTensorGuard guard_y(&zdnn_y);

    // Execute zDNN matrix multiplication (zDNN always expects an "input_c" bias vector/matrix).
    zdnn_status status{};
    if (plan.kind == MatMulPlan::Kind::kUnstacked || plan.kind == MatMulPlan::Kind::kStacked) {
      status = zdnn_matmul_op(&zdnn_a, b_ztensor, &zdnn_c, MATMUL_OP_ADDITION, &zdnn_y);
      ORT_RETURN_IF_ERROR(CheckStatus(status, "zdnn_matmul_op"));
    } else {
      status = zdnn_matmul_bcast_op(&zdnn_a, b_ztensor, &zdnn_c, MATMUL_BCAST_OP_ADDITION, &zdnn_y);
      ORT_RETURN_IF_ERROR(CheckStatus(status, "zdnn_matmul_bcast_op"));
    }

    // Convert result back to ORT tensor
    ORT_RETURN_IF_ERROR(ConvertFromZTensor(zdnn_y, *Y));

    return Status::OK();
  }

  private:
  struct MatMulPlan {
    enum class Kind { kUnstacked, kStacked, kBcast1, kBcast23 };

    Kind kind{};
    zdnn_data_layouts layout_a{};
    zdnn_data_layouts layout_b{};
    zdnn_data_layouts layout_c{};
    zdnn_data_layouts layout_y{};
    TensorShape logical_shape_a;
    TensorShape logical_shape_b;
    TensorShape logical_shape_c;
    TensorShape logical_shape_y;
  };

  // Prepacked RHS operand B (optional). When set, input B is removed as an initializer and Compute() uses `packed_b_`.
  mutable std::optional<ZTensorGuard> packed_b_guard_;
  mutable zdnn_ztensor packed_b_{};
  mutable TensorShape packed_b_shape_;
  mutable bool have_packed_plan_{false};
  mutable MatMulPlan packed_plan_{};
  mutable TensorShape packed_output_shape_;

  static bool AllOnes(const std::vector<int64_t>& dims) {
    for (int64_t d : dims) {
      if (d != 1) return false;
    }
    return true;
  }

  static std::vector<int64_t> AlignBatchDims(const std::vector<int64_t>& batch_dims, size_t out_rank) {
    if (batch_dims.size() >= out_rank) return batch_dims;
    std::vector<int64_t> aligned(out_rank - batch_dims.size(), 1);
    aligned.insert(aligned.end(), batch_dims.begin(), batch_dims.end());
    return aligned;
  }

  static int64_t Product(const std::vector<int64_t>& dims) {
    int64_t p = 1;
    for (int64_t d : dims) {
      p *= d;
    }
    return p;
  }

  static Status BroadcastBatchDims(const std::vector<int64_t>& a_batch,
                                   const std::vector<int64_t>& b_batch,
                                   std::vector<int64_t>& out_batch) {
    const size_t rank_a = a_batch.size();
    const size_t rank_b = b_batch.size();
    const size_t out_rank = std::max(rank_a, rank_b);
    out_batch.assign(out_rank, 1);

    for (size_t i = 0; i < out_rank; ++i) {
      const int64_t dim_a = (i < rank_a) ? a_batch[rank_a - 1 - i] : 1;
      const int64_t dim_b = (i < rank_b) ? b_batch[rank_b - 1 - i] : 1;

      if (dim_a == dim_b) {
        out_batch[out_rank - 1 - i] = dim_a;
      } else if (dim_a == 1) {
        out_batch[out_rank - 1 - i] = dim_b;
      } else if (dim_b == 1) {
        out_batch[out_rank - 1 - i] = dim_a;
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "MatMul batch dimensions are not compatible: dim_a=",
                               dim_a, " dim_b=", dim_b);
      }
    }
    return Status::OK();
  }

  static Status CreatePlan(const TensorShape& shape_a,
                           const TensorShape& shape_b,
                           MatMulPlan& plan,
                           TensorShape& output_shape) {
    const auto& dims_a = shape_a.GetDims();
    const auto& dims_b = shape_b.GetDims();

    if (dims_a.size() < 2 || dims_b.size() < 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "MatMul requires at least 2D tensors. Got shapes: ",
                             shape_a.ToString(), " and ", shape_b.ToString());
    }

    const int64_t M = dims_a[dims_a.size() - 2];
    const int64_t K_a = dims_a[dims_a.size() - 1];
    const int64_t K_b = dims_b[dims_b.size() - 2];
    const int64_t N = dims_b[dims_b.size() - 1];

    if (K_a != K_b) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "MatMul inner dimensions must match. Got K_A=", K_a,
                             " and K_B=", K_b);
    }

    const std::vector<int64_t> a_batch(dims_a.begin(), dims_a.end() - 2);
    const std::vector<int64_t> b_batch(dims_b.begin(), dims_b.end() - 2);

    std::vector<int64_t> out_batch;
    ORT_RETURN_IF_ERROR(BroadcastBatchDims(a_batch, b_batch, out_batch));

    std::vector<int64_t> out_dims = out_batch;
    out_dims.push_back(M);
    out_dims.push_back(N);
    output_shape = TensorShape(out_dims);

    const int64_t stack = Product(out_batch);

    const auto a_batch_aligned = AlignBatchDims(a_batch, out_batch.size());
    const auto b_batch_aligned = AlignBatchDims(b_batch, out_batch.size());

    const bool a_matches_output = (a_batch_aligned == out_batch);
    const bool b_matches_output = (b_batch_aligned == out_batch);
    const bool a_all_ones = AllOnes(a_batch_aligned);
    const bool b_all_ones = AllOnes(b_batch_aligned);

    // We treat "stack==1" as unstacked for zDNN (even if ONNX output has leading 1 dims).
    if (stack == 1) {
      if (!a_matches_output || !b_matches_output) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "MatMul batch dimensions are not compatible after broadcast");
      }

      plan.kind = MatMulPlan::Kind::kUnstacked;
      plan.layout_a = ZDNN_2D;
      plan.layout_b = ZDNN_2D;
      plan.layout_c = ZDNN_1D;
      plan.layout_y = ZDNN_2D;
      plan.logical_shape_a = TensorShape({M, K_a});
      plan.logical_shape_b = TensorShape({K_a, N});
      plan.logical_shape_c = TensorShape({N});
      plan.logical_shape_y = TensorShape({M, N});
      return Status::OK();
    }

    // stack > 1: must map to either stacked matmul_op, or bcast1/bcast23.
    if (a_matches_output && b_matches_output) {
      plan.kind = MatMulPlan::Kind::kStacked;
      plan.layout_a = ZDNN_3DS;
      plan.layout_b = ZDNN_3DS;
      plan.layout_c = ZDNN_2DS;
      plan.layout_y = ZDNN_3DS;
      plan.logical_shape_a = TensorShape({stack, M, K_a});
      plan.logical_shape_b = TensorShape({stack, K_a, N});
      plan.logical_shape_c = TensorShape({stack, N});
      plan.logical_shape_y = TensorShape({stack, M, N});
      return Status::OK();
    }

    if (a_matches_output && b_all_ones) {
      plan.kind = MatMulPlan::Kind::kBcast23;
      plan.layout_a = ZDNN_3DS;
      plan.layout_b = ZDNN_2D;
      plan.layout_c = ZDNN_1D;
      plan.layout_y = ZDNN_3DS;
      plan.logical_shape_a = TensorShape({stack, M, K_a});
      plan.logical_shape_b = TensorShape({K_a, N});
      plan.logical_shape_c = TensorShape({N});
      plan.logical_shape_y = TensorShape({stack, M, N});
      return Status::OK();
    }

    if (a_all_ones && b_matches_output) {
      plan.kind = MatMulPlan::Kind::kBcast1;
      plan.layout_a = ZDNN_2D;
      plan.layout_b = ZDNN_3DS;
      plan.layout_c = ZDNN_2DS;
      plan.layout_y = ZDNN_3DS;
      plan.logical_shape_a = TensorShape({M, K_a});
      plan.logical_shape_b = TensorShape({stack, K_a, N});
      plan.logical_shape_c = TensorShape({stack, N});
      plan.logical_shape_y = TensorShape({stack, M, N});
      return Status::OK();
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "Telum EP: MatMul supports only: "
                           "(1) unstacked 2D, "
                           "(2) stacked with identical batch dims, "
                           "(3) broadcast of a fully-unstacked operand across all batch dims.");
  }

  Status CreateZeroBiasZTensor(int32_t ort_type,
                               const TensorShape& bias_shape,
                               zdnn_data_layouts bias_layout,
                               zdnn_ztensor& bias_ztensor) const {
    const auto zdnn_type = MapONNXTypeToZDNN(ort_type);
    const size_t elem_size = GetZDNNTypeSize(zdnn_type);
    const size_t bytes = static_cast<size_t>(bias_shape.Size()) * elem_size;
    std::vector<uint8_t> zeros(bytes, 0);

    return TensorConverter::ConvertRawToZTensor(zeros.data(), ort_type, bias_shape, bias_ztensor, bias_layout);
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
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    MatMul);

ONNX_OPERATOR_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    MatMul);

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
