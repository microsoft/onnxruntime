// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../telum_kernel_common.h"
#include "core/providers/common.h"

#include <optional>

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
    // Attributes are optional in ONNX and have defaults.
    int64_t temp = 0;
    if (info.GetAttr<int64_t>("transA", &temp).IsOK()) {
      trans_A_ = (temp != 0);
    } else {
      trans_A_ = false;
    }

    temp = 0;
    if (info.GetAttr<int64_t>("transB", &temp).IsOK()) {
      trans_B_ = (temp != 0);
    } else {
      trans_B_ = false;
    }

    alpha_ = 1.0f;
    (void)info.GetAttr<float>("alpha", &alpha_);

    beta_ = 1.0f;
    (void)info.GetAttr<float>("beta", &beta_);
  }

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr /*alloc*/,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* /*prepacked_weights*/) override {
    is_packed = false;

    // Prepack weight matrix B (input index 1) when it is a constant initializer.
    // This avoids repeated (expensive) zDNN transformations per inference.
    if (input_idx == 1) {
      // Reset any existing packed state first.
      packed_b_guard_.reset();

      packed_b_shape_ = tensor.Shape();

      ORT_RETURN_IF_ERROR(ConvertToZTensor(tensor, packed_b_, ZDNN_2D));
      packed_b_guard_.emplace(&packed_b_);

      is_packed = true;
      return Status::OK();
    }

    // Optionally prepack the bias vector C (input index 2) if it can be safely fused:
    // alpha == 1, beta == 1, and C is a bias vector of length N.
    if (input_idx == 2 && alpha_ == 1.0f && beta_ == 1.0f) {
      const auto* node_b = Node().InputDefs().size() > 1 ? Node().InputDefs()[1] : nullptr;
      if (node_b == nullptr || node_b->Shape() == nullptr) {
        return Status::OK();
      }

      // Determine N from B's (static) shape and transB attribute.
      std::vector<int64_t> b_dims;
      for (const auto& dim : node_b->Shape()->dim()) {
        if (!dim.has_dim_value()) return Status::OK();
        b_dims.push_back(dim.dim_value());
      }
      if (b_dims.size() != 2) return Status::OK();

      const int64_t N = trans_B_ ? b_dims[0] : b_dims[1];
      if (!IsBiasVector(tensor, N)) {
        return Status::OK();
      }

      packed_bias_guard_.reset();
      ORT_RETURN_IF_ERROR(TensorConverter::ConvertToZTensorWithShape(tensor, TensorShape({N}), packed_bias_, ZDNN_1D));
      packed_bias_guard_.emplace(&packed_bias_);

      is_packed = true;
      return Status::OK();
    }

    return Status::OK();
  }

  Status Compute(OpKernelContext* context) const override {
    // Get input tensors
    const Tensor* A = context->Input<Tensor>(0);
    const Tensor* B = packed_b_guard_.has_value() ? nullptr : context->Input<Tensor>(1);
    const Tensor* C = context->Input<Tensor>(2);  // Bias (optional)

    ORT_RETURN_IF_NOT(A != nullptr, "Input A is null");
    ORT_RETURN_IF_NOT(B != nullptr || packed_b_guard_.has_value(), "Input B is null");

    // Validate shapes are static
    ORT_RETURN_IF_ERROR(ValidateStaticShape(A->Shape()));
    if (B != nullptr) {
      ORT_RETURN_IF_ERROR(ValidateStaticShape(B->Shape()));
    } else {
      ORT_RETURN_IF_ERROR(ValidateStaticShape(packed_b_shape_));
    }
    if (C != nullptr) {
      ORT_RETURN_IF_ERROR(ValidateStaticShape(C->Shape()));
    }

    // Get shapes
    const auto& shape_A = A->Shape();
    const auto& shape_B = (B != nullptr) ? B->Shape() : packed_b_shape_;

    int64_t M{}, K{}, N{};
    ORT_RETURN_IF_ERROR(ValidateAndGetDims(shape_A, shape_B, M, K, N));

    // Output is always [M, N]
    TensorShape output_shape({M, N});

    // Allocate output tensor
    Tensor* Y = context->Output(0, output_shape);
    ORT_RETURN_IF_NOT(Y != nullptr, "Failed to allocate output tensor");

    // zDNN expects a bias vector (input_c) even when we want "no bias".
    // We'll pass either:
    // - the actual bias vector (when we can fuse it safely), or
    // - a zero vector (and optionally apply beta*C on CPU post-processing).
    const bool can_fuse_bias =
        packed_bias_guard_.has_value() ||
        ((C != nullptr) && (alpha_ == 1.0f) && (beta_ == 1.0f) && IsBiasVector(*C, N));

    // zDNN matmul op shapes:
    //   input_a: ZDNN_2D (m, k)
    //   input_b: ZDNN_2D (k, n)
    //   input_c: ZDNN_1D (n)
    //   output : ZDNN_2D (m, n)
    constexpr zdnn_data_layouts kMatLayout = ZDNN_2D;
    constexpr zdnn_data_layouts kBiasLayout = ZDNN_1D;

    // Convert input tensors to zDNN format
    zdnn_ztensor zdnn_a, zdnn_b, zdnn_c, zdnn_y;
    ORT_RETURN_IF_ERROR(ConvertToZTensor(*A, zdnn_a, kMatLayout));
    ZTensorGuard guard_a(&zdnn_a);

    const zdnn_ztensor* b_ztensor = nullptr;
    std::optional<ZTensorGuard> guard_b;
    if (packed_b_guard_.has_value()) {
      b_ztensor = &packed_b_;
    } else {
      ORT_RETURN_IF_ERROR(ConvertToZTensor(*B, zdnn_b, kMatLayout));
      guard_b.emplace(&zdnn_b);
      b_ztensor = &zdnn_b;
    }

    const zdnn_ztensor* c_ztensor = nullptr;
    std::optional<ZTensorGuard> guard_c;
    if (can_fuse_bias) {
      if (packed_bias_guard_.has_value()) {
        c_ztensor = &packed_bias_;
      } else {
        // Treat [N] or [1,N] as a bias vector of length N.
        ORT_RETURN_IF_ERROR(TensorConverter::ConvertToZTensorWithShape(*C, TensorShape({N}), zdnn_c, kBiasLayout));
        guard_c.emplace(&zdnn_c);
        c_ztensor = &zdnn_c;
      }
    } else {
      ORT_RETURN_IF_ERROR(CreateZeroBiasZTensor(A->GetElementType(), N, zdnn_c));
      guard_c.emplace(&zdnn_c);
      c_ztensor = &zdnn_c;
    }

    ORT_RETURN_IF_ERROR(InitZTensorForOutput(*Y, zdnn_y, kMatLayout));
    ZTensorGuard guard_y(&zdnn_y);

    // Execute zDNN GEMM (MatMul + bias vector addition).
    zdnn_status status{};
    if (trans_A_ || trans_B_) {
      status = zdnn_matmul_transpose_op(&zdnn_a, b_ztensor, c_ztensor,
                                        trans_A_, trans_B_,
                                        MATMUL_OP_ADDITION, &zdnn_y);
      ORT_RETURN_IF_ERROR(CheckStatus(status, "zdnn_matmul_transpose_op (Gemm)"));
    } else {
      status = zdnn_matmul_op(&zdnn_a, b_ztensor, c_ztensor,
                              MATMUL_OP_ADDITION, &zdnn_y);
      ORT_RETURN_IF_ERROR(CheckStatus(status, "zdnn_matmul_op (Gemm)"));
    }

    // Convert result back to ORT tensor
    ORT_RETURN_IF_ERROR(ConvertFromZTensor(zdnn_y, *Y));

    // Apply alpha scaling and beta*C (broadcast) post-processing on CPU when needed.
    // Note: If we fused the bias into zDNN, alpha==beta==1, so we can skip.
    if (alpha_ != 1.0f) {
      ORT_RETURN_IF_ERROR(ScaleInPlace(*Y, alpha_));
    }

    if (!can_fuse_bias && C != nullptr && beta_ != 0.0f) {
      ORT_RETURN_IF_ERROR(AddBroadcastedBiasInPlace(*Y, *C, beta_, M, N));
    }

    return Status::OK();
  }

 private:
  bool trans_A_;
  bool trans_B_;
  float alpha_;
  float beta_;

  // Prepacked weights/bias (optional).
  mutable std::optional<ZTensorGuard> packed_b_guard_;
  mutable zdnn_ztensor packed_b_{};
  mutable TensorShape packed_b_shape_;

  mutable std::optional<ZTensorGuard> packed_bias_guard_;
  mutable zdnn_ztensor packed_bias_{};

  /**
   * @brief Validate GEMM dimensions
   */
  Status ValidateAndGetDims(const TensorShape& shape_A,
                            const TensorShape& shape_B,
                            int64_t& M,
                            int64_t& K,
                            int64_t& N) const {
    const auto& dims_A = shape_A.GetDims();
    const auto& dims_B = shape_B.GetDims();

    // A and B must be 2D
    if (dims_A.size() != 2 || dims_B.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "Gemm requires 2D tensors. Got shapes: ",
                            shape_A.ToString(), " and ", shape_B.ToString());
    }

    // Check dimension compatibility
    M = trans_A_ ? dims_A[1] : dims_A[0];
    const int64_t K_A = trans_A_ ? dims_A[0] : dims_A[1];
    const int64_t K_B = trans_B_ ? dims_B[1] : dims_B[0];
    N = trans_B_ ? dims_B[0] : dims_B[1];

    if (K_A != K_B) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "Gemm inner dimensions must match. Got K_A=", K_A,
                            " and K_B=", K_B);
    }
    K = K_A;
    return Status::OK();
  }

  static bool IsBiasVector(const Tensor& bias, int64_t N) {
    const auto& dims = bias.Shape().GetDims();
    if (dims.empty()) {
      return false;
    }
    if (dims.size() == 1) {
      return dims[0] == N;
    }
    if (dims.size() == 2) {
      return dims[0] == 1 && dims[1] == N;
    }
    return false;
  }

  Status CreateZeroBiasZTensor(int32_t ort_type, int64_t N, zdnn_ztensor& bias_ztensor) const {
    const auto zdnn_type = MapONNXTypeToZDNN(ort_type);
    const size_t elem_size = GetZDNNTypeSize(zdnn_type);
    const size_t bytes = static_cast<size_t>(N) * elem_size;
    std::vector<uint8_t> zeros(bytes, 0);
    return TensorConverter::ConvertRawToZTensor(zeros.data(), ort_type, TensorShape({N}), bias_ztensor, ZDNN_1D);
  }

  static Status ScaleInPlace(Tensor& y, float alpha) {
    if (alpha == 1.0f) return Status::OK();

    const size_t n = static_cast<size_t>(y.Shape().Size());
    if (y.IsDataType<float>()) {
      float* data = y.MutableData<float>();
      for (size_t i = 0; i < n; ++i) {
        data[i] *= alpha;
      }
      return Status::OK();
    }

    if (y.IsDataType<MLFloat16>()) {
      MLFloat16* data = y.MutableData<MLFloat16>();
      for (size_t i = 0; i < n; ++i) {
        data[i] = MLFloat16(static_cast<float>(data[i]) * alpha);
      }
      return Status::OK();
    }

    if (y.IsDataType<BFloat16>()) {
      BFloat16* data = y.MutableData<BFloat16>();
      for (size_t i = 0; i < n; ++i) {
        data[i] = BFloat16(static_cast<float>(data[i]) * alpha);
      }
      return Status::OK();
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported Gemm output type for Telum EP");
  }

  static Status AddBroadcastedBiasInPlace(Tensor& y,
                                         const Tensor& c,
                                         float beta,
                                         int64_t M,
                                         int64_t N) {
    const auto& cdims = c.Shape().GetDims();

    // Gemm supports C broadcastable to [M, N].
    if (cdims.size() > 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Gemm bias must be 0D/1D/2D, got rank ", cdims.size());
    }

    auto get_c_index = [&](int64_t i, int64_t j) -> int64_t {
      if (cdims.empty()) {
        return 0;  // scalar
      }
      if (cdims.size() == 1) {
        const int64_t cN = cdims[0];
        if (cN == 1) return 0;  // scalar
        if (cN == N) return j;
        // allow [M] only when N == 1 (column bias)
        if (cN == M && N == 1) return i;
        return -1;
      }

      // 2D
      const int64_t cM = cdims[0];
      const int64_t cN = cdims[1];

      const int64_t ii = (cM == 1) ? 0 : i;
      const int64_t jj = (cN == 1) ? 0 : j;

      if (!((cM == 1 || cM == M) && (cN == 1 || cN == N))) {
        return -1;
      }

      return ii * cN + jj;
    };

    const size_t y_size = static_cast<size_t>(y.Shape().Size());

    if (y.IsDataType<float>()) {
      float* y_data = y.MutableData<float>();
      const float* c_data = c.Data<float>();

      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          const int64_t c_idx = get_c_index(i, j);
          if (c_idx < 0) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                   "Gemm bias shape ", c.Shape().ToString(),
                                   " is not broadcastable to [", M, ",", N, "]");
          }
          const size_t y_idx = static_cast<size_t>(i * N + j);
          ORT_RETURN_IF_NOT(y_idx < y_size, "Output index out of bounds");
          y_data[y_idx] += beta * c_data[c_idx];
        }
      }
      return Status::OK();
    }

    if (y.IsDataType<MLFloat16>()) {
      MLFloat16* y_data = y.MutableData<MLFloat16>();
      const MLFloat16* c_data = c.Data<MLFloat16>();

      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          const int64_t c_idx = get_c_index(i, j);
          if (c_idx < 0) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                   "Gemm bias shape ", c.Shape().ToString(),
                                   " is not broadcastable to [", M, ",", N, "]");
          }
          const size_t y_idx = static_cast<size_t>(i * N + j);
          ORT_RETURN_IF_NOT(y_idx < y_size, "Output index out of bounds");

          const float yv = static_cast<float>(y_data[y_idx]);
          const float cv = static_cast<float>(c_data[c_idx]);
          y_data[y_idx] = MLFloat16(yv + beta * cv);
        }
      }
      return Status::OK();
    }

    if (y.IsDataType<BFloat16>()) {
      BFloat16* y_data = y.MutableData<BFloat16>();
      const BFloat16* c_data = c.Data<BFloat16>();

      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          const int64_t c_idx = get_c_index(i, j);
          if (c_idx < 0) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                   "Gemm bias shape ", c.Shape().ToString(),
                                   " is not broadcastable to [", M, ",", N, "]");
          }
          const size_t y_idx = static_cast<size_t>(i * N + j);
          ORT_RETURN_IF_NOT(y_idx < y_size, "Output index out of bounds");

          const float yv = static_cast<float>(y_data[y_idx]);
          const float cv = static_cast<float>(c_data[c_idx]);
          y_data[y_idx] = BFloat16(yv + beta * cv);
        }
      }
      return Status::OK();
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported Gemm output type for Telum EP");
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
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Gemm);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    9, 10,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Gemm);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    11, 12,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Gemm);

ONNX_OPERATOR_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Gemm);

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
