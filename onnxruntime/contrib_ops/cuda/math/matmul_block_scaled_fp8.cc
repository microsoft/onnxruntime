// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp8.h"

#include <type_traits>

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime::contrib::cuda {
using namespace onnxruntime::cuda;

namespace {
bool FusedFp8ActivationQdqDisabled() {
  static const bool disabled =
      ParseEnvironmentVariableWithDefault<bool>("ORT_DISABLE_FUSED_FP8_ACT_QDQ", false);
  return disabled;
}

bool RowwiseFp8PrefillEnabled() {
  static const bool enabled =
      ParseEnvironmentVariableWithDefault<bool>("ORT_ENABLE_ROW_WISE_FP8_GEMM", false);
  return enabled;
}

Status LaunchRowwiseFp8Gemm(cublasLtHandle_t cublas_lt,
                            cudaStream_t stream,
                            const void* a_fp8,
                            const void* b_fp8,
                            const float* scale_a,
                            const float* scale_b,
                            void* output,
                            int m,
                            int n,
                            int k,
                            bool is_bf16,
                            void* workspace,
                            size_t workspace_size,
                            bool& handled) {
#if !defined(DISABLE_FLOAT8_TYPES) && CUDA_VERSION >= 12090
  handled = false;
  cublasLtMatmulDesc_t operation_desc = nullptr;
  auto clean_operation_desc = gsl::finally([&operation_desc]() {
    if (operation_desc) {
      cublasLtMatmulDescDestroy(operation_desc);
    }
  });
  cublasLtMatrixLayout_t a_desc = nullptr;
  auto clean_a_desc = gsl::finally([&a_desc]() {
    if (a_desc) {
      cublasLtMatrixLayoutDestroy(a_desc);
    }
  });
  cublasLtMatrixLayout_t b_desc = nullptr;
  auto clean_b_desc = gsl::finally([&b_desc]() {
    if (b_desc) {
      cublasLtMatrixLayoutDestroy(b_desc);
    }
  });
  cublasLtMatrixLayout_t c_desc = nullptr;
  auto clean_c_desc = gsl::finally([&c_desc]() {
    if (c_desc) {
      cublasLtMatrixLayoutDestroy(c_desc);
    }
  });
  cublasLtMatrixLayout_t d_desc = nullptr;
  auto clean_d_desc = gsl::finally([&d_desc]() {
    if (d_desc) {
      cublasLtMatrixLayoutDestroy(d_desc);
    }
  });
  cublasLtMatmulPreference_t preference = nullptr;
  auto clean_preference = gsl::finally([&preference]() {
    if (preference) {
      cublasLtMatmulPreferenceDestroy(preference);
    }
  });

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t trans_a = CUBLAS_OP_N;
  const cublasOperation_t trans_b = CUBLAS_OP_T;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operation_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(scale_a)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operation_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(scale_b)));
  const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operation_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operation_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
  constexpr int8_t fast_accumulation = 1;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      operation_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accumulation, sizeof(fast_accumulation)));

  const cudaDataType_t output_type = is_bf16 ? CUDA_R_16BF : CUDA_R_16F;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, m, k, k));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, n, k, k));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&c_desc, output_type, m, n, n));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&d_desc, output_type, m, n, n));
  const cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
  for (cublasLtMatrixLayout_t desc : {a_desc, b_desc, c_desc, d_desc}) {
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));
  }

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
  cublasLtMatmulHeuristicResult_t heuristic_result{};
  int returned_results = 0;
  const cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
      cublas_lt, operation_desc, a_desc, b_desc, c_desc, d_desc, preference,
      1, &heuristic_result, &returned_results);
  if (heuristic_status == CUBLAS_STATUS_NOT_SUPPORTED ||
      (heuristic_status == CUBLAS_STATUS_SUCCESS && returned_results == 0)) {
    return Status::OK();
  }
  CUBLAS_RETURN_IF_ERROR(heuristic_status);

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(
      cublas_lt, operation_desc, &alpha,
      a_fp8, a_desc, b_fp8, b_desc, &beta,
      output, c_desc, output, d_desc,
      &heuristic_result.algo, workspace, workspace_size, stream));
  handled = true;
  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(cublas_lt);
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(a_fp8);
  ORT_UNUSED_PARAMETER(b_fp8);
  ORT_UNUSED_PARAMETER(scale_a);
  ORT_UNUSED_PARAMETER(scale_b);
  ORT_UNUSED_PARAMETER(output);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(workspace);
  ORT_UNUSED_PARAMETER(workspace_size);
  ORT_UNUSED_PARAMETER(handled);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Row-wise FP8 GEMM requires CUDA 12.9 or later.");
#endif
}
}  // namespace

#if !defined(DISABLE_FLOAT8_TYPES)
ONNX_OPERATOR_KERNEL_EX(
    MatMulBlockQuantizedFp8Weight,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, BFloat16>())
        .TypeConstraint("T1", BuildKernelDefConstraints<Float8E4M3FN>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float>()),
    MatMulBlockQuantizedFp8Weight);
#endif

MatMulBlockQuantizedFp8Weight::MatMulBlockQuantizedFp8Weight(const OpKernelInfo& info)
    : CudaKernel(info), block_size_(info.GetAttrOrDefault<int64_t>("block_size", 128)) {
  ORT_ENFORCE(block_size_ > 0, "block_size must be positive.");
}

template <typename T>
Status MatMulBlockQuantizedFp8Weight::ComputeImpl(OpKernelContext* context) const {
#if defined(DISABLE_FLOAT8_TYPES)
  ORT_UNUSED_PARAMETER(context);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp8Weight requires float8 support.");
#else
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* a = context->Input<Tensor>(0);
  const Tensor* b = context->Input<Tensor>(1);
  const Tensor* b_scale = context->Input<Tensor>(2);
  const Tensor* a_scale = context->Input<Tensor>(3);  // optional
  const Tensor* bias = context->Input<Tensor>(4);     // optional

  const auto& a_shape = a->Shape();
  ORT_ENFORCE(a_shape.NumDimensions() >= 1, "A must have rank at least 1.");

  const int64_t a_rank = a_shape.NumDimensions();
  const int64_t k = a_shape[a_rank - 1];
  const auto& b_shape = b->Shape();
  ORT_ENFORCE(b_shape.NumDimensions() == 2 && b_shape[1] == k,
              "B must have shape [N, K] with K = ", k, ", got ", b_shape.ToString(), ".");
  const int64_t n = b_shape[0];
  const int64_t k_blocks = (k + block_size_ - 1) / block_size_;

  const auto& b_scale_shape = b_scale->Shape();
  ORT_ENFORCE(b_scale_shape.NumDimensions() == 2 && b_scale_shape[0] == n &&
                  b_scale_shape[1] == k_blocks,
              "b_scale must have shape [N, ceil(K/block_size)] = [", n, ", ", k_blocks, "], got ",
              b_scale_shape.ToString(), ".");
  if (a_scale != nullptr) {
    ORT_ENFORCE(a_scale->Shape().Size() == 1, "a_scale must be a scalar.");
  }
  if (bias != nullptr) {
    ORT_ENFORCE(bias->Shape().NumDimensions() == 1 && bias->Shape()[0] == n,
                "bias must have shape [N] = [", n, "], got ", bias->Shape().ToString(), ".");
  }

  constexpr bool transa = false;
  constexpr bool transb = true;
  MatMulComputeHelper helper;
  TensorShape b_logical_shape({n, k});
  ORT_RETURN_IF_ERROR(helper.Compute(a_shape, b_logical_shape, transa, transb));

  Tensor* Y = context->Output(0, helper.OutputShape());
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const int m_i = SafeInt<int>(helper.M());
  const int n_i = SafeInt<int>(helper.N());
  const int k_i = SafeInt<int>(helper.K());

  // Optional W8A8 activation path: statically quantize A to FP8 E4M3 and dequantize back so the
  // GEMM sees the same activation rounding as native W8A8 execution. When a_scale is absent the
  // activation is kept at full FP16/BF16 precision (weight-only W8A16).
  //
  // The decode GEMV absorbs this round trip in-register, so the standalone kernel (and its [M, K]
  // scratch buffer) is only materialized for the cuBLAS path below.
  constexpr int kGemvMaxM = 8;
  const bool use_gemv = m_i > 0 && m_i <= kGemvMaxM && (k_i % 16 == 0) && (block_size_ % 16 == 0);
  const bool fuse_act_qdq = use_gemv && !FusedFp8ActivationQdqDisabled();

  const void* a_ptr = a->DataRaw();
  IAllocatorUniquePtr<CudaT> a_dequant;
  if (a_scale != nullptr && !fuse_act_qdq) {
    a_dequant = GetScratchBuffer<CudaT>(SafeInt<size_t>(m_i) * SafeInt<size_t>(k_i),
                                        GetComputeStream(context));
    ORT_RETURN_IF_ERROR(LaunchQuantizeDequantizeActivationFp8(
        a_dequant.get(),
        a->DataRaw(),
        a_scale->Data<float>(),
        m_i,
        k_i,
        std::is_same<T, BFloat16>::value,
        Stream(context)));
    a_ptr = a_dequant.get();
  }

  // Decode fast path: for small M (autoregressive generation) this is a memory-bound GEMV.
  // A fused warp-per-column kernel reads the FP8 weight directly, avoiding both the [N, K]
  // dequant scratch buffer and the cuBLAS GEMM (which is underutilized at M == 1).
  if (use_gemv) {
    return LaunchMatMulBlockScaledFp8Gemv(
        Y->MutableDataRaw(),
        a_ptr,
        b->DataRaw(),
        b_scale->Data<float>(),
        bias != nullptr ? bias->DataRaw() : nullptr,
        (a_scale != nullptr && fuse_act_qdq) ? a_scale->Data<float>() : nullptr,
        m_i,
        n_i,
        k_i,
        SafeInt<int>(block_size_),
        std::is_same<T, BFloat16>::value,
        GetDeviceProp(),
        Stream(context));
  }

  const bool use_rowwise_fp8 = RowwiseFp8PrefillEnabled() && GetDeviceProp().major >= 9 &&
                               a_scale == nullptr && bias == nullptr && block_size_ >= k_i &&
                               n_i >= 4096 && (n_i % 16 == 0) && (k_i % 16 == 0);
  if (use_rowwise_fp8) {
    auto a_fp8 = GetScratchBuffer<uint8_t>(SafeInt<size_t>(m_i) * SafeInt<size_t>(k_i),
                                           GetComputeStream(context));
    auto row_scales = GetScratchBuffer<float>(m_i, GetComputeStream(context));
    ORT_RETURN_IF_ERROR(LaunchDynamicQuantizeActivationFp8(
        a_fp8.get(), row_scales.get(), a->DataRaw(), m_i, k_i,
        std::is_same<T, BFloat16>::value, Stream(context)));

    constexpr size_t kWorkspaceSize = 32 * 1024 * 1024;
    auto workspace = GetScratchBuffer<uint8_t>(kWorkspaceSize, GetComputeStream(context));
    bool handled = false;
    ORT_RETURN_IF_ERROR(LaunchRowwiseFp8Gemm(
        GetCublasLtHandle(context), Stream(context),
        a_fp8.get(), b->DataRaw(), row_scales.get(), b_scale->Data<float>(), Y->MutableDataRaw(),
        m_i, n_i, k_i, std::is_same<T, BFloat16>::value,
        workspace.get(), kWorkspaceSize, handled));
    if (handled) {
      return Status::OK();
    }
  }

  // Dequantize the FP8 weight into a scratch [N, K] buffer of the activation type, then GEMM.
  IAllocatorUniquePtr<CudaT> b_dequant = GetScratchBuffer<CudaT>(SafeInt<size_t>(n) * SafeInt<size_t>(k),
                                                                 GetComputeStream(context));
  ORT_RETURN_IF_ERROR(LaunchDequantizeBlockScaledFp8(
      b_dequant.get(),
      b->DataRaw(),
      b_scale->Data<float>(),
      SafeInt<int>(n),
      SafeInt<int>(k),
      SafeInt<int>(block_size_),
      std::is_same<T, BFloat16>::value,
      Stream(context)));

  const CudaT alpha = ToCudaType<T>::FromFloat(1.f);
  const CudaT zero = ToCudaType<T>::FromFloat(0.f);

  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      GetCublasHandle(context),
      CUBLAS_OP_T,  // transB: dequantized weight is [N, K] row-major == K-major [K, N]
      CUBLAS_OP_N,  // transA
      n_i,
      m_i,
      k_i,
      &alpha,
      b_dequant.get(),
      helper.Ldb(transb),
      reinterpret_cast<const CudaT*>(a_ptr),
      helper.Lda(transa),
      &zero,
      reinterpret_cast<CudaT*>(Y->MutableDataRaw()),
      helper.Ldc(),
      GetDeviceProp(),
      UseTF32()));

  if (bias != nullptr) {
    ORT_RETURN_IF_ERROR(LaunchAddBiasBlockScaledFp8(
        Y->MutableDataRaw(),
        bias->DataRaw(),
        m_i,
        n_i,
        std::is_same<T, BFloat16>::value,
        Stream(context)));
  }

  return Status::OK();
#endif
}

Status MatMulBlockQuantizedFp8Weight::ComputeInternal(OpKernelContext* context) const {
  const Tensor* a = context->Input<Tensor>(0);
  if (a->IsDataType<MLFloat16>()) {
    return ComputeImpl<MLFloat16>(context);
  }
  if (a->IsDataType<BFloat16>()) {
    return ComputeImpl<BFloat16>(context);
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "MatMulBlockQuantizedFp8Weight only supports FP16 or BF16 activations.");
}

}  // namespace onnxruntime::contrib::cuda
