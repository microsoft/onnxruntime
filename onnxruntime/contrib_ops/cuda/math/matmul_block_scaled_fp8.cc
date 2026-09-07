// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp8.h"

#include <algorithm>
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

size_t DequantScratchLimitBytes() {
  const int64_t limit = ParseEnvironmentVariableWithDefault<int64_t>("ORT_FP8_DEQUANT_SCRATCH_MIB", 256);
  ORT_ENFORCE(limit > 0, "ORT_FP8_DEQUANT_SCRATCH_MIB must be positive.");
  return SafeInt<size_t>(limit) * 1024 * 1024;
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
    : CudaKernel(info),
      block_size_(info.GetAttrOrDefault<int64_t>("block_size", 128)),
      max_dequant_scratch_bytes_(DequantScratchLimitBytes()) {
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

  if (k_i == 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(Y->MutableDataRaw(), 0, Y->SizeInBytes(), Stream(context)));
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
  }

  // Optional W8A8 activation path: statically quantize A to FP8 E4M3 and dequantize back so the
  // GEMM sees the same activation rounding as native W8A8 execution. When a_scale is absent the
  // activation is kept at full FP16/BF16 precision (weight-only W8A16).
  //
  // The decode GEMV absorbs this round trip in-register, so the standalone kernel (and its [M, K]
  // scratch buffer) is only materialized for the cuBLAS path below.
  const bool use_gemv = m_i > 0 &&
                        m_i <= MatMulBlockScaledFp8GemvMaxM(k_i, SafeInt<int>(block_size_), GetDeviceProp()) &&
                        (k_i % 16 == 0) && (block_size_ % 16 == 0);
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

  // Dequantize the FP8 weight into a scratch buffer of the activation type, then GEMM. The scratch
  // is tiled over N because a full [N, K] buffer is 2.37 GiB for a 248320 x 5120 LM head.
  const int64_t rows_per_scratch =
      static_cast<int64_t>(max_dequant_scratch_bytes_ / (static_cast<size_t>(k) * sizeof(CudaT)));
  const int64_t tile_rows = std::clamp<int64_t>(rows_per_scratch, 1, n);

  IAllocatorUniquePtr<CudaT> b_dequant =
      GetScratchBuffer<CudaT>(SafeInt<size_t>(tile_rows) * SafeInt<size_t>(k), GetComputeStream(context));

  const CudaT alpha = ToCudaType<T>::FromFloat(1.f);
  const CudaT zero = ToCudaType<T>::FromFloat(0.f);
  const auto* b_data = static_cast<const uint8_t*>(b->DataRaw());
  const float* b_scale_data = b_scale->Data<float>();

  for (int64_t n_offset = 0; n_offset < n; n_offset += tile_rows) {
    const int rows = SafeInt<int>(std::min<int64_t>(tile_rows, n - n_offset));
    const size_t row_offset = static_cast<size_t>(n_offset);
    ORT_RETURN_IF_ERROR(LaunchDequantizeBlockScaledFp8(
        b_dequant.get(),
        b_data + row_offset * static_cast<size_t>(k),
        b_scale_data + row_offset * static_cast<size_t>(k_blocks),
        rows,
        SafeInt<int>(k),
        SafeInt<int>(block_size_),
        std::is_same<T, BFloat16>::value,
        Stream(context)));

    // The row-major [M, N] output is column-major [N, M] to cuBLAS, so an N tile is a row offset.
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        GetCublasHandle(context),
        CUBLAS_OP_T,  // transB: dequantized weight is [N, K] row-major == K-major [K, N]
        CUBLAS_OP_N,  // transA
        rows,
        m_i,
        k_i,
        &alpha,
        b_dequant.get(),
        helper.Ldb(transb),
        reinterpret_cast<const CudaT*>(a_ptr),
        helper.Lda(transa),
        &zero,
        reinterpret_cast<CudaT*>(Y->MutableDataRaw()) + n_offset,
        helper.Ldc(),
        GetDeviceProp(),
        UseTF32()));
  }

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
