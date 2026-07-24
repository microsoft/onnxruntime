// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp8.h"

#include <type_traits>

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime::contrib::cuda {
using namespace onnxruntime::cuda;

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
  const void* a_ptr = a->DataRaw();
  IAllocatorUniquePtr<CudaT> a_dequant;
  if (a_scale != nullptr) {
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
  constexpr int kGemvMaxM = 8;
  if (m_i > 0 && m_i <= kGemvMaxM && (k_i % 16 == 0) && (block_size_ % 16 == 0)) {
    return LaunchMatMulBlockScaledFp8Gemv(
        Y->MutableDataRaw(),
        a_ptr,
        b->DataRaw(),
        b_scale->Data<float>(),
        bias != nullptr ? bias->DataRaw() : nullptr,
        m_i,
        n_i,
        k_i,
        SafeInt<int>(block_size_),
        std::is_same<T, BFloat16>::value,
        Stream(context));
  }

  // Dequantize the FP8 weight into a scratch [N, K] buffer of the activation type, then GEMM.
  IAllocatorUniquePtr<CudaT> b_dequant = GetScratchBuffer<CudaT>(SafeInt<size_t>(n) * SafeInt<size_t>(k),
                                                                 context->GetComputeStream());
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
