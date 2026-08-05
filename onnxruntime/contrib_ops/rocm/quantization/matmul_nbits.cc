// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/quantization/matmul_nbits.h"
#include "contrib_ops/rocm/quantization/dequantize_nbits_rocm.cuh"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

// ---------------------------------------------------------------------------
// Kernel registration
// ---------------------------------------------------------------------------
#define REGISTER_MATMULNBITS(T)                                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      MatMulNBits,                                                              \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kRocmExecutionProvider,                                                   \
      (*KernelDefBuilder::Create())                                             \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())               \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())         \
          .TypeConstraint("T3", {DataTypeImpl::GetTensorType<uint8_t>(),        \
                                 DataTypeImpl::GetTensorType<T>()}),            \
      MatMulNBits<T>);

REGISTER_MATMULNBITS(float)
REGISTER_MATMULNBITS(MLFloat16)

// ---------------------------------------------------------------------------
// Lightweight input validation (avoids including matmul_nbits_helper.h
// which pulls in non-provider-API headers that conflict in shared library TUs)
// ---------------------------------------------------------------------------
static Status ValidateMatMulNBitsInputs(
    const Tensor* b, const Tensor* scales,
    int64_t N, int64_t K, int64_t block_size, int64_t nbits) {
  if (nbits != 4 && nbits != 8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "MatMulNBits ROCm: only bits=4 and bits=8 are supported, got ", nbits);
  }
  if (block_size < 16 || (block_size & (block_size - 1)) != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "MatMulNBits ROCm: block_size must be a power-of-2 >= 16, got ", block_size);
  }
  ORT_UNUSED_PARAMETER(b);
  ORT_UNUSED_PARAMETER(scales);
  ORT_UNUSED_PARAMETER(N);
  ORT_UNUSED_PARAMETER(K);
  return Status::OK();
}

// ---------------------------------------------------------------------------
// ComputeInternal
// ---------------------------------------------------------------------------
template <typename T>
Status MatMulNBits<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* a           = ctx->Input<Tensor>(0);
  const Tensor* b           = ctx->Input<Tensor>(1);
  const Tensor* scales      = ctx->Input<Tensor>(2);
  const Tensor* zero_points = ctx->Input<Tensor>(3);  // optional
  const Tensor* reorder_idx = ctx->Input<Tensor>(4);  // optional

  ORT_RETURN_IF_ERROR(ValidateMatMulNBitsInputs(b, scales, N_, K_, block_size_, nbits_));

  // g_idx (reorder_idx) reshuffles K-blocks across columns.  The column-wise
  // dequant kernel assumes the standard exporter layout.
  if (reorder_idx != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "MatMulNBits ROCm: reorder_idx (g_idx) is not "
                           "supported. Use the CPU EP for this variant.");
  }

  // Compute output shape.
  constexpr bool transa = false;
  constexpr bool transb = true;
  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, transa, transb));

  Tensor* Y = ctx->Output(0, helper.OutputShape());
  if (Y->Shape().Size() == 0) return Status::OK();

  typedef typename ToHipType<T>::MappedType HipT;

  hipStream_t stream = Stream(ctx);

  // -------------------------------------------------------------------------
  // Step 1: dequantize packed weights → dense fp matrix [N, K_padded]
  // -------------------------------------------------------------------------
  const int64_t K_padded =
      (K_ + block_size_ - 1) / block_size_ * block_size_;

  IAllocatorUniquePtr<T> b_data_ptr = GetScratchBuffer<T>(
      SafeInt<size_t>(N_) * SafeInt<size_t>(K_padded),
      ctx->GetComputeStream());
  HipT* b_dequant = reinterpret_cast<HipT*>(b_data_ptr.get());

  const unsigned char* quant_data = b->Data<uint8_t>();
  const HipT* scales_data =
      reinterpret_cast<const HipT*>(scales->Data<T>());
  const unsigned char* zp_data =
      (zero_points != nullptr) ? zero_points->Data<uint8_t>() : nullptr;

  if (nbits_ == 4) {
    HIP_RETURN_IF_ERROR(LaunchDequantize4Bits<HipT>(
        b_dequant, quant_data, scales_data, zp_data,
        static_cast<int>(K_padded), static_cast<int>(N_),
        static_cast<int>(block_size_), stream));
  } else {
    HIP_RETURN_IF_ERROR(LaunchDequantize8Bits<HipT>(
        b_dequant, quant_data, scales_data, zp_data,
        static_cast<int>(K_padded), static_cast<int>(N_),
        static_cast<int>(block_size_), stream));
  }

  // -------------------------------------------------------------------------
  // Step 2: GEMM  Y = A * B_dequant^T   via hipBLAS
  //
  // hipBLAS is column-major.  ORT tensors are row-major.
  // We call  C = op(B) * op(A) where:
  //   op(B) = B_dequant^T  (HIPBLAS_OP_T),  lda = K_padded
  //   op(A) = A            (HIPBLAS_OP_N),  ldb = K
  //   C = Y^T (row-major Y)                 ldc = N
  // -------------------------------------------------------------------------
  const HipT* a_data =
      reinterpret_cast<const HipT*>(a->Data<T>());
  HipT* y_data =
      reinterpret_cast<HipT*>(Y->MutableData<T>());

  const HipT alpha = ToHipType<T>::FromFloat(1.0f);
  const HipT beta  = ToHipType<T>::FromFloat(0.0f);

  HIPBLAS_RETURN_IF_ERROR(hipblasGemmHelper(
      GetHipblasHandle(ctx),
      HIPBLAS_OP_T,
      HIPBLAS_OP_N,
      static_cast<int>(helper.N()),
      static_cast<int>(helper.M()),
      static_cast<int>(helper.K()),
      &alpha,
      b_dequant,
      static_cast<int>(K_padded),
      a_data,
      helper.Lda(transa),
      &beta,
      y_data,
      helper.Ldc()));

  return Status::OK();
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
