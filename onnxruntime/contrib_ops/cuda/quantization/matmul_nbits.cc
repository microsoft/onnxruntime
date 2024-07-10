// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_nbits.h"

#include <cstdint>

#include "core/common/status.h"
#include "core/framework/float16.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "matmul_nbits.cuh"
#include "dequantize_blockwise.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if !defined(USE_MIGRAPHX) && !defined(USE_ROCM)
template <>
Status MatMulNBits<MLFloat16>::PrepackedGemm(
    cudaStream_t stream,
    int M,
    const Tensor* a,
    const Tensor* b,
    const Tensor* scales,
    const Tensor* zero_points,
    Tensor* Y) const {
  uint8_t const* zero_points_ptr = nullptr;
  size_t zero_points_size = 0;
  if (zero_points != nullptr) {
    zero_points_ptr = zero_points->Data<uint8_t>();
    zero_points_size = zero_points->Shape().Size();
  }

  return blkq4_fp16_gemm_sm80_dispatch<MLFloat16>(
      int(block_size_), column_wise_quant_blk_, int(M), int(N_), int(K_), stream,
      a->Data<MLFloat16>(), a->Shape().Size(),
      b->Data<uint8_t>(), b->Shape().Size(),
      scales->Data<MLFloat16>(), scales->Shape().Size(),
      zero_points_ptr, zero_points_size,
      Y->MutableData<MLFloat16>(), Y->Shape().Size());
}
#endif  // !defined(USE_MIGRAPHX) && !defined(USE_ROCM)

template <typename T>
Status MatMulNBits<T>::ComputeInternal(OpKernelContext* ctx) const {
  using CudaT = typename onnxruntime::cuda::ToCudaType<T>::MappedType;

  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b = ctx->Input<Tensor>(1);
  const Tensor* scales = ctx->Input<Tensor>(2);
  const Tensor* zero_points = ctx->Input<Tensor>(3);
  const Tensor* reorder_idx = ctx->Input<Tensor>(4);

  constexpr bool transa = false;
  constexpr bool transb = true;
  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(
      helper.Compute(a->Shape(), b_shape, transa, transb));

  Tensor* Y = ctx->Output(0, helper.OutputShape());
  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0) return Status::OK();

  if (prepack_ > 0) {
#if !defined(USE_MIGRAPHX) && !defined(USE_ROCM)
    ORT_RETURN_IF(reorder_idx != nullptr,
                  "Internal Error: Prepacked gemm does not support reorder index. Fix the prepacking logic!");
    ORT_RETURN_IF(zero_points != nullptr && zero_points->IsDataType<T>(),
                  "Internal Error: Prepacked gemm does not support zero points of type T. Fix the prepacking logic!");
    return PrepackedGemm(
        static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()),
        static_cast<int>(helper.M()), a, b, scales, zero_points, Y);
#else
    ORT_RETURN_IF(true, "Prepacked gemm is not supported for MatMulNBits op.");
#endif  // !defined(USE_MIGRAPHX) && !defined(USE_ROCM)
  }

  const auto* a_data = a->Data<T>();
  const uint8_t* blob_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<T>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->DataRaw();
  const auto* reorder_idx_data = reorder_idx == nullptr ? nullptr : reorder_idx->Data<int32_t>();

  bool is_4bit_done = (reorder_idx_data == nullptr) &&
                      (!zero_points || !zero_points->IsDataType<T>()) &&
                      TryMatMul4Bits(
                          reinterpret_cast<CudaT*>(Y->MutableData<T>()),
                          reinterpret_cast<const CudaT*>(a_data),
                          blob_data,
                          reinterpret_cast<const CudaT*>(scales_data),
                          static_cast<const uint8_t*>(zero_points_data),
                          SafeInt<int>(helper.M()),
                          SafeInt<int>(helper.N()),
                          SafeInt<int>(helper.K()),
                          SafeInt<int>(block_size_),
                          SafeInt<int>(GetDeviceProp().sharedMemPerBlock),
                          static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()));

  if (is_4bit_done) {
    return Status::OK();
  }

  int64_t K_padded = (K_ + block_size_ - 1) / block_size_ * block_size_;
  IAllocatorUniquePtr<T> b_data_ptr = GetScratchBuffer<T>(N_ * K_padded, ctx->GetComputeStream());
  auto* b_data = b_data_ptr.get();
  if (column_wise_quant_blk_) {
    if (reorder_idx) {
      ORT_ENFORCE(K_padded == reorder_idx->Shape()[0], "K_padded != g_idx->Shape()[0]");
    }
    // column-wise block
    if ((zero_points && zero_points->IsDataType<T>())) {
      ORT_RETURN_IF_ERROR(Dequantize4Bits(
          reinterpret_cast<CudaT*>(b_data),
          blob_data,
          reinterpret_cast<const CudaT*>(scales_data),
          (const CudaT*)zero_points_data,
          reorder_idx_data,
          SafeInt<int>(K_padded),
          SafeInt<int>(N_),
          SafeInt<int>(block_size_),
          static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())));
    } else {
      ORT_RETURN_IF_ERROR(Dequantize4Bits(
          reinterpret_cast<CudaT*>(b_data),
          blob_data,
          reinterpret_cast<const CudaT*>(scales_data),
          (const uint8_t*)zero_points_data,
          reorder_idx_data,
          SafeInt<int>(K_padded),
          SafeInt<int>(N_),
          SafeInt<int>(block_size_),
          static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())));
    }
  } else {
    // row-wise block
    K_padded = K_;

    ORT_RETURN_IF_ERROR(DequantizeBlockwise4b(
        reinterpret_cast<CudaT*>(b_data),
        blob_data,
        reinterpret_cast<const CudaT*>(scales_data),
        (const uint8_t*)zero_points_data,
        SafeInt<int>(block_size_),
        column_wise_quant_blk_,
        SafeInt<int>(K_),
        SafeInt<int>(N_),
        static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())));
  }
#if 0
cudaStreamSynchronize(static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()));
T* b_data_cpu = new T[K_ * N_];
cudaMemcpy(b_data_cpu, b_data, K_ * N_ * sizeof(T), cudaMemcpyDeviceToHost);
delete[] b_data_cpu;
#endif

  const CudaT alpha = onnxruntime::cuda::ToCudaType<T>::FromFloat(1.f);
  const CudaT zero = onnxruntime::cuda::ToCudaType<T>::FromFloat(0.f);

  if (helper.OutputOffsets().size() == 1) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        GetCublasHandle(ctx),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        SafeInt<int>(helper.N()),
        SafeInt<int>(helper.M()),
        SafeInt<int>(helper.K()),
        &alpha,
        reinterpret_cast<const CudaT*>(b_data),
        SafeInt<int>(K_padded),
        reinterpret_cast<const CudaT*>(a_data),
        helper.Lda(transa),
        &zero,
        reinterpret_cast<CudaT*>(Y->MutableData<T>()),
        helper.Ldc(),
        GetDeviceProp(),
        UseTF32()));
  }

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulNBits,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulNBits<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulNBits,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulNBits<MLFloat16>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
