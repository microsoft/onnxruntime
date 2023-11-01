// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define MatMulFp32Q4 operator, it is basically
// matmul float32 with right hand side being a 2-D matrix
// pre-packed and block-compacted into int4
//

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "matmul_nbits.cuh"
#include "dequantize_blockwise.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;

template <typename T>
class MatMulNBits final : public CudaKernel {
 public:
  MatMulNBits(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("bits", &nbits_));
    ORT_ENFORCE(nbits_ == 4, "Only 4b quantization is supported for MatMulNBits op, additional bits support is planned.");
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t nbits_;
};

template <typename T>
Status MatMulNBits<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b = ctx->Input<Tensor>(1);
  const Tensor* scales = ctx->Input<Tensor>(2);
  const Tensor* zero_points = ctx->Input<Tensor>(3);

  const auto* a_data = a->Data<T>();
  const uint8_t* blob_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<T>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->Data<uint8_t>();

  typedef typename ToCudaType<T>::MappedType CudaT;

  constexpr bool transa = false;
  constexpr bool transb = true;
  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(
      helper.Compute(a->Shape(), b_shape, transa, transb));

  Tensor* Y = ctx->Output(0, helper.OutputShape());
  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0) return Status::OK();

  bool is_4bit_done = TryMatMul4Bits(
      reinterpret_cast<CudaT*>(Y->MutableData<T>()),
      reinterpret_cast<const CudaT*>(a_data),
      blob_data,
      reinterpret_cast<const CudaT*>(scales_data),
      zero_points_data,
      SafeInt<int>(helper.M()),
      SafeInt<int>(helper.N()),
      SafeInt<int>(helper.K()),
      SafeInt<int>(block_size_),
      SafeInt<int>(GetDeviceProp().sharedMemPerBlock),
      static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()));
  if (!is_4bit_done) {
    IAllocatorUniquePtr<T> b_data_ptr = GetScratchBuffer<T>(N_ * K_, ctx->GetComputeStream());
    auto* b_data = b_data_ptr.get();
    ORT_RETURN_IF_ERROR(DequantizeBlockwise4b(
        reinterpret_cast<CudaT*>(b_data),
        blob_data,
        reinterpret_cast<const CudaT*>(scales_data),
        zero_points_data,
        SafeInt<int>(block_size_),
        true,
        SafeInt<int>(K_),
        SafeInt<int>(N_),
        static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())));
#if 0
  cudaStreamSynchronize(static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()));
  T* b_data_cpu = new T[K_ * N_];
  cudaMemcpy(b_data_cpu, b_data, K_ * N_ * sizeof(T), cudaMemcpyDeviceToHost);
  delete[] b_data_cpu;
#endif

    const CudaT alpha = ToCudaType<T>::FromFloat(1.f);
    const CudaT zero = ToCudaType<T>::FromFloat(0.f);

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
          SafeInt<int>(K_),
          reinterpret_cast<const CudaT*>(a_data),
          helper.Lda(transa),
          &zero,
          reinterpret_cast<CudaT*>(Y->MutableData<T>()),
          helper.Ldc(),
          GetDeviceProp()));
    }
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
