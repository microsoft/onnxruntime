// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Status MatMul<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* left_X = ctx->Input<Tensor>(0);
  const Tensor* right_X = ctx->Input<Tensor>(1);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(left_X->Shape(), right_X->Shape()));

  Tensor* Y = ctx->Output(0, helper.OutputShape());
  ORT_RETURN_IF_NOT(strcmp(Y->Location().name, CUDA) == 0, "Output should be allocated on CUDA");

  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  if (helper.OutputOffsets().size() == 1) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        Base::CublasHandle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(helper.N()),
        static_cast<int>(helper.M()),
        static_cast<int>(helper.K()),
        &one,
        reinterpret_cast<const CudaT*>(right_X->template Data<T>()),
        static_cast<int>(helper.N()),
        reinterpret_cast<const CudaT*>(left_X->template Data<T>()),
        static_cast<int>(helper.K()),
        &zero,
        reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
        static_cast<int>(helper.N())));
    return Status::OK();
  }
  int device_id = 0;
  CudaAsyncBuffer<const CudaT*> left_arrays(this, device_id, helper.LeftOffsets().size());
  CudaAsyncBuffer<const CudaT*> right_arrays(this, device_id, helper.RightOffsets().size());
  CudaAsyncBuffer<CudaT*> output_arrays(this, device_id, helper.OutputOffsets().size());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const CudaT*>(left_X->template Data<T>()), helper.LeftOffsets(), left_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const CudaT*>(right_X->template Data<T>()), helper.RightOffsets(), right_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<CudaT*>(Y->template MutableData<T>()), helper.OutputOffsets(), output_arrays.CpuSpan());
  ORT_RETURN_IF_ERROR(left_arrays.CopyToGpu());
  ORT_RETURN_IF_ERROR(right_arrays.CopyToGpu());
  ORT_RETURN_IF_ERROR(output_arrays.CopyToGpu());

  // note that onnxruntime OrtValue is row major, while cublas is column major,
  // so swap left/right operands
  CUBLAS_RETURN_IF_ERROR(cublasGemmBatchedHelper(
      Base::CublasHandle(),
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      static_cast<int>(helper.N()),
      static_cast<int>(helper.M()),
      static_cast<int>(helper.K()),
      &one,
      right_arrays.GpuPtr(),
      static_cast<int>(helper.N()),
      left_arrays.GpuPtr(),
      static_cast<int>(helper.K()),
      &zero,
      output_arrays.GpuPtr(),
      static_cast<int>(helper.N()),
      static_cast<int>(helper.OutputOffsets().size())));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
