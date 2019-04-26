// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "transpose.h"
#include "transpose_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Transpose,                                                  \
      kOnnxDomain,                                                \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Transpose<T>);

template <typename T>
Status Transpose<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X_ptr = ctx->Input<Tensor>(0);
  if (X_ptr == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *X_ptr;
  const TensorShape& input_shape = X.Shape();
  const std::vector<int64_t>& input_dims = input_shape.GetDims();
  size_t rank = input_dims.size();

  std::vector<int64_t> output_dims(rank);
  std::vector<int64_t> default_perm(rank);
  const std::vector<int64_t>* p_perm = nullptr;
  const auto& status = ComputeOutputShape(X, output_dims, default_perm, p_perm);
  if (!status.IsOK())
    return status;

  TensorShape output_shape{output_dims};
  Tensor* Y = ctx->Output(0, output_shape);
  int device_id = 0;
  CudaAsyncBuffer<int64_t> input_strides(this, device_id, rank);
  CudaAsyncBuffer<int64_t> perm(this, device_id, *p_perm);
  CudaAsyncBuffer<fast_divmod> fdm_output_strides(this, device_id, rank);
  ORT_ENFORCE(TensorPitches::Calculate(input_strides.CpuSpan(), input_dims));
  ORT_ENFORCE(CalculateFdmStrides(fdm_output_strides.CpuSpan(), output_dims));

  ORT_RETURN_IF_ERROR(input_strides.CopyToGpu());
  ORT_RETURN_IF_ERROR(perm.CopyToGpu());
  ORT_RETURN_IF_ERROR(fdm_output_strides.CopyToGpu());

  TransposeImpl(
      rank,
      input_strides.GpuPtr(),
      perm.GpuPtr(),
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(X.template Data<T>()),
      fdm_output_strides.GpuPtr(),
      reinterpret_cast<typename ToCudaType<T>::MappedType*>(Y->template MutableData<T>()),
      output_shape.Size());

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Transpose<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace cuda
}  // namespace onnxruntime
