// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/hip/tensor/gather_grad.h"
#include "orttraining/training_ops/hip/tensor/gather_grad_impl.h"
#include "orttraining/training_ops/hip/tensor/thrustallocator.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace hip {

ONNX_OPERATOR_KERNEL_EX(
    GatherGrad,
    kOnnxDomain,
    9,
    kHipExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    GatherGrad);

#define TYPED_GRAD_FUNCTION_CALL(T)                                                                    \
  if (T_type == DataTypeImpl::GetType<T>()) {                                                          \
    const T* grad_data = grad->template Data<T>();                                                     \
    T* output_data = output->template MutableData<T>();                                                \
    if (Tin_type == DataTypeImpl::GetType<int32_t>()) {                                                \
      auto original_indices = GetScratchBuffer<int32_t>(static_cast<size_t>(indices->Shape().Size())); \
      const int32_t* indices_data = indices->template Data<int32_t>();                                 \
      GatherGradImpl(reinterpret_cast<const ToHipType<T>::MappedType*>(grad_data),                    \
                     indices_data,                                                                     \
                     indices->Shape().Size(),                                                          \
                     num_weights,                                                                      \
                     stride,                                                                           \
                     original_indices.get(),                                                           \
                     reinterpret_cast<typename ToHipType<T>::MappedType*>(output_data),               \
                     num_inputs,                                                                       \
                     param_itrs,                                                                       \
                     thrust_alloc);                                                                    \
      return Status::OK();                                                                             \
    }                                                                                                  \
    if (Tin_type == DataTypeImpl::GetType<int64_t>()) {                                                \
      auto original_indices = GetScratchBuffer<int64_t>(static_cast<size_t>(indices->Shape().Size())); \
      const int64_t* indices_data = indices->template Data<int64_t>();                                 \
      GatherGradImpl(reinterpret_cast<const ToHipType<T>::MappedType*>(grad_data),                    \
                     indices_data,                                                                     \
                     indices->Shape().Size(),                                                          \
                     num_weights,                                                                      \
                     stride,                                                                           \
                     original_indices.get(),                                                           \
                     reinterpret_cast<typename ToHipType<T>::MappedType*>(output_data),               \
                     num_inputs,                                                                       \
                     param_itrs,                                                                       \
                     thrust_alloc);                                                                    \
      return Status::OK();                                                                             \
    }                                                                                                  \
  }

Status GatherGrad::ComputeInternal(OpKernelContext* context) const {
  const Tensor* shape = context->Input<Tensor>(0);
  const TensorShape data_shape(shape->template Data<int64_t>(), shape->Shape().Size());
  const Tensor* indices = context->Input<Tensor>(1);
  const Tensor* grad = context->Input<Tensor>(2);

  Tensor* output = context->Output(0, data_shape);
  HIP_RETURN_IF_ERROR(hipMemset(output->MutableDataRaw(), 0, output->SizeInBytes()));
  MLDataType T_type = grad->DataType();
  MLDataType Tin_type = indices->DataType();

  auto axis = HandleNegativeAxis(axis_, data_shape.NumDimensions());
  AllocatorPtr tmp_allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&tmp_allocator));
  hipGetLastError();  //TODO: This is hack, need to be removed
  ThrustAllocator thrust_alloc(tmp_allocator.get());
  int64_t stride = data_shape.SizeFromDimension(axis + 1);
  int64_t num_weights = data_shape.Size() / stride;
  auto new_indices = GetScratchBuffer<int32_t>(static_cast<size_t>(indices->Shape().Size()));
  const int64_t num_inputs = data_shape.SizeFromDimension(axis);
  const int64_t param_itrs = data_shape.SizeFromDimension(0) / num_inputs;

  TYPED_GRAD_FUNCTION_CALL(float)
#if !defined(__HIP_ARCH__) || __HIP_ARCH__ >= 700
  TYPED_GRAD_FUNCTION_CALL(MLFloat16)
#endif
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for Tind not supported yet in GatherGrad.");
}

}  // namespace hip
}  // namespace onnxruntime
