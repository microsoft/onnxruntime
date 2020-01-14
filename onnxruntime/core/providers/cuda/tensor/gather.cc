// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/gather_impl.h"
#include "core/providers/cuda/tensor/gather.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"
#include "core/providers/cuda/tensor/thrustallocator.h"

namespace onnxruntime {
namespace cuda {
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gather,
    kOnnxDomain,
    1, 10,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

// explicit negative axis support
ONNX_OPERATOR_KERNEL_EX(
    Gather,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

#define TYPED_FUNCTION_CALL(T)                                                  \
  if (utils::IsPrimitiveDataType<T>(T_type)) {                                  \
    T* output_data = p.output_tensor->template MutableData<T>();                \
    const T* input_data = p.input_tensor->template Data<T>();                   \
    if (utils::IsPrimitiveDataType<int32_t>(Tin_type)) {                        \
      if (p.output_tensor->Shape().Size() > 0) {                                \
        GatherImpl(                                                             \
            input_block_size,                                                   \
            indices_max,                                                        \
            p.indices_tensor->template Data<int32_t>(),                         \
            &divmod_output_block_size,                                          \
            &divmod_block_size,                                                 \
            reinterpret_cast<const ToCudaType<T>::MappedType*>(input_data),     \
            reinterpret_cast<typename ToCudaType<T>::MappedType*>(output_data), \
            p.output_tensor->Shape().Size());                                   \
      }                                                                         \
      return Status::OK();                                                      \
    }                                                                           \
    if (utils::IsPrimitiveDataType<int64_t>(Tin_type)) {                        \
      if (p.output_tensor->Shape().Size() > 0) {                                \
        GatherImpl(                                                             \
            input_block_size,                                                   \
            indices_max,                                                        \
            p.indices_tensor->template Data<int64_t>(),                         \
            &divmod_output_block_size,                                          \
            &divmod_block_size,                                                 \
            reinterpret_cast<const ToCudaType<T>::MappedType*>(input_data),     \
            reinterpret_cast<typename ToCudaType<T>::MappedType*>(output_data), \
            p.output_tensor->Shape().Size());                                   \
      }                                                                         \
      return Status::OK();                                                      \
    }                                                                           \
  }

Status Gather::ComputeInternal(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(context, p));

  const TensorShape& input_shape = p.input_tensor->Shape();

  const int64_t block_size = input_shape.SizeFromDimension(p.axis + 1);
  size_t N = p.indices_tensor->Shape().Size();
  const int64_t input_block_size = input_shape.SizeFromDimension(p.axis);
  const int64_t output_block_size = N * block_size;
  const int64_t indices_max = input_shape[p.axis];

  const fast_divmod divmod_output_block_size(gsl::narrow_cast<int>(output_block_size));
  const fast_divmod divmod_block_size(gsl::narrow_cast<int>(block_size));

  MLDataType T_type = p.input_tensor->DataType();
  MLDataType Tin_type = p.indices_tensor->DataType();

  TYPED_FUNCTION_CALL(int8_t)
  TYPED_FUNCTION_CALL(int16_t)
  TYPED_FUNCTION_CALL(int32_t)
  TYPED_FUNCTION_CALL(int64_t)
  TYPED_FUNCTION_CALL(uint8_t)
  TYPED_FUNCTION_CALL(uint16_t)
  TYPED_FUNCTION_CALL(uint32_t)
  TYPED_FUNCTION_CALL(uint64_t)
  TYPED_FUNCTION_CALL(MLFloat16)
  TYPED_FUNCTION_CALL(float)
  TYPED_FUNCTION_CALL(double)
  TYPED_FUNCTION_CALL(bool)

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for Tind not supported yet in Gather.");
}

ONNX_OPERATOR_KERNEL_EX(
    GatherGrad,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
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
      GatherGradImpl(reinterpret_cast<const ToCudaType<T>::MappedType*>(grad_data),                    \
                     indices_data,                                                                     \
                     indices->Shape().Size(),                                                          \
                     num_weights,                                                                      \
                     stride,                                                                           \
                     original_indices.get(),                                                           \
                     reinterpret_cast<typename ToCudaType<T>::MappedType*>(output_data),               \
                     num_inputs,                                                                       \
                     param_itrs,                                                                       \
                     thrust_alloc);                                                                    \
      return Status::OK();                                                                             \
    }                                                                                                  \
    if (Tin_type == DataTypeImpl::GetType<int64_t>()) {                                                \
      auto original_indices = GetScratchBuffer<int64_t>(static_cast<size_t>(indices->Shape().Size())); \
      const int64_t* indices_data = indices->template Data<int64_t>();                                 \
      GatherGradImpl(reinterpret_cast<const ToCudaType<T>::MappedType*>(grad_data),                    \
                     indices_data,                                                                     \
                     indices->Shape().Size(),                                                          \
                     num_weights,                                                                      \
                     stride,                                                                           \
                     original_indices.get(),                                                           \
                     reinterpret_cast<typename ToCudaType<T>::MappedType*>(output_data),               \
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
  CUDA_RETURN_IF_ERROR(cudaMemset(output->MutableDataRaw(), 0, output->SizeInBytes()));
  MLDataType T_type = grad->DataType();
  MLDataType Tin_type = indices->DataType();

  auto axis = HandleNegativeAxis(axis_, data_shape.NumDimensions());
  AllocatorPtr tmp_allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&tmp_allocator));
  cudaGetLastError();  //TODO: This is hack, need to be removed
  ThrustAllocator thrust_alloc(tmp_allocator.get());
  int64_t stride = data_shape.SizeFromDimension(axis + 1);
  int64_t num_weights = data_shape.Size() / stride;
  auto new_indices = GetScratchBuffer<int32_t>(static_cast<size_t>(indices->Shape().Size()));
  const int64_t num_inputs = data_shape.SizeFromDimension(axis);
  const int64_t param_itrs = data_shape.SizeFromDimension(0) / num_inputs;

  TYPED_GRAD_FUNCTION_CALL(float)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
  TYPED_GRAD_FUNCTION_CALL(MLFloat16)
#endif
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for Tind not supported yet in GatherGrad.");
}

}  // namespace cuda
}  // namespace onnxruntime
