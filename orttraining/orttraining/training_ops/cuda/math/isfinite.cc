// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "isfinite.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#define REGISTER_ISFINITE_KERNEL_TYPED(T)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      IsFinite,                                                       \
      kMSDomain,                                                      \
      1,                                                              \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      KernelDefBuilder()                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()), \
      IsFiniteOp<T>);

template <typename TSrc>
Status IsFiniteOp<TSrc>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<TSrc>::MappedType CudaTSrc;
  const Tensor& input = *context->Input<Tensor>(0);
  Tensor& output = *context->Output(0, input.Shape());
  IsFinite(
      reinterpret_cast<const CudaTSrc*>(input.Data<TSrc>()),
      output.MutableData<bool>(), input.Shape().Size());

  return Status::OK();
}

REGISTER_ISFINITE_KERNEL_TYPED(MLFloat16)
REGISTER_ISFINITE_KERNEL_TYPED(float)
REGISTER_ISFINITE_KERNEL_TYPED(double)

#define REGISTER_ISALLFINITE_KERNEL_TYPED(T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      IsAllFinite,                                                   \
      kMSDomain,                                                     \
      1,                                                             \
      T,                                                             \
      kCudaExecutionProvider,                                        \
      KernelDefBuilder()                                             \
          .OutputMemoryType<OrtMemTypeCPUOutput>(0)                  \
          .TypeConstraint("V", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()), \
      IsAllFiniteOp<T>);

template <typename TSrc>
Status IsAllFiniteOp<TSrc>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<TSrc>::MappedType TSrcCuda;

  // Get Input tensor count.
  const auto total_tensor_count = context->InputCount();

  // Allocate GPU memory to capture the result computed by GPU kernel.
  // The GPU result will be copied later to the output which locates
  // on CPU memory.
  IAllocatorUniquePtr<bool> deviceOutput = GetScratchBuffer<bool>(1);
  CUDA_RETURN_IF_ERROR(cudaMemset(deviceOutput.get(), int(true), sizeof(bool)));

  std::vector<std::vector<void*>> grouped_tensor_pointers(total_tensor_count);
  std::vector<int> tensor_sizes(total_tensor_count);

  for (int i = 0; i < total_tensor_count; ++i) {
    const auto& input = context->Input<Tensor>(i);
    grouped_tensor_pointers[i] = {const_cast<TSrc*>(input->Data<TSrc>())};
    tensor_sizes[i] = static_cast<int>(input->Shape().Size());
  }

  typedef IsAllFiniteFunctor<TSrcCuda> TFunctor;
  TFunctor functor;

  // Check if all values are finite and write true to deviceOutput.
  // Otherwise, false will be written.
  launch_multi_tensor_functor<1, TFunctor, bool*>(
      2048 * 32, tensor_sizes, grouped_tensor_pointers, functor, deviceOutput.get());

  // Copy GPU result in deviceOutput to CPU memory.
  // Per this operator's schema, it's output is in CPU memory.
  Tensor& output = *context->Output(0, {});
  CUDA_RETURN_IF_ERROR(
      cudaMemcpy(
          output.MutableData<bool>(),
          deviceOutput.get(),
          sizeof(bool),
          cudaMemcpyDeviceToHost));

  return Status::OK();
}

REGISTER_ISALLFINITE_KERNEL_TYPED(MLFloat16)
REGISTER_ISALLFINITE_KERNEL_TYPED(float)
REGISTER_ISALLFINITE_KERNEL_TYPED(double)

}  // namespace cuda
}  // namespace onnxruntime
