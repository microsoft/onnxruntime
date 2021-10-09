// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/isfinite.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#define REGISTER_ISALLFINITE_KERNEL_TYPED(T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      IsAllFinite,                                                   \
      kMSDomain,                                                     \
      1,                                                             \
      T,                                                             \
      kCudaExecutionProvider,                                        \
      (*KernelDefBuilder::Create())                                  \
          .TypeConstraint("V", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()), \
      IsAllFiniteOp<T>);

template <typename TSrc>
Status IsAllFiniteOp<TSrc>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<TSrc>::MappedType TSrcCuda;

  // Get Input tensor count.
  const auto total_tensor_count = context->InputCount();

  // Initialize the output to true.  GPU kernel will set it
  // to false if any value in any tensor is non-finite.
  Tensor& output = *context->Output(0, {});
  auto* output_data = reinterpret_cast<ToCudaType<bool>::MappedType*>(output.template MutableData<bool>());
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(output_data, int(true), sizeof(bool), Stream()));

  std::vector<std::vector<void*>> grouped_tensor_pointers(total_tensor_count);
  std::vector<int> tensor_sizes(total_tensor_count);

  for (int i = 0; i < total_tensor_count; ++i) {
    const auto& input = context->Input<Tensor>(i);
    grouped_tensor_pointers[i] = {const_cast<TSrc*>(input->Data<TSrc>())};
    tensor_sizes[i] = static_cast<int>(input->Shape().Size());
  }

  typedef IsAllFiniteFunctor<TSrcCuda> TFunctor;
  TFunctor functor;

  // Check if all values are finite and write true to output.
  // Otherwise, false will be written.
  launch_multi_tensor_functor<1, TFunctor>(
      Stream(), 2048 * 32, tensor_sizes, grouped_tensor_pointers, functor, output_data, isinf_only_, isnan_only_);

  return Status::OK();
}

REGISTER_ISALLFINITE_KERNEL_TYPED(MLFloat16)
REGISTER_ISALLFINITE_KERNEL_TYPED(float)
REGISTER_ISALLFINITE_KERNEL_TYPED(double)

}  // namespace cuda
}  // namespace onnxruntime
