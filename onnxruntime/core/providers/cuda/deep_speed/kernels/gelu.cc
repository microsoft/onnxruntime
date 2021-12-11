// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gelu.h"
#include "common.h"

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {
namespace deep_speed {

#define REGISTER_BIAS_GELU_KERNEL_TYPED(T)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      BiasGelu,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .MayInplace(0, 0)                                       \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BiasGelu<T>);

REGISTER_BIAS_GELU_KERNEL_TYPED(float)

template <typename T>
BiasGelu<T>::BiasGelu(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
}

template <typename T>
Status BiasGelu<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const auto* input_tensor = context->Input<Tensor>(0);
  const auto* input_buffer = input_tensor->Data<T>();
  const auto& input_shape = input_tensor->Shape();

  auto* output_tensor = context->Output(0, input_shape);
  auto* output_buffer = output_tensor->MutableData<T>();

  const size_t num_elements = input_shape.Size();

  auto stream = Stream();

  if (input_buffer != output_buffer) {
    CUDA_RETURN_IF_ERROR(cudaMemcpy(output_buffer, input_buffer,
                                    num_elements * sizeof(CudaT), cudaMemcpyDeviceToDevice));
  }

  // TODO: Call DeepSpeed here

  return Status::OK();
}

}  // namespace deep_speed
}  // namespace cuda
}  // namespace onnxruntime
