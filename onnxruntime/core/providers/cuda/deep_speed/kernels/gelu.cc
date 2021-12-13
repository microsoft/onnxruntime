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
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_buffer, input_buffer,
                                         num_elements * sizeof(CudaT), cudaMemcpyDeviceToDevice, stream));
  }

  /*
  size_t input_rank = input_shape.NumDimensions();

  unsigned batch_size = 1;  // Assume input is scalar initially
  unsigned hidden_dim = 1;  // Assume input is scalar initially

  if (input_rank > 0) {  // Input is not a scalar
    hidden_dim = static_cast<unsigned>(input_shape[input_rank - 1]);
    batch_size = static_cast<unsigned>(num_elements) / hidden_dim;
  }

  const auto* bias_buffer = context->Input<Tensor>(1)->Data<T>();

  // TODO: Add more shape checks for input and bias inputs


  DeepSpeedAPI::bias_gelu(output_buffer,
                          // TODO: DeepSpeed lib only takes non-const buffers. Can we ask them to take const pointers ?
                          const_cast<T*>(bias_buffer),
                          batch_size,
                          hidden_dim,
                          true,  // Currently this kernel only supports float
                          stream);
*/
  return Status::OK();
}

}  // namespace deep_speed
}  // namespace cuda
}  // namespace onnxruntime
