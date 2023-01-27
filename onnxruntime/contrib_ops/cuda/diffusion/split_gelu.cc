// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/diffusion/split_gelu.h"
#include "contrib_ops/cuda/diffusion/split_gelu_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      SplitGelu,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SplitGelu<T>);

REGISTER_KERNEL_TYPED(MLFloat16);
REGISTER_KERNEL_TYPED(float);

using namespace ONNX_NAMESPACE;

template <typename T>
SplitGelu<T>::SplitGelu(const OpKernelInfo& op_info) : CudaKernel(op_info) {
}

template <typename T>
Status SplitGelu<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);

  const auto& input_dims = input->Shape().GetDims();
  if (input_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input is expected to have 3 dimensions, got ", input_dims.size());
  }

  TensorShapeVector output_shape = input->Shape().AsShapeVector();
  output_shape[2] = input_dims[2] / 2;
  Tensor* output = context->Output(0, output_shape);

  typedef typename ToCudaType<T>::MappedType CudaT;
  const int32_t grid_size = static_cast<int32_t>(input_dims[0] * input_dims[1]);
  const int32_t half_hidden_size = static_cast<int32_t>(input_dims[2] / 2);
  LaunchSplitGeluKernel<CudaT>(Stream(context), grid_size, half_hidden_size,
                               reinterpret_cast<const CudaT*>(input->Data<T>()),
                               reinterpret_cast<CudaT*>(output->MutableData<T>()));

  CUDA_RETURN_IF_ERROR(cudaPeekAtLastError());
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
