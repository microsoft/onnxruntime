// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "moe.h"
#include "moe_kernel.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MoEBlock,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MoEBlock<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
MoEBlock<T>::MoEBlock(const OpKernelInfo& info) : CudaKernel(info) {
}

template <typename T>
Status MoEBlock<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* gated_output = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(3);
  const Tensor* fc1_experts_bias = context->Input<Tensor>(4);
  const Tensor* fc2_experts_bias = context->Input<Tensor>(5);

  // shape check
  const auto& input_dims = input->Shape().GetDims();
  const int64_t num_rows = input_dims[0];
  const int64_t hidden_size = input_dims[1];

  const auto& fc1_experts_weights_dims = fc1_experts_weights->Shape().GetDims();
  const int64_t num_experts = fc1_experts_weights_dims[0];
  const int64_t inter_size = fc1_experts_weights_dims[2];

  Tensor* output = context->Output(0, input->Shape());

  typedef typename ToCudaType<T>::MappedType CudaT;
  //context->GetComputeStream();
  // scale can be nullptr
  // skip can be nullptr later(todo);

  ORT_UNUSED_PARAMETER(input);
  ORT_UNUSED_PARAMETER(gated_output);
  ORT_UNUSED_PARAMETER(fc1_experts_weights);
  ORT_UNUSED_PARAMETER(fc2_experts_weights);
  ORT_UNUSED_PARAMETER(fc1_experts_bias);
  ORT_UNUSED_PARAMETER(fc2_experts_bias);
  ORT_UNUSED_PARAMETER(output);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
