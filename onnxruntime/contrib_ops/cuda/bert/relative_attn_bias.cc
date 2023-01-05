// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "relative_attn_bias.h"
#include "relative_attn_bias_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      RelPosAttnBiasGen,                                          \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      RelPosAttnBias<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
RelPosAttnBias<T>::RelPosAttnBias(const OpKernelInfo& info) : CudaKernel(info) {
  is_bidirectional_ = info.GetAttrOrDefault<int64_t>("is_bidirectional", 0) == 1;

  int64_t max_distance = 0;
  ORT_ENFORCE(info.GetAttr("max_distance", &max_distance).IsOK());
  max_distance_ = static_cast<int>(max_distance);
}

template <typename T>
Status RelPosAttnBias<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* bias_table = context->Input<Tensor>(0);
  const Tensor* sequence_length = context->Input<Tensor>(1);

  const auto& bias_table_dims = bias_table->Shape().GetDims();
  const int64_t num_buckets = bias_table_dims[0];
  const int64_t num_heads = bias_table_dims[1];

  const int64_t seq_len =  *sequence_length->Data<int64_t>();

  Tensor* output = context->Output(0, {1, num_heads, seq_len, seq_len});

  return LaunchRelPosAttnBiasKernel<T>(Stream(context),
                                       output->template MutableData<T>(),
                                       bias_table->template Data<T>(),
                                       static_cast<int>(num_heads),
                                       static_cast<int>(seq_len),
                                       static_cast<int>(num_buckets),
                                       max_distance_,
                                       is_bidirectional_);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
