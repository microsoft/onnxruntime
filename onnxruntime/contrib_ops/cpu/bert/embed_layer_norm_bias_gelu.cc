// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm_bias_gelu.h"

namespace onnxruntime {
namespace contrib {

// This op is internal-only, so register outside of onnx:
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      EmbedLayerNormBiasGelu,                                     \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      EmbedLayerNormBiasGelu<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
EmbedLayerNormBiasGelu<T>::EmbedLayerNormBiasGelu(
    const OpKernelInfo& op_kernel_info) : EmbedLayerNormBase(op_kernel_info) {}

#pragma warning(disable: 4100)
template <typename T>
Status EmbedLayerNormBiasGelu<T>::Compute(OpKernelContext* context) const {
  //ORT_RETURN_IF_ERROR(embed_layer_norm::CheckInputs(context));

  //bool is_signed_inputs = false;
  //ORT_RETURN_IF_ERROR(CheckQuantizedInputs(context, &is_signed_inputs));

  //if (is_signed_inputs) {
  //  return ComputeInternal<T, int8_t>(context, epsilon());
  //} else {
  //  return ComputeInternal<T, uint8_t>(context, epsilon());
  //}

  return Status::OK();
}
#pragma warning(default: 4100)

}  // namespace contrib
}  // namespace onnxruntime
