// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qembed_layer_norm_bias_gelu.h"

#pragma warning(disable : 4189 4100)
namespace onnxruntime {
namespace contrib {

namespace {
}  // namespace

// This op is internal-only, so register outside of onnx:
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      QEmbedLayerNormBiasGelu,                                    \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      QEmbedLayerNormBiasGelu<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
QEmbedLayerNormBiasGelu<T>::QEmbedLayerNormBiasGelu(
    const OpKernelInfo& op_kernel_info)
    : EmbedLayerNormBase(op_kernel_info) {
}

template <typename T>
Status QEmbedLayerNormBiasGelu<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);

  const TensorShape& output_shape = input->Shape();
  Tensor* skip_layer_norm_output = context->Output(0, output_shape);
  Tensor* output = context->Output(1, output_shape);

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

}  // namespace contrib
}  // namespace onnxruntime
#pragma warning(default : 4189 4100)