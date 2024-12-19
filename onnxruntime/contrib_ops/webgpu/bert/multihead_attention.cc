// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/webgpu/bert/attention_common.h"
#include "contrib_ops/webgpu/bert/multihead_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

#include "core/providers/webgpu/webgpu_supported_types.h"

using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::multihead_attention_helper;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    MultiHeadAttention,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    MultiHeadAttention);

MultiHeadAttention::MultiHeadAttention(const OpKernelInfo& info)
    : WebGpuKernel(info), AttentionBase(info, false) {
  ORT_ENFORCE(!is_unidirectional_, "Unidirectional MHA does not support webgpu kernel");
}

Status MultiHeadAttention::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* query = context.Input(0);
  const Tensor* key = context.Input(1);
  const Tensor* value = context.Input(2);
  const Tensor* bias = context.Input(3);
  const Tensor* key_padding_mask = context.Input(4);
  const Tensor* attention_bias = context.Input(5);
  const Tensor* past_key = context.Input(6);
  const Tensor* past_value = context.Input(7);

  if (query->Shape().GetDims().size() == 5) {
    ORT_NOT_IMPLEMENTED("Packed QKV of shape (B, L, N, 3, H) not implemented for webgpu");
  }
  if (key != nullptr && key->Shape().GetDims().size() == 5) {
    ORT_NOT_IMPLEMENTED("Packed KV not implemented for webgpu");
  }
  if (key_padding_mask) {
    ORT_NOT_IMPLEMENTED("input `key_padding_mask` not implemented for webgpu");
  }

  AttentionParameters params;
  ORT_RETURN_IF_ERROR(multihead_attention_helper::CheckInputs<Tensor>(query, key, value,
                                                                      bias, key_padding_mask, attention_bias, past_key, past_value, nullptr, &params,
                                                                      num_heads_, mask_filter_value_, scale_, is_unidirectional_, false, kMultiHeadAttention,
                                                                      context.DeviceLimits().maxComputeInvocationsPerWorkgroup));
  WebgpuAttentionParameters parameters(params);
  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size_);
  output_shape[1] = static_cast<int64_t>(parameters.sequence_length_);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size_);
  Tensor* output = context.Output(0, output_shape);

  // If optional outputs aren't needed, present_key and present_value will be null
  std::vector<int64_t> present_dims{
      parameters.batch_size_,
      parameters.num_heads_,
      parameters.total_sequence_length_,
      parameters.head_size_,
  };
  TensorShape present_shape(present_dims);
  Tensor* present_key = context.Output(1, present_shape);
  Tensor* present_value = context.Output(2, present_shape);

  TensorShapeVector q_new_dims({parameters.batch_size_, parameters.num_heads_,
                                parameters.sequence_length_, parameters.head_size_});
  TensorShape q_new_shape(q_new_dims);
  Tensor Q = context.CreateGPUTensor(query->DataType(), q_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(
      context, parameters.num_heads_, parameters.sequence_length_, parameters.head_size_, query, bias, 0, &Q));

  if (parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH) {  // key and value in BNSH format
    return ApplyAttention(&Q, key, value, attention_bias, past_key, past_value, output, present_key,
                          present_value, parameters, context);
  }

  TensorShapeVector k_new_dims({parameters.batch_size_, parameters.num_heads_,
                                parameters.kv_sequence_length_, parameters.head_size_});
  TensorShape k_new_shape(k_new_dims);
  Tensor K = context.CreateGPUTensor(key->DataType(), k_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.num_heads_, parameters.kv_sequence_length_,
                                        parameters.head_size_, key, bias, parameters.hidden_size_, &K));

  TensorShapeVector v_new_dims({parameters.batch_size_, parameters.num_heads_,
                                parameters.kv_sequence_length_, parameters.v_head_size_});
  TensorShape v_new_shape(v_new_dims);
  Tensor V = context.CreateGPUTensor(value->DataType(), v_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.num_heads_, parameters.kv_sequence_length_,
                                        parameters.v_head_size_, value, bias, 2 * parameters.hidden_size_, &V));

  // Compute the attention score and apply the score to V
  return ApplyAttention(&Q, &K, &V, attention_bias, past_key, past_value, output, present_key,
                        present_value, parameters, context);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
