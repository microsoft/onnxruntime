// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/group_query_attention_helper.h"
#include "contrib_ops/webgpu/bert/attention_common.h"
#include "contrib_ops/webgpu/bert/group_query_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

#include "core/providers/webgpu/webgpu_supported_types.h"

using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::group_query_attention_helper;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    GroupQueryAttention,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .MayInplace(3, 1)
        .MayInplace(4, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 6),
    GroupQueryAttention);

Status GroupQueryAttention::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* query = context.Input<Tensor>(0);
  const Tensor* key = context.Input<Tensor>(1);
  const Tensor* value = context.Input<Tensor>(2);
  const Tensor* past_key = context.Input<Tensor>(3);
  const Tensor* past_value = context.Input<Tensor>(4);
  const Tensor* seqlen_k = context.Input<Tensor>(5);
  const Tensor* total_seqlen_tensor = context.Input<Tensor>(6);
  const Tensor* cos_cache = context.Input<Tensor>(7);
  const Tensor* sin_cache = context.Input<Tensor>(8);

  GroupQueryAttentionParameters params;
  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckInputs(query,
                                                                key,
                                                                value,
                                                                past_key,
                                                                past_value,
                                                                cos_cache,
                                                                sin_cache,
                                                                &params,
                                                                num_heads_,
                                                                kv_num_heads_,
                                                                seqlen_k,
                                                                total_seqlen_tensor,
                                                                scale_,
                                                                softcap_));
  WebgpuAttentionParameters parameters(params);
  if (parameters.is_packed_qkv_) {
    ORT_NOT_IMPLEMENTED("Packed QKV of shape (B, L, N, 3, H) not implemented for webgpu-ep.");
  }
  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size_);
  output_shape[1] = static_cast<int64_t>(parameters.sequence_length_);
  output_shape[2] = static_cast<int64_t>(parameters.hidden_size_);
  Tensor* output = context.Output(0, output_shape);
  std::vector<int64_t> present_dims{
      parameters.batch_size_,
      kv_num_heads_,
      parameters.seqlen_present_kv_cache_,
      parameters.head_size_};
  std::vector<int64_t> present_kv_shape(present_dims);
  Tensor* present_key = context.Output(1, present_kv_shape);
  Tensor* present_value = context.Output(2, present_kv_shape);
  parameters.past_present_share_buffer_ = present_key != nullptr && present_value != nullptr && past_key != nullptr && past_value != nullptr && past_key->DataRaw() == present_key->DataRaw() && past_value->DataRaw() == present_value->DataRaw();

  TensorShapeVector q_new_dims({parameters.batch_size_, parameters.num_heads_,
                                parameters.sequence_length_, parameters.head_size_});
  TensorShape q_new_shape(q_new_dims);
  Tensor Q = context.CreateGPUTensor(query->DataType(), q_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(
      context, parameters.num_heads_, parameters.sequence_length_, parameters.head_size_, query, nullptr, 0, &Q));
  if (parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH) {  // key and value in BNSH format
    return ApplyAttention(&Q, key, value, nullptr, past_key, past_value, output, present_key,
                          present_value, parameters, context, seqlen_k);
  }

  TensorShapeVector k_new_dims({parameters.batch_size_, parameters.kv_num_heads_,
                                parameters.kv_sequence_length_, parameters.head_size_});
  TensorShape k_new_shape(k_new_dims);
  Tensor K = context.CreateGPUTensor(key->DataType(), k_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.kv_num_heads_, parameters.kv_sequence_length_,
                                        parameters.head_size_, key, nullptr, 0, &K));

  TensorShapeVector v_new_dims({parameters.batch_size_, parameters.kv_num_heads_,
                                parameters.kv_sequence_length_, parameters.v_head_size_});
  TensorShape v_new_shape(v_new_dims);
  Tensor V = context.CreateGPUTensor(value->DataType(), v_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.kv_num_heads_, parameters.kv_sequence_length_,
                                        parameters.v_head_size_, value, nullptr, 0, &V));
  return ApplyAttention(&Q, &K, &V, nullptr, past_key, past_value, output, present_key,
                        present_value, parameters, context, seqlen_k);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
