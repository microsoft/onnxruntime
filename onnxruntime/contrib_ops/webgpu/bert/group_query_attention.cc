// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/group_query_attention_helper.h"
#include "contrib_ops/webgpu/bert/group_query_attention.h"
#include "core/providers/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/bert/webgpu_attention_common.h"

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
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    GroupQueryAttention);

Status GroupQueryAttention::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);
  const Tensor* seqlens_k = context->Input<Tensor>(5);
  const Tensor* total_seqlen_tensor = context->Input<Tensor>(6);
  const Tensor* cos_cache = context->Input<Tensor>(7);
  const Tensor* sin_cache = context->Input<Tensor>(8);

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
                                                                seqlens_k,
                                                                total_seqlen_tensor,
                                                                scale_,
                                                                softcap_));
  WebgpuAttentionParameters parameters(params);
  if (parameters.is_packed_qkv) {
    ORT_NOT_IMPLEMENTED("Packed QKV of shape (B, L, N, 3, H) not implemented for webgpu-ep.");
  }
  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(parameters.sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context.Output(0, output_shape);
  const int present_kv_seqlen = parameters.seqlen_present_kv_cache;
  std::vector<int64_t> present_kv_shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(kv_num_heads_), static_cast<int64_t>(present_kv_seqlen), static_cast<int64_t>(head_size)});
  Tensor* present_k = context->Output(1, present_kv_shape);
  Tensor* present_v = context->Output(2, present_kv_shape);

  TensorShapeVector q_new_dims({parameters.batch_size, parameters.num_heads,
                                parameters.sequence_length, parameters.head_size});
  TensorShape q_new_shape(q_new_dims);
  Tensor Q = context.CreateGPUTensor(query->DataType(), q_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(
      context, parameters.num_heads, parameters.sequence_length, parameters.head_size, query, bias, 0, &Q));
  TensorShapeVector k_new_dims({parameters.batch_size, parameters.num_heads,
                                parameters.kv_sequence_length, parameters.head_size});
  if (parameters.qkv_format == Q_K_V_BSNH_BNSH_BNSH) {  // key and value in BNSH format
    return ApplyAttention(&Q, &K, &V, attention_bias, past_key, past_value, output, present_key,
                        present_value, parameters, context);
  }
  TensorShape k_new_shape(k_new_dims);
  Tensor K = context.CreateGPUTensor(key->DataType(), k_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.num_heads, parameters.kv_sequence_length,
                                        parameters.head_size, key, bias, parameters.hidden_size, &K));

  TensorShapeVector v_new_dims({parameters.batch_size, parameters.num_heads,
                                parameters.kv_sequence_length, parameters.v_head_size});
  TensorShape v_new_shape(v_new_dims);
  Tensor V = context.CreateGPUTensor(value->DataType(), v_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.num_heads, parameters.kv_sequence_length,
                                        parameters.v_head_size, value, bias, 2 * parameters.hidden_size, &V));
  return ApplyAttention(&Q, &K, &V, attention_bias, past_key, past_value, output, present_key,
                        present_value, parameters, seqlens_k, total_seqlens_tensor, context);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
