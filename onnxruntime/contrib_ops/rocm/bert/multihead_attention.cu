// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/multihead_attention.h"

#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh"
#include "core/platform/env_var_utils.h"
#include "core/providers/rocm/rocm_common.h"

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MultiHeadAttention,                                         \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MultiHeadAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
MultiHeadAttention<T>::MultiHeadAttention(const OpKernelInfo& info)
    : RocmKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);

  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
}

template <typename T>
Status MultiHeadAttention<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_ENFORCE(
      GetTuningContext()->IsTunableOpEnabled(),
      "MultiHeadAttention of ROCm EP is only supported if tunable op is used and tuning is enabled.");

  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);
  const Tensor* key_padding_mask = context->Input<Tensor>(4);
  const Tensor* relative_position_bias = context->Input<Tensor>(5);
  const Tensor* past_key = context->Input<Tensor>(6);
  const Tensor* past_value = context->Input<Tensor>(7);

  // TODO: Add support for bias, key_padding_mask and attention cache.
  ORT_ENFORCE(bias == nullptr && key_padding_mask == nullptr && past_key == nullptr && past_value == nullptr,
              "bias, key_padding_mask and attention cache is not supported");

  auto& device_prop = GetDeviceProp();
  AttentionParameters attn;
  ORT_RETURN_IF_ERROR(
      multihead_attention_helper::CheckInputs<Tensor>(
          query, key, value, bias,
          key_padding_mask, relative_position_bias,
          past_key, past_value, /*past_seq_len=*/nullptr,
          &attn,
          num_heads_, mask_filter_value_, scale_,
          false, device_prop.maxThreadsPerBlock));
  // TODO: support more qkv formats
  ORT_ENFORCE(attn.qkv_format == Q_KV_BSNH_BSN2H || attn.qkv_format == QKV_BSN3H, "Got ", attn.qkv_format);

  int sequence_length = attn.sequence_length;

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(attn.batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(attn.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{
      attn.batch_size,
      attn.num_heads,
      attn.total_sequence_length,
      attn.head_size,
  };
  TensorShape present_shape(present_dims);
  Tensor* present_key = context->Output(1, present_shape);
  Tensor* present_value = context->Output(2, present_shape);
  // TODO: Add support for attention cache
  ORT_ENFORCE(present_key == nullptr && present_value == nullptr, "attention cache is not supported");

  using HipT = typename ToHipType<T>::MappedType;
  using AttentionTunableOp = GemmSoftmaxGemmPermuteTunableOp<HipT>;
  auto workspace_bytes = AttentionTunableOp::GetWorkspaceNumBytes(&attn);
  auto workspace = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

  GemmSoftmaxGemmPermuteParams<HipT> params;
  params.tuning_ctx = GetTuningContext();
  params.stream = Stream(context);
  params.handle = GetRocblasHandle(context);
  params.attention = &attn;
  params.device_prop = &device_prop;
  params.scale = scale_ == 0 ? 1.0f / sqrt(attn.head_size) : scale_;
  std::tie(params.q_buffer, params.k_buffer, params.v_buffer) = GetQkvBuffers<HipT>(
      &attn,
      query->DataRaw(),
      key == nullptr ? nullptr : key->DataRaw(),
      value == nullptr ? nullptr : value->DataRaw());
  params.out_buffer = reinterpret_cast<HipT*>(output->MutableDataRaw());

  if (relative_position_bias != nullptr) {
    params.bias_buffer = reinterpret_cast<const HipT*>(relative_position_bias->DataRaw());
  }

  params.workspace_buffer = reinterpret_cast<HipT*>(workspace.get());
  return AttentionTunableOp{}(&params);
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
