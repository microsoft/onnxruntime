// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/group_query_attention_helper.h"
#include "contrib_ops/webgpu/bert/attention_common.h"
#include "contrib_ops/webgpu/bert/group_query_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/bert/rotary_embedding.h"
#include "contrib_ops/webgpu/bert/flash_attention.h"

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/shader_helper.h"

using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::group_query_attention_helper;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status SplitPackedQKVWithRotaryEmbeddingProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& packed_qkv = sh.AddInput("packed_qkv", ShaderUsage::UseUniform);
  const auto& seqlens = sh.AddInput("seqlens", ShaderUsage::UseUniform);
  const auto& cos_cache = sh.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = sh.AddInput("sin_cache", ShaderUsage::UseUniform);

  const auto& query = sh.AddOutput("query", ShaderUsage::UseUniform);
  const auto& key = sh.AddOutput("key", ShaderUsage::UseUniform);
  const auto& val = sh.AddOutput("val", ShaderUsage::UseUniform);

  return WGSL_TEMPLATE_APPLY(sh, "bert/split_packed_qkv_with_rotary_embedding.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(interleaved, interleaved_),
                             WGSL_TEMPLATE_VARIABLE(cos_cache, cos_cache),
                             WGSL_TEMPLATE_VARIABLE(key, key),
                             WGSL_TEMPLATE_VARIABLE(packed_qkv, packed_qkv),
                             WGSL_TEMPLATE_VARIABLE(query, query),
                             WGSL_TEMPLATE_VARIABLE(seqlens, seqlens),
                             WGSL_TEMPLATE_VARIABLE(sin_cache, sin_cache),
                             WGSL_TEMPLATE_VARIABLE(val, val));
}

// Split packed QKV with Q/K rotary embedding fusion
Status RunSplitPackedQKVWithRotaryEmbedding(onnxruntime::webgpu::ComputeContext& context,
                                            const WebgpuAttentionParameters& params,
                                            const Tensor* packedQKV,
                                            const Tensor* seqlen_k,
                                            const Tensor* cos_cache,
                                            const Tensor* sin_cache,
                                            Tensor* query,
                                            Tensor* key,
                                            Tensor* val) {
  const auto half_rotary_embedding_dim = gsl::narrow_cast<uint32_t>(cos_cache->Shape()[1]);
  const auto head_size = params.head_size_;

  int components = 1;
  // Currently we only support vectorization when RoPE is not interleaved
  if (!params.rotary_interleaved_) {
    if ((params.head_size_ % 4 == 0) && (half_rotary_embedding_dim % 4 == 0)) {
      components = 4;
    } else if ((params.head_size_ % 2 == 0) && (half_rotary_embedding_dim % 2 == 0)) {
      components = 2;
    }
  }

  // Adjust dimensions for vectorization
  const auto half_rotary_embedding_dim_vec = half_rotary_embedding_dim / components;
  const auto head_size_vec = head_size / components;

  // Dispatch: batch_size * sequence_length * num_heads * (half_rotary_dim + need_copy_dim)
  // work_per_head = half_rotary_dim + (head_size - 2 * half_rotary_dim)
  //               = head_size - half_rotary_dim
  const auto work_per_head_vec = head_size_vec - half_rotary_embedding_dim_vec;
  auto dispatch_size = static_cast<uint32_t>(params.batch_size_ * params.sequence_length_ * params.num_heads_ * work_per_head_vec);

  SplitPackedQKVWithRotaryEmbeddingProgram program(params.rotary_interleaved_);
  program
      .CacheHint(params.rotary_interleaved_)
      .AddInput({packedQKV, ProgramTensorMetadataDependency::TypeAndRank, components})
      .AddInputs({
          {seqlen_k, ProgramTensorMetadataDependency::TypeAndRank},
          {cos_cache, ProgramTensorMetadataDependency::Rank, components},
          {sin_cache, ProgramTensorMetadataDependency::Rank, components},
      })
      .AddOutputs({{query, ProgramTensorMetadataDependency::None, components},
                   {key, ProgramTensorMetadataDependency::None, components},
                   {val, ProgramTensorMetadataDependency::None, components}})
      .AddUniformVariables({
          {static_cast<uint32_t>(params.sequence_length_)},
          {static_cast<uint32_t>(params.hidden_size_ / components)},
          {static_cast<uint32_t>(params.kv_hidden_size_ / components)},
          {static_cast<uint32_t>(params.num_heads_)},
          {static_cast<uint32_t>(params.kv_num_heads_)},
          {static_cast<uint32_t>(head_size_vec)},
          {static_cast<uint32_t>(half_rotary_embedding_dim_vec)},
          {static_cast<uint32_t>(dispatch_size)},
      })
      .SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

// Fused Q/K rotary embedding
Status RunFusedQKRotaryEmbedding(onnxruntime::webgpu::ComputeContext& context,
                                 const WebgpuAttentionParameters& params,
                                 const Tensor* query_in,
                                 const Tensor* key_in,
                                 const Tensor* seqlen_k,
                                 const Tensor* cos_cache,
                                 const Tensor* sin_cache,
                                 Tensor* query_out,
                                 Tensor* key_out) {
  const auto half_rotary_embedding_dim = gsl::narrow_cast<uint32_t>(cos_cache->Shape()[1]);
  const auto head_size = params.head_size_;

  // Build Q domain
  const auto hidden_size_q = params.hidden_size_;
  const TensorShape q_global_shape({params.batch_size_, params.sequence_length_,
                                    hidden_size_q / head_size,
                                    static_cast<int64_t>(head_size - half_rotary_embedding_dim)});
  const auto rank = q_global_shape.NumDimensions();
  std::vector<uint32_t> q_global_dims(rank);
  std::vector<uint32_t> q_global_strides(rank);
  for (size_t j = 0; j < rank; ++j) {
    q_global_dims[j] = gsl::narrow_cast<uint32_t>(q_global_shape[j]);
    q_global_strides[j] = gsl::narrow_cast<uint32_t>(q_global_shape.SizeFromDimension(j + 1));
  }

  // Build K domain
  const auto hidden_size_k = params.kv_hidden_size_;
  const TensorShape k_global_shape({params.batch_size_, params.sequence_length_,
                                    hidden_size_k / head_size,
                                    static_cast<int64_t>(head_size - half_rotary_embedding_dim)});
  std::vector<uint32_t> k_global_dims(rank);
  for (size_t j = 0; j < rank; ++j) {
    k_global_dims[j] = gsl::narrow_cast<uint32_t>(k_global_shape[j]);
  }

  const auto q_domain_size = gsl::narrow_cast<uint32_t>(q_global_shape.Size());

  const auto q_input_output_strides = std::vector<uint32_t>(
      {gsl::narrow_cast<uint32_t>(query_in->Shape().SizeFromDimension(1)),
       gsl::narrow_cast<uint32_t>(hidden_size_q),
       gsl::narrow_cast<uint32_t>(head_size),
       1u});

  const auto k_input_output_strides = std::vector<uint32_t>(
      {gsl::narrow_cast<uint32_t>(key_in->Shape().SizeFromDimension(1)),
       gsl::narrow_cast<uint32_t>(hidden_size_k),
       gsl::narrow_cast<uint32_t>(head_size),
       1u});

  // Dispatch computations only over the Q domain, and fuse K write operations using a head-index-based condition.
  FusedQKRotaryEmbeddingProgram program(params.rotary_interleaved_);
  program
      .CacheHint(params.rotary_interleaved_)
      .AddInputs({
          {query_in, ProgramTensorMetadataDependency::TypeAndRank},
          {key_in, ProgramTensorMetadataDependency::Rank},
          {seqlen_k, ProgramTensorMetadataDependency::TypeAndRank},
          {cos_cache, ProgramTensorMetadataDependency::Rank},
          {sin_cache, ProgramTensorMetadataDependency::Rank},
      })
      .AddOutputs({
          {query_out, ProgramTensorMetadataDependency::None},
          {key_out, ProgramTensorMetadataDependency::None},
      })
      .SetDispatchGroupSize((q_domain_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          {params.scale_},
          {gsl::make_span(q_global_dims)},
          {gsl::make_span(q_global_strides)},
          {gsl::make_span(q_input_output_strides)},
          {gsl::make_span(k_global_dims)},
          {gsl::make_span(k_input_output_strides)},
          {q_domain_size},
      });

  return context.RunProgram(program);
}

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
  const Tensor* position_ids = context.Input<Tensor>(9);  // TODO: support sliding window
  const Tensor* attention_bias = context.Input<Tensor>(10);
  const Tensor* head_sink = context.Input<Tensor>(11);

  GroupQueryAttentionParameters params = {};
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
                                                                softcap_,
                                                                0,
                                                                context.DeviceLimits().maxComputeInvocationsPerWorkgroup));
  params.use_smooth_softmax = use_smooth_softmax_;
  params.rotary_interleaved = rotary_interleaved_;

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckCustomAttentionInputs(position_ids,
                                                                               attention_bias,
                                                                               head_sink,
                                                                               params));

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckNoQKOutput(
      context.OutputCount(),
      static_cast<int>(Info().GetAttrOrDefault<int64_t>("qk_output", static_cast<int64_t>(QKOutputType::NO_OUTPUT)))));

  WebgpuAttentionParameters parameters(params);
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

  ORT_ENFORCE(parameters.total_sequence_length_ <= parameters.seqlen_present_kv_cache_, "Total sequence length cannot be greater than the existing KV cache length.");

  Tensor qSplit;
  Tensor kSplit;
  Tensor vSplit;

  Tensor qRotary;
  Tensor kRotary;

  // Use a sliding window if the total sequence exceeds the window's length.
  bool use_sliding_window = (local_window_size_ != -1 && local_window_size_ < parameters.total_sequence_length_);
  bool will_use_flash_attention = false;
  if (head_sink == nullptr && !use_smooth_softmax_ && !use_sliding_window) {
    // Create a temporary parameters copy with is_packed_qkv_ set to false to check if flash attention can be applied after unpacking
    WebgpuAttentionParameters temp_params = parameters;
    temp_params.is_packed_qkv_ = false;
    will_use_flash_attention = CanApplyFlashAttention(nullptr, temp_params, context);
  }

  if (parameters.is_packed_qkv_ && do_rotary_) {
    // Use the ultimate fused operation when FlashAttention and static KV cache is enabled.
    if (will_use_flash_attention && parameters.past_present_share_buffer_) {
      // Directly call ApplyFlashAttention with fused split/rotary/copyKV enabled
      // query points to packed QKV, K and V are nullptr since they're not needed
      return ApplyFlashAttention(query, nullptr, nullptr, attention_bias, output, past_key, present_key, past_value,
                                 present_value, parameters, context, seqlen_k, cos_cache, sin_cache);
    }
    // Fused: splitQKV + rotary QK
    qSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.hidden_size_}));
    kSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.kv_hidden_size_}));
    vSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.kv_hidden_size_}));
    ORT_RETURN_IF_ERROR(RunSplitPackedQKVWithRotaryEmbedding(context, parameters,
                                                             query, seqlen_k,
                                                             cos_cache, sin_cache,
                                                             &qSplit, &kSplit, &vSplit));
    parameters.is_packed_qkv_ = false;
    parameters.qkv_format_ = Q_K_V_BSNH;
    query = &qSplit;
    key = &kSplit;
    value = &vSplit;
  } else {
    if (parameters.is_packed_qkv_) {
      // splitQKV
      qSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.hidden_size_}));
      kSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.kv_hidden_size_}));
      vSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.kv_hidden_size_}));
      ORT_RETURN_IF_ERROR(SplitPackedQKV(context, parameters, query, &qSplit, &kSplit, &vSplit, parameters.kv_hidden_size_));
      parameters.is_packed_qkv_ = false;
      parameters.qkv_format_ = Q_K_V_BSNH;
      query = &qSplit;
      key = &kSplit;
      value = &vSplit;
    }
    if (do_rotary_) {
      // rotary QK
      qRotary = context.CreateGPUTensor(query->DataType(), query->Shape());
      kRotary = context.CreateGPUTensor(key->DataType(), key->Shape());
      ORT_RETURN_IF_ERROR(RunFusedQKRotaryEmbedding(context, parameters,
                                                    query, key,
                                                    seqlen_k,
                                                    cos_cache, sin_cache,
                                                    &qRotary, &kRotary));
      query = &qRotary;
      key = &kRotary;
    }
  }

  if (will_use_flash_attention) {
    return ApplyFlashAttention(query, key, value, attention_bias, output, past_key, present_key, past_value,
                               present_value, parameters, context, seqlen_k);
  }

  TensorShapeVector q_new_dims({parameters.batch_size_, parameters.num_heads_,
                                parameters.sequence_length_, parameters.head_size_});
  TensorShape q_new_shape(q_new_dims);
  Tensor Q = context.CreateGPUTensor(query->DataType(), q_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(
      context, parameters.num_heads_, parameters.sequence_length_, parameters.head_size_, query, nullptr, 0, &Q));
  if (parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH) {  // key and value in BNSH format
    return ApplyAttention(&Q, key, value, attention_bias, past_key, past_value, output, present_key,
                          present_value, nullptr, parameters, context, head_sink, seqlen_k, local_window_size_);
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
  return ApplyAttention(&Q, &K, &V, attention_bias, past_key, past_value, output, present_key,
                        present_value, nullptr, parameters, context, head_sink, seqlen_k, local_window_size_);
}

KernelCreateInfo CreateGroupQueryAttentionKernelInfo(bool enable_graph_capture) {
  KernelDefBuilder builder;
  builder.SetName("GroupQueryAttention")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Provider(kWebGpuExecutionProvider)
      .TypeConstraint("T", WebGpuSupportedFloatTypes())
      .MayInplace(3, 1)
      .MayInplace(4, 2);

  // Only set InputMemoryType to CPU when graph capture is disabled
  if (!enable_graph_capture) {
    builder.InputMemoryType(OrtMemTypeCPUInput, 6);
  }

  return KernelCreateInfo(
      builder.Build(),
      [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
        out = std::make_unique<GroupQueryAttention>(info);
        return Status::OK();
      });
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
