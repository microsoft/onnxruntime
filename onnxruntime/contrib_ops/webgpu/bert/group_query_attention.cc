// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/group_query_attention_helper.h"
#include "contrib_ops/webgpu/bert/attention_common.h"
#include "contrib_ops/webgpu/bert/group_query_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/bert/rotary_embedding.h"
#include "contrib_ops/webgpu/bert/flash_attention.h"

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

Status SplitPackedQKVProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& packed_qkv = sh.AddInput("packed_qkv", ShaderUsage::UseOffsetToIndices | ShaderUsage::UseUniform);
  const auto& query = sh.AddOutput("query", ShaderUsage::UseSetByIndices | ShaderUsage::UseUniform);
  const auto& key = sh.AddOutput("key", ShaderUsage::UseSetByIndices | ShaderUsage::UseUniform);
  const auto& value = sh.AddOutput("val", ShaderUsage::UseSetByIndices | ShaderUsage::UseUniform);
  sh.MainFunctionBody() << "  let packed_qkv_indices = " << packed_qkv.OffsetToIndices("global_idx") << ";\n"
                        << "  let input_data = " << packed_qkv.GetByOffset("global_idx") << ";\n"
                        << "  let index = " << packed_qkv.IndicesGet("packed_qkv_indices", "2") << ";\n"
                        << "  if (index < uniforms.hidden_size) {\n"
                        << "    " << query.SetByIndices("packed_qkv_indices", "input_data") << ";\n"
                        << "  } else if (index < (uniforms.hidden_size + uniforms.kv_hidden_size)) {\n"
                        << "    var key_indices = packed_qkv_indices;\n"
                        << "   " << key.IndicesSet("key_indices", "2", "u32(index - uniforms.hidden_size)") << ";\n"
                        << "   " << key.SetByIndices("key_indices", "input_data") << ";\n"
                        << "  } else {\n"
                        << "    var val_indices = packed_qkv_indices;\n"
                        << "   " << value.IndicesSet("val_indices", "2", "u32(index - uniforms.hidden_size - uniforms.kv_hidden_size)") << ";\n"
                        << "   " << value.SetByIndices("val_indices", "input_data") << ";\n"
                        << "  }";
  return Status::OK();
}

Status SplitPackedQKV(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& params, const Tensor* packedQKV, Tensor* query, Tensor* key, Tensor* val) {
  SplitPackedQKVProgram program;
  auto input_size = packedQKV->Shape().Size();
  program
      .AddInput({packedQKV, ProgramTensorMetadataDependency::Rank})
      .AddOutputs({{query, ProgramTensorMetadataDependency::Rank}, {key, ProgramTensorMetadataDependency::Rank}, {val, ProgramTensorMetadataDependency::Rank}})
      .AddUniformVariables({
          {static_cast<uint32_t>(params.hidden_size_)},
          {static_cast<uint32_t>(params.kv_hidden_size_)},
      })
      .SetDispatchGroupSize((input_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

Status GeneratePositionIDsProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& output = sh.AddOutput("output", ShaderUsage::UseUniform);
  const auto& seqlens = sh.AddInput("seqlens", ShaderUsage::UseUniform);
  sh.MainFunctionBody() << "  var pos_id: i32 = 0;\n"
                        << "  let batch_idx = global_idx / uniforms.sequence_length;\n"
                        << "  let sequence_idx = i32(global_idx % uniforms.sequence_length);\n"
                        << "  let seqlen = " << seqlens.GetByOffset("batch_idx") << ";\n";
  if (is_first_prompt_) {
    sh.MainFunctionBody() << "  let total_seqlen = seqlen + 1;\n"
                          << "  if (sequence_idx < total_seqlen) {\n"
                          << "    pos_id = sequence_idx;\n"
                          << "  } else {\n"
                          << "    pos_id = 1;\n"
                          << "  }\n"
                          << "  " << output.SetByOffset("global_idx", "pos_id") << "\n";
  } else if (is_subsequent_prompt_) {
    sh.MainFunctionBody() << "  let total_seqlen = seqlen + 1;\n"
                          << "  let past_seqlen = total_seqlen - i32(uniforms.sequence_length);\n"
                          << "  if (past_seqlen + sequence_idx < total_seqlen) {\n"
                          << "    pos_id = past_seqlen + sequence_idx;\n"
                          << "  } else {\n"
                          << "    pos_id = 1;\n"
                          << "  }\n"
                          << "  " << output.SetByOffset("global_idx", "pos_id") << "\n";
  } else {
    sh.MainFunctionBody() << "  if (global_idx < uniforms.batch_size) {\n"
                          << "    " << output.SetByOffset("global_idx", "seqlen") << "\n"
                          << "  }\n";
  }
  return Status::OK();
}

Status GeneratePositionIDs(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& params, const Tensor* seqlens, Tensor* output_tensor) {
  GeneratePositionIDsProgram program(params.is_first_prompt_, params.is_subsequent_prompt_);
  auto output_size = params.batch_size_ * params.sequence_length_;
  program.CacheHint(params.is_first_prompt_, params.is_subsequent_prompt_)
      .AddInput({seqlens, ProgramTensorMetadataDependency::Rank})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::Rank})
      .AddUniformVariables({{static_cast<uint32_t>(params.batch_size_)}, {static_cast<uint32_t>(params.sequence_length_)}})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

Status RunRotaryEmbedding(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& params, const Tensor* input, const Tensor* pos_ids, const Tensor* cos_cache, const Tensor* sin_cache, Tensor* output, bool is_query_input) {
  const auto half_rotary_embedding_dim = gsl::narrow_cast<uint32_t>(cos_cache->Shape()[1]);
  const auto head_size = params.head_size_;
  const auto hidden_size = is_query_input ? params.hidden_size_ : params.kv_hidden_size_;
  const TensorShape global_shape({params.batch_size_, params.sequence_length_, hidden_size / head_size, static_cast<int64_t>(head_size - half_rotary_embedding_dim)});
  const auto rank = global_shape.NumDimensions();
  std::vector<uint32_t> global_dims(rank);
  std::vector<uint32_t> global_strides(rank);
  for (size_t j = 0; j < rank; ++j) {
    global_dims[j] = gsl::narrow_cast<uint32_t>(global_shape[j]);
    global_strides[j] = gsl::narrow_cast<uint32_t>(global_shape.SizeFromDimension(j + 1));
  }
  const auto input_output_strides = std::vector<uint32_t>({gsl::narrow_cast<uint32_t>(input->Shape().SizeFromDimension(1)), gsl::narrow_cast<uint32_t>(hidden_size), gsl::narrow_cast<uint32_t>(head_size), 1});
  const auto output_size = gsl::narrow_cast<const uint32_t>(global_shape.Size());

  RotaryEmbeddingProgram program(params.rotary_interleaved_);
  program
      .CacheHint(params.rotary_interleaved_)
      .AddInputs({{input, ProgramTensorMetadataDependency::Rank},
                  {pos_ids, ProgramTensorMetadataDependency::Rank},
                  {cos_cache, ProgramTensorMetadataDependency::Rank},
                  {sin_cache, ProgramTensorMetadataDependency::Rank}})
      .AddOutput(output)
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{params.scale_},
                            {gsl::make_span(global_dims)},
                            {gsl::make_span(global_strides)},
                            {gsl::make_span(input_output_strides)}})
      .AddIndices(TensorShape{1, 1});
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
                                                                softcap_));
  params.use_smooth_softmax = use_smooth_softmax_;

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

  if (!do_rotary_ &&
      head_sink == nullptr && !use_smooth_softmax_ &&
      local_window_size_ == -1 &&
      CanApplyFlashAttention(attention_bias, present_key, present_value, parameters, context)) {
    return ApplyFlashAttention(query, key, value, attention_bias, output, past_key, present_key, past_value,
                               present_value, parameters, context);
  }

  Tensor qSplit;
  Tensor kSplit;
  Tensor vSplit;
  if (parameters.is_packed_qkv_) {
    qSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.hidden_size_}));
    kSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.kv_hidden_size_}));
    vSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.kv_hidden_size_}));
    ORT_RETURN_IF_ERROR(SplitPackedQKV(context, parameters, query, &qSplit, &kSplit, &vSplit));
    parameters.is_packed_qkv_ = false;
    query = &qSplit;
    key = &kSplit;
    value = &vSplit;
  }

  Tensor qRotary;
  Tensor kRotary;
  if (do_rotary_) {
    qRotary = context.CreateGPUTensor(query->DataType(), query->Shape());
    kRotary = context.CreateGPUTensor(key->DataType(), key->Shape());
    auto pos_ids_shape = TensorShape({parameters.batch_size_, parameters.sequence_length_});
    Tensor pos_ids = context.CreateGPUTensor(DataTypeImpl::GetType<int64_t>(), pos_ids_shape);
    ORT_RETURN_IF_ERROR(GeneratePositionIDs(context, parameters, seqlen_k, &pos_ids));
    ORT_RETURN_IF_ERROR(RunRotaryEmbedding(context, parameters, query, &pos_ids, cos_cache, sin_cache, &qRotary, /* is_query_input = */ true));
    ORT_RETURN_IF_ERROR(RunRotaryEmbedding(context, parameters, key, &pos_ids, cos_cache, sin_cache, &kRotary, /* is_query_input = */ false));
    query = &qRotary;
    key = &kRotary;
  }

  TensorShapeVector q_new_dims({parameters.batch_size_, parameters.num_heads_,
                                parameters.sequence_length_, parameters.head_size_});
  TensorShape q_new_shape(q_new_dims);
  Tensor Q = context.CreateGPUTensor(query->DataType(), q_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(
      context, parameters.num_heads_, parameters.sequence_length_, parameters.head_size_, query, nullptr, 0, &Q));
  if (parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH) {  // key and value in BNSH format
    return ApplyAttention(&Q, key, value, attention_bias, past_key, past_value, output, present_key,
                          present_value, parameters, context, head_sink, seqlen_k, local_window_size_);
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
                        present_value, parameters, context, head_sink, seqlen_k, local_window_size_);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
