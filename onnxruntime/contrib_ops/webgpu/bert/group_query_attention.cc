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

Status RunRotaryEmbedding(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& params, const Tensor* input, const Tensor* pos_ids, const Tensor* cos_cache, const Tensor* sin_cache, Tensor* output, bool is_packed, bool is_query_input) {
  const auto half_rotary_embedding_dim = gsl::narrow<uint32_t>(cos_cache->Shape()[1]);
  const auto head_size = params.head_size_;
  const auto hidden_size = is_packed ? params.hidden_size_ + 2 * params.kv_hidden_size_ : is_query_input ? params.hidden_size_ : params.kv_hidden_size_;
  const auto input_output_hidden_size = is_packed ? params.hidden_size_ + 2 * params.kv_hidden_size_ : is_query_input ? params.hidden_size_ : params.kv_hidden_size_;
  const TensorShape global_shape({params.batch_size_, params.sequence_length_, hidden_size / head_size, static_cast<int64_t>(head_size - half_rotary_embedding_dim)});
  const auto rank = global_shape.NumDimensions();
  std::vector<uint32_t> global_dims(rank);
  std::vector<uint32_t> global_strides(rank);
  for (size_t j = 0; j < rank; ++j) {
    global_dims[j] = gsl::narrow<uint32_t>(global_shape[j]);
    global_strides[j] = gsl::narrow<uint32_t>(global_shape.SizeFromDimension(j + 1));
  }
  const auto input_output_strides = std::vector<uint32_t>({gsl::narrow<uint32_t>(input->Shape().SizeFromDimension(1)), gsl::narrow<uint32_t>(input_output_hidden_size), gsl::narrow<uint32_t>(head_size), 1});
  const auto output_size = gsl::narrow<const uint32_t>(global_shape.Size());

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
                            {gsl::make_span(input_output_strides)},
                            {static_cast<uint32_t>(is_packed ? 1 : 0)},
                            {static_cast<uint32_t>(is_query_input ? params.num_heads_ : params.kv_num_heads_)},
                            {static_cast<uint32_t>(is_query_input ? params.kv_num_heads_ : 0)}})
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

  if (!do_rotary_ && CanApplyFlashAttention(nullptr /* bias */, present_key, present_value, parameters, context)) {
    return ApplyFlashAttention(query, key, value, nullptr /* attention_bias */, output, past_key, present_key, past_value,
                               present_value, parameters, context);
  }

  Tensor qRotary;
  Tensor kRotary;
  if (do_rotary_) {
    qRotary = context.CreateGPUTensor(query->DataType(), query->Shape());
    auto pos_ids_shape = TensorShape({parameters.batch_size_, parameters.sequence_length_});
    Tensor pos_ids = context.CreateGPUTensor(DataTypeImpl::GetType<int64_t>(), pos_ids_shape);
    ORT_RETURN_IF_ERROR(GeneratePositionIDs(context, parameters, seqlen_k, &pos_ids));
    ORT_RETURN_IF_ERROR(RunRotaryEmbedding(context, parameters, query, &pos_ids, cos_cache, sin_cache, &qRotary, parameters.is_packed_qkv_, /* is_query_input = */ true));
    if (!parameters.is_packed_qkv_) {
      kRotary = context.CreateGPUTensor(key->DataType(), key->Shape());
      ORT_RETURN_IF_ERROR(RunRotaryEmbedding(context, parameters, key, &pos_ids, cos_cache, sin_cache, &kRotary, parameters.is_packed_qkv_, /* is_query_input = */ false));
      key = &kRotary;
    }
    query = &qRotary;
  }

  TensorShapeVector q_new_dims({parameters.batch_size_, parameter.is_packed_qkv_ ? parameters.num_heads_ + 2* parameters.kv_num_heads_ : parameters.num_heads_,
                                parameters.sequence_length_, parameters.head_size_});
  TensorShape q_new_shape(q_new_dims);
  Tensor Q = context.CreateGPUTensor(query->DataType(), q_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(
      context, parameter.is_packed_qkv_ ? parameters.num_heads_ + 2* parameters.kv_num_heads_ , parameters.sequence_length_, parameters.head_size_, query, nullptr, 0, &Q));
  if (parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH) {  // key and value in BNSH format
    return ApplyAttention(&Q, key, value, nullptr, past_key, past_value, output, present_key,
                          present_value, parameters, context, seqlen_k);
  }
  query = &Q;
  Tensor K;
  Tensor V;
  if (key) {
    TensorShapeVector k_new_dims({parameters.batch_size_, parameters.kv_num_heads_,
                                  parameters.kv_sequence_length_, parameters.head_size_});
    TensorShape k_new_shape(k_new_dims);
    K = context.CreateGPUTensor(key->DataType(), k_new_shape);
    ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.kv_num_heads_, parameters.kv_sequence_length_,
                                          parameters.head_size_, key, nullptr, 0, &K));
    key = &K;
  }
  if (value) {
    TensorShapeVector v_new_dims({parameters.batch_size_, parameters.kv_num_heads_,
                                  parameters.kv_sequence_length_, parameters.v_head_size_});
    TensorShape v_new_shape(v_new_dims);
    V = context.CreateGPUTensor(value->DataType(), v_new_shape);
    ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.kv_num_heads_, parameters.kv_sequence_length_,
                                          parameters.v_head_size_, value, nullptr, 0, &V));
    value = &V;
  }
  return ApplyAttention(query, key, value, nullptr, past_key, past_value, output, present_key,
                        present_value, parameters, context, seqlen_k);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
