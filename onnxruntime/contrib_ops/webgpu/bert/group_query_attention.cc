// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/group_query_attention_helper.h"
#include "contrib_ops/webgpu/bert/attention_common.h"
#include "contrib_ops/webgpu/bert/group_query_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/bert/rotary_embedding.h"

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
  sh.AddInput("seqlens", ShaderUsage::UseUniform);
  sh.AddOutput("output", ShaderUsage::UseUniform);
  sh.MainFunctionBody() << "let batch_idx = global_idx / uniforms.sequence_length;\n"
                        << "let sequence_idx = i32(global_idx % uniforms.sequence_length);\n"
                        << "var pos_id: u32 = 0u;\n"
                        << "if (is_first_prompt == 0) {\n"
                        << "  let total_seqlen = ${seqLensInputHelper.getByOffset('batch_idx')} + 1;\n"
                        << "  let past_seqlen = total_seqlen - i32(uniforms.sequence_length);\n"
                        << "  if (past_seqlen + sequence_idx < total_seqlen) {\n"
                        << "    pos_id = u32(past_seqlen + sequence_idx);\n"
                        << "  } else {\n"
                        << "    pos_id = 1u;\n"
                        << "  }\n"
                        << "}\n"
                        << "output[global_idx] = pos_id;\n";
  return Status::OK();
}

Status GeneratePositionIDs(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& params, const Tensor* seqlens, Tensor* output_tensor) {
  GeneratePositionIDsProgram program(params);
  program.AddInput(seqlens)
      .AddOutput(output_tensor)
      .AddUniformVariables({{static_cast<uint32_t>(params.batch_size_)},
                            {static_cast<uint32_t>(params.sequence_length_)},
                            {static_cast<uint32_t>(params.num_heads_)},
                            {static_cast<uint32_t>(params.head_size_)},
                            {static_cast<uint32_t>(params.rotary_dim_)},
                            {static_cast<uint32_t>(params.rotary_interleaved_)},
                            {static_cast<uint32_t>(params.is_first_prompt_ ? 0 : 1)},
                            {static_cast<uint32_t>(params.total_sequence_length_)}});
  return context.RunProgram(program);
}

Status RunRotaryEmbedding(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& params, const Tensor* input, const Tensor* pos_ids, const Tensor* cos_cache, const Tensor* sin_cache, Tensor* output) {
  const auto half_rotary_embedding_dim = gsl::narrow<uint32_t>(cos_cache->Shape()[1]);

  const TensorShape global_shape({params.batch_size_, params.sequence_length_, params.num_heads_, params.head_size_ - half_rotary_embedding_dim});
  const auto rank = global_shape.NumDimensions();
  std::vector<uint32_t> global_dims(rank);
  std::vector<uint32_t> global_strides(rank);
  for (size_t j = 0; j < rank; ++j) {
    global_dims[j] = gsl::narrow<uint32_t>(global_shape[j]);
    global_strides[j] = gsl::narrow<uint32_t>(global_shape.SizeFromDimension(j + 1));
  }
  const auto input_output_strides = std::vector<uint32_t>({gsl::narrow<uint32_t>(params.batch_size_), gsl::narrow<uint32_t>(params.hidden_size_), gsl::narrow<uint32_t>(params.head_size_), 1});
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
  const Tensor* seqlens_k = context.Input<Tensor>(5);
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
                                                                seqlens_k,
                                                                total_seqlen_tensor,
                                                                scale_,
                                                                softcap_));
  WebgpuAttentionParameters parameters(params);
  if (parameters.is_packed_qkv_) {
    ORT_NOT_IMPLEMENTED("Packed QKV of shape (B, L, N, 3, H) not implemented for webgpu-ep.");
  }
  if (do_rotary_) {
    Tensor q = context.CreateGPUTensor(query->DataType(), query->Shape());
    Tensor k = context.CreateGPUTensor(key->DataType(), key->Shape());
    TensorShape pos_ids_shape = parameters.is_first_prompt_ ? TensorShape({1}) : TensorShape({parameters.batch_size_ * parameters.sequence_length_});
    Tensor pos_ids = context.CreateGPUTensor(DataTypeImpl::GetType<int64_t>(), pos_ids_shape);
    ORT_RETURN_IF_ERROR(GeneratePositionIDs(context, parameters, seqlens_k, &pos_ids));

    ORT_RETURN_IF_ERROR(RunRotaryEmbedding(context, parameters, query, &pos_ids, cos_cache, sin_cache, &q));

    ORT_RETURN_IF_ERROR(RunRotaryEmbedding(context, parameters, key, &pos_ids, cos_cache, sin_cache, &k));

    query = &q;
    key = &k;
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
                          present_value, parameters, context, seqlens_k);
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
                        present_value, parameters, context, seqlens_k);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
