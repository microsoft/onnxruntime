// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/llm/rotary_embedding.h"
#include "contrib_ops/webgpu/bert/rotary_embedding.h"
#include "core/providers/webgpu/generator/range.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    RotaryEmbedding,
    kOnnxDomain,
    23,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int64_t>()),
    RotaryEmbedding);

RotaryEmbedding::RotaryEmbedding(const OpKernelInfo& info) : WebGpuKernel(info) {
  rotary_embedding_dim_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0));
  num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("num_heads", 0));
  interleaved_ = (info.GetAttrOrDefault<int64_t>("interleaved", 0) == 1);
}

Status RotaryEmbedding::ComputeInternal(ComputeContext& context) const {
  // ONNX inputs:  X(0), cos_cache(1), sin_cache(2), position_ids(3, optional)
  const auto* input = context.Input<Tensor>(0);
  const auto* cos_cache = context.Input<Tensor>(1);
  const auto* sin_cache = context.Input<Tensor>(2);
  const auto* position_ids = context.Input<Tensor>(3);  // optional

  const auto input_shape = input->Shape();
  auto* output = context.Output(0, input_shape);

  const auto batch_size = onnxruntime::narrow<uint32_t>(input_shape[0]);
  const auto batch_stride = onnxruntime::narrow<uint32_t>(input_shape.SizeFromDimension(1));
  const auto sequence_length = onnxruntime::narrow<uint32_t>(input_shape[input_shape.NumDimensions() - 2]);
  const auto hidden_size = batch_stride / sequence_length;
  const auto half_rotary_embedding_dim = onnxruntime::narrow<uint32_t>(cos_cache->Shape()[cos_cache->Shape().NumDimensions() - 1]);

  // Compute head_size: when rotary_embedding_dim is not set, head_size = rotary_dim (= 2 * half).
  // When rotary_embedding_dim is set, derive head_size from the 4D input shape or num_heads attribute.
  uint32_t head_size;
  if (rotary_embedding_dim_ == 0) {
    head_size = half_rotary_embedding_dim * 2;
  } else if (input_shape.NumDimensions() == 4) {
    // 4D input: [batch, num_heads, seq, head_size]
    head_size = onnxruntime::narrow<uint32_t>(input_shape[3]);
  } else {
    ORT_ENFORCE(num_heads_ > 0,
                "Attribute 'num_heads' must be provided when 'rotary_embedding_dim' is specified "
                "and input is not rank-4 (batch, num_heads, sequence, head).");
    head_size = hidden_size / num_heads_;
  }

  const TensorShape global_shape({batch_size,
                                  sequence_length,
                                  hidden_size / head_size,
                                  head_size - half_rotary_embedding_dim});

  const auto rank = global_shape.NumDimensions();
  std::vector<uint32_t> global_dims(rank);
  std::vector<uint32_t> global_strides(rank);
  for (size_t j = 0; j < rank; ++j) {
    global_dims[j] = onnxruntime::narrow<uint32_t>(global_shape[j]);
    global_strides[j] = onnxruntime::narrow<uint32_t>(global_shape.SizeFromDimension(j + 1));
  }

  const auto output_size = onnxruntime::narrow<const uint32_t>(global_shape.Size());
  const auto input_output_strides =
      input_shape.NumDimensions() == 3
          ? std::vector<uint32_t>({batch_stride, hidden_size, head_size, 1})
          : (input_shape.NumDimensions() == 4
                 ? std::vector<uint32_t>({batch_stride, head_size, sequence_length * head_size, 1})
                 : std::vector<uint32_t>({}));

  // The contrib RotaryEmbeddingProgram expects inputs in order:
  //   input(0), position_ids(1), cos_cache(2), sin_cache(3)
  // The ONNX op has: X(0), cos_cache(1), sin_cache(2), position_ids(3, optional)

  if (position_ids != nullptr) {
    // position_ids provided: cos/sin cache is 2D (max_pos, D/2)
    // position_ids bounds validation is handled by shader-side defense-in-depth checks
    // (OOB position_ids → pass-through input unchanged). Host-side value scanning is not possible
    // because WebGPU program inputs must be GPU buffers (InputMemoryType(OrtMemTypeCPUInput) is
    // incompatible with AddInputs).
    // Note: ONNX RotaryEmbedding has no base-offset mode (format 0) — position_ids is always
    // a 2D tensor (batch_size, sequence_length) when provided.

    contrib::webgpu::RotaryEmbeddingProgram program{interleaved_};
    program
        .CacheHint(interleaved_)
        .AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank},
                    {position_ids, ProgramTensorMetadataDependency::Rank},
                    {cos_cache, ProgramTensorMetadataDependency::Rank},
                    {sin_cache, ProgramTensorMetadataDependency::Rank}})
        .AddOutput({output, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({{1.0f},
                              {gsl::make_span(global_dims)},
                              {gsl::make_span(global_strides)},
                              {gsl::make_span(input_output_strides)}})
        .AddIndices(TensorShape{1, 1});
    return context.RunProgram(program);
  }

  // position_ids NOT provided: cos/sin cache is 3D (B, S, D/2)
  // Reshape to 2D (B*S, D/2) and generate sequential position_ids.
  const auto total_seq = batch_size * sequence_length;
  const TensorShape cache_2d_shape({static_cast<int64_t>(total_seq),
                                    static_cast<int64_t>(half_rotary_embedding_dim)});

  // Generate position_ids [0, 1, ..., B*S-1] reshaped as (B, S) on GPU using RangeProgram
  const TensorShape pos_ids_shape({static_cast<int64_t>(batch_size),
                                   static_cast<int64_t>(sequence_length)});
  Tensor pos_ids_tensor = context.CreateGPUTensor(DataTypeImpl::GetType<int64_t>(), pos_ids_shape);
  {
    RangeProgram range_program{ONNX_NAMESPACE::TensorProto_DataType_INT64};
    int32_t start_i32 = 0;
    int32_t delta_i32 = 1;
    range_program
        .AddOutput({&pos_ids_tensor, ProgramTensorMetadataDependency::Type})
        .SetDispatchGroupSize((total_seq + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({
            total_seq,
            std::bit_cast<uint32_t>(start_i32),
            std::bit_cast<uint32_t>(delta_i32),
        });
    ORT_RETURN_IF_ERROR(context.RunProgram(range_program));
  }

  contrib::webgpu::RotaryEmbeddingProgram program{interleaved_};
  program
      .CacheHint(interleaved_)
      .AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank},
                  {&pos_ids_tensor, ProgramTensorMetadataDependency::Rank},
                  {cos_cache, ProgramTensorMetadataDependency::Rank, cache_2d_shape, 1},
                  {sin_cache, ProgramTensorMetadataDependency::Rank, cache_2d_shape, 1}})
      .AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{1.0f},
                            {gsl::make_span(global_dims)},
                            {gsl::make_span(global_strides)},
                            {gsl::make_span(input_output_strides)}})
      .AddIndices(TensorShape{1, 1});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
