#include <optional>

#include "contrib_ops/webgpu/bert/sparse_attention_selection_merge.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime::contrib::webgpu {

ONNX_OPERATOR_KERNEL_EX(SparseAttentionSelectionMerge, kMSDomain, 1, kWebGpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
                        SparseAttentionSelectionMerge);

Status SparseAttentionSelectionMergeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& base = shader.AddInput("base", ShaderUsage::UseValueTypeAlias);
  const auto& counts = shader.AddInput("counts", ShaderUsage::UseValueTypeAlias);
  const auto& rows = shader.AddInput("rows", ShaderUsage::UseValueTypeAlias);
  const auto& starts = shader.AddInput("starts", ShaderUsage::UseValueTypeAlias);
  const auto& ends = shader.AddInput("ends", ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);
  const auto& output_counts = shader.AddOutput("output_counts", ShaderUsage::UseValueTypeAlias);
  const auto& status = shader.AddOutput("status", ShaderUsage::UseValueTypeAlias);
  const bool use_hash = hash_capacity_ <= 4096;
  if (use_hash) {
    shader.AdditionalImplementation() << "var<workgroup> keys: array<i32, " << hash_capacity_ << ">;\n";
  }
  auto& body = shader.MainFunctionBody();
  body << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.queries")
       << "let offset = global_idx * uniforms.capacity;\n"
       << output_counts.SetByOffset("global_idx", "0") << status.SetByOffset("global_idx", "0")
       << "for (var column = 0u; column < uniforms.capacity; column++) {\n"
       << output.SetByOffset("offset + column", "-1") << "}\n"
       << "let row = " << rows.GetByOffset("global_idx") << ";\n"
       << "let start = " << starts.GetByOffset("global_idx") << ";\n"
       << "let end = " << ends.GetByOffset("global_idx") << ";\n"
       << "if (row < 0 || u32(row) >= uniforms.rows || start < 0 || end < start) {\n"
       << status.SetByOffset("global_idx", "1") << "return; }\n"
       << "let count = " << counts.GetByOffset("u32(row)") << ";\n"
       << "if (count < 0 || u32(count) > uniforms.base_capacity) {\n"
       << status.SetByOffset("global_idx", "1") << "return; }\n"
       << "let base_offset = u32(row) * uniforms.base_capacity;\n"
       << "for (var column = 0u; column < u32(count); column++) { if ("
       << base.GetByOffset("base_offset + column") << " < 0) {\n"
       << status.SetByOffset("global_idx", "1") << "return; }}\n";
  if (use_hash) {
    body << "for (var slot = 0u; slot < " << hash_capacity_ << "u; slot++) { keys[slot] = -1; }\n";
  }
  body << "var emitted = 0u; var overlap = 0u; var overflow = false;\n"
       << "for (var column = 0u; column < u32(count); column++) {\n"
       << "let value = " << base.GetByOffset("base_offset + column") << ";\n"
       << "var duplicate = false;\n";
  if (use_hash) {
    body << "var slot = (u32(value) * 2654435761u) & " << hash_capacity_ - 1 << "u;\n"
         << "for (var probe = 0u; probe < " << hash_capacity_ << "u; probe++) {\n"
         << "if (keys[slot] == value) { duplicate = true; break; }\n"
         << "if (keys[slot] == -1) { keys[slot] = value; break; }\n"
         << "slot = (slot + 1u) & " << hash_capacity_ - 1 << "u; }\n";
  } else {
    body << "for (var previous = 0u; previous < emitted; previous++) { if ("
         << output.GetByOffset("offset + previous") << " == value) { duplicate = true; break; }}\n";
  }
  body << "if (!duplicate) { if (emitted == uniforms.capacity) { overflow = true; break; }\n"
       << output.SetByOffset("offset + emitted", "value")
       << "emitted++; if (value >= start && value < end) { overlap++; }} }\n"
       << "if (overflow || u32(end - start) - overlap > uniforms.capacity - emitted) {\n"
       << "for (var column = 0u; column < uniforms.capacity; column++) {\n"
       << output.SetByOffset("offset + column", "-1") << "}\n"
       << status.SetByOffset("global_idx", "2") << "return; }\n"
       << "let base_emitted = emitted;\n"
       << "for (var value = start; value < end; value++) {\n"
       << "var duplicate = false;\n";
  if (use_hash) {
    body << "var slot = (u32(value) * 2654435761u) & " << hash_capacity_ - 1 << "u;\n"
         << "for (var probe = 0u; probe < " << hash_capacity_ << "u; probe++) {\n"
         << "if (keys[slot] == value) { duplicate = true; break; }\n"
         << "if (keys[slot] == -1) { break; }\n"
         << "slot = (slot + 1u) & " << hash_capacity_ - 1 << "u; }\n";
  } else {
    body << "for (var previous = 0u; previous < base_emitted; previous++) { if ("
         << output.GetByOffset("offset + previous") << " == value) { duplicate = true; break; }}\n";
  }
  body << "if (!duplicate) {\n"
       << output.SetByOffset("offset + emitted", "value")
       << "emitted++; }}\n"
       << output_counts.SetByOffset("global_idx", "i32(emitted)");
  return Status::OK();
}

Status SparseAttentionSelectionMerge::ComputeInternal(ComputeContext& context) const {
  const auto* base = context.Input(0);
  const auto* counts = context.Input(1);
  const auto* rows = context.Input(2);
  const auto* starts = context.Input(3);
  const auto* ends = context.Input(4);
  ORT_RETURN_IF(context.Input(5) || context.Input(6), "append_range does not accept additional index inputs");
  selection_merge::Dimensions dimensions;
  ORT_RETURN_IF_ERROR(selection_merge::Validate(base, counts, rows, starts, ends, capacity_, dimensions));
  ORT_RETURN_IF(base->Shape().Size() > UINT32_MAX || static_cast<uint64_t>(dimensions.queries) * capacity_ > UINT32_MAX,
                "WebGPU selection tensor offsets must fit uint32");
  auto* output = context.Output(0, TensorShape({dimensions.queries, capacity_}));
  auto* output_counts = context.Output(1, TensorShape({dimensions.queries}));
  auto* status = context.Output(2, TensorShape({dimensions.queries}));
  if (dimensions.queries == 0) return Status::OK();
  std::optional<Tensor> empty_binding;
  if (base->Shape().Size() == 0 || counts->Shape().Size() == 0) {
    empty_binding.emplace(context.CreateGPUTensor(DataTypeImpl::GetType<int32_t>(), TensorShape({1})));
  }
  const auto* base_binding = base->Shape().Size() == 0 ? &empty_binding.value() : base;
  const auto* counts_binding = counts->Shape().Size() == 0 ? &empty_binding.value() : counts;
  SparseAttentionSelectionMergeProgram program(dimensions.hash_capacity);
  program.SetWorkgroupSize(1).SetDispatchGroupSize(dimensions.queries).CacheHint(std::to_string(dimensions.hash_capacity)).AddInputs({{base_binding, ProgramTensorMetadataDependency::None}, {counts_binding, ProgramTensorMetadataDependency::None}, {rows, ProgramTensorMetadataDependency::None}, {starts, ProgramTensorMetadataDependency::None}, {ends, ProgramTensorMetadataDependency::None}}).AddOutputs({{output, ProgramTensorMetadataDependency::None}, {output_counts, ProgramTensorMetadataDependency::None}, {status, ProgramTensorMetadataDependency::None}}).AddUniformVariables({{static_cast<uint32_t>(dimensions.rows)}, {static_cast<uint32_t>(dimensions.capacity)}, {static_cast<uint32_t>(dimensions.queries)}, {static_cast<uint32_t>(capacity_)} });
  return context.RunProgram(program);
}

}  // namespace onnxruntime::contrib::webgpu