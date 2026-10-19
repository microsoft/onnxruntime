#include <optional>

#include "contrib_ops/webgpu/bert/packed_sparse_attention_indexer_merge.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime::contrib::webgpu {

ONNX_OPERATOR_KERNEL_EX(PackedSparseAttentionIndexerMerge, kMSDomain, 1, kWebGpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
                        PackedSparseAttentionIndexerMerge);

Status PackedSparseAttentionIndexerMergeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& base = shader.AddInput("base", ShaderUsage::UseValueTypeAlias);
  const auto& counts = shader.AddInput("counts", ShaderUsage::UseValueTypeAlias);
  const auto& rows = shader.AddInput("rows", ShaderUsage::UseValueTypeAlias);
  const auto& starts = shader.AddInput("starts", ShaderUsage::UseValueTypeAlias);
  const auto& ends = shader.AddInput("ends", ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);
  const auto& output_counts = shader.AddOutput("output_counts", ShaderUsage::UseValueTypeAlias);
  const auto& status = shader.AddOutput("status", ShaderUsage::UseValueTypeAlias);
  shader.AdditionalImplementation()
      << "var<workgroup> positions: array<u32, 256>;\n"
      << "var<workgroup> base_offset_shared: u32;\n"
      << "var<workgroup> count_shared: u32;\n"
      << "var<workgroup> start_shared: i32;\n"
      << "var<workgroup> end_shared: i32;\n"
      << "var<workgroup> result_shared: u32;\n"
      << "var<workgroup> emitted_shared: u32;\n"
      << "var<workgroup> overlap_shared: atomic<u32>;\n"
      << "var<workgroup> invalid_shared: atomic<u32>;\n";
  auto& body = shader.MainFunctionBody();
  body << "if (workgroup_idx >= uniforms.queries) { return; }\n"
       << "let offset = workgroup_idx * uniforms.capacity;\n"
       << "for (var column = local_idx; column < uniforms.capacity; column += workgroup_size_x) {\n"
       << output.SetByOffset("offset + column", "-1") << "}\n"
       << "storageBarrier();\n"
       << "if (local_idx == 0u) {\n"
       << output_counts.SetByOffset("workgroup_idx", "0") << status.SetByOffset("workgroup_idx", "0")
       << "let row = " << rows.GetByOffset("workgroup_idx") << ";\n"
       << "let start = " << starts.GetByOffset("workgroup_idx") << ";\n"
       << "let end = " << ends.GetByOffset("workgroup_idx") << ";\n"
       << "if (row < 0 || u32(row) >= uniforms.rows || start < 0 || end < start) {\n"
       << "result_shared = 1u;\n"
       << "} else {\n"
       << "let count = " << counts.GetByOffset("u32(row)") << ";\n"
       << "if (count < 0 || u32(count) > uniforms.base_capacity) {\n"
       << "result_shared = 1u;\n"
       << "} else {\n"
       << "base_offset_shared = u32(row) * uniforms.base_capacity;\n"
       << "count_shared = u32(count); start_shared = start; end_shared = end;\n"
       << "}}}\n"
       << "if (workgroupUniformLoad(&result_shared) != 0u) {\n"
       << "if (local_idx == 0u) { " << status.SetByOffset("workgroup_idx", "1") << "} return; }\n"
       << "let base_offset = workgroupUniformLoad(&base_offset_shared);\n"
       << "let count = workgroupUniformLoad(&count_shared);\n"
       << "let start = workgroupUniformLoad(&start_shared);\n"
       << "let end = workgroupUniformLoad(&end_shared);\n"
       << "for (var column = local_idx; column < count; column += workgroup_size_x) { if ("
       << base.GetByOffset("base_offset + column") << " < 0) { atomicOr(&invalid_shared, 1u); }}\n"
       << "workgroupBarrier();\n"
       << "if (local_idx == 0u) { result_shared = atomicLoad(&invalid_shared); }\n"
       << "if (workgroupUniformLoad(&result_shared) != 0u) {\n"
       << "if (local_idx == 0u) { " << status.SetByOffset("workgroup_idx", "1") << "} return; }\n";

  // Each chunk is compacted in input order; storage barriers publish its output to range membership scans.
  auto emit_compaction = [&](bool base_phase) {
    body << "positions[local_idx] = select(0u, 1u, keep);\n"
         << "workgroupBarrier();\n"
         << "if (local_idx == 0u) {\n"
         << "var emitted = emitted_shared;\n"
         << "for (var lane = 0u; lane < workgroup_size_x; lane++) {\n"
         << "let flag = positions[lane]; positions[lane] = emitted; emitted += flag; }\n"
         << "emitted_shared = emitted; }\n"
         << "workgroupBarrier();\n"
         << "if (keep && positions[local_idx] < uniforms.capacity) {\n"
         << output.SetByOffset("offset + positions[local_idx]", "value");
    if (base_phase) {
      body << "if (value >= start && value < end) { atomicAdd(&overlap_shared, 1u); }\n";
    }
    body << "}\n"
         << "storageBarrier();\n"
         << "workgroupBarrier();\n";
  };

  body << "for (var chunk = 0u; chunk < count; chunk += workgroup_size_x) {\n"
       << "let column = chunk + local_idx;\n"
       << "var value = 0i; var keep = column < count;\n"
       << "if (keep) {\n"
       << "value = " << base.GetByOffset("base_offset + column") << ";\n"
       << "for (var previous = 0u; previous < column; previous++) { if ("
       << base.GetByOffset("base_offset + previous") << " == value) { keep = false; break; }}}\n";
  emit_compaction(true);
  body << "if (workgroupUniformLoad(&emitted_shared) > uniforms.capacity) { break; }}\n"
       << "let base_emitted = workgroupUniformLoad(&emitted_shared);\n"
       << "if (local_idx == 0u) {\n"
       << "if (base_emitted > uniforms.capacity || "
          "u32(end - start) - atomicLoad(&overlap_shared) > uniforms.capacity - base_emitted) {\n"
       << "result_shared = 2u; }}\n"
       << "if (workgroupUniformLoad(&result_shared) != 0u) {\n"
       << "for (var column = local_idx; column < uniforms.capacity; column += workgroup_size_x) {\n"
       << output.SetByOffset("offset + column", "-1") << "}\n"
       << "if (local_idx == 0u) { " << status.SetByOffset("workgroup_idx", "2") << "} return; }\n"
       << "let range_length = u32(end - start);\n"
       << "for (var chunk = 0u; chunk < range_length; chunk += workgroup_size_x) {\n"
       << "let column = chunk + local_idx;\n"
       << "var value = 0i; var keep = column < range_length;\n"
       << "if (keep) {\n"
       << "value = start + i32(column);\n"
       << "for (var previous = 0u; previous < base_emitted; previous++) { if ("
       << output.GetByOffset("offset + previous") << " == value) { keep = false; break; }}}\n";
  emit_compaction(false);
  body << "}\n"
       << "if (local_idx == 0u) {\n"
       << output_counts.SetByOffset("workgroup_idx", "i32(emitted_shared)") << "}\n";
  return Status::OK();
}

Status PackedSparseAttentionIndexerMerge::ComputeInternal(ComputeContext& context) const {
  const auto* base = context.Input(0);
  const auto* counts = context.Input(1);
  const auto* rows = context.Input(2);
  const auto* starts = context.Input(3);
  const auto* ends = context.Input(4);
  ORT_RETURN_IF(context.Input(5) || context.Input(6), "append_range does not accept additional index inputs");
  indexer_merge::Dimensions dimensions;
  ORT_RETURN_IF_ERROR(indexer_merge::Validate(base, counts, rows, starts, ends, capacity_, dimensions));
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
  PackedSparseAttentionIndexerMergeProgram program;
  program.SetWorkgroupSize(256)
      .SetDispatchGroupSize(dimensions.queries)
      .AddInputs({{base_binding, ProgramTensorMetadataDependency::Type},
                  {counts_binding, ProgramTensorMetadataDependency::Type},
                  {rows, ProgramTensorMetadataDependency::Type},
                  {starts, ProgramTensorMetadataDependency::Type},
                  {ends, ProgramTensorMetadataDependency::Type}})
      .AddOutputs({{output, ProgramTensorMetadataDependency::Type},
                   {output_counts, ProgramTensorMetadataDependency::Type},
                   {status, ProgramTensorMetadataDependency::Type}})
      .AddUniformVariables({{static_cast<uint32_t>(dimensions.rows)},
                            {static_cast<uint32_t>(dimensions.capacity)},
                            {static_cast<uint32_t>(dimensions.queries)},
                            {static_cast<uint32_t>(capacity_)}});
  return context.RunProgram(program);
}

}  // namespace onnxruntime::contrib::webgpu