#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "contrib_ops/cpu/sparse/packed_sparse_attention_indexer_merge_common.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

class PackedSparseAttentionIndexerMergeProgram final : public Program<PackedSparseAttentionIndexerMergeProgram> {
 public:
  explicit PackedSparseAttentionIndexerMergeProgram(int32_t hash_capacity)
      : Program{"PackedSparseAttentionIndexerMerge"}, hash_capacity_(hash_capacity) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"rows", ProgramUniformVariableDataType::Uint32},
                                          {"base_capacity", ProgramUniformVariableDataType::Uint32},
                                          {"queries", ProgramUniformVariableDataType::Uint32},
                                          {"capacity", ProgramUniformVariableDataType::Uint32});

 private:
  int32_t hash_capacity_;
};

class PackedSparseAttentionIndexerMerge final : public WebGpuKernel {
 public:
  explicit PackedSparseAttentionIndexerMerge(const OpKernelInfo& info)
      : WebGpuKernel(info), capacity_(indexer_merge::ReadCapacity(info)) {
    ORT_ENFORCE(info.GetAttrOrDefault<std::string>("policy_mode", "append_range") == "append_range",
                "WebGPU PackedSparseAttentionIndexerMerge supports only append_range");
  }
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t capacity_;
};

}  // namespace onnxruntime::contrib::webgpu