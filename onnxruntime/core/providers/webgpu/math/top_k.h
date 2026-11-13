// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

namespace detail {

inline bool SupportsAppleHierarchicalTopK(std::string_view vendor, wgpu::BackendType backend,
                                          uint32_t subgroup_min, uint32_t subgroup_max,
                                          const wgpu::Limits& limits, uint64_t groups) {
  return vendor == "apple" && backend == wgpu::BackendType::Metal &&
         subgroup_min == 32 && subgroup_max == 32 &&
         limits.maxComputeWorkgroupSizeX >= 32 &&
         limits.maxComputeInvocationsPerWorkgroup >= 32 &&
         groups <= limits.maxComputeWorkgroupsPerDimension;
}

}  // namespace detail

class TopKProgram final : public Program<TopKProgram> {
 public:
  TopKProgram(uint32_t wg, uint32_t shared_size, bool largest, bool is_fp16)
      : Program{"TopK"}, wg_{wg}, shared_size_{shared_size}, largest_{largest}, is_fp16_{is_fp16} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"cols", ProgramUniformVariableDataType::Int32},
      {"k", ProgramUniformVariableDataType::Int32},
      {"padding_bits", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t wg_;
  uint32_t shared_size_;
  bool largest_;
  bool is_fp16_;
};

// Global-memory bitonic sort programs for cols > 2048
class TopKInitProgram final : public Program<TopKInitProgram> {
 public:
  explicit TopKInitProgram(bool is_fp16)
      : Program{"TopKInit"}, is_fp16_{is_fp16} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"cols", ProgramUniformVariableDataType::Int32},
      {"padded_cols", ProgramUniformVariableDataType::Int32},
      {"padding_bits", ProgramUniformVariableDataType::Uint32});

 private:
  bool is_fp16_;
};

class TopKSortStepProgram final : public Program<TopKSortStepProgram> {
 public:
  TopKSortStepProgram(bool largest)
      : Program{"TopKSortStep"}, largest_{largest} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"gap", ProgramUniformVariableDataType::Uint32},
      {"padded_cols", ProgramUniformVariableDataType::Uint32},
      {"total_threads", ProgramUniformVariableDataType::Uint32});

 private:
  bool largest_;
};

class TopKOutputProgram final : public Program<TopKOutputProgram> {
 public:
  TopKOutputProgram()
      : Program{"TopKOutput"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"padded_cols", ProgramUniformVariableDataType::Int32},
      {"k", ProgramUniformVariableDataType::Int32});
};

class TopK final : public WebGpuKernel {
 public:
  TopK(const OpKernelInfo& info) : WebGpuKernel{info} {
    opset_ = info.node().SinceVersion();
    info.GetAttrOrDefault<int64_t>("axis", &axis_, -1);
    info.GetAttrOrDefault<int64_t>("largest", &largest_, 1);
    info.GetAttrOrDefault<int64_t>("sorted", &sorted_, 1);
    if (opset_ <= 9) {
      int64_t k_temp = 0;
      ORT_ENFORCE(info.GetAttr<int64_t>("k", &k_temp).IsOK());
      attr_k_ = k_temp;
    }
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_ = -1;
  int64_t largest_ = 1;
  int64_t sorted_ = 1;
  int64_t attr_k_ = 0;
  int opset_;
};

}  // namespace webgpu
}  // namespace onnxruntime
