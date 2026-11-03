// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "static_node_replay.h"

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;

class HyperConnectionPreMixProgram final : public Program<HyperConnectionPreMixProgram> {
 public:
  explicit HyperConnectionPreMixProgram(int gate_layout, bool fuse_sigmoid = false)
      : Program{"HyperConnectionPreMix"}, gate_layout_(gate_layout), fuse_sigmoid_(fuse_sigmoid) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"count", ProgramUniformVariableDataType::Uint32},
      {"branches", ProgramUniformVariableDataType::Uint32},
      {"hidden", ProgramUniformVariableDataType::Uint32},
      {"reduction_scale", ProgramUniformVariableDataType::Float32});

 private:
  int gate_layout_;
  bool fuse_sigmoid_;
};

class HyperConnectionPreMix : public WebGpuKernel {
 public:
  explicit HyperConnectionPreMix(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 protected:
  HyperConnectionPreMix(const OpKernelInfo& info, bool fuse_sigmoid);

 private:
  mutable onnxruntime::webgpu::StaticNodeReplay static_replay_{"HyperConnectionPreMix", {0, 1}};
  int64_t num_branches_;
  float reduction_scale_;
  bool fuse_sigmoid_;
};

class HyperConnectionPreMixSigmoidF16 final : public HyperConnectionPreMix {
 public:
  explicit HyperConnectionPreMixSigmoidF16(const OpKernelInfo& info)
      : HyperConnectionPreMix(info, true) {}
};

}  // namespace onnxruntime::contrib::webgpu
