// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;

class HyperConnectionPostMixProgram final : public Program<HyperConnectionPostMixProgram> {
 public:
  HyperConnectionPostMixProgram(int gate_layout, bool has_stream_mix, bool fuse_gate = false,
                                bool fuse_shared = false)
      : Program{"HyperConnectionPostMix"},
        gate_layout_(gate_layout),
        has_stream_mix_(has_stream_mix),
        fuse_gate_(fuse_gate),
        fuse_shared_(fuse_shared) {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"count", ProgramUniformVariableDataType::Uint32},
      {"branches", ProgramUniformVariableDataType::Uint32},
      {"hidden", ProgramUniformVariableDataType::Uint32});

 private:
  int gate_layout_;
  bool has_stream_mix_;
  bool fuse_gate_;
  bool fuse_shared_;
};

class HyperConnectionPostMix : public WebGpuKernel {
 public:
  explicit HyperConnectionPostMix(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 protected:
  HyperConnectionPostMix(const OpKernelInfo& info, bool fuse_gate, bool fuse_shared = false);

 private:
  int64_t num_branches_;
  bool fuse_gate_;
  bool fuse_shared_;
};

class HyperConnectionPostMixGateF16 final : public HyperConnectionPostMix {
 public:
  explicit HyperConnectionPostMixGateF16(const OpKernelInfo& info)
      : HyperConnectionPostMix(info, true) {}
};

class HyperConnectionSharedPostMixF16 final : public HyperConnectionPostMix {
 public:
  explicit HyperConnectionSharedPostMixF16(const OpKernelInfo& info)
      : HyperConnectionPostMix(info, true, true) {}
};

}  // namespace onnxruntime::contrib::webgpu
