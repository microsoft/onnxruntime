// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

// Forward declaration
class Cast;

// Type constraint functions for Cast operator
const std::vector<MLDataType>& CastOpTypeConstraints();
const std::vector<MLDataType>& CastOpTypeConstraintsWithoutInt64();

// Create Cast kernel info with appropriate type constraints based on graph capture support
template <int StartVersion, int EndVersion = StartVersion>
KernelCreateInfo CreateCastKernelInfo(bool enable_graph_capture) {
  const auto& type_constraints = enable_graph_capture ? CastOpTypeConstraints() : CastOpTypeConstraintsWithoutInt64();

  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    auto cast_kernel = std::make_unique<Cast>(info);
    out = std::move(cast_kernel);
    return Status::OK();
  };

  if constexpr (StartVersion == EndVersion) {
    // Non-versioned kernel
    return {
        KernelDefBuilder()
            .SetName("Cast")
            .SetDomain(kOnnxDomain)
            .SinceVersion(StartVersion)
            .Provider(kWebGpuExecutionProvider)
            .TypeConstraint("T1", type_constraints)
            .TypeConstraint("T2", type_constraints)
            .Build(),
        kernel_create_fn};
  } else {
    // Versioned kernel
    return {
        KernelDefBuilder()
            .SetName("Cast")
            .SetDomain(kOnnxDomain)
            .SinceVersion(StartVersion, EndVersion)
            .Provider(kWebGpuExecutionProvider)
            .TypeConstraint("T1", type_constraints)
            .TypeConstraint("T2", type_constraints)
            .Build(),
        kernel_create_fn};
  }
}

class CastProgram final : public Program<CastProgram> {
 public:
  CastProgram(int32_t to, bool is_from_int64) : Program{"Cast"}, to_{to}, is_from_int64_{is_from_int64} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"vec_size", ProgramUniformVariableDataType::Uint32});

 private:
  int32_t to_;
  bool is_from_int64_;
};

class Cast final : public WebGpuKernel {
 public:
  Cast(const OpKernelInfo& info) : WebGpuKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = onnxruntime::narrow<int32_t>(to);

    // ignore attribute 'saturate' as float8 is not supported in WebGPU
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int32_t to_;
};

}  // namespace webgpu
}  // namespace onnxruntime
