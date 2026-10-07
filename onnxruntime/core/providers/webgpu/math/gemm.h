// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_GEMM_NAIVE_PROGRAM_CONFIG(F) \
  F(bool, transA_)                          \
  F(bool, transB_)                          \
  F(bool, need_handle_bias_)                \
  F(bool, need_handle_matmul_)

struct GemmNaiveProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_GEMM_NAIVE_PROGRAM_CONFIG);
    Config(bool transA, bool transB, bool need_handle_bias, bool need_handle_matmul)
        : transA_{transA},
          transB_{transB},
          need_handle_bias_{need_handle_bias},
          need_handle_matmul_{need_handle_matmul} {}
  };
  static constexpr std::string_view name = "GemmNaive";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_GEMM_NAIVE_PROGRAM_CONFIG

using GemmNaiveProgram = ConfiguredProgram<GemmNaiveProgramShader>;

class Gemm final : public WebGpuKernel {
 public:
  // Abstract base class for alternative optimized Gemm implementations.
  // Implementations can provide optimized computation paths by deriving from this class.
  class GemmOptImpl {
   public:
    explicit GemmOptImpl(const Gemm& parent) : parent_(parent) {}
    virtual ~GemmOptImpl() = default;

    // Called during Compute phase to execute implementation-specific computation.
    // @param context       The WebGPU compute context.
    // @param handled       Output parameter. Set to true if this implementation handled the computation.
    // @return Status::OK() on success, or an error status on failure.
    virtual Status Compute(ComputeContext& context,
                           /*out*/ bool& handled) = 0;

   protected:
    const Gemm& parent_;
  };

  Gemm(const OpKernelInfo& info) : WebGpuKernel(info) {
    int64_t transA_temp = 0;
    info.GetAttrOrDefault("transA", &transA_temp, static_cast<int64_t>(0));
    transA_ = transA_temp != 0;

    int64_t transB_temp = 0;
    info.GetAttrOrDefault("transB", &transB_temp, static_cast<int64_t>(0));
    transB_ = transB_temp != 0;

    info.GetAttrOrDefault("alpha", &alpha_, 1.0f);
    info.GetAttrOrDefault("beta", &beta_, 1.0f);
  }

  Status ComputeInternal(ComputeContext& context) const override;

  bool TransA() const { return transA_; }
  bool TransB() const { return transB_; }
  float Alpha() const { return alpha_; }
  float Beta() const { return beta_; }

 private:
  bool transA_;
  bool transB_;
  float alpha_;
  float beta_;

  // Alternative optimized implementation (lazily created on the first Compute call,
  // once the device capabilities can be queried from the compute context). A null
  // impl_ after initialization means this device has no vendor-optimized path.
  mutable std::unique_ptr<GemmOptImpl> impl_;
  mutable std::once_flag impl_init_flag_;
};

}  // namespace webgpu
}  // namespace onnxruntime
