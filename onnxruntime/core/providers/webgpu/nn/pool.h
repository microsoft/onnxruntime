// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_POOL_PROGRAM_CONFIG(F) \
  F(bool, is_max_pool_)               \
  F(bool, is_nhwc_)                   \
  F(size_t, kernel_rank_)             \
  F(bool, count_include_pad_)         \
  F(bool, use_parallel_reduction_)

struct PoolProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_POOL_PROGRAM_CONFIG);
  static Config MakeConfig(bool is_max_pool, bool is_nhwc, const TensorShapeVector& kernel_shape,
                           bool count_include_pad, bool use_parallel_reduction) {
    Config result{};
    result.is_max_pool_ = bool{is_max_pool};
    result.is_nhwc_ = bool{is_nhwc};
    result.kernel_rank_ = size_t{kernel_shape.size()};
    result.count_include_pad_ = bool{count_include_pad};
    result.use_parallel_reduction_ = bool{use_parallel_reduction};

    return result;
  }
  static constexpr std::string_view name = "Pool";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_POOL_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                       \
  U(Uint32, kernel_size)                       \
  U(Uint32, kernel_strides)                    \
  U(Uint32, pads)                              \
  U(Uint32, strides)                           \
  U(Uint32, dilations)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_POOL_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_POOL_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_POOL_PROGRAM_CONFIG

using PoolProgram = ConfiguredProgram<PoolProgramShader>;

template <typename PoolType, bool is_nhwc>
class Pool : public WebGpuKernel, public PoolBase {
 public:
  explicit Pool(const OpKernelInfo& info) : WebGpuKernel(info), PoolBase(info) {}

  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace onnxruntime
