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
  F(bool, is_float16_)                \
  F(bool, count_include_pad_)         \
  F(bool, use_parallel_reduction_)

struct PoolProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_POOL_PROGRAM_CONFIG);
    Config(bool is_max_pool, bool is_nhwc, const TensorShapeVector& kernel_shape, bool is_float16,
           bool count_include_pad, bool use_parallel_reduction)
        : is_max_pool_{is_max_pool},
          is_nhwc_{is_nhwc},
          kernel_rank_{kernel_shape.size()},
          is_float16_{is_float16},
          count_include_pad_{count_include_pad},
          use_parallel_reduction_{use_parallel_reduction} {}
  };
  static constexpr std::string_view name = "Pool";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"kernel_size", ProgramUniformVariableDataType::Uint32},
                                          {"kernel_strides", ProgramUniformVariableDataType::Uint32},
                                          {"pads", ProgramUniformVariableDataType::Uint32},
                                          {"strides", ProgramUniformVariableDataType::Uint32},
                                          {"dilations", ProgramUniformVariableDataType::Uint32});
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
