// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {
namespace webgpu {

class PoolProgram final : public Program<PoolProgram> {
 public:
  PoolProgram(bool is_max_pool, bool is_nhwc, const TensorShapeVector& kernel_shape, bool is_float16,
              bool count_include_pad, bool are_small_output_big_kernel)
      : Program{"Pool"},
        is_max_pool_{is_max_pool},
        is_nhwc_{is_nhwc},
        kernel_shape_{kernel_shape},
        is_float16_{is_float16},
        count_include_pad_{count_include_pad},
        are_small_output_big_kernel_{are_small_output_big_kernel} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"kernel_size", ProgramUniformVariableDataType::Uint32},
                                          {"kernel_strides", ProgramUniformVariableDataType::Uint32},
                                          {"pads", ProgramUniformVariableDataType::Uint32},
                                          {"strides", ProgramUniformVariableDataType::Uint32},
                                          {"dilations", ProgramUniformVariableDataType::Uint32});

 private:
  // Whether it is max pool or average pool.
  const bool is_max_pool_;

  const bool is_nhwc_;
  const TensorShapeVector kernel_shape_;
  const bool is_float16_;
  const bool count_include_pad_;
  const bool are_small_output_big_kernel_;
};

template <typename PoolType, bool is_nhwc>
class Pool : public WebGpuKernel, public PoolBase {
 public:
  explicit Pool(const OpKernelInfo& info) : WebGpuKernel(info), PoolBase(info) {}

  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace onnxruntime
