// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

// OneDNN blocked format types
// IMPORTANT: keep enum order/numeric values in sync with format_transform.wgsl.template format macro values.
enum class BlockedFormat {
  Plain,      // Standard NCHW layout
  nChw4c,     // Blocked with 4-channel blocks
  ABcd16a4b,  // 2D blocked format: blocks A dimension with 16, B dimension with 4
};

class FormatTransformProgram final : public Program<FormatTransformProgram> {
 public:
  FormatTransformProgram(BlockedFormat src_format, BlockedFormat dst_format,
                         const TensorShape& input_shape);

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  BlockedFormat src_format_;
  BlockedFormat dst_format_;
  TensorShape input_shape_;
};

// Internal operator for format transformation between plain and blocked formats
class FormatTransform final : public WebGpuKernel {
 public:
  // Helper that converts a plain-format tensor into the requested blocked layout on the GPU.
  // The returned tensor uses the provided allocator, and the shader dispatch is issued via context.
  static Status TransformPlainToBlocked(ComputeContextBase& context,
                                        const Tensor& input,
                                        BlockedFormat dst_format,
                                        AllocatorPtr alloc,
                                        std::unique_ptr<Tensor>& output);

  FormatTransform(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  BlockedFormat src_format_;
  BlockedFormat dst_format_;
};

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
