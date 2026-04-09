// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/webgpu/quantization/quantization_utils.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::webgpu {

class QuantizeLinearProgram final : public Program<QuantizeLinearProgram> {
 public:
  QuantizeLinearProgram(util::QuantizationType quantization_type, bool has_zero_point,
                        uint32_t workgroup_size, int32_t y_element_data_type);

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"data_size", ProgramUniformVariableDataType::Uint32},
      {"axis_stride", ProgramUniformVariableDataType::Uint32},
      {"scale_dim_on_axis", ProgramUniformVariableDataType::Uint32},
  );

 private:
  const util::QuantizationType quantization_type_;
  const bool has_zero_point_;
  const uint32_t workgroup_size_;
  const bool y_is_signed_;
  const util::U32PackingMode y_packing_mode_;
};

class QuantizeLinear final : public WebGpuKernel {
 public:
  explicit QuantizeLinear(const OpKernelInfo& info)
      : WebGpuKernel{info},
        axis_{info.GetAttrOrDefault<int64_t>("axis", 1)},
        block_size_{info.GetAttrOrDefault<int64_t>("block_size", 0)} {
    ORT_ENFORCE(block_size_ >= 0, "'block_size' must be non-negative.");

    if (info.GetAttrOrDefault<int64_t>("output_dtype", 0) != int64_t{0}) {
      ORT_NOT_IMPLEMENTED("Explicitly specified 'output_dtype' is not yet supported.");
    }

    if (info.GetAttrOrDefault<int64_t>("precision", 0) != int64_t{0}) {
      ORT_NOT_IMPLEMENTED("Explicitly specified 'precision' is not yet supported.");
    }

    // Note: We also don't handle the "saturate" attribute yet since it only applies to float8 quantization.
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  const int64_t axis_;
  const int64_t block_size_;
};

}  // namespace onnxruntime::webgpu
