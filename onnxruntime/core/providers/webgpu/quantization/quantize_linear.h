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
  QuantizeLinearProgram(util::QuantizationType quantization_type, bool has_zero_point)
      : Program<QuantizeLinearProgram>{"QuantizeLinear"},
        quantization_type_{quantization_type},
        has_zero_point_{has_zero_point} {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

 private:
  const util::QuantizationType quantization_type_;
  const bool has_zero_point_;
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