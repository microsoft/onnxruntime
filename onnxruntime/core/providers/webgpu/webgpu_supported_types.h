// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/tensor/shape_op.h"

namespace onnxruntime {
namespace webgpu {

using SupportedNumberTypes =
    TypeList<
        float,
        MLFloat16,
        int32_t,
        uint32_t>;

using SupportedFloats =
    TypeList<
        float,
        MLFloat16>;

inline const std::vector<MLDataType>& WebGpuSupportedNumberTypes() {
  static const std::vector<MLDataType> supportedDataTypes = BuildKernelDefConstraintsFromTypeList<SupportedNumberTypes>();
  return supportedDataTypes;
}

inline const std::vector<MLDataType>& WebGpuSupportedFloatTypes() {
  static const std::vector<MLDataType> supportedDataTypes = BuildKernelDefConstraintsFromTypeList<SupportedFloats>();
  return supportedDataTypes;
}

}  // namespace webgpu
}  // namespace onnxruntime
