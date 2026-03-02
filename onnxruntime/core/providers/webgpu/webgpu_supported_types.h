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

inline const std::vector<MLDataType>& GetOpTypeConstraints(bool enable_int64 = false, bool enable_bool = false) {
  static std::vector<MLDataType> base_types{
      DataTypeImpl::GetTensorType<MLFloat16>(),
      DataTypeImpl::GetTensorType<float>(),
      DataTypeImpl::GetTensorType<int32_t>(),
      DataTypeImpl::GetTensorType<uint32_t>()};

  if (enable_int64 && enable_bool) {
    static std::vector<MLDataType> types_with_int64_bool = []() {
      auto types = base_types;
      types.push_back(DataTypeImpl::GetTensorType<int64_t>());
      types.push_back(DataTypeImpl::GetTensorType<bool>());
      return types;
    }();
    return types_with_int64_bool;
  } else if (enable_int64) {
    static std::vector<MLDataType> types_with_int64 = []() {
      auto types = base_types;
      types.push_back(DataTypeImpl::GetTensorType<int64_t>());
      return types;
    }();
    return types_with_int64;
  } else if (enable_bool) {
    static std::vector<MLDataType> types_with_bool = []() {
      auto types = base_types;
      types.push_back(DataTypeImpl::GetTensorType<bool>());
      return types;
    }();
    return types_with_bool;
  } else {
    return base_types;
  }
}

}  // namespace webgpu
}  // namespace onnxruntime
