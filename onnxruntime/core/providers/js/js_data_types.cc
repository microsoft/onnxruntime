// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/shape_op.h"

namespace onnxruntime {
namespace js {

using SupportedTypes =
    TypeList<
        float,
#ifdef ENABLE_WEBASSEMBLY_FLOAT16
        MLFloat16,
#endif
        int32_t,
        uint32_t>;

using SupportedFloats =
#ifdef ENABLE_WEBASSEMBLY_FLOAT16
    TypeList<
        float,
        MLFloat16>;
#else
    TypeList<float>;
#endif

const std::vector<MLDataType>& JsepSupportedDataTypes() {
  static const std::vector<MLDataType> supportedDataTypes = BuildKernelDefConstraintsFromTypeList<SupportedTypes>();
  return supportedDataTypes;
}

const std::vector<MLDataType>& JsepSupportedFloatTypes() {
  static const std::vector<MLDataType> supportedDataTypes = BuildKernelDefConstraintsFromTypeList<SupportedFloats>();
  return supportedDataTypes;
}

}  // namespace js
}  // namespace onnxruntime