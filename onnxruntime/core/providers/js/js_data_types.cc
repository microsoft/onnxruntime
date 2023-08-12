// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/cpu/tensor/shape_op.h"

namespace onnxruntime {
namespace js {

using SupportedTypes =
    TypeList<
        float,
        int32_t,
        uint32_t>;

std::vector<MLDataType> JsepSupportedDataTypes() {
  return BuildKernelDefConstraintsFromTypeList<SupportedTypes>();
}
}
}