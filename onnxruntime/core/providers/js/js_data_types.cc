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

const std::vector<MLDataType> JsepSupportedDataTypes = BuildKernelDefConstraintsFromTypeList<SupportedTypes>();

}  // namespace js
}  // namespace onnxruntime