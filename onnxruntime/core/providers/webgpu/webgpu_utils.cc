// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {

int64_t GetMaxComponents(int64_t size) {
    if (size % 4 == 0) {
        return 4;
    } else if (size % 2 == 0) {
        return 2;
    }
    return 1;
}

}  // namespace webgpu
}  // namespace onnxruntime
