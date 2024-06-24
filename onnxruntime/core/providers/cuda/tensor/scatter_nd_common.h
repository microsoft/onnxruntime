// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {

struct ElementCountsAndInputDimsSpanOrGpu {
  int64_t stack_ptr[12];
  int64_t* gpu_ptr;
};

}  // namespace cuda
}  // namespace onnxruntime
