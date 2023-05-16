// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_fp16.h>

#include "core/common/common.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template<typename T>
struct GemmSoftmaxGemmPermuteParams;

namespace internal {

// Status LaunchConvertMask(const GemmSoftmaxGemmPermuteParams<half>* params);
Status LaunchAddBiasAndConvertMask(const GemmSoftmaxGemmPermuteParams<half>* params);

}
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
