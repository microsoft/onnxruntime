// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace cuda {

template <class TOUT, class TIN>
Status CudaCast(cudaStream_t stream, const TIN* input, TOUT* output, size_t num_of_element);

}  // namespace cuda
}  // namespace onnxruntime
