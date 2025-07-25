// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cuda_fp16.h>
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, typename QK>
Status CopyQK(cudaStream_t stream,
              const int qk_size,
              const T* input,
              QK* output);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
