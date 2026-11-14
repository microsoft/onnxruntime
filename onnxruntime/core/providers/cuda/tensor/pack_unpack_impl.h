// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/pack_unpack_impl.h"

namespace onnxruntime {
namespace cuda {

Status LaunchPacking(cudaStream_t stream, bool unpack, bool word32, const void* input, void* output,
                     uint64_t work, packing::Parameters parameters, packing::Layout layout);

}  // namespace cuda
}  // namespace onnxruntime
