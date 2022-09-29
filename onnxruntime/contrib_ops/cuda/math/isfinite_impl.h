// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include "core/providers/cuda/multi_tensor/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T>
struct IsAllFiniteFunctor {
  void operator()(cudaStream_t stream, ChunkGroup<1> chunks, bool* output, const bool isinf_only, const bool isnan_only);
};

}
}  // namespace onnxruntime