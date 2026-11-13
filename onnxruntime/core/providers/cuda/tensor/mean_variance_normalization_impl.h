// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace onnxruntime::cuda {

inline int64_t MeanVarianceNormalizationChunks(int64_t rows, int64_t columns) {
  return rows < 256 && columns > 4096 ? std::min<int64_t>((columns - 1) / 4096 + 1, 256) : 1;
}

template <typename T>
void MeanVarianceNormalizationImpl(cudaStream_t stream, const T* input, T* output,
                                   int64_t rows, int64_t columns, int64_t inner, float epsilon, void* workspace);

}  // namespace onnxruntime::cuda
