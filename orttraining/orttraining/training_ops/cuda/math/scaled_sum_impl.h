// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// #include <cuda_runtime.h>

namespace onnxruntime {
namespace cuda {

template <typename T>
void ScaledSumImpl(cudaStream_t stream,
                   int64_t input_element_count,
                   const std::vector<const T*>& inputs,
                   const std::vector<const T*>& scales,
                   T* output_data);

}  // namespace cuda
}  // namespace onnxruntime
