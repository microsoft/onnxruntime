// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

namespace onnxruntime {
namespace cuda {

// Launches a CUDA kernel that performs a label encoding lookup using binary search
// on sorted key arrays. For each input element, the kernel searches for the element
// in the sorted keys array and writes the corresponding value, or the default value
// if not found.
//
// For floating-point key types, NaN values receive special handling: all NaN inputs
// map to the same value (the value associated with the NaN key, if present).
template <typename TKey, typename TValue>
void LabelEncoderImpl(
    cudaStream_t stream,
    const TKey* input,
    TValue* output,
    int64_t num_elements,
    const TKey* keys,
    const TValue* values,
    int64_t num_keys,
    TValue default_value,
    int64_t nan_key_index);  // -1 if no NaN key exists

}  // namespace cuda
}  // namespace onnxruntime
