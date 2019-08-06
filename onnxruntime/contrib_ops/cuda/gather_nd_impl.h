// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {
template <typename T>
void GatherNDImpl(
    const size_t N,  //The number of copie
    const void* input_data,
    void* output_data,
    const size_t nums_of_elements,
    const int64_t* element_offset);

template <typename T>
void GatherNDGradImpl(
    const size_t N,  //The number of copies
    const void* input_data,
    void* output_data,
    const size_t nums_of_elements,
    const int64_t* element_offset);

}  // namespace cuda
}  // namespace onnxruntime
