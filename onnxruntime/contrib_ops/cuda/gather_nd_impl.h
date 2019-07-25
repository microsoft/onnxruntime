// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {
template <typename Tind>
void GatherNDImpl(
    const int64_t* element_index_counts,
    const Tind* indice,  //The address of the indices
    const int64_t last_indice_dimension,
    const size_t N,  //The number of copie
    void* input_data,
    void* output_data,
    const size_t nums_of_elements,
    const size_t element_bytes,
    int64_t axis_);

template <typename T, typename Tind>
void GatherNDGradImpl(
    const int64_t* element_index_counts,
    const Tind* indice,  //The address of the indices
    const int64_t last_indice_dimension,
    const size_t N,  //The number of copies
    void* input_data,
    void* output_data,
    const size_t nums_of_elements,
    const int64_t axis_);

}  // namespace cuda
}  // namespace onnxruntime