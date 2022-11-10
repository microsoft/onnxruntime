// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void LaunchTopK(
    const T* input,
    int batch_size,
    int num_beams,
    int vocab_size,
    int K,
    T* output_values,
    int32_t* output_indices,
    T* output_values_tmp,
    int32_t* output_indices_tmp,
    cudaStream_t stream);

template <typename T>
void BeamSearchTopK(
    const T* input,
    int32_t batch_size,
    int32_t num_beams,
    int32_t vocab_size,
    int32_t k,
    T* tmp_values_1st_stage,
    int32_t* tmp_indices_1st_stage,
    T* tmp_values_2st_stage,
    int32_t* tmp_indices_2st_stage,
    T* output_values,
    int32_t* output_tokens,
    int32_t* output_indices,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime