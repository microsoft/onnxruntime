/*
 The implementation of this file is based on qkvToContext plugin in TensorRT demo:
 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT/

Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include <type_traits>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/math/softmax.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status ComputeSoftmax(
    cudaStream_t stream,
    const int all_sequence_length,
    const int sequence_length,
    const int batch_size,
    const int num_heads,
    const T* add_before_softmax,
    const T* input,
    T* output,
    bool is_unidirectional);

template <typename T>
Status ComputeSoftmaxWithMask1D(
    cudaStream_t stream,
    const int all_sequence_length,
    const int sequence_length,
    const int batch_size,
    const int num_heads,
    const int* mask_index,
    const int* mask_start,
    const T* add_before_softmax,
    const T* input,
    T* output,
    const bool is_unidirectional);

template <typename T>
Status ComputeSoftmaxWithRawMask(
    cudaStream_t stream,
    const int all_sequence_length,
    const int sequence_length,
    const int batch_size,
    const int num_heads,
    const int* attention_mask,
    const bool* key_padding_mask,
    const T* add_before_softmax,
    const T* input,
    T* output,
    const bool is_unidirectional,
    const float rsqrt_head_size,
    const int mask_dimension,
    const int max_sequence_length,
    const bool use_persistent_softmax,
    T* persistent_softmax_workspace,
    const float mask_filter_value);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
