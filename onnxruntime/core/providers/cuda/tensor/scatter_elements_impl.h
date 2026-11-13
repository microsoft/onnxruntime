// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

struct GatherScatterElementsArgs;

template <typename T, typename TIndex>
Status ScatterElementsImpl(cudaStream_t stream, const T* input_data, const TIndex* indices_data, const T* updates_data,
                           T* output_data, const GatherScatterElementsArgs& args);

Status ScatterElementsMeanImpl(cudaStream_t stream, const void* indices, size_t index_element_size,
                               const GatherScatterElementsArgs& args, TArray<int64_t> indices_strides,
                               int64_t* heads, int64_t* next,
                               bool deterministic);

}  // namespace cuda
}  // namespace onnxruntime
