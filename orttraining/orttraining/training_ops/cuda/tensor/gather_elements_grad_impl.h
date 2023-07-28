// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

struct GatherScatterElementsArgs;

template <typename T, typename TIndex>
Status GatherElementsGradNonDeterministicImpl(cudaStream_t stream, const TIndex* indices_data,
                                              const T* updates_data, T* output_data,
                                              const GatherScatterElementsArgs& args);

}  // namespace cuda
}  // namespace onnxruntime
