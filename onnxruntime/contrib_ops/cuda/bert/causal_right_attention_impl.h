// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include "core/common/status.h"

namespace onnxruntime::contrib::cuda {

size_t CausalRightAttentionWorkspaceSize(int batch, int heads, int queries, int keys);

template <typename T>
void ConfigureCausalRightAttention();

template <typename T>
Status LaunchCausalRightAttention(cudaStream_t stream, const T* query, const T* key, const T* value,
                                  T* output, void* workspace, int batch, int heads, int queries, int keys,
                                  float scale);

}  // namespace onnxruntime::contrib::cuda
