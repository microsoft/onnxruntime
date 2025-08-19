// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

namespace onnxruntime::contrib::cuda {

template <typename T, bool IsInterLeaved, bool HasLimit>
void invokeSwiGLU(T* output, T const* input, int intermediate_size, int num_rows, float alpha, float limit, cudaStream_t stream);

}  // namespace onnxruntime::contrib::cuda