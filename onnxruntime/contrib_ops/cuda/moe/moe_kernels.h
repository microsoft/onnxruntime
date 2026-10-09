// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace onnxruntime::contrib::cuda {

void LaunchRemapMoeExpertIndices(const int* expert_indices, int* remapped_expert_indices,
                                 const int* expert_map, size_t count, cudaStream_t stream);

void LaunchAddMoeFp16Output(half* output, const half* contribution,
                            size_t count, cudaStream_t stream);
void LaunchAddMoeBf16Output(__nv_bfloat16* output, const __nv_bfloat16* contribution,
                            size_t count, cudaStream_t stream);

}  // namespace onnxruntime::contrib::cuda
