/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "contrib_ops/cuda/llm/common/cuda_fp8_utils.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"

#include <cassert>

namespace onnxruntime::llm {
namespace common {

// cudaMemcpyAsync with extra check via ASan for D2H copy
cudaError_t cudaMemcpyAsyncSanitized(
    void* dst, void const* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream = nullptr);

}  // namespace common
}  // namespace onnxruntime::llm
