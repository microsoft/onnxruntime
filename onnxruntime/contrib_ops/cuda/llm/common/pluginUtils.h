/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "contrib_ops/cuda/llm/runtime/nv_infer_runtime.h"

#include "contrib_ops/cuda/llm/common/logger.h"

namespace onnxruntime::llm::plugins::utils
{
using DimType64 = int64_t;

inline DimType64 computeMDimension(bool transA, nvinfer1::Dims const& dims)
{
    DimType64 M{1};
    if (transA)
    {
        for (int i = dims.nbDims - 1; i > 0; --i)
        {
            M *= dims.d[i];
        }
    }
    else
    {
        for (int i = 0; i < dims.nbDims - 1; ++i)
        {
            M *= dims.d[i];
        }
    }
    return M;
}

inline DimType64 computeNDimension(bool transB, nvinfer1::Dims const& dims)
{
    DimType64 N{1};
    if (transB)
    {
        for (int32_t i = 0; i < dims.nbDims - 1; ++i)
        {
            N *= dims.d[i];
        }
    }
    else
    {
        for (int32_t i = dims.nbDims - 1; i > 0; --i)
        {
            N *= dims.d[i];
        }
    }
    return N;
}

// inline std::int32_t logErrorReturn0(char const* variable)
// {
//     TLLM_LOG_ERROR("Value of %s is out of range for int32_t", variable);
//     return 0;
// }

// #define TLLM_INT32_CAST(value)                                                                                         \
//     ((value > 0x7FFFFFFFLL || value < -0x80000000LL) ? onnxruntime::llm::plugins::utils::logErrorReturn0(#value)           \
//                                                      : static_cast<int32_t>(value))

} // namespace onnxruntime::llm::plugins::utils
