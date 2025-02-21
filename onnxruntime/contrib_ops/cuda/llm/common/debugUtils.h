/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "contrib_ops/cuda/llm/runtime/bufferManager.h"

namespace onnxruntime::llm::runtime::utils {

template <typename T>
bool tensorHasInvalid(ITensor const& tensor, BufferManager const& manager, std::string const& infoStr);

bool tensorHasInvalid(
    size_t M, size_t K, nvinfer1::DataType type, void const* data, cudaStream_t stream, std::string const& infoStr);

int stallStream(
    char const* name, std::optional<cudaStream_t> stream = std::nullopt, std::optional<int> delay = std::nullopt);

}  // namespace onnxruntime::llm::runtime::utils
