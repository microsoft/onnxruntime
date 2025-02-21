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

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace onnxruntime::llm::runtime {

#define FMT_DIM "%ld"

// typedefs
// Note that we use signed size types as recommended by TensorRT:
// https://github.com/NVIDIA/TensorRT/blob/main/CODING-GUIDELINES.md#signed-vs-unsigned-integers
using SizeType32 = std::int32_t;
using SizeType64 = std::int64_t;

enum class RequestType : std::int32_t {
  kCONTEXT = 0,
  kGENERATION = 1
};

// Token ID type
using TokenIdType = std::int32_t;

using LoraTaskIdType = std::uint64_t;
using TokenExtraIdType = std::uint64_t;
using VecTokenExtraIds = std::vector<TokenExtraIdType>;

struct UniqueToken {
  TokenIdType tokenId;
  TokenExtraIdType tokenExtraId;

  bool operator==(UniqueToken const& other) const noexcept {
    return (tokenId == other.tokenId && tokenExtraId == other.tokenExtraId);
  }
};

using VecUniqueTokens = std::vector<UniqueToken>;

template <typename T>
using StringPtrMap = std::unordered_map<std::string, std::shared_ptr<T>>;

}  // namespace onnxruntime::llm::runtime
