/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#if ENABLE_BF16
#include <cuda_bf16.h>
#endif // ENABLE_BF16
#include <cuda_fp16.h>

#include <memory>  // std::make_unique
#include <sstream> // std::stringstream
#include <string>
#include <unordered_set>
#include <vector>

namespace onnxruntime::llm::common
{
#if ENABLE_BF16
static inline std::basic_ostream<char>& operator<<(std::basic_ostream<char>& stream, __nv_bfloat16 const& val)
{
    stream << __bfloat162float(val);
    return stream;
}
#endif // ENABLE_BF16

static inline std::basic_ostream<char>& operator<<(std::basic_ostream<char>& stream, __half const& val)
{
    stream << __half2float(val);
    return stream;
}

// Add forward declaration before printElement functions
template <typename... Args>
std::ostream& operator<<(std::ostream& os, std::tuple<Args...> const& t);


template <typename... Args>
std::string to_string(std::tuple<Args...> const& t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}

inline std::string fmtstr(std::string const& s)
{
    return s;
}

inline std::string fmtstr(std::string&& s)
{
    return s;
}

#if defined(_MSC_VER)
std::string fmtstr(char const* format, ...);
#else
std::string fmtstr(char const* format, ...) __attribute__((format(printf, 1, 2)));
#endif

// __PRETTY_FUNCTION__ is used for neat debugging printing but is not supported on Windows
// The alternative is __FUNCSIG__, which is similar but not identical
#if defined(_WIN32)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif


} // namespace onnxruntime::llm::common
