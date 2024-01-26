/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "contrib_ops/cuda/bert/tensorrt_llm_fmha/common/stringUtils.h"
//#include "contrib_ops/cuda/bert/tensorrt_llm_fmha/common/tllmException.h"
#include "core/common/common.h"

#include <string>

namespace tensorrt_llm::common
{
[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "")
{
    fprintf(stderr, "CUDA Error: %s %s %d\n", info.c_str(), file, line);
    ORT_THROW(info);
}

} // namespace tensorrt_llm::common

extern bool CHECK_DEBUG_ENABLED;

#if defined(_WIN32)
#define TLLM_LIKELY(x) (__assume((x) == 1), (x))
#else
#define TLLM_LIKELY(x) __builtin_expect((x), 1)
#endif

#define TLLM_CHECK(val)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        TLLM_LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                               \
                                            : tensorrt_llm::common::throwRuntimeError(__FILE__, __LINE__, #val);       \
    } while (0)

#define TLLM_CHECK_WITH_INFO(val, info, ...)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        TLLM_LIKELY(static_cast<bool>(val))                                                                            \
        ? ((void) 0)                                                                                                   \
        : tensorrt_llm::common::throwRuntimeError(                                                                     \
            __FILE__, __LINE__, tensorrt_llm::common::fmtstr(info, ##__VA_ARGS__));                                    \
    } while (0)

#define TLLM_CHECK_DEBUG(val)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (CHECK_DEBUG_ENABLED)                                                                                       \
        {                                                                                                              \
            TLLM_LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                           \
                                                : tensorrt_llm::common::throwRuntimeError(__FILE__, __LINE__, #val);   \
        }                                                                                                              \
    } while (0)

#define TLLM_CHECK_DEBUG_WITH_INFO(val, info)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (CHECK_DEBUG_ENABLED)                                                                                       \
        {                                                                                                              \
            TLLM_LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                           \
                                                : tensorrt_llm::common::throwRuntimeError(__FILE__, __LINE__, info);   \
        }                                                                                                              \
    } while (0)

#define TLLM_THROW(...)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        throw NEW_TLLM_EXCEPTION(__VA_ARGS__);                                                                         \
    } while (0)

// #define TLLM_WRAP(ex)                                                                                                  \
//     NEW_TLLM_EXCEPTION("%s: %s", tensorrt_llm::common::TllmException::demangle(typeid(ex).name()).c_str(), ex.what())
