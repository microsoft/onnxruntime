/*
 * Copyright (c) 1993-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/runtime/nv_infer_runtime.h"
#include <map>

namespace onnxruntime::llm::common
{

constexpr static size_t getDTypeSize(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT32: [[fallthrough]];
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kBF16: [[fallthrough]];
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL: [[fallthrough]];
    case nvinfer1::DataType::kUINT8: [[fallthrough]];
    case nvinfer1::DataType::kINT8: [[fallthrough]];
    case nvinfer1::DataType::kFP8: return 1;
    case nvinfer1::DataType::kINT4: TLLM_THROW("Cannot determine size of INT4 data type");
    case nvinfer1::DataType::kFP4: TLLM_THROW("Cannot determine size of FP4 data type");
    default: TLLM_THROW("Unknown dtype %d", static_cast<int>(type));
    }
    return 0;
}

constexpr static size_t getDTypeSizeInBits(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kINT64: return 64;
    case nvinfer1::DataType::kINT32: [[fallthrough]];
    case nvinfer1::DataType::kFLOAT: return 32;
    case nvinfer1::DataType::kBF16: [[fallthrough]];
    case nvinfer1::DataType::kHALF: return 16;
    case nvinfer1::DataType::kBOOL: [[fallthrough]];
    case nvinfer1::DataType::kUINT8: [[fallthrough]];
    case nvinfer1::DataType::kINT8: [[fallthrough]];
    case nvinfer1::DataType::kFP8: return 8;
    case nvinfer1::DataType::kINT4: [[fallthrough]];
    case nvinfer1::DataType::kFP4: return 4;
    default: TLLM_THROW("Unknown dtype %d", static_cast<int>(type));
    }
    return 0;
}

[[maybe_unused]] static std::string getDtypeString(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return "fp32"; break;
    case nvinfer1::DataType::kHALF: return "fp16"; break;
    case nvinfer1::DataType::kINT8: return "int8"; break;
    case nvinfer1::DataType::kINT32: return "int32"; break;
    case nvinfer1::DataType::kBOOL: return "bool"; break;
    case nvinfer1::DataType::kUINT8: return "uint8"; break;
    case nvinfer1::DataType::kFP8: return "fp8"; break;
    case nvinfer1::DataType::kBF16: return "bf16"; break;
    case nvinfer1::DataType::kINT64: return "int64"; break;
    case nvinfer1::DataType::kINT4: return "int4"; break;
    case nvinfer1::DataType::kFP4: return "fp4"; break;
    default: throw std::runtime_error("Unsupported data type"); break;
    }

    return "";
}

} // namespace onnxruntime::llm::common
