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

#include "contrib_ops/cuda/llm/runtime/memoryCounters.h"

#include "contrib_ops/cuda/llm/common/stringUtils.h"

#include <array>
#include <cmath>

namespace tc = onnxruntime::llm::common;

namespace
{

auto constexpr kByteUnits = std::array{"B", "KB", "MB", "GB", "TB", "PB", "EB"};

std::string doubleBytesToString(double bytes, int precision)
{
    std::uint32_t unitIdx{0};

    while (std::abs(bytes) >= 1024.0 && unitIdx < kByteUnits.size() - 1)
    {
        bytes /= 1024.0;
        ++unitIdx;
    }
    auto const format = "%." + std::to_string(precision) + "f %s";
    return tc::fmtstr(format.c_str(), bytes, kByteUnits[unitIdx]);
}

} // namespace

namespace onnxruntime::llm::runtime
{
std::string MemoryCounters::bytesToString(SizeType32 bytes, int precision)
{
    return doubleBytesToString(static_cast<double>(bytes), precision);
}

std::string MemoryCounters::bytesToString(DiffType bytes, int precision)
{
    return doubleBytesToString(static_cast<double>(bytes), precision);
}

std::string MemoryCounters::toString() const
{
    return onnxruntime::llm::common::fmtstr("[MemUsage] GPU %s, CPU %s, Pinned %s", bytesToString(this->getGpu()).c_str(),
        bytesToString(this->getCpu()).c_str(), bytesToString(this->getPinned()).c_str());
}

void MemoryCounters::allocate(MemoryType memoryType, MemoryCounters::SizeType32 size)
{
    switch (memoryType)
    {
    case MemoryType::kGPU: allocate<MemoryType::kGPU>(size); break;
    case MemoryType::kCPU: allocate<MemoryType::kCPU>(size); break;
    case MemoryType::kPINNED: allocate<MemoryType::kPINNED>(size); break;
    case MemoryType::kPINNEDPOOL: allocate<MemoryType::kPINNEDPOOL>(size); break;
    default: TLLM_THROW("Unknown memory type");
    }
}

void MemoryCounters::deallocate(MemoryType memoryType, MemoryCounters::SizeType32 size)
{
    switch (memoryType)
    {
    case MemoryType::kGPU: deallocate<MemoryType::kGPU>(size); break;
    case MemoryType::kCPU: deallocate<MemoryType::kCPU>(size); break;
    case MemoryType::kPINNED: deallocate<MemoryType::kPINNED>(size); break;
    case MemoryType::kPINNEDPOOL: deallocate<MemoryType::kPINNEDPOOL>(size); break;
    default: TLLM_THROW("Unknown memory type");
    }
}

MemoryCounters& MemoryCounters::getInstance()
{
    static MemoryCounters mInstance;
    return mInstance;
}
} // namespace onnxruntime::llm::runtime
