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

#if !defined(CUDART_VERSION) || (CUDART_VERSION < 11050)
#error CUDA >= 11.5 is required
#else
#include <cub/cub.cuh>
#endif

#include "contrib_ops/cuda/llm/common/debugUtils.h"

#include "contrib_ops/cuda/llm/common/cudaUtils.h"
#include "contrib_ops/cuda/llm/common/memoryUtils.h"
#include <cfloat>
#include <string>

namespace
{
template <typename T, int blockSize>
__global__ void checkTensorInvalidKernel(T const* data, std::size_t size, int* foundInvalid)
{
    auto tidx = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t found = 0;

    for (auto idx = tidx; idx < size; idx += blockDim.x * gridDim.x)
    {
        auto value = static_cast<float>(data[idx]);
        if (isnan(value) || isinf(value))
        {
            found = 1;
            break;
        }
    }

    typedef cub::BlockReduce<int32_t, blockSize> BlockReduceT;

    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduceT::TempStorage tempStorage;

    // Compute block-wide maximum
    int blockFound = BlockReduceT(tempStorage).Reduce(found, cub::Max());

    // Have thread 0 write out block's result
    if (threadIdx.x == 0)
    {
        atomicCAS(foundInvalid, 0, blockFound);
    }
}

__global__ void stallStreamKernel(int const microSeconds)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    for (int i = 0; i < microSeconds; ++i)
    {
        __nanosleep(1000);
    }
#endif
}
} // namespace

using namespace onnxruntime::llm::runtime;
namespace tc = onnxruntime::llm::common;

namespace onnxruntime::llm::runtime::utils
{

template <typename T>
void invokeCheckTensorInvalidKernel(T const* data, std::size_t size, int* foundInvalid, cudaStream_t stream)
{
    constexpr uint32_t kThreadsPerCta = 256;
    checkTensorInvalidKernel<T, kThreadsPerCta>
        <<<tc::ceilDiv(size, kThreadsPerCta), kThreadsPerCta, 0, stream>>>(data, size, foundInvalid);
}

template void invokeCheckTensorInvalidKernel(
    float const* data, std::size_t size, int* foundInvalid, cudaStream_t stream);
template void invokeCheckTensorInvalidKernel(
    half const* data, std::size_t size, int* foundInvalid, cudaStream_t stream);
template void invokeCheckTensorInvalidKernel(
    __nv_bfloat16 const* data, std::size_t size, int* foundInvalid, cudaStream_t stream);
template void invokeCheckTensorInvalidKernel(
    __nv_fp8_e4m3 const* data, std::size_t size, int* foundInvalid, cudaStream_t stream);

template <typename T>
void printLogitsKeyInfo(ITensor const& tensor, std::string const& infoStr)
{
    auto const& shape = tensor.getShape();
    auto const volume = ITensor::volume(shape);

    BufferManager::ITensorPtr host{};
    T const* hostData;
    if (tensor.getMemoryType() == MemoryType::kGPU)
    {
        auto streamPtr = std::make_shared<CudaStream>();
        BufferManager manager{streamPtr};
        host = manager.copyFrom(tensor, MemoryType::kCPU);
        streamPtr->synchronize();
        hostData = bufferCast<T>(*host);
    }
    else
    {
        hostData = bufferCast<T>(tensor);
    }

    std::stringstream ss;
    ss << infoStr;
    ss << " Shape: " << shape;
    ss << "; Top 5: ";
    for (int64_t ki = 0; ki < 5; ++ki)
    {
        ss << static_cast<float>(hostData[ki]) << ", ";
    }

    ss << " Last 5: ";
    for (int64_t ki = volume - 6; ki < volume; ++ki)
    {
        ss << static_cast<float>(hostData[ki]) << ", ";
    }

    // find max, min, avg
    double mSum = 0.f;
    float mMax = -FLT_MAX;
    float mMin = FLT_MAX;

    for (int64_t ki = 0; ki < volume; ++ki)
    {
        float value = static_cast<float>(hostData[ki]);
        mSum += value;
        if (value > mMax)
        {
            mMax = value;
        }
        if (value < mMin)
        {
            mMin = value;
        }
    }
    float mAvg = mSum / volume;

    ss << " avg: " << mAvg << ", min: " << mMin << ", max: " << mMax << std::endl;

    TLLM_LOG_TRACE(ss.str());
}

template void printLogitsKeyInfo<float>(ITensor const& tensor, std::string const& infoStr);
template void printLogitsKeyInfo<half>(ITensor const& tensor, std::string const& infoStr);
#ifdef ENABLE_BF16
template void printLogitsKeyInfo<__nv_bfloat16>(ITensor const& tensor, std::string const& infoStr);
#endif

#ifdef ENABLE_FP8
template void printLogitsKeyInfo<__nv_fp8_e4m3>(ITensor const& tensor, std::string const& infoStr);
#endif

template <typename T>
bool tensorHasInvalid(ITensor const& tensor, BufferManager const& manager, std::string const& infoStr)
{
    printLogitsKeyInfo<T>(tensor, infoStr);
    auto foundInvalid = BufferManager::pinnedPool(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    auto foundInvalidPtr = bufferCast<int32_t>(*foundInvalid);
    foundInvalidPtr[0] = 0;
    auto const size = tensor.getSize();
    invokeCheckTensorInvalidKernel(bufferCast<T>(tensor), size, foundInvalidPtr, manager.getStream().get());
    manager.getStream().synchronize();
    return static_cast<bool>(foundInvalidPtr[0]);
}

template bool tensorHasInvalid<float>(ITensor const& tensor, BufferManager const& manager, std::string const& infoStr);
template bool tensorHasInvalid<half>(ITensor const& tensor, BufferManager const& manager, std::string const& infoStr);

#ifdef ENABLE_BF16
template bool tensorHasInvalid<__nv_bfloat16>(
    ITensor const& tensor, BufferManager const& manager, std::string const& infoStr);
#endif

#ifdef ENABLE_FP8    
template bool tensorHasInvalid<__nv_fp8_e4m3>(
    ITensor const& tensor, BufferManager const& manager, std::string const& infoStr);
#endif

bool tensorHasInvalid(
    size_t M, size_t K, nvinfer1::DataType type, void const* data, cudaStream_t stream, std::string const& infoStr)
{
    auto tensorView = ITensor::wrap(
        const_cast<void*>(data), type, ITensor::makeShape({static_cast<int32_t>(M), static_cast<int32_t>(K)}));
    auto manager = BufferManager(std::make_shared<CudaStream>(stream));
    if (type == nvinfer1::DataType::kFLOAT)
    {
        return tensorHasInvalid<float>(*tensorView, manager, infoStr);
    }
    else if (type == nvinfer1::DataType::kHALF)
    {
        return tensorHasInvalid<half>(*tensorView, manager, infoStr);
    }
#ifdef ENABLE_BF16    
    else if (type == nvinfer1::DataType::kBF16)
    {
        return tensorHasInvalid<__nv_bfloat16>(*tensorView, manager, infoStr);
    }
#endif
#ifdef ENABLE_FP8    
    else if (type == nvinfer1::DataType::kFP8)
    {
        return tensorHasInvalid<__nv_fp8_e4m3>(*tensorView, manager, infoStr);
    }
#endif    
    else
    {
        TLLM_THROW("Not supported type for Nan check");
    }
}

int stallStream(char const* name, std::optional<cudaStream_t> stream, std::optional<int> delay)
{
    int delay_val = 0;
    if (delay)
    {
        delay_val = delay.value();
    }
    else
    {
        char const* const env = std::getenv(name);
        if (env != nullptr)
        {
            delay_val = std::stoi(env);
        }
    }
    if (stream && delay_val > 0)
    {
        stallStreamKernel<<<1, 32, 0, stream.value()>>>(delay_val);
    }
    return delay_val;
}

} // namespace onnxruntime::llm::runtime::utils
