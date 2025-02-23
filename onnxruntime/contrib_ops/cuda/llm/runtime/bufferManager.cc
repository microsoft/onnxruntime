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
#include "bufferManager.h"
#include "cudaMemPool.h"
#include "contrib_ops/cuda/llm/common/assert.h"
#include "contrib_ops/cuda/llm/common/cudaUtils.h"
#include "tllmBuffers.h"

#include <cstring>
#include <cuda_runtime_api.h>
#include <memory>

namespace tc = onnxruntime::llm::common;

namespace onnxruntime::llm::runtime
{

BufferManager::BufferManager(CudaStreamPtr stream, bool trimPool)
    : mStream{std::move(stream)}
    , mTrimPool{trimPool}
{
    TLLM_CHECK_WITH_INFO(static_cast<bool>(mStream), "Undefined CUDA stream");
    mPool = CudaMemPool::getPrimaryPoolForDevice(mStream->getDevice());
}

BufferManager::IBufferPtr BufferManager::gpu(std::size_t size, nvinfer1::DataType type) const
{
    if (static_cast<bool>(mPool))
    {
        return std::make_unique<DeviceBuffer>(size, type, CudaAllocatorAsync{mStream, mPool});
    }
    // When memory pools are not supported, fallback to synchronous memory allocations.
    return gpuSync(size, type);
}

BufferManager::ITensorPtr BufferManager::gpu(nvinfer1::Dims dims, nvinfer1::DataType type) const
{
    if (static_cast<bool>(mPool))
    {
        return std::make_unique<DeviceTensor>(dims, type, CudaAllocatorAsync{mStream, mPool});
    }
    // When memory pools are not supported, fallback to synchronous memory allocations.
    return gpuSync(dims, type);
}

BufferManager::IBufferPtr BufferManager::gpuSync(std::size_t size, nvinfer1::DataType type)
{
    return std::make_unique<StaticDeviceBuffer>(size, type, CudaAllocator{});
}

BufferManager::ITensorPtr BufferManager::gpuSync(nvinfer1::Dims dims, nvinfer1::DataType type)
{
    return std::make_unique<StaticDeviceTensor>(dims, type, CudaAllocator{});
}

BufferManager::IBufferPtr BufferManager::cpu(std::size_t size, nvinfer1::DataType type)
{
    return std::make_unique<HostBuffer>(size, type);
}

BufferManager::ITensorPtr BufferManager::cpu(nvinfer1::Dims dims, nvinfer1::DataType type)
{
    return std::make_unique<HostTensor>(dims, type);
}

BufferManager::IBufferPtr BufferManager::pinned(std::size_t size, nvinfer1::DataType type)
{
    return std::make_unique<PinnedBuffer>(size, type);
}

BufferManager::ITensorPtr BufferManager::pinned(nvinfer1::Dims dims, nvinfer1::DataType type)
{
    return std::make_unique<PinnedTensor>(dims, type);
}

BufferManager::IBufferPtr BufferManager::pinnedPool(std::size_t size, nvinfer1::DataType type)
{
    return std::make_unique<PinnedPoolBuffer>(size, type);
}

BufferManager::ITensorPtr BufferManager::pinnedPool(nvinfer1::Dims dims, nvinfer1::DataType type)
{
    return std::make_unique<PinnedPoolTensor>(dims, type);
}

BufferManager::IBufferPtr BufferManager::managed(std::size_t size, nvinfer1::DataType type)
{
    return std::make_unique<UVMBuffer>(size, type);
}

BufferManager::ITensorPtr BufferManager::managed(nvinfer1::Dims dims, nvinfer1::DataType type)
{
    return std::make_unique<UVMTensor>(dims, type);
}

// BufferManager::ITensorPtr BufferManager::ipcNvls(std::set<int> ranks, nvinfer1::Dims dims, nvinfer1::DataType type)
// {
//     return std::make_unique<MulticastTensor>(dims, type, ranks);
// }

void BufferManager::setZero(IBuffer& buffer) const
{
    setMem(buffer, 0);
}

void BufferManager::setMem(IBuffer& buffer, int32_t value) const
{
    if (buffer.getMemoryType() == MemoryType::kGPU)
    {
        TLLM_CUDA_CHECK(cudaMemsetAsync(buffer.data(), value, buffer.getSizeInBytes(), mStream->get()));
    }
    else
    {
        std::memset(buffer.data(), value, buffer.getSizeInBytes());
    }
}

void BufferManager::copy(void const* src, IBuffer& dst, MemoryType srcType) const
{
    if (dst.getSizeInBytes() > 0)
    {
        if (srcType != MemoryType::kGPU && dst.getMemoryType() != MemoryType::kGPU)
        {
            std::memcpy(dst.data(), src, dst.getSizeInBytes());
        }
        else
        {
            TLLM_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src, dst.getSizeInBytes(), cudaMemcpyDefault, mStream->get()));
        }
    }
}

void BufferManager::copy(IBuffer const& src, void* dst, MemoryType dstType) const
{
    if (src.getSizeInBytes() > 0)
    {
        if (src.getMemoryType() != MemoryType::kGPU && dstType != MemoryType::kGPU)
        {
            std::memcpy(dst, src.data(), src.getSizeInBytes());
        }
        else
        {
            TLLM_CUDA_CHECK(cudaMemcpyAsync(dst, src.data(), src.getSizeInBytes(), cudaMemcpyDefault, mStream->get()));
        }
    }
}

void BufferManager::copy(IBuffer const& src, IBuffer& dst) const
{
    TLLM_CHECK_WITH_INFO(src.getDataType() == dst.getDataType(),
        tc::fmtstr("Incompatible data types: %s != %s", src.getDataTypeName(), dst.getDataTypeName()));
    TLLM_CHECK_WITH_INFO(src.getSizeInBytes() == dst.getSizeInBytes(),
        tc::fmtstr("Incompatible buffer sizes: %lu != %lu", src.getSizeInBytes(), dst.getSizeInBytes()));
    copy(src, dst.data(), dst.getMemoryType());
}

BufferManager::IBufferPtr BufferManager::allocate(
    MemoryType memoryType, std::size_t size, nvinfer1::DataType type) const
{
    switch (memoryType)
    {
    case MemoryType::kCPU: return cpu(size, type);
    case MemoryType::kGPU: return gpu(size, type);
    case MemoryType::kPINNED: return pinned(size, type);
    case MemoryType::kUVM: return managed(size, type);
    case MemoryType::kPINNEDPOOL: return pinnedPool(size, type);
    }

    TLLM_THROW("Unknown memory type");
}

BufferManager::ITensorPtr BufferManager::allocate(
    MemoryType memoryType, nvinfer1::Dims dims, nvinfer1::DataType type) const
{
    switch (memoryType)
    {
    case MemoryType::kCPU: return cpu(dims, type);
    case MemoryType::kGPU: return gpu(dims, type);
    case MemoryType::kPINNED: return pinned(dims, type);
    case MemoryType::kUVM: return managed(dims, type);
    case MemoryType::kPINNEDPOOL: return pinnedPool(dims, type);
    }

    TLLM_THROW("Unknown memory type");
}

BufferManager::IBufferPtr BufferManager::copyFrom(IBuffer const& src, MemoryType memoryType) const
{
    auto dst = allocate(memoryType, src.getSize(), src.getDataType());
    copy(src, *dst);
    return dst;
}

BufferManager::ITensorPtr BufferManager::copyFrom(ITensor const& src, MemoryType memoryType) const
{
    auto dst = allocate(memoryType, src.getShape(), src.getDataType());
    copy(src, *dst);
    return dst;
}

CudaStream const& BufferManager::getStream() const
{
    return *mStream;
}

std::size_t BufferManager::memoryPoolReserved() const
{
    if (!static_cast<bool>(mPool))
    {
        TLLM_LOG_TRACE(
            "Operation '%s' trivially returns zero on systems without memory pool support.", __PRETTY_FUNCTION__);
        return 0;
    }
    mStream->synchronize();
    return mPool->memoryPoolReserved();
}

std::size_t BufferManager::memoryPoolUsed() const
{
    if (!static_cast<bool>(mPool))
    {
        TLLM_LOG_TRACE(
            "Operation '%s' trivially returns zero on systems without memory pool support.", __PRETTY_FUNCTION__);
        return 0;
    }
    mStream->synchronize();
    return mPool->memoryPoolUsed();
}

std::size_t BufferManager::memoryPoolFree() const
{
    if (!static_cast<bool>(mPool))
    {
        TLLM_LOG_TRACE(
            "Operation '%s' trivially returns zero on systems without memory pool support.", __PRETTY_FUNCTION__);
        return 0;
    }
    mStream->synchronize();
    return mPool->memoryPoolReserved() - mPool->memoryPoolUsed();
}

void BufferManager::memoryPoolTrimTo(std::size_t size)
{
    if (!static_cast<bool>(mPool))
    {
        TLLM_LOG_TRACE("Operation '%s' does not do anything on this system as it does not support memory pools.",
            __PRETTY_FUNCTION__);
        return;
    }
    mPool->memoryPoolTrimTo(size);
}
} // namespace onnxruntime::llm::runtime
