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

#include "contrib_ops/cuda/llm/runtime/cudaMemPool.h"
#include "contrib_ops/cuda/llm/common/assert.h"
#include "contrib_ops/cuda/llm/common/cudaUtils.h"
#include "contrib_ops/cuda/llm/common/logger.h"
#include <array>
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>

namespace onnxruntime::llm::runtime
{

CudaMemPool::CudaMemPool(cudaMemPool_t pool)
{
    TLLM_CHECK_WITH_INFO(pool != nullptr, "Pointer to cudaMemPool cannot be nullptr.");
    mPool = PoolPtr{pool, Deleter{}};
}

std::size_t CudaMemPool::memoryPoolReserved() const
{
    std::size_t reserved = 0;
    TLLM_CUDA_CHECK(cudaMemPoolGetAttribute(mPool.get(), cudaMemPoolAttrReservedMemCurrent, &reserved));
    return reserved;
}

std::size_t CudaMemPool::memoryPoolUsed() const
{
    std::size_t used = 0;
    TLLM_CUDA_CHECK(cudaMemPoolGetAttribute(mPool.get(), cudaMemPoolAttrUsedMemCurrent, &used));
    return used;
}

std::size_t CudaMemPool::memoryPoolFree() const
{
    return memoryPoolReserved() - memoryPoolUsed();
}

void CudaMemPool::memoryPoolTrimTo(std::size_t size)
{
    TLLM_CUDA_CHECK(::cudaMemPoolTrimTo(mPool.get(), size));
}

cudaMemPool_t CudaMemPool::getPool() const
{
    return mPool.get();
}

bool CudaMemPool::supportsMemoryPool(int deviceId)
{
    int32_t value{};
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMemoryPoolsSupported, deviceId));
    return value != 0;
}

void CudaMemPool::Deleter::operator()(cudaMemPool_t pool) const
{
    TLLM_CUDA_CHECK_FREE_RESOURCE(::cudaMemPoolDestroy(pool));
    TLLM_LOG_TRACE("Destroyed pool %p", pool);
}

namespace
{

std::shared_ptr<CudaMemPool> createPrimaryDevicePool(int deviceId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    ::cudaMemPool_t memPool = nullptr;
    ::cudaMemPoolProps poolProps{};
    poolProps.allocType = ::cudaMemAllocationTypePinned;
    poolProps.handleTypes = ::cudaMemHandleTypeNone;
    poolProps.location.type = ::cudaMemLocationTypeDevice;
    poolProps.location.id = deviceId;
    TLLM_CUDA_CHECK(::cudaMemPoolCreate(&memPool, &poolProps));
    // set memory pool threshold to avoid shrinking the pool
    auto maxThreshold = std::numeric_limits<std::uint64_t>::max();
    TLLM_CUDA_CHECK(cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &maxThreshold));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return std::make_shared<CudaMemPool>(memPool);
}

/// @brief The maximum number of devices per node this feature supports. Increase when/if this value becomes too small.
/// Will add a tiny memory usage increase.
constexpr size_t maxDevicePerNode = 64;

/// @brief Ensures thread safe initialization of the primary device pools.
std::mutex primaryDevicePoolsMutex{};

/// @brief The primary device memory pool for each discovered device, if the device in question supports memory pools,
/// initialized on first use, write is mutually exclusive through primaryDevicePoolsMutex.
std::array<std::shared_ptr<CudaMemPool>, maxDevicePerNode> primaryDevicePools{};

/// @brief Whether or not initializing the primary device pool at each device ID has been attempted. If true, then it is
/// safe to just return whatever is at the same index in primaryDevicePools without locking. Also, prevents repeatedly
/// trying to initialize the memory pool for a device, if the first attempt failed.
std::array<bool, maxDevicePerNode> primaryDevicePoolInitAttempted{};

} // namespace

std::shared_ptr<CudaMemPool> CudaMemPool::getPrimaryPoolForDevice(int deviceId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // If we've already attempted, successfully or not, to initialize the pool for that device, we just return whatever
    // we have at the device index.
    if (primaryDevicePoolInitAttempted.at(deviceId))
    {
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return primaryDevicePools.at(deviceId);
    }

    // Otherwise, we grab the lock as we will need to initialize the pool, and it should be done in a thread safe way.
    {
        std::lock_guard lockGuard{primaryDevicePoolsMutex};

        // Check again that pool has not been initialized while this thread was waiting on the lock.
        if (primaryDevicePoolInitAttempted.at(deviceId))
        {
            TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
            return primaryDevicePools.at(deviceId);
        }

        // Check for mem pool support on that device.
        if (!CudaMemPool::supportsMemoryPool(deviceId))
        {
            primaryDevicePoolInitAttempted.at(deviceId) = true;
            TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
            return {};
        }

        // Creating the mem pool can throw, needs to be handled.
        try
        {
            primaryDevicePools.at(deviceId) = createPrimaryDevicePool(deviceId);
            primaryDevicePoolInitAttempted.at(deviceId) = true;
            TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
            return primaryDevicePools.at(deviceId);
        }
        catch (std::exception const& exception)
        {
            TLLM_LOG_ERROR("Failed to initialized memory pool for device %i.", deviceId);
            TLLM_LOG_EXCEPTION(exception);
            primaryDevicePoolInitAttempted.at(deviceId) = true;
            TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
            return {};
        }
    }
}

} // namespace onnxruntime::llm::runtime
