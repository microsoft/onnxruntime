#pragma once

#include <winrt/windows.media.h>
#include "VideoFrameToTensorConverter.h"
#include "TensorToVideoFrameConverter.h"

namespace Windows::AI::MachineLearning
{

// Forward Declare
class ConverterResourceStore;

struct CONVERTER_RESOURCE_DESC
{
    DWORD PixelFormat;
    int Width;
    int Height;
    LUID Luid;
    
    bool operator==(_In_ CONVERTER_RESOURCE_DESC& desc)
    {
        // Converter resources DON'T match if
        // 1) the resources have different dimensions
        // 2) the resources are on different devices
        // 3) the resources have different pixel formats
        if (desc.Width != Width ||
            desc.Height != Height ||
            desc.Luid.HighPart != Luid.HighPart ||
            desc.Luid.LowPart != Luid.LowPart ||
            desc.PixelFormat != PixelFormat)
        {
            return false;
        }

        return true;
    }
};

class ConverterResources : public std::enable_shared_from_this<ConverterResources>
{
    using Pool = std::weak_ptr<ConverterResourceStore>;

public:
    template <typename... TArgs>
    static std::shared_ptr<ConverterResources> Create(Pool pool, CONVERTER_RESOURCE_DESC& descriptor)
    {
        return std::make_shared<ConverterResources>(pool, descriptor);
    }

    ConverterResources(Pool& pool, CONVERTER_RESOURCE_DESC& descriptor);

    void ReturnToCache();

public:
    CONVERTER_RESOURCE_DESC Descriptor;

    std::unique_ptr<Windows::AI::MachineLearning::Internal::VideoFrameToTensorConverter> Tensorizer;
    std::unique_ptr<Windows::AI::MachineLearning::Internal::TensorToVideoFrameConverter> Detensorizer;
private:
    Pool m_pool;
};

// This class retains  tensorization and detensorization
// resources in a store, and evicts the oldest resource object
// when the size of the pool is maxed out. Objects in the pool
// can be reused for caching purposes to enhance performance during
// tensorization.
class ConverterResourceStore : public std::enable_shared_from_this<ConverterResourceStore>
{
    struct PoolObject
    {
        std::shared_ptr<ConverterResources> Resource;
        uint64_t StoreId;
    };

public:
    template <typename... TArgs>
    static std::shared_ptr<ConverterResourceStore> Create(TArgs&&... args)
    {
        return std::make_shared<ConverterResourceStore>(std::forward<TArgs>(args)...);
    }

    ConverterResourceStore(size_t nCacheSize);

    std::shared_ptr<ConverterResources> Fetch(CONVERTER_RESOURCE_DESC& descriptor);
    void Store(std::shared_ptr<ConverterResources> object);

private:
    std::shared_ptr<ConverterResources> FetchAndRemoveObject(CONVERTER_RESOURCE_DESC& desc);
    void EvictOldestPoolObject();

private:
    std::vector<PoolObject> m_objects;
    size_t m_cacheSize;
    std::mutex m_mutex;
    uint64_t storeId = 0;
};

class PoolObjectWrapper
{
public:
    template <typename... TArgs>
    static std::shared_ptr<PoolObjectWrapper> Create(TArgs&&... args)
    {
        return std::make_shared<PoolObjectWrapper>(std::forward<TArgs>(args)...);
    }

    explicit PoolObjectWrapper(std::shared_ptr<ConverterResources>&& resources) :
        m_resources(resources)
    {
    }

    ~PoolObjectWrapper()
    {
        if (m_resources)
        {
            m_resources->ReturnToCache();
        }
    }

    std::shared_ptr<ConverterResources> Get()
    {
        return m_resources;
    }

private:
    std::shared_ptr<ConverterResources> m_resources;
};

} // Windows::AI::MachineLearning