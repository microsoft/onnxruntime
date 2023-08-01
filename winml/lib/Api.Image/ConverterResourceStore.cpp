// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Api.Image/pch.h"
#include "inc/ConverterResourceStore.h"

#include <winrt\windows.media.h>
#include <winrt\Windows.Graphics.Imaging.h>

#include <d3d11on12.h>
#include <d3d11_4.h>

using namespace _winml;

ConverterResources::ConverterResources(Pool& pool, ConverterResourceDescription& descriptor)
  : Descriptor(descriptor),
    Tensorizer(std::make_unique<VideoFrameToTensorConverter>()),
    Detensorizer(std::make_unique<TensorToVideoFrameConverter>()),
    m_pool(pool) {
}

void ConverterResources::ReturnToCache() {
  if (auto pool = m_pool.lock()) {
    pool->Store(shared_from_this());
  }
}

ConverterResourceStore::ConverterResourceStore(size_t nCacheSize) : m_cacheSize(nCacheSize) {
}

std::shared_ptr<ConverterResources> ConverterResourceStore::Fetch(ConverterResourceDescription& descriptor) {
  std::lock_guard<std::mutex> lock(m_mutex);

  auto resource = FetchAndRemoveObject(descriptor);

  if (resource == nullptr) {
    // Create the resource
    resource = ConverterResources::Create(shared_from_this(), descriptor);
  }

  return resource;
}

std::shared_ptr<ConverterResources> ConverterResourceStore::FetchAndRemoveObject(ConverterResourceDescription& desc) {
  // Iterate through the resources and find all the resources which are completed and unallocate
  auto foundIt = std::find_if(std::begin(m_objects), std::end(m_objects), [&](const auto& cachedObject) {
    return desc == cachedObject.Resource->Descriptor;
  });

  if (foundIt == std::end(m_objects)) {
    return nullptr;
  } else {
    std::shared_ptr<ConverterResources> object = foundIt->Resource;
    // Remove the item from the cache so that it is not reused by another call
    m_objects.erase(foundIt);

    return object;
  }
}

void ConverterResourceStore::Store(std::shared_ptr<ConverterResources> object) {
  std::lock_guard<std::mutex> lock(m_mutex);

  auto foundIt = std::find_if(std::begin(m_objects), std::end(m_objects), [&](const auto& cachedObject) {
    return object == cachedObject.Resource;
  });

  if (foundIt == std::end(m_objects)) {
    // If the resource is not already cached
    if (m_objects.size() < m_cacheSize) {
      // If the cache has free slots, then use one
      m_objects.push_back(PoolObject{object, storeId++});
    } else {
      // If the cache has no free slots, then evict the oldest
      EvictOldestPoolObject();

      m_objects.push_back(PoolObject{object, storeId++});
    }
  }
}

void ConverterResourceStore::EvictOldestPoolObject() {
  auto oldestIt =
    std::min_element(std::begin(m_objects), std::end(m_objects), [&](const auto& left, const auto& right) {
      return left.StoreId < right.StoreId;
    });

  // Remove the oldest item from the cache
  m_objects.erase(oldestIt);
}
