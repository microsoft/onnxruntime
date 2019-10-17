// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "mkldnn.hpp"
#include <unordered_map>
#include <list>

#define MKLDNN_EP_LRU_CACHE_DEFAULT_SIZE 500

namespace onnxruntime {
namespace mkl_dnn {

template <typename T>
static mkldnn::memory::data_type MklDnnType();

// Add more types here as needed.
template <>
mkldnn::memory::data_type MklDnnType<float>() {
  return mkldnn::memory::data_type::f32;
}

static mkldnn::engine& GetEngine() {
  static mkldnn::engine cpu_engine = mkldnn::engine(mkldnn::engine::kind::cpu, 0);
  return cpu_engine;
}

static void AddDimsToKey(std::string& key, const mkldnn::memory::dims& dims) {
  key.append(1, '#');
  for (size_t i = 0; i < dims.size(); i++) {
    key.append(std::to_string(dims[i]));
    key.append(1, '_');
  }
  key.append(1, '#');
}

class PrimitiveBase {
 public:
  virtual ~PrimitiveBase() = default;
};

template <typename T>
class PrimitivePool {
 public:
  PrimitivePool() {
    // Get cache size from environment
    std::string tempSize;
#ifdef _WIN32
    char* buf{nullptr};
    size_t bufSize = 0;
    if (!_dupenv_s(&buf, &bufSize, "ONNXRUNTIME_MKLDNN_LRU_CACHE_SIZE") && buf) {
      tempSize = buf;
      free(buf);
    }
#else
    if (std::getenv("ONNXRUNTIME_NGRAPH_LRU_CACHE_SIZE")) {
      tempSize = std::getenv("ONNXRUNTIME_NGRAPH_LRU_CACHE_SIZE");
    }
#endif
    auto& cacheSize = PrimitivePool<T>::GetCacheSize();
    cacheSize = tempSize.empty() ? MKLDNN_EP_LRU_CACHE_DEFAULT_SIZE : std::stoi(tempSize);
  };

  ~PrimitivePool() = default;

  void SetPrimitive(const std::string& key, std::unique_ptr<PrimitiveBase> primitive) {
    auto& map = PrimitivePool<T>::GetMap();
    auto& keyCache = PrimitivePool<T>::GetKeyCache();
    auto& cacheSize = PrimitivePool<T>::GetCacheSize();

    if (keyCache.size() == cacheSize) {
      // Delete least recently used element
      std::string last = keyCache.back();

      // Pop the last element
      keyCache.pop_back();

      // Erase the last element from cache
      map.erase(map.find(last));
    }

    auto iter = map.find(key);
    // We should not find a primitive already using this key.
    ORT_ENFORCE(iter == map.end(), "duplicate key: " + key);
    map.insert(std::make_pair(key, std::move(primitive)));
  }

  PrimitiveBase* GetPrimitive(const std::string& key) {
    const auto& map = PrimitivePool<T>::GetMap();
    auto& keyCache = PrimitivePool<T>::GetKeyCache();

    auto iter = map.find(key);
    if (iter != map.end()) {
      keyCache.remove(key);
      keyCache.push_front(key);

      return iter->second.get();
    } else {
      return nullptr;
    }
  }

 private:
  // For thread safety, the map needs to be kept in thread local storage.
  static inline std::unordered_map<std::string, std::unique_ptr<PrimitiveBase>>& GetMap() {
    static thread_local std::unordered_map<std::string, std::unique_ptr<PrimitiveBase>> map;
    return map;
  }

  static inline std::list<std::string>& GetKeyCache() {
    static thread_local std::list<std::string> keyCache;
    return keyCache;
  }

  static inline size_t& GetCacheSize() {
    static thread_local size_t cacheSize;
    return cacheSize;
  }
};
}  // namespace mkl_dnn
}  // namespace onnxruntime
