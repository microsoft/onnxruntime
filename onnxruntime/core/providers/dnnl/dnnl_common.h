// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/shared_library/provider_api.h"
#include "gsl/gsl-lite.hpp"
#include "dnnl.hpp"
#include "dnnl_debug.h"
#include <unordered_map>
#include <list>

namespace onnxruntime {
namespace ort_dnnl {

template <typename T>
static dnnl::memory::data_type DnnnType();

// Add more types here as needed.
template <>
dnnl::memory::data_type DnnnType<float>() {
  return dnnl::memory::data_type::f32;
}

class DnnlEngineInstance {
 private:
  static DnnlEngineInstance* instance;
  std::unordered_map<dnnl::engine::kind, dnnl::engine> dnnl_engine_map;

  DnnlEngineInstance() {
    dnnl::engine engine;
    try {
      engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
      if (engine) {
        dnnl_engine_map.insert(std::make_pair(dnnl::engine::kind::cpu, engine));
      }
    } catch (const std::exception& e) {
      LOGS_DEFAULT(ERROR) << e.what() << std::endl;
      throw;
    }
    try {
      engine = dnnl::engine(dnnl::engine::kind::gpu, 0);
      if (engine) {
        dnnl_engine_map.insert(std::make_pair(dnnl::engine::kind::gpu, engine));
      }
    } catch (const std::exception& e) {
      LOGS_DEFAULT(INFO) << e.what() << std::endl;
    }
  }

 public:
  DnnlEngineInstance(DnnlEngineInstance& other) = delete;
  void operator=(const DnnlEngineInstance&) = delete;
  static DnnlEngineInstance* getInstance() {
    if (!instance)
      instance = new DnnlEngineInstance();
    return instance;
  }

  const std::unordered_map<dnnl::engine::kind, dnnl::engine>& getEngineMap() {
    return dnnl_engine_map;
  }

  const dnnl::engine& getEngine(dnnl::engine::kind kind) {
    std::unordered_map<dnnl::engine::kind, dnnl::engine>::iterator iter = dnnl_engine_map.find(kind);
    return iter->second;
  }
};

static void AddDimsToKey(std::string& key, const dnnl::memory::dims& dims) {
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
  PrimitivePool() = default;
  ~PrimitivePool() = default;

  void SetPrimitive(const std::string& key, std::unique_ptr<PrimitiveBase> primitive) {
    auto& map = PrimitivePool<T>::GetMap();
    auto iter = map.find(key);
    // We should not find a primitive already using this key.
    ORT_ENFORCE(iter == map.end(), "duplicate key: " + key);
    map.insert(std::make_pair(key, std::move(primitive)));
  }

  PrimitiveBase* GetPrimitive(const std::string& key) {
    const auto& map = PrimitivePool<T>::GetMap();
    auto iter = map.find(key);
    if (iter != map.end()) {
      return iter->second.get();
    } else {
      return nullptr;
    }
  }

 private:
  // For thread safety, the map needs to be kept in thread local storage.
  static inline std::unordered_map<std::string, std::unique_ptr<PrimitiveBase>>& GetMap() {
    using MapType = std::unordered_map<std::string, std::unique_ptr<PrimitiveBase>>;
    static thread_local DeleteOnUnloadPtr<MapType> map(new MapType());
    return *map;
  }
};

}  // namespace ort_dnnl
}  // namespace onnxruntime
