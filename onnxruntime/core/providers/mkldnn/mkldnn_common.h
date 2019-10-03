// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "mkldnn.hpp"
#include <unordered_map>

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
  static mkldnn::engine cpu_engine = mkldnn::engine(mkldnn::engine::cpu, 0);
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
    static thread_local std::unordered_map<std::string, std::unique_ptr<PrimitiveBase>> map;
    return map;
  }
};

// Struct which encapsulates parameters for MKLDNN memory reorder primitive.
struct MemoryReorderParams {
  const mkldnn::memory& src;
  const mkldnn::memory& dst;

  MemoryReorderParams(const mkldnn::memory& src, const mkldnn::memory& dst) : src(src), dst(dst) {}

  // Used as the key for MemoryReorder primitive reuse pool.
  std::string ToString() const {
    std::string key;
    key.reserve(64);
    key.append("reorder_");
    const auto& src_desc = src.get_primitive_desc().desc().data;
    const auto& dst_desc = dst.get_primitive_desc().desc().data;
    mkldnn::memory::dims src_dims(src_desc.dims, &src_desc.dims[src_desc.ndims]);
    mkldnn::memory::dims dst_dims(dst_desc.dims, &dst_desc.dims[dst_desc.ndims]);
    key.append(std::to_string(src_desc.format));
    key.append(1, '_');
    key.append(std::to_string(dst_desc.data_type));
    AddDimsToKey(key, src_dims);
    key.append(std::to_string(dst_desc.format));
    key.append(1, '_');
    key.append(std::to_string(dst_desc.data_type));
    AddDimsToKey(key, dst_dims);
    return key;
  }
};

// Encapsulates an MKLDNN memory reorder primitive.
// these are needed to convert the source/weight/destination memory layout
// to one that is identified by MKLDNN to be optimal for performance.
class MemoryReorderPrimitive : public PrimitiveBase {
 public:
  explicit MemoryReorderPrimitive(const MemoryReorderParams& params) : cpu_engine_(GetEngine()) {
    Initialize(params);
  }
  ~MemoryReorderPrimitive() = default;

  std::shared_ptr<mkldnn::primitive> GetPrimitive() {
    return context_.primitive;
  }

  void SetMemory(const MemoryReorderParams& params) {
    context_.src_mem->set_data_handle(params.src.get_data_handle());
    context_.dst_mem->set_data_handle(params.dst.get_data_handle());
  }

 private:
  struct MemoryReorderContext {
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> dst_mem;
    std::shared_ptr<mkldnn::primitive> primitive;
    MemoryReorderContext() : src_mem(nullptr), dst_mem(nullptr), primitive(nullptr) {
    }
  } context_;

  mkldnn::engine& cpu_engine_;

  void Initialize(const MemoryReorderParams& params) {
    context_.src_mem = std::make_shared<mkldnn::memory>(
        mkldnn::memory({params.src.get_primitive_desc().desc(), cpu_engine_}, nullptr));
    context_.dst_mem = std::make_shared<mkldnn::memory>(
        mkldnn::memory({params.dst.get_primitive_desc().desc(), cpu_engine_}, nullptr));
    context_.primitive = std::make_shared<mkldnn::reorder>(
        mkldnn::reorder(*context_.src_mem, *context_.dst_mem));
  }
};

// Pool which allows for reuse of MKLDNN memory reorder primitives which are expensive to instantiate.
// To address thread safety, the primitives are stored in a map on thread local storage.
template <typename T>
class MemoryReorderPrimitivePool : public PrimitivePool<T> {
 public:
  static MemoryReorderPrimitivePool& GetInstance() {
    static MemoryReorderPrimitivePool pool;
    return pool;
  }

  static MemoryReorderPrimitive* Get(const MemoryReorderParams& params) {
    MemoryReorderPrimitive* primitive = static_cast<MemoryReorderPrimitive*>(
        MemoryReorderPrimitivePool<T>::GetInstance().GetPrimitive(params.ToString()));
    if (primitive == nullptr) {
      auto reorder_primitive = onnxruntime::make_unique<MemoryReorderPrimitive>(params);
      primitive = reorder_primitive.get();
      MemoryReorderPrimitivePool<T>::GetInstance().SetPrimitive(params.ToString(), std::move(reorder_primitive));
    }
    primitive->SetMemory(params);
    return primitive;
  }

 private:
  MemoryReorderPrimitivePool() = default;
  ~MemoryReorderPrimitivePool() = default;
};

template <typename T>
static void DoReorder(const MemoryReorderParams& params) {
  std::vector<mkldnn::primitive> net;
  net.push_back(*(MemoryReorderPrimitivePool<T>::Get(params)->GetPrimitive()));
  mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
}

}  // namespace mkl_dnn
}  // namespace onnxruntime
