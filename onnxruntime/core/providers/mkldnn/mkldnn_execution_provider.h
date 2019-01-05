// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <map>
#include <memory.h>
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/graph_transformer.h"

namespace mkldnn {
struct memory;
};

namespace onnxruntime {

// Information needed to construct MKL-DNN execution providers.
struct MKLDNNExecutionProviderInfo {
  bool create_arena{true};

  explicit MKLDNNExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}
  MKLDNNExecutionProviderInfo() = default;
};

// Logical device representation.
class MKLDNNExecutionProvider : public IExecutionProvider {
 public:
  explicit MKLDNNExecutionProvider(const MKLDNNExecutionProviderInfo& info);
  virtual ~MKLDNNExecutionProvider();

  std::string Type() const override {
    return onnxruntime::kMklDnnExecutionProvider;
  }

  Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  std::shared_ptr<mkldnn::memory> GetWeightsMemory(const std::string& weight_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = weights_mem_map_.find(weight_key);
    if (iter != weights_mem_map_.end())
      return iter->second;
    return nullptr;
  }

  void SetWeightsMemory(const std::string& weight_key,
                        const std::shared_ptr<mkldnn::memory>& filter_dst_mem) {
    std::lock_guard<std::mutex> lock(mutex_);
    weights_mem_map_[weight_key] = filter_dst_mem;
  }

  std::mutex& GetMutex() {
    return conv_mutex_;
  }

 private:
  // mkldnn weights(filer data) memory blocks from first iteration
  // saved by weights name
  std::map<std::string, std::shared_ptr<mkldnn::memory>> weights_mem_map_;
  mutable std::mutex mutex_;
  std::mutex conv_mutex_;
};

}  // namespace onnxruntime
