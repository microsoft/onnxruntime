// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <map>

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/graph_transformer.h"
#include "mkldnn.hpp"

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

  std::shared_ptr<mkldnn::memory> GetWeightMemory(std::string weightName) {

    auto iter = weights_mem_map.find(weightName);
    if (iter != weights_mem_map.end())
      return iter->second;
    return nullptr;
  }

  std::map<std::string, std::shared_ptr<mkldnn::memory>>& GetWeightsMap() {
    return weights_mem_map;
  }
 
private:
  // mkldnn formatted weights(filer data) memory from first iteration 
  // saved by weights name
  std::map<std::string, std::shared_ptr<mkldnn::memory>> weights_mem_map;
};

}  // namespace onnxruntime
