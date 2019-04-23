// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <map>
#include <list>
#include <memory.h>

#include "core/platform/ort_mutex.h"
#include "core/graph/constants.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/providers/mkldnn/subgraph/subgraph.h"

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

struct MKLContext {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocator = nullptr;

  MKLContext(AllocateFunc allocate_func, DestroyFunc release_func, AllocatorHandle alloc)
      : allocate_func(allocate_func),
        release_func(release_func),
        allocator(alloc) {}
};

// Logical device representation.
class MKLDNNExecutionProvider : public IExecutionProvider {
 public:
  explicit MKLDNNExecutionProvider(const MKLDNNExecutionProviderInfo& info);
  virtual ~MKLDNNExecutionProvider();

  Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  std::shared_ptr<mkldnn::memory> GetWeightsMemoryBuffer(const std::string& weight_key) {
    auto iter = weights_mem_map_.find(weight_key);
    if (iter != weights_mem_map_.end())
      return iter->second;
    return nullptr;
  }

  void SetWeightsMemoryBuffer(const std::string& weight_key,
                              const std::shared_ptr<mkldnn::memory>& filter_dst_mem) {
    weights_mem_map_.insert(std::make_pair(weight_key, filter_dst_mem));
  }

  OrtMutex& GetMutex() {
    return mutex_;
  }

  void SaveAllocatedMemory(IAllocatorUniquePtr<void> buffer) {
    // keep reordered memory buffers in scope.
    reordered_buffers_.push_back(std::move(buffer));
  }

 private:
  // mkldnn weights(filer data) memory blocks from first iteration
  // saved by weights name
  std::unordered_map<std::string, std::shared_ptr<mkldnn::memory>> weights_mem_map_;
  // Save reordered memory buffers in list so that memory is not freed.
  std::vector<IAllocatorUniquePtr<void>> reordered_buffers_;
  OrtMutex mutex_;

  // SUBGRAPH
 private:
  bool RunSubgraph(const onnxruntime::GraphViewer& graph_viewer,
                                            const std::vector<const KernelRegistry*>& kernel_registries,
                                            std::vector<std::unique_ptr<ComputeCapability>>& result) const;
  void CreateMetaDef(SubgraphVariables& sub_var, const NodeAttributes& subgraph_attributes,
                     std::vector<std::unique_ptr<ComputeCapability>>& result) const;

 public:
  const std::shared_ptr<Subgraph> GetMklSubgraph(const std::string& subgraph_id) {
    auto iter = mkl_subgraphs_.find(subgraph_id);
    if (iter != mkl_subgraphs_.end())
      return iter->second;
    return nullptr;
  }

 private:
  mutable std::unordered_map<std::string, std::shared_ptr<Subgraph>> mkl_subgraphs_;
};

}  // namespace onnxruntime
