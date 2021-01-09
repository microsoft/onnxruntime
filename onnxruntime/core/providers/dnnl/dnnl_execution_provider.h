// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <map>
#include <list>
#include <memory.h>

#include "core/platform/ort_mutex.h"
#include "core/providers/dnnl/subgraph/subgraph.h"
#include "core/platform/ort_mutex.h"

namespace dnnl {
struct memory;
};

namespace onnxruntime {

// Information needed to construct DNNL execution providers.
struct DNNLExecutionProviderInfo {
  bool create_arena{true};

  explicit DNNLExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}
  DNNLExecutionProviderInfo() = default;
};

// Logical device representation.
class DNNLExecutionProvider : public Provider_IExecutionProvider {
 public:
  explicit DNNLExecutionProvider(const DNNLExecutionProviderInfo& info);
  virtual ~DNNLExecutionProvider();

  virtual std::shared_ptr<Provider_KernelRegistry> Provider_GetKernelRegistry() const override;

  std::shared_ptr<dnnl::memory> GetWeightsMemoryBuffer(const std::string& weight_key) {
    auto iter = weights_mem_map_.find(weight_key);
    if (iter != weights_mem_map_.end())
      return iter->second;
    return nullptr;
  }

  void SetWeightsMemoryBuffer(const std::string& weight_key,
                              const std::shared_ptr<dnnl::memory>& filter_dst_mem) {
    weights_mem_map_.insert(std::make_pair(weight_key, filter_dst_mem));
  }

  OrtMutex& GetMutex() {
    return mutex_;
  }

  void SaveAllocatedMemory(IAllocatorUniquePtr<void> buffer) {
    // keep reordered memory buffers in scope.
    reordered_buffers_.push_back(std::move(buffer));
  }

  std::shared_ptr<dnnl::memory> GetBiasMemoryBuffer(const std::string& key) {
    auto iter = bias_mem_map_.find(key);
    if (iter != bias_mem_map_.end())
      return iter->second;
    return nullptr;
  }

  // Conv+BathNorm fusion. save scaled bias memory.
  void SetBiasMemoryBuffer(const std::string& key,
                           const std::shared_ptr<dnnl::memory>& bias_mem) {
    bias_mem_map_.insert(std::make_pair(key, bias_mem));
  }

  void SaveAllocatedBiasMemory(IAllocatorUniquePtr<void> buffer) {
    // keep reordered memory buffers in scope.
    biass_buffers_.push_back(std::move(buffer));
  }

  std::vector<std::unique_ptr<Provider_ComputeCapability>>
  Provider_GetCapability(const onnxruntime::Provider_GraphViewer& graph,
                         const std::vector<const Provider_KernelRegistry*>& /*kernel_registries*/) const override;

  common::Status Provider_Compile(const std::vector<onnxruntime::Provider_Node*>& fused_nodes,
                                  std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  // dnnl weights(filer data) memory blocks from first iteration
  // saved by weights name
  std::unordered_map<std::string, std::shared_ptr<dnnl::memory>> weights_mem_map_;
  // Save reordered memory buffers in list so that memory is not freed.
  std::vector<IAllocatorUniquePtr<void>> reordered_buffers_;

  // conv+batchnorm fusion. normalized bias memory blocks from first iteration
  std::unordered_map<std::string, std::shared_ptr<dnnl::memory>> bias_mem_map_;
  // Conv+BathNorm fusion bias memory buffer.
  std::vector<IAllocatorUniquePtr<void>> biass_buffers_;
  OrtMutex mutex_;

  // SUBGRAPH
 private:
  static int GetOnnxOpSet(const Provider_GraphViewer& graph_viewer) {
    const auto& dm_to_ver = graph_viewer.DomainToVersionMap();
    return dm_to_ver.at(kOnnxDomain);
  }

  std::string GetGraphName(const onnxruntime::Provider_GraphViewer& graph_viewer) const {
    std::string graph_name;

    int opset = GetOnnxOpSet(graph_viewer);

    int index = 0;
    if (graph_viewer.MaxNodeIndex() > 0) {
      auto first_node = graph_viewer.GetNode(index);
      while (first_node == nullptr) {
        index++;
        first_node = graph_viewer.GetNode(index);
      }
      auto first_node_outputs = first_node->OutputDefs();
      graph_name = graph_viewer.Name() + "_opset-" + std::to_string(opset) + "_" + first_node_outputs[0]->Name();
    }
    return graph_name;
  }

  bool UseSubgraph(const onnxruntime::Provider_GraphViewer& graph_viewer) const;

  // Some dimensions are not supported by DNNL
  // example: Pool with NumDimensions <= 3 is not supported
  // Fall back to CPU implementation
  bool IsDimensionSupported(const Provider_Node* node) const {
    bool supported = true;
    if (node->OpType() == "BatchNormalization") {
      auto node_inputs = node->InputDefs();
      if (node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim_size() == 3) {
        supported = false;
      }
    }
    if (node->OpType().find("Pool") != std::string::npos) {
      auto node_inputs = node->InputDefs();
      if (node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim_size() <= 3) {
        supported = false;
      }

      if (node->OutputDefs().size() > 1)
        supported = false;
    }
    return supported;
  }

  void CreateOrUpdateDnnlNode(const Provider_Node* node,
                              std::shared_ptr<ort_dnnl::Subgraph>& subgraph_ptr,
                              ort_dnnl::Subgraph::SubgraphVariables& sub_var,
                              bool fused,
                              std::map<std::string, size_t>& output_to_source_node_map,
                              Provider_NodeAttributes& subgraph_attributes) const;

  // Create Dnnl node, update inputs, outputs and parent nodes
  // collect attribtes
  void CreateMetaDef(const onnxruntime::Provider_GraphViewer& graph_viewer,
                     const Provider_NodeAttributes& subgraph_attributes,
                     std::shared_ptr<ort_dnnl::Subgraph>& subgraph_ptr,
                     ort_dnnl::Subgraph::SubgraphVariables& sub_var,
                     std::vector<std::unique_ptr<Provider_ComputeCapability>>& result) const;

 public:
  const std::shared_ptr<ort_dnnl::Subgraph> GetDnnlSubgraph(const std::string& subgraph_id) {
    return mkl_subgraphs_[subgraph_id];
  }

 private:
  mutable int subgraph_index_ = 0;

  // supported Dnnl Operators
  std::set<std::string> dnnl_ops_ = {"Conv", "BatchNormalization", "Relu", "Sum",
                                     "AveragePool", "GlobalMaxPool", "GlobalAveragePool", "MaxPool", "LRN"};

  mutable std::unordered_map<std::string, std::shared_ptr<ort_dnnl::Subgraph>> mkl_subgraphs_;
};

}  // namespace onnxruntime
