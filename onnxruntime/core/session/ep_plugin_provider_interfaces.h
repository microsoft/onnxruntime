// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/providers/providers.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
struct EpNode;
struct EpValueInfo;

/// <summary>
/// IExecutionProviderFactory that wraps a OrtEpFactory. Required for SessionOptionsAppendExecutionProvider_V2.
/// </summary>
struct PluginExecutionProviderFactory : public IExecutionProviderFactory {
 public:
  PluginExecutionProviderFactory(OrtEpFactory& ep_factory, gsl::span<const OrtEpDevice* const> ep_devices);

  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger) override;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    ORT_NOT_IMPLEMENTED("CreateProvider without parameters is not supported.");
  }

 private:
  OrtEpFactory& ep_factory_;
  std::vector<const OrtHardwareDevice*> devices_;
  std::vector<const OrtKeyValuePairs*> ep_metadata_;
};

/// <summary>
/// Functor that deletes an instance of OrtEp. Used to create an std::unique_ptr<OrtEp, OrtEpDeleter>.
/// </summary>
struct OrtEpDeleter {
  explicit OrtEpDeleter(OrtEpFactory& ort_ep_factory) : ort_ep_factory_(ort_ep_factory) {}
  void operator()(OrtEp* ort_ep) {
    ort_ep_factory_.ReleaseEp(&ort_ep_factory_, ort_ep);
  }
  OrtEpFactory& ort_ep_factory_;
};

/// <summary>
/// Type that represents a std::unique_ptr for an instance of OrtEp.
/// </summary>
using UniqueOrtEp = std::unique_ptr<OrtEp, OrtEpDeleter>;

/// <summary>
/// IExecutionProvider that wraps an instance of OrtEp.
/// </summary>
class PluginExecutionProvider : public IExecutionProvider {
 public:
  explicit PluginExecutionProvider(UniqueOrtEp ep);
  ~PluginExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const IKernelLookup& kernel_lookup,
                const GraphOptimizerRegistry& graph_optimizer_registry,
                IResourceAccountant* resource_accountant = nullptr) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  struct FusedNodeState {
    FusedNodeState() = default;
    FusedNodeState(FusedNodeState&& other) = default;
    FusedNodeState(const FusedNodeState& other) = delete;
    Status AddFusedNode(const Node& fused_node, /*out*/ EpNode*& added_ep_node);

    std::vector<std::unique_ptr<EpNode>> nodes;
    std::unordered_map<std::string, std::unique_ptr<EpValueInfo>> value_infos;
  };

  UniqueOrtEp ort_ep_;
  std::vector<OrtNodeComputeInfo*> api_node_compute_infos_;

  // Fused nodes have to be valid throughout model inference because they may be cached in NodeComputeInfo instances.
  // For each fused node, the Compile() function creates EpNode and EpValueInfo instances on the heap,
  // which are then passed to the underlying OrtEp instance. This class stores this "fused node state"
  // so that it is not destroyed until the EP itself is destroyed.
  std::vector<FusedNodeState> fused_node_states_;
};
}  // namespace onnxruntime
