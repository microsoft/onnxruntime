// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/execution_provider.h"
#include "core/providers/providers.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
struct EpNode;
struct EpValueInfo;
class NodeArg;
class PluginExecutionProvider;

/// <summary>
/// IExecutionProviderFactory that wraps a OrtEpFactory. Required for SessionOptionsAppendExecutionProvider_V2.
/// </summary>
struct PluginExecutionProviderFactory : public IExecutionProviderFactory {
 public:
  PluginExecutionProviderFactory(OrtEpFactory& ep_factory, gsl::span<const OrtEpDevice* const> ep_devices);

  // Constructor that accepts hw devices and ep metadata that have already been extracted from the given OrtEpDevice
  // instances. It is an error to call this constructor with hw devices or ep metadata that do not correspond to the
  // correct EP devices (e.g., hw_devices[i] and ep_metadata[i] should be extracted from ep_devices[i]).
  PluginExecutionProviderFactory(OrtEpFactory& ep_factory,
                                 gsl::span<const OrtEpDevice* const> ep_devices,
                                 gsl::span<const OrtHardwareDevice* const> hw_devices,
                                 gsl::span<const OrtKeyValuePairs* const> ep_metadata);

  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger) override;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    ORT_NOT_IMPLEMENTED("CreateProvider without parameters is not supported.");
  }

  /// <summary>
  /// Alternative version of CreateProvider that returns a Status.
  /// </summary>
  /// <param name="session_options">The session options to pass to the EP factory.</param>
  /// <param name="logger">The session logger. Stored by the OrtEp.</param>
  /// <param name="plugin_ep">Output parameter set to the newly created PluginExecutionProvider.</param>
  /// <returns>A status indicating success or an error.</returns>
  Status CreatePluginExecutionProvider(const OrtSessionOptions& session_options,
                                       const OrtLogger& logger,
                                       /*out*/ std::unique_ptr<PluginExecutionProvider>& plugin_ep);

 private:
  OrtEpFactory& ep_factory_;
  InlinedVector<const OrtEpDevice*> devices_;
  InlinedVector<const OrtHardwareDevice*> hardware_devices_;
  InlinedVector<const OrtKeyValuePairs*> ep_metadata_;
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
 private:
  using Base = IExecutionProvider;

 public:
  explicit PluginExecutionProvider(UniqueOrtEp ep, const OrtSessionOptions& session_options, OrtEpFactory& ep_factory,
                                   gsl::span<const OrtEpDevice* const> ep_devices,
                                   std::shared_ptr<KernelRegistry> kernel_registry,
                                   const logging::Logger& logger);
  ~PluginExecutionProvider();

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const IKernelLookup& kernel_lookup,
                const GraphOptimizerRegistry& graph_optimizer_registry,
                IResourceAccountant* resource_accountant = nullptr) const override;

  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  DataLayout GetPreferredLayout() const override;

  std::optional<bool> ShouldConvertDataLayoutForOp(std::string_view node_domain,
                                                   std::string_view node_op_type,
                                                   DataLayout target_data_layout) const override;

  Status OnRunStart(const RunOptions& run_options) override;

  Status OnRunEnd(bool sync_stream, const RunOptions& run_options) override;

  Status SetEpDynamicOptions(gsl::span<const char* const> keys,
                             gsl::span<const char* const> values) override;

  const InlinedVector<const Node*> GetEpContextNodes() const override;

  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

  void RegisterStreamHandlers(IStreamCommandHandleRegistry&, AllocatorMap&) const override;

  // create per-session allocators
  // longer term we should prefer shared allocators in Environment and only create per-session allocators as
  // needed based on matching against allocator_mem_infos_.
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  std::string GetCompiledModelCompatibilityInfo(const onnxruntime::GraphViewer& graph_viewer) const override;

  Status ValidateCompiledModelCompatibilityInfo(const std::string& compatibility_info,
                                                OrtCompiledModelCompatibility& model_compatibility) const override;

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
  OrtEpFactory& ep_factory_;
  std::vector<const OrtEpDevice*> ep_devices_;
  std::vector<const OrtMemoryInfo*> allocator_mem_infos_;
  bool generate_ep_ctx_model_ = false;

  std::vector<OrtNodeComputeInfo*> api_node_compute_infos_;

  // Fused nodes have to be valid throughout model inference because they may be cached in NodeComputeInfo instances.
  // For each fused node, the Compile() function creates EpNode and EpValueInfo instances on the heap,
  // which are then passed to the underlying OrtEp instance. This class stores this "fused node state"
  // so that it is not destroyed until the EP itself is destroyed.
  std::vector<FusedNodeState> fused_node_states_;

  // Stores the EPContext Nodes created from the OrtNode instances returned by the underlying plugin EP.
  // Need to store both the Node and NodeArg instances so that they are available when the GraphPartitioner
  // calls IExecutionProvider::GetEpContextNodes().
  std::vector<std::unique_ptr<Node>> ep_context_nodes_;
  std::vector<std::unique_ptr<NodeArg>> ep_context_node_args_;

  std::shared_ptr<KernelRegistry> kernel_registry_;
};
}  // namespace onnxruntime
