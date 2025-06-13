// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_plugin_provider_interfaces.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "core/framework/compute_capability.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/graph/ep_api_types.h"
#include "core/session/ort_apis.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_ep_types.h"
#include "core/session/abi_logger.h"
#include "core/session/allocator_adapters.h"
#include "core/providers/partitioning_utils.h"

namespace onnxruntime {

namespace {
// TODO(adrianlizarraga): Use better solution when PR 25036 is merged.
using UniqueOrtStatus = std::unique_ptr<OrtStatus, decltype(&OrtApis::ReleaseStatus)>;
}  // namespace

//
// PluginExecutionProviderFactory
//

PluginExecutionProviderFactory::PluginExecutionProviderFactory(OrtEpFactory& ep_factory,
                                                               gsl::span<const OrtEpDevice* const> ep_devices)
    : ep_factory_{ep_factory} {
  devices_.reserve(ep_devices.size());
  ep_metadata_.reserve(ep_devices.size());

  for (const auto* ep_device : ep_devices) {
    devices_.push_back(ep_device->device);
    ep_metadata_.push_back(&ep_device->ep_metadata);
  }
}

std::unique_ptr<IExecutionProvider>
PluginExecutionProviderFactory::CreateProvider(const OrtSessionOptions& session_options,
                                               const OrtLogger& session_logger) {
  OrtEp* ort_ep = nullptr;
  UniqueOrtStatus status(ep_factory_.CreateEp(&ep_factory_, devices_.data(), ep_metadata_.data(), devices_.size(),
                                              &session_options, &session_logger, &ort_ep),
                         OrtApis::ReleaseStatus);

  if (status != nullptr) {
    ORT_THROW("Error creating execution provider: ", ToStatus(status.get()).ToString());
  }

  auto ep_wrapper = std::make_unique<PluginExecutionProvider>(UniqueOrtEp(ort_ep, OrtEpDeleter(ep_factory_)));
  ep_wrapper->SetLogger(session_logger.ToInternal());

  return ep_wrapper;
}

/// <summary>
/// Functor used to generate a Metadef name for a subgraph supported by a plugin EP.
/// The generated name is a concatenation of a prefix (i.e., the EP name) with
/// the model's hash and a unique ID.
/// </summary>
struct PluginEpMetaDefNameFunctor {
  explicit PluginEpMetaDefNameFunctor(const ModelMetadefIdGenerator& generator,
                                      const GraphViewer& graph_viewer,
                                      const std::string& prefix)
      : generator_(generator), graph_viewer_(graph_viewer), prefix_(prefix) {}

  std::string operator()() {
    uint64_t model_hash = 0;
    int id = generator_.GenerateId(graph_viewer_, model_hash);
    return MakeString(prefix_, "_", model_hash, "_", id);
  }

  const ModelMetadefIdGenerator& generator_;
  const GraphViewer& graph_viewer_;
  const std::string& prefix_;
};

//
// PluginExecutionProvider
//

PluginExecutionProvider::PluginExecutionProvider(UniqueOrtEp ep)
    : IExecutionProvider(ep->GetName(ep.get()), OrtDevice()),  // TODO: What to do about OrtDevice for plugins?
      ort_ep_(std::move(ep)) {
}

PluginExecutionProvider::~PluginExecutionProvider() {
  if (ort_ep_ && !api_node_compute_infos_.empty()) {
    ort_ep_->ReleaseNodeComputeInfos(ort_ep_.get(), api_node_compute_infos_.data(),
                                     api_node_compute_infos_.size());
  }
}

std::vector<std::unique_ptr<ComputeCapability>>
PluginExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& kernel_lookup,
                                       const GraphOptimizerRegistry& graph_optimizer_registry,
                                       IResourceAccountant* resource_accountant) const {
  ORT_UNUSED_PARAMETER(graph_optimizer_registry);  // TODO: Add support
  ORT_UNUSED_PARAMETER(resource_accountant);       // TODO: Add support? Not used by prioritized EPs
  ORT_UNUSED_PARAMETER(kernel_lookup);             // TODO: Add support? Not used by prioritized EPs, so probably not needed?

  std::unique_ptr<EpGraph> ep_graph = nullptr;
  if (Status status = EpGraph::Create(graph_viewer, /*parent_node*/ nullptr, ep_graph); !status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Failed to create OrtGraph: " << status.ToString();
    return {};
  }

  OrtEpGraphSupportInfo api_graph_support_info(*ep_graph);
  UniqueOrtStatus status(ort_ep_->GetCapability(ort_ep_.get(), ep_graph->ToExternal(), &api_graph_support_info),
                         OrtApis::ReleaseStatus);

  if (status != nullptr) {
    LOGS_DEFAULT(ERROR) << "OrtEp::GetCapability() failed with error: " << ToStatus(status.get()).ToString();
    return {};
  }

  std::vector<std::unique_ptr<ComputeCapability>> result;
  result.reserve(api_graph_support_info.node_groupings.size());
  if (api_graph_support_info.node_groupings.empty()) {
    return {};
  }

  ModelMetadefIdGenerator generator;

  // Create ComputeCapability instances from OrtEpGraphSupportInfo::NodeGrouping instances.
  for (const OrtEpGraphSupportInfo::NodeGrouping& node_grouping : api_graph_support_info.node_groupings) {
    if (node_grouping.kind == OrtEpGraphSupportInfo::NodeGroupingKind::kSingleAssignedNode) {
      auto indexed_sub_graph = std::make_unique<IndexedSubGraph>();

      indexed_sub_graph->nodes.push_back(node_grouping.nodes[0]->node.Index());
      result.push_back(std::make_unique<ComputeCapability>(std::move(indexed_sub_graph)));
    } else if (node_grouping.kind == OrtEpGraphSupportInfo::NodeGroupingKind::kFusedNode) {
      std::unordered_set<const Node*> node_set;
      node_set.reserve(node_grouping.nodes.size());
      for (const EpNode* ep_node : node_grouping.nodes) {
        node_set.insert(&ep_node->node);
      }

      // We now require the OrtEp to only provide individual groups of supported nodes that each maps to exactly
      // one ComputeCapability. Calling utils::CreateSupportedPartitions() may create multiple ComputeCapability
      // instances, and if so, log an error and return.
      std::vector<std::unique_ptr<ComputeCapability>> capabilities = utils::CreateSupportedPartitions(
          graph_viewer, node_set, /*stop_ops*/ {}, PluginEpMetaDefNameFunctor(generator, graph_viewer, this->Type()),
          this->Type(), this->Type(), /*node_unit_map*/ nullptr);

      if (capabilities.size() > 1) {
        LOGS_DEFAULT(ERROR) << "OrtEp::GetCapability() set nodes that cannot be fused together. "
                            << "Please ensure that the nodes provided to EpGraphSupportInfo_AddFusedNodes() do not "
                            << "have an unsupported node in any path between two of the supported nodes.";
        return {};
      }

      for (auto& capability : capabilities) {
        result.push_back(std::move(capability));
      }
    } else {
      LOGS_DEFAULT(ERROR) << "PluginExecutionProvider::GetCapability() has invalid NodeGroupingKind: "
                          << static_cast<int>(node_grouping.kind);
      return {};
    }
  }

  return result;
}

PluginExecutionProvider::FusedNodeState& PluginExecutionProvider::PushFusedNodeState(size_t num_fused_nodes) {
  fused_node_states_.push_back(FusedNodeState());
  FusedNodeState& fused_node_state = fused_node_states_.back();
  fused_node_state.nodes.reserve(num_fused_nodes);
  return fused_node_state;
}

Status PluginExecutionProvider::FusedNodeState::AddFusedNode(const Node& fused_node, /*out*/ EpNode* added_ep_node) {
  std::unique_ptr<EpNode> unique_ep_fused_node = nullptr;
  ORT_RETURN_IF_ERROR(EpNode::Create(fused_node, /*parent graph*/ nullptr, this->value_infos, unique_ep_fused_node));
  this->nodes.push_back(std::move(unique_ep_fused_node));
  added_ep_node = this->nodes.back().get();
  return Status::OK();
}

common::Status PluginExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                std::vector<NodeComputeInfo>& node_compute_infos) {
  const logging::Logger* logger = GetLogger();
  const size_t num_graphs = fused_nodes_and_graphs.size();
  std::vector<std::unique_ptr<EpGraph>> api_graphs_holder;
  std::vector<const OrtGraph*> api_graphs;
  std::vector<OrtNodeComputeInfo*> api_node_compute_infos(num_graphs, nullptr);
  std::vector<const OrtNode*> api_fused_nodes;
  FusedNodeState& fused_node_state = PushFusedNodeState(num_graphs);

  api_graphs_holder.reserve(num_graphs);
  api_graphs.reserve(num_graphs);
  api_fused_nodes.reserve(num_graphs);
  api_node_compute_infos_.reserve(api_node_compute_infos_.size() + num_graphs);

  // Wrap GraphViewers into OrtGraphs and fused Nodes into OrtNodes.
  for (const FusedNodeAndGraph& node_and_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_viewer = node_and_graph.filtered_graph;
    const Node& fused_node = node_and_graph.fused_node;

    std::unique_ptr<EpGraph> ep_graph = nullptr;
    ORT_RETURN_IF_ERROR(EpGraph::Create(graph_viewer, /*parent_node*/ nullptr, ep_graph));
    api_graphs.push_back(ep_graph->ToExternal());
    api_graphs_holder.push_back(std::move(ep_graph));

    EpNode* ep_fused_node = nullptr;
    ORT_RETURN_IF_ERROR(fused_node_state.AddFusedNode(fused_node, ep_fused_node));
    api_fused_nodes.push_back(ep_fused_node->ToExternal());
  }

  UniqueOrtStatus status(ort_ep_->Compile(ort_ep_.get(), api_graphs.data(), api_fused_nodes.data(), num_graphs,
                                          api_node_compute_infos.data()),
                         OrtApis::ReleaseStatus);
  ORT_RETURN_IF_ERROR(ToStatus(status.get()));

  // Save OrtNodeComputeInfo created by OrtEp instance. They're freed when this IExecutionProvider
  // is destroyed.
  for (size_t i = 0; i < num_graphs; i++) {
    if (api_node_compute_infos[i] != nullptr) {
      api_node_compute_infos_.push_back(api_node_compute_infos[i]);
    }
  }

  // Initialize node_compute_infos as wrappers to api_node_compute_infos.
  for (size_t i = 0; i < num_graphs; i++) {
    OrtNodeComputeInfo* api_node_compute_info = api_node_compute_infos[i];
    ORT_RETURN_IF(api_node_compute_info == nullptr, "OrtEp::Compile() did not set a valid OrtNodeComputeInfo ",
                  "instance for graph at index ", i);

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [api_node_compute_info, logger](ComputeContext* context,
                                                                     FunctionState* compute_state) -> int {
      UniqueOrtStatus status(api_node_compute_info->CreateComputeState(api_node_compute_info,
                                                                       reinterpret_cast<OrtNodeComputeContext*>(context),
                                                                       compute_state),
                             OrtApis::ReleaseStatus);
      const bool success = status == nullptr;
      if (!success) {
        LOGS(*logger, ERROR) << "OrtNodeComputeInfo::CreateComputeState() failed with error: "
                             << ToStatus(status.get()).ErrorMessage();
      }
      return success ? 0 : 1;
    };

    compute_info.release_state_func = [api_node_compute_info](FunctionState compute_state) -> void {
      api_node_compute_info->DestroyComputeState(api_node_compute_info, compute_state);
    };

    compute_info.compute_func = [api_node_compute_info](FunctionState compute_state,
                                                        const OrtApi* /*c_api*/,
                                                        OrtKernelContext* kernel_context) -> Status {
      UniqueOrtStatus status(api_node_compute_info->Compute(api_node_compute_info, compute_state, kernel_context),
                             OrtApis::ReleaseStatus);
      return ToStatus(status.get());
    };

    node_compute_infos.push_back(std::move(compute_info));
  }

  return Status::OK();
}
}  // namespace onnxruntime
