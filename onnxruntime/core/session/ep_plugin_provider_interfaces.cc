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
#include "core/framework/plugin_data_transfer.h"
#include "core/graph/ep_api_types.h"
#include "core/graph/model_editor_api_types.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_ep_types.h"
#include "core/session/abi_logger.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/allocator_adapters.h"
#include "core/session/ort_apis.h"
#include "core/providers/partitioning_utils.h"

namespace onnxruntime {

//
// PluginExecutionProviderFactory
//

PluginExecutionProviderFactory::PluginExecutionProviderFactory(OrtEpFactory& ep_factory,
                                                               gsl::span<const OrtEpDevice* const> ep_devices)
    : ep_factory_{ep_factory},
      devices_{ep_devices.begin(), ep_devices.end()} {
  hardware_devices_.reserve(ep_devices.size());
  ep_metadata_.reserve(ep_devices.size());

  for (const auto* ep_device : ep_devices) {
    hardware_devices_.push_back(ep_device->device);
    ep_metadata_.push_back(&ep_device->ep_metadata);
  }
}

std::unique_ptr<IExecutionProvider>
PluginExecutionProviderFactory::CreateProvider(const OrtSessionOptions& session_options,
                                               const OrtLogger& session_logger) {
  OrtEp* ort_ep = nullptr;
  Status status = ToStatusAndRelease(ep_factory_.CreateEp(&ep_factory_, hardware_devices_.data(), ep_metadata_.data(),
                                                          hardware_devices_.size(), &session_options, &session_logger,
                                                          &ort_ep));

  if (!status.IsOK()) {
    ORT_THROW("Error creating execution provider: ", status.ToString());
  }

  return std::make_unique<PluginExecutionProvider>(UniqueOrtEp(ort_ep, OrtEpDeleter(ep_factory_)),
                                                   session_options, ep_factory_, devices_,
                                                   *session_logger.ToInternal());
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

static OrtDevice GetOrtDeviceForPluginEp(gsl::span<const OrtEpDevice* const> ep_devices) {
  // Get the OrtDevice from OrtEpDevice.device_memory_info if it is set. Otherwise, we set it to CPU.
  // If there are multiple OrtEpDevice instances, the device_memory_info must be consistent for all.

  ORT_ENFORCE(!ep_devices.empty());  // Should not be possible to create an EP without OrtEpDevices.

  const OrtMemoryInfo* device_memory_info = ep_devices[0]->device_memory_info;

  // Check assertion that all OrtEpDevice instances must have equivalent device_memory_infos
  bool all_match = std::all_of(ep_devices.begin() + 1, ep_devices.end(),
                               [mem_a = device_memory_info](const OrtEpDevice* ep_device) {
                                 const OrtMemoryInfo* mem_b = ep_device->device_memory_info;

                                 if (mem_a == mem_b) {
                                   return true;  // Point to the same OrtMemoryInfo instance.
                                 }

                                 if (mem_a == nullptr || mem_b == nullptr) {
                                   return false;  // One is nullptr and the other is not.
                                 }

                                 // Both non-null but point to different instances. Use operator==.
                                 return *mem_a == *mem_b;
                               });
  if (!all_match) {
    ORT_THROW("Error creating execution provider '", ep_devices[0]->ep_name,
              "': expected all OrtEpDevice instances to use the same device_memory_info.");
  }

  return device_memory_info != nullptr ? device_memory_info->device : OrtDevice();
}

PluginExecutionProvider::PluginExecutionProvider(UniqueOrtEp ep, const OrtSessionOptions& session_options,
                                                 OrtEpFactory& ep_factory,
                                                 gsl::span<const OrtEpDevice* const> ep_devices,
                                                 const logging::Logger& logger)
    : IExecutionProvider(ep->GetName(ep.get()), GetOrtDeviceForPluginEp(ep_devices), logger),
      ort_ep_(std::move(ep)),
      ep_factory_(ep_factory),
      ep_devices_(ep_devices.begin(), ep_devices.end()) {
  generate_ep_ctx_model_ = session_options.value.GetEpContextGenerationOptions().enable;

  for (const auto* ep_device : ep_devices_) {
    if (ep_device->device_memory_info != nullptr) {
      allocator_mem_infos_.push_back(ep_device->device_memory_info);
    }

    if (ep_device->host_accessible_memory_info != nullptr) {
      allocator_mem_infos_.push_back(ep_device->host_accessible_memory_info);
    }
  }
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
  if (Status status = EpGraph::Create(graph_viewer, ep_graph); !status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Failed to create OrtGraph: " << status.ToString();
    return {};
  }

  OrtEpGraphSupportInfo api_graph_support_info(*ep_graph);
  Status status = ToStatusAndRelease(ort_ep_->GetCapability(ort_ep_.get(), ep_graph->ToExternal(), &api_graph_support_info));

  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "OrtEp::GetCapability() failed with error: " << status.ToString();
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

      indexed_sub_graph->nodes.push_back(node_grouping.nodes[0]->GetInternalNode().Index());
      result.push_back(std::make_unique<ComputeCapability>(std::move(indexed_sub_graph)));
    } else if (node_grouping.kind == OrtEpGraphSupportInfo::NodeGroupingKind::kFusedNode) {
      std::unordered_set<const Node*> node_set;
      node_set.reserve(node_grouping.nodes.size());

      for (const EpNode* ep_node : node_grouping.nodes) {
        node_set.insert(&ep_node->GetInternalNode());
      }

      // We now require the OrtEp to only provide individual groups of supported nodes that each maps to exactly
      // one ComputeCapability. Calling utils::CreateSupportedPartitions() may create multiple ComputeCapability
      // instances, and if so, log an error and return.
      //
      // TODO(adrianlizarraga): Do not use the heavy-weight CreateSupportedPartitions just to check if the user
      // provided a single partition. Use utils::MakeCapability() and create a new helper to check that there are no
      // unsupported nodes in any path between supported nodes.
      std::vector<std::unique_ptr<ComputeCapability>> capabilities = utils::CreateSupportedPartitions(
          graph_viewer, node_set, /*stop_ops*/ {}, PluginEpMetaDefNameFunctor(generator, graph_viewer, this->Type()),
          this->Type(), this->Type(), /*node_unit_map*/ nullptr,
          node_grouping.fusion_options.drop_constant_initializers);

      if (capabilities.size() > 1) {
        LOGS_DEFAULT(ERROR) << "OrtEp::GetCapability() set nodes that cannot be fused together. "
                            << "Please ensure that the nodes provided to EpGraphSupportInfo_AddFusedNodes() do not "
                            << "have an unsupported node in any path between two of the supported nodes.";
        return {};
      }

      // Enforce that the nodes in node_set match the nodes in capabilities[0]
      // TODO(adrianlizarraga): This check can be removed when we stop using utils::CreateSupportedPartitions() above.
      std::vector<NodeIndex>& capability_node_indices = capabilities[0]->sub_graph->nodes;
      std::unordered_set<NodeIndex> capability_node_indices_set(capability_node_indices.begin(),
                                                                capability_node_indices.end());

      ORT_ENFORCE(node_set.size() == capability_node_indices_set.size());
      ORT_ENFORCE(std::all_of(node_set.begin(), node_set.end(), [&capability_node_indices_set](const Node* node) {
        return capability_node_indices_set.count(node->Index()) != 0;
      }));

      result.push_back(std::move(capabilities[0]));
    } else {
      LOGS_DEFAULT(ERROR) << "PluginExecutionProvider::GetCapability() has invalid NodeGroupingKind: "
                          << static_cast<int>(node_grouping.kind);
      return {};
    }
  }

  return result;
}

Status PluginExecutionProvider::FusedNodeState::AddFusedNode(const Node& fused_node, /*out*/ EpNode*& added_ep_node) {
  std::unique_ptr<EpNode> unique_ep_fused_node = nullptr;
  ORT_RETURN_IF_ERROR(EpNode::Create(fused_node, /*parent graph*/ nullptr, this->value_infos, unique_ep_fused_node));
  this->nodes.push_back(std::move(unique_ep_fused_node));
  added_ep_node = this->nodes.back().get();
  return Status::OK();
}

/// <summary>
/// Converts the EPContext nodes provided by the plugin EP (OrtNode instances) to onnxruntime::Node instances.
/// Note that the EP plugin uses the model editor API to create the OrtNode instances.
/// </summary>
/// <param name="ep_name">Name of the plugin EP.</param>
/// <param name="fused_nodes">fused nodes provided by ORT.</param>
/// <param name="plugin_ep_context_nodes">EPContext nodes provided by the plugin EP.</param>
/// <param name="result_nodes">Output parameter set to the resulting array of EPContext nodes.</param>
/// <param name="result_node_args">Output parameter that stores the NodeArgs used by the EPContext nodes.</param>
/// <returns>A status indicating success or an error.</returns>
static Status ConvertEpContextNodes(const std::string& ep_name, const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes, const std::vector<OrtNode*> plugin_ep_context_nodes,
                                    /*out*/ std::vector<std::unique_ptr<Node>>& result_nodes,
                                    /*out*/ std::vector<std::unique_ptr<NodeArg>>& result_node_args) {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  if (plugin_ep_context_nodes.empty()) {
    return Status::OK();  // No EPContext nodes.
  }

  std::vector<std::unique_ptr<Node>> ep_context_nodes_holder;
  std::vector<std::unique_ptr<NodeArg>> ep_context_node_args_holder;

  ep_context_nodes_holder.reserve(plugin_ep_context_nodes.size());
  int index = -1;
  for (const OrtNode* ort_node : plugin_ep_context_nodes) {
    ++index;
    auto& fused_node_filtered_graph = fused_nodes[index].filtered_graph;
    ORT_RETURN_IF_NOT(ort_node != nullptr, ep_name, ": OrtEp::Compile() returned a NULL EPContext node.");

    const ModelEditorNode* editor_node = ModelEditorNode::ToInternal(ort_node);
    ORT_RETURN_IF_NOT(editor_node != nullptr, ep_name, ": OrtEp::Compile() returned OrtNode objects ",
                      "that were not created with OrtModelEditorApi.");

    // Create NodeArg for each input/output.
    std::vector<NodeArg*> input_node_args;
    std::vector<NodeArg*> output_node_args;

    input_node_args.reserve(editor_node->input_names.size());
    output_node_args.reserve(editor_node->output_names.size());

    for (const std::string& input_name : editor_node->input_names) {
      auto node_arg_on_fused_graph = fused_node_filtered_graph.get().GetNodeArg(input_name);
      const ONNX_NAMESPACE::TypeProto* p_arg_type = node_arg_on_fused_graph ? node_arg_on_fused_graph->TypeAsProto() : nullptr;
      auto node_arg = std::make_unique<NodeArg>(input_name, p_arg_type);  // Graph.Resolve() cannot set type because EP Context OP does not have proper shape inference function available.
      input_node_args.push_back(node_arg.get());
      ep_context_node_args_holder.push_back(std::move(node_arg));
    }

    for (const std::string& output_name : editor_node->output_names) {
      auto node_arg_on_fused_graph = fused_node_filtered_graph.get().GetNodeArg(output_name);
      const ONNX_NAMESPACE::TypeProto* p_arg_type = node_arg_on_fused_graph ? node_arg_on_fused_graph->TypeAsProto() : nullptr;
      auto node_arg = std::make_unique<NodeArg>(output_name, p_arg_type);  // Graph.Resolve() cannot set type because EP Context OP does not have proper shape infererence function avaliable.
      output_node_args.push_back(node_arg.get());
      ep_context_node_args_holder.push_back(std::move(node_arg));
    }

    // Create a name -> attribute map.
    NodeAttributes attributes;
    attributes.reserve(editor_node->attributes.size());

    for (const ONNX_NAMESPACE::AttributeProto& attr : editor_node->attributes) {
      attributes.emplace(attr.name(), attr);
    }

    // Create Node
    auto internal_node = std::make_unique<Node>(editor_node->node_name,
                                                editor_node->operator_name,
                                                "EPContext node for " + ep_name,
                                                input_node_args,
                                                output_node_args,
                                                &attributes,
                                                editor_node->domain_name);

    ep_context_nodes_holder.push_back(std::move(internal_node));
  }

  result_nodes = std::move(ep_context_nodes_holder);
  result_node_args = std::move(ep_context_node_args_holder);

  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(ep_name);
  ORT_UNUSED_PARAMETER(plugin_ep_context_nodes);
  ORT_UNUSED_PARAMETER(result_nodes);
  ORT_UNUSED_PARAMETER(result_node_args);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Creating EPContext models is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
}

Status PluginExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                        std::vector<NodeComputeInfo>& node_compute_infos) {
  const logging::Logger* logger = GetLogger();
  const size_t num_graphs = fused_nodes_and_graphs.size();
  std::vector<std::unique_ptr<EpGraph>> api_graphs_holder;
  std::vector<const OrtGraph*> api_graphs;
  std::vector<OrtNodeComputeInfo*> api_node_compute_infos(num_graphs, nullptr);
  std::vector<const OrtNode*> api_fused_nodes;

  // Push a new FusedNodeState to store the EpNode instances that we'll create to wrap the original fused nodes.
  // Fused nodes must be valid throughout model inference because they may be cached in NodeComputeInfo instances.
  fused_node_states_.push_back(FusedNodeState());
  FusedNodeState& fused_node_state = fused_node_states_.back();

  fused_node_state.nodes.reserve(num_graphs);
  api_graphs_holder.reserve(num_graphs);
  api_graphs.reserve(num_graphs);
  api_fused_nodes.reserve(num_graphs);
  api_node_compute_infos_.reserve(api_node_compute_infos_.size() + num_graphs);

  // Wrap GraphViewers into OrtGraphs and fused Nodes into OrtNodes.
  for (const FusedNodeAndGraph& node_and_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_viewer = node_and_graph.filtered_graph;
    const Node& fused_node = node_and_graph.fused_node;

    std::unique_ptr<EpGraph> ep_graph = nullptr;
    ORT_RETURN_IF_ERROR(EpGraph::Create(graph_viewer, ep_graph));
    api_graphs.push_back(ep_graph->ToExternal());
    api_graphs_holder.push_back(std::move(ep_graph));

    EpNode* ep_fused_node = nullptr;
    ORT_RETURN_IF_ERROR(fused_node_state.AddFusedNode(fused_node, ep_fused_node));
    api_fused_nodes.push_back(ep_fused_node->ToExternal());
  }

  // Provide an output buffer for the plugin EP to store EPContext nodes if it needs to (i.e., enabled in session options).
  std::vector<std::unique_ptr<OrtNode, decltype(&OrtApis::ReleaseNode)>> plugin_ep_context_nodes_holder;
  std::vector<OrtNode*> plugin_ep_context_nodes;
  plugin_ep_context_nodes_holder.reserve(num_graphs);
  plugin_ep_context_nodes.resize(num_graphs, nullptr);

  Status compile_status = ToStatusAndRelease(ort_ep_->Compile(ort_ep_.get(), api_graphs.data(), api_fused_nodes.data(),
                                                              num_graphs, api_node_compute_infos.data(),
                                                              plugin_ep_context_nodes.data()));

  // Store any EPContext nodes provided by the plugin EP in std::unique_ptr so that they are always properly released.
  for (OrtNode* ort_node : plugin_ep_context_nodes) {
    auto unique_ort_node = std::unique_ptr<OrtNode, decltype(&OrtApis::ReleaseNode)>(ort_node, OrtApis::ReleaseNode);
    plugin_ep_context_nodes_holder.push_back(std::move(unique_ort_node));
  }

  // Save OrtNodeComputeInfo created by OrtEp instance. They're freed when this IExecutionProvider
  // is destroyed.
  for (size_t i = 0; i < num_graphs; i++) {
    if (api_node_compute_infos[i] != nullptr) {
      api_node_compute_infos_.push_back(api_node_compute_infos[i]);
    }
  }

  ORT_RETURN_IF_ERROR(compile_status);

  // Initialize node_compute_infos as wrappers to api_node_compute_infos.
  for (size_t i = 0; i < num_graphs; i++) {
    OrtNodeComputeInfo* api_node_compute_info = api_node_compute_infos[i];
    ORT_RETURN_IF(api_node_compute_info == nullptr, "OrtEp::Compile() did not set a valid OrtNodeComputeInfo ",
                  "instance for graph at index ", i);

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [api_node_compute_info, logger](ComputeContext* context,
                                                                     FunctionState* compute_state) -> int {
      Status status = ToStatusAndRelease(
          api_node_compute_info->CreateState(api_node_compute_info,
                                             reinterpret_cast<OrtNodeComputeContext*>(context),
                                             compute_state));
      const bool success = status.IsOK();
      if (!success) {
        LOGS(*logger, ERROR) << "OrtNodeComputeInfo::CreateComputeState() failed with error: "
                             << status.ErrorMessage();
      }

      return success ? 0 : 1;
    };

    compute_info.release_state_func = [api_node_compute_info](FunctionState compute_state) -> void {
      api_node_compute_info->ReleaseState(api_node_compute_info, compute_state);
    };

    compute_info.compute_func = [api_node_compute_info](FunctionState compute_state,
                                                        const OrtApi* /*c_api*/,
                                                        OrtKernelContext* kernel_context) -> Status {
      ORT_RETURN_IF_ERROR(ToStatusAndRelease((api_node_compute_info->Compute(api_node_compute_info, compute_state,
                                                                             kernel_context))));
      return Status::OK();
    };

    node_compute_infos.push_back(std::move(compute_info));
  }

  // Convert the EPContext nodes provided by the plugin EP into onnxruntime::Node instances.
  // We store the converted Node and NodeArg instances as members to ensure they can be returned to the ORT graph
  // partitioner via a call to IExecutionProvider::GetEpContextNodes().
  if (generate_ep_ctx_model_) {
    ORT_RETURN_IF_ERROR(ConvertEpContextNodes(Type(), fused_nodes_and_graphs, plugin_ep_context_nodes,
                                              /*out*/ ep_context_nodes_, /*out*/ ep_context_node_args_));
  }

  return Status::OK();
}

const InlinedVector<const Node*> PluginExecutionProvider::GetEpContextNodes() const {
  InlinedVector<const Node*> result;

  for (const std::unique_ptr<Node>& node : ep_context_nodes_) {
    result.push_back(node.get());
  }

  return result;
}

namespace {

struct DataLayoutMapping {
  DataLayout data_layout;
  OrtEpDataLayout api_data_layout;
};

// Maps enum values between `onnxruntime::DataLayout` and `OrtEpDataLayout`.
constexpr std::array kDataLayoutMappings{
    DataLayoutMapping{DataLayout::NCHW, OrtEpDataLayout::OrtEpDataLayout_NCHW},
    DataLayoutMapping{DataLayout::NHWC, OrtEpDataLayout::OrtEpDataLayout_NHWC},
};

}  // namespace

DataLayout PluginExecutionProvider::GetPreferredLayout() const {
  if (ort_ep_->GetPreferredDataLayout == nullptr) {
    return Base::GetPreferredLayout();
  }

  OrtEpDataLayout api_data_layout{};

  ORT_THROW_IF_ERROR(ToStatusAndRelease(ort_ep_->GetPreferredDataLayout(ort_ep_.get(), &api_data_layout)));

  const auto data_layout_mapping = std::find_if(kDataLayoutMappings.begin(), kDataLayoutMappings.end(),
                                                [api_data_layout](const DataLayoutMapping& mapping) {
                                                  return mapping.api_data_layout == api_data_layout;
                                                });

  ORT_ENFORCE(data_layout_mapping != kDataLayoutMappings.end(),
              "OrtEp::GetPreferredDataLayout() returned an invalid data layout: ", static_cast<int>(api_data_layout));

  return data_layout_mapping->data_layout;
}

std::optional<bool> PluginExecutionProvider::ShouldConvertDataLayoutForOp(std::string_view node_domain,
                                                                          std::string_view node_op_type,
                                                                          DataLayout target_data_layout) const {
  if (ort_ep_->ShouldConvertDataLayoutForOp == nullptr) {
    return Base::ShouldConvertDataLayoutForOp(node_domain, node_op_type, target_data_layout);
  }

  const auto data_layout_mapping = std::find_if(kDataLayoutMappings.begin(), kDataLayoutMappings.end(),
                                                [target_data_layout](const DataLayoutMapping& mapping) {
                                                  return mapping.data_layout == target_data_layout;
                                                });

  ORT_ENFORCE(data_layout_mapping != kDataLayoutMappings.end(),
              "Unable to map target_data_layout (", static_cast<int>(target_data_layout), ") to OrtEpDataLayout.");

  // Ensure domain and op type strings are null-terminated.
  const std::string node_domain_str{node_domain}, node_op_type_str{node_op_type};
  int should_convert = -1;

  ORT_THROW_IF_ERROR(ToStatusAndRelease(
      ort_ep_->ShouldConvertDataLayoutForOp(ort_ep_.get(),
                                            node_domain_str.c_str(), node_op_type_str.c_str(),
                                            data_layout_mapping->api_data_layout,
                                            &should_convert)));

  if (should_convert > 0) {
    return true;
  } else if (should_convert == 0) {
    return false;
  } else {
    return std::nullopt;
  }
}

Status PluginExecutionProvider::OnRunStart(const RunOptions& run_options) {
  if (ort_ep_->OnRunStart == nullptr) {
    return Base::OnRunStart(run_options);
  }

  return ToStatusAndRelease(ort_ep_->OnRunStart(ort_ep_.get(), &run_options));
}

Status PluginExecutionProvider::OnRunEnd(bool sync_stream, const RunOptions& run_options) {
  if (ort_ep_->OnRunEnd == nullptr) {
    return Base::OnRunEnd(sync_stream, run_options);
  }

  return ToStatusAndRelease(ort_ep_->OnRunEnd(ort_ep_.get(), &run_options, sync_stream));
}

Status PluginExecutionProvider::SetEpDynamicOptions(gsl::span<const char* const> keys,
                                                    gsl::span<const char* const> values) {
  if (ort_ep_->SetDynamicOptions == nullptr) {
    return Base::SetEpDynamicOptions(keys, values);
  }

  ORT_RETURN_IF_NOT(keys.size() == values.size(),
                    "The number of keys (", keys.size(), ") and number of values (", values.size(),
                    ") must be the same.");

  return ToStatusAndRelease(ort_ep_->SetDynamicOptions(ort_ep_.get(), keys.data(), values.data(), keys.size()));
}
std::unique_ptr<onnxruntime::IDataTransfer> PluginExecutionProvider::GetDataTransfer() const {
  OrtDataTransferImpl* data_transfer_impl = nullptr;
  OrtStatus* status = ep_factory_.CreateDataTransfer(&ep_factory_, &data_transfer_impl);
  if (status != nullptr) {
    ORT_THROW("Error creating data transfer: ", ToStatusAndRelease(status).ToString());
  }

  if (data_transfer_impl == nullptr) {
    return {};
  }

  return std::make_unique<plugin_ep::DataTransfer>(*data_transfer_impl);
}

std::vector<AllocatorPtr> PluginExecutionProvider::CreatePreferredAllocators() {
  std::vector<AllocatorPtr> allocators;
  allocators.reserve(allocator_mem_infos_.size());

  for (const auto* memory_info : allocator_mem_infos_) {
    OrtAllocator* ort_allocator_ptr = nullptr;
    OrtStatus* ort_status = ep_factory_.CreateAllocator(&ep_factory_, memory_info, nullptr, &ort_allocator_ptr);

    // throw or log? start with throw
    if (ort_status != nullptr) {
      ORT_THROW("Error creating allocator: ", ToStatusAndRelease(ort_status).ToString());
    }

    auto ort_allocator = OrtAllocatorUniquePtr(
        ort_allocator_ptr,
        [this](OrtAllocator* allocator) {
          ep_factory_.ReleaseAllocator(&ep_factory_, allocator);
        });
    allocators.push_back(std::make_shared<IAllocatorImplWrappingOrtAllocator>(std::move(ort_allocator)));
  }

  return allocators;
}

}  // namespace onnxruntime
