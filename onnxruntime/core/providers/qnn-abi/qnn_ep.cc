#include "qnn_ep.h"

#include "qnn_ep_factory.h"

#include "core/providers/qnn-abi/ort_api.h"
#include <unordered_map>
#include <vector>
#include <memory>
#include <unordered_set>
#include <iostream>

namespace onnxruntime {

QnnEp::QnnEp(const QnnEpFactory& factory, const std::string& name,
           const Config& config, const OrtLogger* logger)
    : OrtEp{},
    ApiPtrs{static_cast<const ApiPtrs&>(factory)},
    factory_{factory},
    name_{name},
    config_{config},
    logger_{logger},
    context_cache_enabled_{config.enable_ep_context},
    share_ep_contexts_{config.share_ep_contexts}{
        std::cout << "DEBUG: QnnEp constructor called with name: " << name << std::endl;
        GetName = GetNameImpl;
        GetCapability = GetCapabilityImpl;
}
QnnEp::~QnnEp() = default;

const char* ORT_API_CALL QnnEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* qnn_ep = static_cast<const QnnEp*>(this_ptr);
  return qnn_ep->name_.c_str();
}

OrtStatus* ORT_API_CALL QnnEp::GetCapabilityImpl(OrtEp* this_ptr,
                                                const OrtGraph* graph,
                                                OrtEpGraphSupportInfo* graph_support_info) {

    QnnEp* ep = static_cast<QnnEp*>(this_ptr);

    //   std::vector<std::unique_ptr<ComputeCapability>> result;

//   if (graph_viewer.IsSubgraph()) {
//     return result;
//   }
//   const size_t num_nodes_in_graph = static_cast<size_t>(graph_viewer.NumberOfNodes());

//   const auto& logger = *GetLogger();
//   bool is_qnn_ctx_model = qnn::GraphHasEpContextNode(graph_viewer);

//   const auto gen_metadef_name = [&]() {
//     uint64_t model_hash;
//     int metadef_id = metadef_id_generator_->GenerateId(graph_viewer, model_hash);
//     return MakeString(QNN, context_node_name_prefix_, "_", model_hash, "_", metadef_id);
//   };

//   // share ep contexts is enabled
//   // check the ep_shared_contexts to see if it contains all the graphs in the context model
//   // directly use the resource from ep_shared_contexts if it has all the graphs needed by the current session
//   // no need to setup QNN backend
//   if (is_qnn_ctx_model && share_ep_contexts_ && SharedContext::GetInstance().HasSharedQnnModels()) {
//     if (EpSharedContextsHasAllGraphs(graph_viewer, logger)) {
//       PartitionCtxModel(graph_viewer, num_nodes_in_graph, result, gen_metadef_name, logger);
//       return result;
//     }
//   }

//   std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> context_bin_map;
//   if (enable_vtcm_backup_buffer_sharing_) {
//     std::unordered_set<const Node*> ep_ctx_nodes;
//     GetMainEPCtxNodes(graph_viewer, ep_ctx_nodes, logger);

//     onnxruntime::PathString context_model_path;
//     GetContextOnnxModelFilePath(context_cache_path_cfg_, graph_viewer.ModelPath().native(), context_model_path);

//     std::filesystem::path parent_path = std::filesystem::path(context_model_path).parent_path();

//     for (auto& ep_ctx_node : ep_ctx_nodes) {
//       NodeAttrHelper node_helper(*ep_ctx_node);
//       std::string context_bin_filepath(parent_path.string());
//       context_bin_filepath.append("/").append(node_helper.Get(qnn::EP_CACHE_CONTEXT, ""));

//       if (context_bin_map.find(context_bin_filepath) == context_bin_map.end()) {
//         context_bin_map.emplace(context_bin_filepath, std::make_unique<std::vector<std::string>>());
//         // Push context bin filepath for lookup between sessions
//         context_bin_map.at(context_bin_filepath)->push_back(context_bin_filepath);
//       }
//       context_bin_map.at(context_bin_filepath)->push_back(ep_ctx_node->Name());
//     }
//   }

//   // It will load the QnnSystem lib if is_qnn_ctx_model=true, and
//   // delay the Qnn context creation to Compile() using the cached context binary
//   // or generate context cache enable, need to use use QnnSystem lib to parse the binary to get the max spill fill buffer size
//   auto rt = qnn_backend_manager_->SetupBackend(logger, is_qnn_ctx_model,
//                                                context_cache_enabled_ && enable_spill_fill_buffer_,
//                                                share_ep_contexts_,
//                                                enable_vtcm_backup_buffer_sharing_,
//                                                context_bin_map);

//   context_bin_map.clear();

//   if (Status::OK() != rt) {
//     LOGS(logger, ERROR) << "QNN SetupBackend failed " << rt.ErrorMessage();
//     return result;
//   }

//   if (IsNpuBackend(qnn_backend_manager_->GetQnnBackendType())) {
//     // Set the power config id and the default power mode from provider option for main thread,
//     // otherwise it will mess up the power mode if user just create session without run it.
//     GetPerThreadContext();
//   }

//   // Report error if QNN CPU backend is loaded while CPU fallback is disabled
//   if (disable_cpu_ep_fallback_ && qnn_backend_manager_->GetQnnBackendType() == qnn::QnnBackendType::CPU) {
//     LOGS(logger, ERROR) << "Qnn CPU backend is loaded while CPU fallback is disabled.";
//     return result;
//   }

//   if ((context_cache_enabled_ || is_qnn_ctx_model) && !IsQpuBackend(qnn_backend_manager_->GetQnnBackendType())) {
//     LOGS(logger, ERROR) << "Qnn context cache only works for HTP/DSP/GPU backend.";
//     return result;
//   }

//   // For model with EPContext, make sure each partition only has one single EPContext node
//   if (is_qnn_ctx_model) {
//     PartitionCtxModel(graph_viewer, num_nodes_in_graph, result, gen_metadef_name, logger);
//     return result;
//   }

//   // Get all the NodeUnits in the graph_viewer
//   std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
//   std::unordered_map<const Node*, const NodeUnit*> node_unit_map;

//   std::tie(node_unit_holder, node_unit_map) = GetQDQNodeUnits(graph_viewer, logger);

//   // remove is_qnn_ctx_model related code
//   const auto supported_nodes = GetSupportedNodes(graph_viewer, node_unit_map,
//                                                  node_unit_holder.size(), logger);

//   // Helper function that returns a string that lists all unsupported nodes.
//   // Ex: { name: mul_123, type: Mul }, {}, ...
//   auto get_unsupported_node_names = [&node_unit_holder, &supported_nodes]() -> std::string {
//     std::stringstream ss;
//     const size_t num_node_units = node_unit_holder.size();

//     for (size_t i = 0; i < num_node_units; ++i) {
//       const auto& node_unit = node_unit_holder[i];

//       if (supported_nodes.find(&node_unit->GetNode()) == supported_nodes.end()) {
//         ss << "{ name: " << node_unit->Name() << ", type: " << node_unit->OpType() << " }";
//         if (i == num_node_units - 1) {
//           ss << ", ";
//         }
//       }
//     }

//     return ss.str();
//   };

//   if (supported_nodes.empty()) {
//     LOGS(logger, INFO) << "Number of partitions supported by QNN EP: 0";
//     return result;
//   }

//   size_t num_of_supported_nodes = 0;

//   // Create partitions from supported nodes.
//   std::vector<std::unique_ptr<ComputeCapability>> partitions = utils::CreateSupportedPartitions(
//       graph_viewer, supported_nodes, {}, gen_metadef_name, QNN, kQnnExecutionProvider, &node_unit_map);

//   // Filter out partitions that consist of a single QuantizeLinear or DequantizeLinear node.
//   // We also count the number of supported nodes in all valid partitions.
//   for (auto& partition : partitions) {
//     bool is_valid_partition = true;
//     size_t nodes_in_partition = 0;

//     if (partition && ComputeCapability__SubGraph(*partition)) {
//       const auto& subgraph = ComputeCapability__SubGraph(*partition);
//       const auto& subgraph_nodes = IndexedSubGraph__Nodes(*subgraph);

//       nodes_in_partition = subgraph_nodes.size();

//       if (nodes_in_partition == 1 && !is_qnn_ctx_model) {
//         const Node* node = graph_viewer.GetNode(subgraph_nodes[0]);

//         if (!node) {
//           LOGS(logger, ERROR) << "QNN EP: Invalid node in partition of one node.";
//           is_valid_partition = false;
//         } else if (node->OpType() == "QuantizeLinear" || node->OpType() == "DequantizeLinear") {
//           LOGS(logger, WARNING) << "QNN EP does not support a single Quantize/Dequantize node in a partition.";
//           is_valid_partition = false;
//         }
//       }
//     } else {
//       LOGS(logger, ERROR) << "QNN EP: Invalid partition.";
//       is_valid_partition = false;
//     }

//     if (is_valid_partition) {
//       result.push_back(std::move(partition));
//       num_of_supported_nodes += nodes_in_partition;
//     }
//   }  // for

//   const size_t num_of_partitions = result.size();
//   const auto summary_msg = MakeString("Number of partitions supported by QNN EP: ", num_of_partitions,
//                                       ", number of nodes in the graph: ", num_nodes_in_graph,
//                                       ", number of nodes supported by QNN: ", num_of_supported_nodes);
//   LOGS(logger, INFO) << summary_msg;

//   // Print list of unsupported nodes to the ERROR logger if the CPU EP
//   // has been disabled for this inference session.
//   if (!is_qnn_ctx_model && disable_cpu_ep_fallback_ && num_nodes_in_graph != num_of_supported_nodes) {
//     LOGS(logger, ERROR) << "Unsupported nodes in QNN EP: " << get_unsupported_node_names();
//   }

    std::vector<const OrtNode*> supported_nodes = {};
    graph;
        // qnn::QnnEpBridge::ValidateNodesWithIsOpSupported(graph, ep->ort_api, ep->logger_, graph_support_info);

    if (supported_nodes.empty()) {
        return nullptr;
    }

    OrtNodeFusionOptions node_fusion_options = {};
    node_fusion_options.ort_version_supported = ORT_API_VERSION;
    node_fusion_options.drop_constant_initializers = true;
    RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                                                    supported_nodes.data(),
                                                                    supported_nodes.size(),
                                                                    &node_fusion_options));
    return nullptr;

}

}
