// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn-abi/qnn_ep.h"

#include <unordered_map>
#include <vector>
#include <memory>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <optional>

#include "HTP/QnnHtpGraph.h"

#include "core/providers/qnn-abi/qnn_ep_factory.h"
// #include "core/providers/qnn-abi/shared_context.h"
#include "core/providers/qnn-abi/builder/qnn_backend_manager.h"
#include "core/providers/qnn-abi/builder/qnn_configs_helper.h"
#include "core/providers/qnn-abi/builder/qnn_model.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn-abi/qnn_ep_utils.h"

// Forward declarations for NodeUnit-related classes
namespace onnxruntime {

QnnEp::QnnEp(QnnEpFactory& factory, const std::string& name,
             const Config& config, const OrtLogger& logger)
    : OrtEp{},
      ApiPtrs{static_cast<const ApiPtrs&>(factory)},
      factory_{factory},
      name_{name},
      config_{config},
      logger_{logger} {
  // context_cache_enabled_{config.enable_ep_context},
  // share_ep_contexts_{config.share_ep_contexts},
  // enable_vtcm_backup_buffer_sharing_{config.enable_vtcm_backup_buffer_sharing},
  // context_node_name_prefix_{""},
  // context_cache_path_cfg_{""}{
  std::cout << "DEBUG: QnnEp constructor called with name: " << name << std::endl;
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;

  // Initialize the backend manager
  qnn::QnnBackendManagerConfig backend_config;
  backend_config.backend_path = "QnnCpu.dll";  // Default backend path
  backend_config.profiling_file_path = "";
  backend_config.device_id = 0;
  backend_config.soc_model = 0;
  qnn_backend_manager_ = qnn::QnnBackendManager::Create(backend_config, *logger.ToInternal());
}

QnnEp::~QnnEp() = default;

const char* ORT_API_CALL QnnEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* qnn_ep = static_cast<const QnnEp*>(this_ptr);
  return qnn_ep->name_.c_str();
}

// // Logs information about the supported/unsupported nodes.
// static void LogNodeSupport(const logging::Logger& logger,
//                            logging::Severity log_severity,
//                            logging::DataType log_data_type,
//                            const onnxruntime::CodeLocation& call_site,
//                            const qnn::IQnnNodeGroup& qnn_node_group,
//                            Status support_status) {
//   if (!logger.OutputIsEnabled(log_severity, log_data_type)) {
//     return;
//   }

//   size_t num_nodes = 0;
//   std::ostringstream oss;
//   for (const NodeUnit* node_unit : qnn_node_group.GetNodeUnits()) {
//     for (const Node* node : node_unit->GetAllNodesInGroup()) {
//       oss << "\tOperator type: " << node->OpType()
//           << " Node name: " << node->Name()
//           << " Node index: " << node->Index() << std::endl;
//       num_nodes += 1;
//     }
//   }
//   if (!support_status.IsOK()) {
//     oss << "\tREASON : " << support_status.ErrorMessage() << std::endl;
//   }

//   auto log_capture = Factory<logging::Capture>::Create(logger, log_severity,
//                                                        logging::Category::onnxruntime,
//                                                        log_data_type, call_site);
//   log_capture->Stream()
//       << (support_status.IsOK() ? "Validation PASSED " : "Validation FAILED ") << "for " << num_nodes
//       << " nodes in " << qnn_node_group.Type() << " (" << qnn_node_group.GetTargetNodeUnit()->OpType() << ") :"
//       << std::endl
//       << oss.str();
// }

OrtStatus* QnnEp::GetSupportedNodes(OrtEp* this_ptr,
                                    const OrtGraph* graph,
                                    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                    const size_t node_unit_size,
                                    const logging::Logger& logger,
                                    std::vector<const OrtNode*>& supported_nodes) const {
  const QnnEp* ep = static_cast<const QnnEp*>(this_ptr);

  OrtArrayOfConstObjects* graph_inputs = nullptr;
  OrtArrayOfConstObjects* graph_outputs = nullptr;
  ep->ort_api.Graph_GetInputs(graph, &graph_inputs);
  ep->ort_api.Graph_GetOutputs(graph, &graph_outputs);

  // Util function that initializes a table that maps a graph input or output name to its index.
  auto init_input_output_index_map = [&](std::unordered_map<std::string, size_t>& index_map,
                                         OrtArrayOfConstObjects* inouts) {
    size_t num_elements;
    ep->ort_api.ArrayOfConstObjects_GetSize(inouts, &num_elements);
    for (size_t idx = 0; idx < num_elements; ++idx) {
      const void* inout = nullptr;
      ep->ort_api.ArrayOfConstObjects_GetElementAt(inouts, idx, &inout);
      const char* name = nullptr;
      ep->ort_api.GetValueInfoName(static_cast<const OrtValueInfo*>(inout), &name);

      index_map.emplace(name, idx);
    }
  };

  std::unordered_map<std::string, size_t> model_input_index_map;
  // TODO: Handle initializers as inputs.
  init_input_output_index_map(model_input_index_map, graph_inputs);

  std::unordered_map<std::string, size_t> model_output_index_map;
  init_input_output_index_map(model_output_index_map, graph_outputs);

  auto qnn_model_wrapper = qnn::QnnModelWrapper(*graph,
                                                ApiPtrs{ep->ort_api, ep->ep_api, ep->model_editor_api},
                                                logger,
                                                qnn_backend_manager_->GetQnnInterface(),
                                                qnn_backend_manager_->GetQnnBackendHandle(),
                                                model_input_index_map,
                                                model_output_index_map,
                                                qnn_backend_manager_->GetQnnBackendType(),
                                                model_settings_);

  std::vector<std::unique_ptr<qnn::IQnnNodeGroup>> qnn_node_groups;
  qnn_node_groups.reserve(node_unit_size);

  Status status = qnn::GetQnnNodeGroups(qnn_node_groups, qnn_model_wrapper, node_unit_map, node_unit_size, logger);
  if (!status.IsOK()) {
    return this->ort_api.CreateStatus(ORT_EP_FAIL, status.ErrorMessage().c_str());
  }

  for (const std::unique_ptr<qnn::IQnnNodeGroup>& qnn_node_group : qnn_node_groups) {
    const bool supported = qnn_node_group->IsSupported(qnn_model_wrapper, logger).IsOK();

    // constexpr auto log_severity = logging::Severity::kINFO;
    // constexpr auto log_data_type = logging::DataType::SYSTEM;
    // if (logger.OutputIsEnabled(log_severity, log_data_type)) {
    //   LogNodeSupport(logger, log_severity, log_data_type, ORT_WHERE, *qnn_node_group, status);
    // }

    if (supported) {
      for (const OrtNodeUnit* node_unit : qnn_node_group->GetNodeUnits()) {
        for (const OrtNode* node : node_unit->GetAllNodesInGroup()) {
          supported_nodes.push_back(node);
        }
      }
    }
  }

  ep->ort_api.ReleaseArrayOfConstObjects(graph_inputs);
  ep->ort_api.ReleaseArrayOfConstObjects(graph_outputs);
  return nullptr;
}

void QnnEp::InitQnnHtpGraphConfigs(
    qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t>& configs_builder) const {
  if (qnn_backend_manager_->GetQnnBackendType() == qnn::QnnBackendType::HTP) {
    if (htp_graph_finalization_opt_mode_ != qnn::HtpGraphFinalizationOptimizationMode::kDefault) {
      gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_opt_config = configs_builder.PushCustomConfig();
      htp_graph_opt_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
      htp_graph_opt_config->optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
      htp_graph_opt_config->optimizationOption.floatValue = static_cast<float>(htp_graph_finalization_opt_mode_);

      gsl::not_null<QnnGraph_Config_t*> graph_opt_config = configs_builder.PushConfig();
      graph_opt_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_opt_config->customConfig = htp_graph_opt_config;
    }

    if (vtcm_size_in_mb_ > 0) {
      gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_opt_config_vtcm = configs_builder.PushCustomConfig();
      htp_graph_opt_config_vtcm->option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
      htp_graph_opt_config_vtcm->vtcmSizeInMB = static_cast<uint32_t>(vtcm_size_in_mb_);

      gsl::not_null<QnnGraph_Config_t*> graph_opt_config_vtcm = configs_builder.PushConfig();
      graph_opt_config_vtcm->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_opt_config_vtcm->customConfig = htp_graph_opt_config_vtcm;
    }

    if (enable_HTP_FP16_precision_) {
      gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_precision_config = configs_builder.PushCustomConfig();
      htp_graph_precision_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
      htp_graph_precision_config->precision = QNN_PRECISION_FLOAT16;

      gsl::not_null<QnnGraph_Config_t*> graph_precision_config = configs_builder.PushConfig();
      graph_precision_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_precision_config->customConfig = htp_graph_precision_config;
    }
  }
}

// bool QnnEp::EpSharedContextsHasAllGraphs(const OrtGraph* graph) {
//     OrtArrayOfConstObjects* graph_nodes = nullptr;
//     if (ort_api.Graph_GetNodes(graph, &graph_nodes) != nullptr) {
//         return false;
//     }

//     size_t num_nodes = 0;
//     if (ort_api.ArrayOfConstObjects_GetSize(graph_nodes, &num_nodes) != nullptr) {
//         ort_api.ReleaseArrayOfConstObjects(graph_nodes);
//         return false;
//     }

//     const void* const* node_data = nullptr;
//     if (ort_api.ArrayOfConstObjects_GetData(graph_nodes, &node_data) != nullptr) {
//         ort_api.ReleaseArrayOfConstObjects(graph_nodes);
//         return false;
//     }

//     bool all_graphs_found = true;

//     for (size_t i = 0; i < num_nodes; ++i) {
//         const OrtNode* node = static_cast<const OrtNode*>(node_data[i]);
//         const char* op_type = nullptr;

//         if (ort_api.Node_GetOperatorType(node, &op_type) == nullptr && op_type != nullptr) {
//             if (std::string(op_type) == "EPContext") {
//                 // Check the 'source' attribute to verify it's from QNN
//                 const OrtOpAttr* source_attr = nullptr;
//                 if (ort_api.Node_GetAttributeByName(node, "source", &source_attr) == nullptr && source_attr != nullptr) {
//                     char source_buffer[256] = {0};
//                     size_t source_len = 0;
//                     if (ort_api.ReadOpAttr(source_attr, ORT_OP_ATTR_STRING, source_buffer, sizeof(source_buffer) - 1, &source_len) == nullptr) {
//                         std::string cache_source(source_buffer, source_len);

//                         // Convert to lowercase for comparison
//                         std::transform(cache_source.begin(), cache_source.end(), cache_source.begin(),
//                                      [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

//                         if (cache_source == "qnnexecutionprovider" || cache_source == "qnn") {
//                             // Get the graph name (node name)
//                             const char* node_name = nullptr;
//                             if (ort_api.Node_GetName(node, &node_name) == nullptr && node_name != nullptr) {
//                                 std::string graph_name(node_name);
//                                 bool has_shared_qnn_model = SharedContext::GetInstance().HasQnnModel(graph_name);
//                                 if (!has_shared_qnn_model) {
//                                     // Log the missing graph (equivalent to LOGS(logger, VERBOSE))
//                                     if (logger_ != nullptr) {
//                                         std::string log_message = "Graph: " + graph_name + " from EpContext node not found from shared EP contexts.";
//                                         ort_api.Logger_LogMessage(logger_, ORT_LOGGING_LEVEL_VERBOSE, log_message.c_str(),
//                                                                 ORT_FILE, __LINE__, __FUNCTION__);
//                                     }
//                                     all_graphs_found = false;
//                                     break;
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     ort_api.ReleaseArrayOfConstObjects(graph_nodes);
//     return all_graphs_found;
// }

// // Helper function to get main EPContext nodes - equivalent to GetMainEPCtxNodes
// void QnnEp::GetMainEPCtxNodes(QnnEp* ep, const OrtGraph* graph, std::unordered_set<const OrtNode*>& ep_context_nodes) {
//     OrtArrayOfConstObjects* graph_nodes = nullptr;
//     if (ep->ort_api.Graph_GetNodes(graph, &graph_nodes) != nullptr) {
//         return;
//     }

//     size_t num_nodes = 0;
//     if (ep->ort_api.ArrayOfConstObjects_GetSize(graph_nodes, &num_nodes) != nullptr) {
//         ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
//         return;
//     }

//     const void* const* node_data = nullptr;
//     if (ep->ort_api.ArrayOfConstObjects_GetData(graph_nodes, &node_data) != nullptr) {
//         ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
//         return;
//     }

//     for (size_t i = 0; i < num_nodes; ++i) {
//         const OrtNode* node = static_cast<const OrtNode*>(node_data[i]);
//         const char* op_type = nullptr;

//         if (ep->ort_api.Node_GetOperatorType(node, &op_type) == nullptr && op_type != nullptr) {
//             if (std::string(op_type) == "EPContext") {
//                 // Check main_context attribute
//                 const OrtOpAttr* main_context_attr = nullptr;
//                 if (ep->ort_api.Node_GetAttributeByName(node, "main_context", &main_context_attr) == nullptr && main_context_attr != nullptr) {
//                     int64_t is_main_context = 0;
//                     size_t out_size = 0;
//                     if (ep->ort_api.ReadOpAttr(main_context_attr, ORT_OP_ATTR_INT, &is_main_context, sizeof(is_main_context), &out_size) == nullptr) {
//                         // Check source attribute
//                         const OrtOpAttr* source_attr = nullptr;
//                         if (ep->ort_api.Node_GetAttributeByName(node, "source", &source_attr) == nullptr && source_attr != nullptr) {
//                             char source_buffer[256] = {0};
//                             size_t source_len = 0;
//                             if (ep->ort_api.ReadOpAttr(source_attr, ORT_OP_ATTR_STRING, source_buffer, sizeof(source_buffer) - 1, &source_len) == nullptr) {
//                                 std::string cache_source(source_buffer, source_len);

//                                 // Convert to lowercase for comparison
//                                 std::transform(cache_source.begin(), cache_source.end(), cache_source.begin(),
//                                              [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

//                                 if (is_main_context && (cache_source == "qnnexecutionprovider" || cache_source == "qnn")) {
//                                     // Log the found EPContext node
//                                     if (ep->logger_ != nullptr) {
//                                         const char* node_name = nullptr;
//                                         size_t node_id = 0;
//                                         ep->ort_api.Node_GetName(node, &node_name);
//                                         ep->ort_api.Node_GetId(node, &node_id);

//                                         std::string log_message = "EPContext Node found: [1] index: [" + std::to_string(node_id) +
//                                                                 "] name: [" + (node_name ? node_name : "unknown") + "]";
//                                         ep->ort_api.Logger_LogMessage(ep->logger_, ORT_LOGGING_LEVEL_VERBOSE, log_message.c_str(),
//                                                                 ORT_FILE, __LINE__, __FUNCTION__);
//                                     }
//                                     ep_context_nodes.insert(node);
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
// }

void QnnEp::PartitionCtxModel(const OrtEp* this_ptr, const OrtGraph* graph, size_t num_nodes_in_graph,
                              OrtEpGraphSupportInfo* graph_support_info) {
  const auto* ep = static_cast<const QnnEp*>(this_ptr);
  auto logger = *(ep->logger_.ToInternal());
  // Get all nodes from the graph
  OrtArrayOfConstObjects* graph_nodes = nullptr;
  if (ep->ort_api.Graph_GetNodes(graph, &graph_nodes) != nullptr) {
    return;
  }

  size_t num_nodes = 0;
  if (ep->ort_api.ArrayOfConstObjects_GetSize(graph_nodes, &num_nodes) != nullptr) {
    ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
    return;
  }

  const void* const* node_data = nullptr;
  if (ep->ort_api.ArrayOfConstObjects_GetData(graph_nodes, &node_data) != nullptr) {
    ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
    return;
  }

  std::vector<const OrtNode*> supported_nodes;
  std::vector<std::vector<const OrtNode*>> supported_groups;

  for (size_t i = 0; i < num_nodes; ++i) {
    const OrtNode* node = static_cast<const OrtNode*>(node_data[i]);
    const char* op_type = nullptr;

    if (ep->ort_api.Node_GetOperatorType(node, &op_type) && op_type != nullptr) {
      if (std::string(op_type) == "EPContext") {
        const OrtOpAttr* source_attr = nullptr;
        if (ep->ort_api.Node_GetAttributeByName(node, "source", &source_attr) && source_attr != nullptr) {
          char source_buffer[256] = {0};
          size_t source_len = 0;
          if (ep->ort_api.ReadOpAttr(source_attr, ORT_OP_ATTR_STRING, source_buffer, sizeof(source_buffer) - 1, &source_len)) {
            std::string cache_source(source_buffer, source_len);

            std::transform(cache_source.begin(), cache_source.end(), cache_source.begin(),
                           [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

            if (cache_source == "qnnexecutionprovider" || cache_source == "qnn") {
              const char* node_name = nullptr;
              size_t node_id = 0;
              ep->ort_api.Node_GetName(node, &node_name);
              ep->ort_api.Node_GetId(node, &node_id);

              std::string log_message = "Node supported: [1] index: [" + std::to_string(node_id) +
                                        "] name: [" + (node_name ? node_name : "unknown") +
                                        "] Operator type: [EPContext] index: [" + std::to_string(node_id) + "]";
              LOGS(logger, VERBOSE) << log_message;

              supported_nodes.push_back(node);

              std::vector<const OrtNode*> supported_group{node};
              supported_groups.emplace_back(std::move(supported_group));
            }
          }
        }
      }
    }
  }

  for (const auto& supported_partition : supported_groups) {
    if (!supported_partition.empty()) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;

      ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                                   supported_partition.data(),
                                                   supported_partition.size(),
                                                   &node_fusion_options);
    }
  }

  const size_t num_of_partitions = supported_groups.size();

  std::string summary_msg = "Number of cf supported by QNN EP: " + std::to_string(num_of_partitions) +
                            ", number of nodes in the graph: " + std::to_string(num_nodes_in_graph) +
                            ", number of nodes supported by QNN: " + std::to_string(num_of_partitions);
  LOGS(logger, INFO) << summary_msg;

  ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
}

// Helper function to get context ONNX model file path - equivalent to GetContextOnnxModelFilePath
void QnnEp::GetContextOnnxModelFilePath(const std::string& user_context_cache_path,
                                       const std::string& model_path_string,
                                       std::string& context_model_path) {
    // always try the path set by user first, it's the only way to set it if load model from memory
    if (!user_context_cache_path.empty()) {
        context_model_path = user_context_cache_path;
    } else if (!model_path_string.empty()) {  // model loaded from file
        context_model_path = model_path_string;
    }
}

OrtStatus* ORT_API_CALL QnnEp::GetCapabilityImpl(OrtEp* this_ptr,
                                                 const OrtGraph* graph,
                                                 OrtEpGraphSupportInfo* graph_support_info) {
  graph_support_info;
  QnnEp* ep = static_cast<QnnEp*>(this_ptr);

  const OrtNode* parent_node = nullptr;
  RETURN_IF_ERROR(ep->ort_api.Graph_GetParentNode(graph, &parent_node));
  if (parent_node != nullptr) {
    return nullptr;
  }

  OrtArrayOfConstObjects* graph_nodes = nullptr;
  RETURN_IF_ERROR(ep->ort_api.Graph_GetNodes(graph, &graph_nodes));

  size_t num_nodes_in_graph = 0;
  RETURN_IF_ERROR(ep->ort_api.ArrayOfConstObjects_GetSize(graph_nodes, &num_nodes_in_graph));

  if (num_nodes_in_graph == 0) {
    ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
    return nullptr;
  }

  // bool is_qnn_ctx_model = qnn::GraphHasEpContextNode(graph);
  bool is_qnn_ctx_model = false;

  // auto gen_metadef_name = [ep, graph]() -> std::string {
  //     return ep->MakeMetadefName(graph);
  // };

  OrtArrayOfConstObjects* graph_inputs = nullptr;
  OrtArrayOfConstObjects* graph_outputs = nullptr;
  RETURN_IF_ERROR(ep->ort_api.Graph_GetInputs(graph, &graph_inputs));
  RETURN_IF_ERROR(ep->ort_api.Graph_GetOutputs(graph, &graph_outputs));

  if (is_qnn_ctx_model && ep->config_.share_ep_contexts && false) {  // SharedContext::GetInstance().HasSharedQnnModels()) {
                                                                     // if (ep->EpSharedContextsHasAllGraphs(graph)) {
    ep->PartitionCtxModel(this_ptr, graph, num_nodes_in_graph, graph_support_info);
    ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
    ep->ort_api.ReleaseArrayOfConstObjects(graph_inputs);
    ep->ort_api.ReleaseArrayOfConstObjects(graph_outputs);
    return nullptr;
    // }
  }

  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> context_bin_map;
  if (ep->enable_vtcm_backup_buffer_sharing_) {
    std::unordered_set<const OrtNode*> ep_ctx_nodes;
    // GetMainEPCtxNodes(ep, graph, ep_ctx_nodes);

    std::string model_path_string = "";
    std::string context_model_path;
    ep->GetContextOnnxModelFilePath(ep->context_cache_path_cfg_, model_path_string, context_model_path);

    std::filesystem::path parent_path = std::filesystem::path(context_model_path).parent_path();

    for (auto& ep_ctx_node : ep_ctx_nodes) {
      // Get the ep_cache_context attribute from the node
      const OrtOpAttr* ep_cache_context_attr = nullptr;
      if (ep->ort_api.Node_GetAttributeByName(ep_ctx_node, "ep_cache_context", &ep_cache_context_attr) == nullptr && ep_cache_context_attr != nullptr) {
        char context_buffer[512] = {0};
        size_t context_len = 0;
        if (ep->ort_api.ReadOpAttr(ep_cache_context_attr, ORT_OP_ATTR_STRING, context_buffer, sizeof(context_buffer) - 1, &context_len) == nullptr) {
          std::string context_bin_filepath(parent_path.string());
          context_bin_filepath.append("/").append(std::string(context_buffer, context_len));

          if (context_bin_map.find(context_bin_filepath) == context_bin_map.end()) {
            context_bin_map.emplace(context_bin_filepath, std::make_unique<std::vector<std::string>>());
            // Push context bin filepath for lookup between sessions
            context_bin_map.at(context_bin_filepath)->push_back(context_bin_filepath);
          }

          // Add the node name to the context bin map
          const char* node_name = nullptr;
          if (ep->ort_api.Node_GetName(ep_ctx_node, &node_name) == nullptr && node_name != nullptr) {
            context_bin_map.at(context_bin_filepath)->push_back(std::string(node_name));
          }
        }
      }
    }
  }
  auto logger = *(ep->logger_.ToInternal());

  Status rt = ep->qnn_backend_manager_->SetupBackend(is_qnn_ctx_model,
                                                     ep->context_cache_enabled_ && false,  // enable_spill_fill_buffer_ (not implemented)
                                                     ep->share_ep_contexts_,
                                                     ep->enable_vtcm_backup_buffer_sharing_,
                                                     context_bin_map);

  context_bin_map.clear();

  if (Status::OK() != rt) {
    LOGS(logger, ERROR) << "QNN SetupBackend failed " << rt.ErrorMessage();
    ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
    ep->ort_api.ReleaseArrayOfConstObjects(graph_inputs);
    ep->ort_api.ReleaseArrayOfConstObjects(graph_outputs);
    return nullptr;
  }

  if (qnn::IsNpuBackend(ep->qnn_backend_manager_->GetQnnBackendType())) {
    // Set the power config id and the default power mode from provider option for main thread,
    // otherwise it will mess up the power mode if user just create session without run it.
    ep->GetPerThreadContext();
  }

  // Report error if QNN CPU backend is loaded while CPU fallback is disabled
  if (ep->config_.disable_cpu_ep_fallback && ep->qnn_backend_manager_->GetQnnBackendType() == qnn::QnnBackendType::CPU) {
    LOGS(logger, ERROR) << "Qnn CPU backend is loaded while CPU fallback is disabled.";
    ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
    ep->ort_api.ReleaseArrayOfConstObjects(graph_inputs);
    ep->ort_api.ReleaseArrayOfConstObjects(graph_outputs);
    return nullptr;
  }

  if ((ep->context_cache_enabled_ || is_qnn_ctx_model) && !qnn::IsQpuBackend(ep->qnn_backend_manager_->GetQnnBackendType())) {
    LOGS(logger, ERROR) << "Qnn context cache only works for HTP/DSP/GPU backend.";
    ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
    ep->ort_api.ReleaseArrayOfConstObjects(graph_inputs);
    ep->ort_api.ReleaseArrayOfConstObjects(graph_outputs);
    return nullptr;
  }

  if (is_qnn_ctx_model) {
    ep->PartitionCtxModel(this_ptr, graph, num_nodes_in_graph, graph_support_info);
    ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
    ep->ort_api.ReleaseArrayOfConstObjects(graph_inputs);
    ep->ort_api.ReleaseArrayOfConstObjects(graph_outputs);
    return nullptr;
  }

  // Get node data for processing
  const void* const* node_data = nullptr;
  RETURN_IF_ERROR(ep->ort_api.ArrayOfConstObjects_GetData(graph_nodes, &node_data));

  // Get node units for the ABI layer
  std::vector<std::unique_ptr<OrtNodeUnit>> node_unit_holder;
  std::unordered_map<const OrtNode*, const OrtNodeUnit*> node_unit_map;

  std::tie(node_unit_holder, node_unit_map) = GetAllOrtNodeUnits(ep->ort_api, graph, logger);
  std::cout << "DEBUG: #nodes: " << node_unit_holder.size() << std::endl;

  // Analyze nodes for QNN support
  std::vector<const OrtNode*> supported_nodes;
  ep->GetSupportedNodes(this_ptr, graph, node_unit_map, node_unit_holder.size(), logger, supported_nodes);

  // Clean up intermediate resources
  ep->ort_api.ReleaseArrayOfConstObjects(graph_inputs);
  ep->ort_api.ReleaseArrayOfConstObjects(graph_outputs);

  // Helper function that returns a string that lists all unsupported nodes.
  // Ex: { name: mul_123, type: Mul }, {}, ...
  // auto get_unsupported_node_names = [&node_unit_holder, &supported_nodes]() -> std::string {
  //     std::stringstream ss;
  //     const size_t num_node_units = node_unit_holder.size();

  //     for (size_t i = 0; i < num_node_units; ++i) {
  //     const auto& node_unit = node_unit_holder[i];

  //     if (supported_nodes.find(&node_unit->GetNode()) == supported_nodes.end()) {
  //         ss << "{ name: " << node_unit->Name() << ", type: " << node_unit->OpType() << " }";
  //         if (i == num_node_units - 1) {
  //         ss << ", ";
  //         }
  //     }
  //     }

  //     return ss.str();
  // };

  // If no supported nodes, return empty
  if (supported_nodes.empty()) {
    std::cout << "DEBUG: No supported nodes." << std::endl;
    ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
    return nullptr;
  }

  size_t num_of_supported_nodes = supported_nodes.size();
  std::cout << "DEBUG: #supported nodes " << num_of_supported_nodes << std::endl;

  OrtNodeFusionOptions node_fusion_options = {};
  node_fusion_options.ort_version_supported = ORT_API_VERSION;
  node_fusion_options.drop_constant_initializers = true;
  RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                                               supported_nodes.data(),
                                                               supported_nodes.size(),
                                                               &node_fusion_options));

  // Clean up
  ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);

  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEp::CompileImpl(_In_ OrtEp* this_ptr,
                                           _In_ const OrtGraph** graphs,
                                           _In_ const OrtNode** fused_nodes,
                                           _In_ size_t count,
                                           _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                           _Out_writes_(count) OrtNode** ep_context_nodes) {
  QnnEp* ep = static_cast<QnnEp*>(this_ptr);
  auto logger = *(ep->logger_.ToInternal());

  ep_context_nodes;

  for (size_t graph_idx = 0; graph_idx < count; ++graph_idx) {
    const OrtGraph* graph = graphs[graph_idx];
    const OrtNode* fused_node = fused_nodes[graph_idx];

    const char* name = nullptr;
    ep->ort_api.Node_GetName(fused_node, &name);
    const std::string fused_node_name{name};

    std::unique_ptr<qnn::QnnModel> qnn_model = std::make_unique<qnn::QnnModel>(
        ep->qnn_backend_manager_.get(), ApiPtrs{ep->ort_api, ep->ep_api, ep->model_editor_api});

    qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t> htp_graph_configs_builder(
        QNN_GRAPH_CONFIG_INIT, QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT);
    ep->InitQnnHtpGraphConfigs(htp_graph_configs_builder);

    std::vector<const QnnGraph_Config_t*> all_graph_configs;
    const QnnGraph_Config_t** htp_configs = htp_graph_configs_builder.GetQnnConfigs();
    if (htp_configs) {
      // Reserve enough for configs + nullptr
      all_graph_configs.reserve(htp_graph_configs_builder.GetSize() + 1);
      for (const QnnGraph_Config_t** config = htp_configs; *config; ++config) {
        all_graph_configs.push_back(*config);
      }
    }

    qnn::QnnSerializerConfig* qnn_serializer_config = ep->qnn_backend_manager_->GetQnnSerializerConfig();
    if (qnn_serializer_config) {
      // We don't bother reserving here to keep the API simpler. Also note that if we're here,
      // we're likely debugging and not waiting for inference.
      qnn_serializer_config->SetGraphName(fused_node_name);
      const QnnGraph_Config_t** serializer_configs = qnn_serializer_config->Configure();
      if (serializer_configs) {
        for (const QnnGraph_Config_t** config = serializer_configs; *config; ++config) {
          all_graph_configs.push_back(*config);
        }
      }
    }

    const QnnGraph_Config_t** all_graph_configs_ptr = nullptr;
    if (!all_graph_configs.empty()) {
      all_graph_configs.push_back(nullptr);
      all_graph_configs_ptr = all_graph_configs.data();
    }

    std::string json_graph_filepath;

    if (ep->dump_json_qnn_graph_) {
      namespace fs = std::filesystem;
      fs::path path = fs::path(ep->json_qnn_graph_dir_) / fs::path(fused_node_name + ".json");
      json_graph_filepath = path.string();
    }

    RETURN_IF_NOT_OK(qnn_model->ComposeGraph(*graph,
                                             *fused_node,
                                             ep->model_settings_,
                                             logger,
                                             all_graph_configs_ptr,
                                             json_graph_filepath),
                     ep->ort_api);
    RETURN_IF_NOT_OK(qnn_model->FinalizeGraphs(logger), ep->ort_api);
    RETURN_IF_NOT_OK(qnn_model->SetupQnnInputOutput(logger), ep->ort_api);

    ep->qnn_models_.emplace(fused_node_name, std::move(qnn_model));

    auto node_compute_info = std::make_unique<QnnNodeComputeInfo>(*ep);
    node_compute_infos[graph_idx] = node_compute_info.release();
  }

  std::cout << "DEBUG: QNN CompileImpl completed!" << std::endl;
  return nullptr;
}

void ORT_API_CALL QnnEp::ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                         OrtNodeComputeInfo** node_compute_infos,
                                                         size_t num_node_compute_infos) {
  ORT_UNUSED_PARAMETER(this_ptr);
  for (size_t idx = 0; idx < num_node_compute_infos; ++idx) {
    delete node_compute_infos[idx];
  }
}

QnnEp::PerThreadContext::PerThreadContext(qnn::QnnBackendManager* qnn_backend_manager,
                                          uint32_t device_id,
                                          uint32_t core_id,
                                          qnn::HtpPerformanceMode default_htp_performance_mode,
                                          uint32_t default_rpc_control_latency)
    : qnn_backend_manager_(qnn_backend_manager) {
  Status rt = qnn_backend_manager_->CreateHtpPowerCfgId(device_id, core_id, htp_power_config_id_);
  is_htp_power_config_id_valid_ = rt.IsOK();

  // Set default performance mode and latency for each thread as default
  // so user doesn't need to set it for every session run
  if (is_htp_power_config_id_valid_) {
    if (qnn::HtpPerformanceMode::kHtpDefault != default_htp_performance_mode) {
      ORT_IGNORE_RETURN_VALUE(qnn_backend_manager_->SetHtpPowerConfig(htp_power_config_id_,
                                                                      default_htp_performance_mode));
    }
    if (default_rpc_control_latency > 0) {
      ORT_IGNORE_RETURN_VALUE(qnn_backend_manager_->SetRpcControlLatency(htp_power_config_id_,
                                                                         default_rpc_control_latency));
    }
  }
}

QnnEp::PerThreadContext::~PerThreadContext() {
  if (is_htp_power_config_id_valid_) {
    ORT_IGNORE_RETURN_VALUE(qnn_backend_manager_->DestroyHTPPowerConfigID(htp_power_config_id_));
  }
}

QnnEp::PerThreadContext& QnnEp::GetPerThreadContext() {
  const auto& per_thread_context_cache = PerThreadContextCache();

  // Try to use cached context
  auto cached_context_it = per_thread_context_cache->find(this);
  if (cached_context_it != per_thread_context_cache->end()) {
    auto cached_context = cached_context_it->second.lock();
    if (cached_context) {
      return *cached_context;
    }
  }

  // Get context and update cache
  std::shared_ptr<PerThreadContext> context;
  {
    std::lock_guard<std::mutex> lock(context_state_.mutex);

    // Get or create a context
    if (context_state_.retired_context_pool.empty()) {
      uint32_t core_id = 0;
      context = std::make_shared<PerThreadContext>(qnn_backend_manager_.get(), device_id_, core_id,
                                                   default_htp_performance_mode_, default_rpc_control_latency_);
    } else {
      context = context_state_.retired_context_pool.back();
      context_state_.retired_context_pool.pop_back();
    }

    // Insert into active_contexts
    context_state_.active_contexts.insert(context);

    // Insert into caches_to_update_on_destruction
    context_state_.caches_to_update_on_destruction.insert(per_thread_context_cache);
  }

  per_thread_context_cache->insert(std::make_pair(this, context));

  return *context;
}

void QnnEp::ReleasePerThreadContext() {
  const auto& per_thread_context_cache = PerThreadContextCache();

  auto cached_context_it = per_thread_context_cache->find(this);
  if (cached_context_it != per_thread_context_cache->end()) {
    auto cached_context = cached_context_it->second.lock();
    if (cached_context) {
      {
        std::lock_guard<std::mutex> lock(context_state_.mutex);
        context_state_.active_contexts.erase(cached_context);
        context_state_.retired_context_pool.push_back(cached_context);
      }

      per_thread_context_cache->erase(cached_context_it);
    }
  }
}

// OrtStatus* QnnEp::OnRunStart(const OrtGraph* graph, const OrtRunOptions* run_options) {
//     // Check backend type - only proceed for HTP or DSP backends
//     auto backend_type = qnn_backend_manager_->GetQnnBackendType();
//     if (qnn::QnnBackendType::HTP != backend_type && qnn::QnnBackendType::DSP != backend_type) {
//         return nullptr; // Equivalent to Status::OK()
//     }

//     // Get config options from run options
//     // This is equivalent to the ConfigOptions& config_options = RunOptions__GetConfigOptions(run_options)
//     // in the original implementation

//     // Log that we're in OnRunStart
//     if (logger_ != nullptr) {
//         ort_api.Logger_LogMessage(logger_, ORT_LOGGING_LEVEL_VERBOSE,
//                                 "QnnEp::OnRunStart called",
//                                 ORT_FILE, __LINE__, __FUNCTION__);
//     }

//     return nullptr;
// }

// OrtStatus* QnnEp::OnRunEnd(const OrtGraph* graph, const OrtRunOptions* run_options, bool sync_stream) {
//     // Check backend type - only proceed for HTP or DSP backends
//     auto backend_type = qnn_backend_manager_->GetQnnBackendType();
//     if (qnn::QnnBackendType::HTP != backend_type && qnn::QnnBackendType::DSP != backend_type) {
//         return nullptr; // Equivalent to Status::OK()
//     }

//     // Log that we're in OnRunEnd
//     if (logger_ != nullptr) {
//         ort_api.Logger_LogMessage(logger_, ORT_LOGGING_LEVEL_VERBOSE,
//                                 "QnnEp::OnRunEnd called",
//                                 ORT_FILE, __LINE__, __FUNCTION__);
//     }

//     return nullptr;
// }

QnnEp::QnnNodeComputeInfo::QnnNodeComputeInfo(QnnEp& ep) : ep(ep) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

OrtStatus* QnnEp::QnnNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                      OrtNodeComputeContext* compute_context,
                                                      void** compute_state) {
  auto* node_compute_info = static_cast<QnnNodeComputeInfo*>(this_ptr);
  QnnEp& ep = node_compute_info->ep;

  std::string fused_node_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
  auto qnn_model_it = ep.qnn_models_.find(fused_node_name);
  if (qnn_model_it == ep.qnn_models_.end()) {
    std::string message = "Unable to get QnnModel with name " + fused_node_name;
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
  }

  *compute_state = qnn_model_it->second.get();
  return nullptr;
}

OrtStatus* QnnEp::QnnNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr,
                                                  void* compute_state,
                                                  OrtKernelContext* kernel_context) {
  auto* node_compute_info = static_cast<QnnNodeComputeInfo*>(this_ptr);
  QnnEp& ep = node_compute_info->ep;

  qnn::QnnModel* model = reinterpret_cast<qnn::QnnModel*>(compute_state);
  RETURN_IF_NOT_OK(model->ExecuteGraph(*kernel_context, *ep.logger_.ToInternal()), ep.ort_api);

  return nullptr;
}

void QnnEp::QnnNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state) {
  // The 'state' is a qnn::QnnModel managed by unique_ptr.
  ORT_UNUSED_PARAMETER(this_ptr);
  ORT_UNUSED_PARAMETER(compute_state);
}

}  // namespace onnxruntime
