// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
#include "core/graph/constants.h"
#include "core/providers/qnn/builder/qnn_model.h"

#include <iostream>
#include <fstream>
#include <filesystem>

namespace onnxruntime {
namespace qnn {

bool GraphHasEpContextNode(const onnxruntime::GraphViewer& graph_viewer) {
  // It's an Onnx model with Qnn context cache binary if it has a node with EPContext type
  for (const auto& node : graph_viewer.Nodes()) {
    if (EPCONTEXT_OP == node.OpType()) {
      return true;
    }
  }
  return false;
}

bool IsFusedGraphHasCtxNode(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    const onnxruntime::GraphViewer& graph_viewer(fused_node_graph.filtered_graph);
    bool has_qnn_ep_context_node = GraphHasEpContextNode(graph_viewer);
    if (has_qnn_ep_context_node) {
      return true;
    }
  }
  return false;
}

Status GetMainContextNode(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                          QnnBackendManager* qnn_backend_manager,
                          const logging::Logger& logger,
                          int& main_context_pos,
                          std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models) {
  main_context_pos = -1;
  for (size_t i = 0; i < fused_nodes_and_graphs.size(); ++i) {
    const onnxruntime::GraphViewer& graph_viewer(fused_nodes_and_graphs[i].filtered_graph);
    const auto& ep_context_node = graph_viewer.Nodes().begin();
    ORT_RETURN_IF_NOT(EPCONTEXT_OP == ep_context_node->OpType(), "Should only filter in the EPContext node.");
    qnn_models.emplace(ep_context_node->Name(),
                       std::make_unique<qnn::QnnModel>(logger, qnn_backend_manager));
    NodeAttrHelper node_helper(*ep_context_node);
    int64_t is_main_context = node_helper.Get(MAIN_CONTEXT, static_cast<int64_t>(0));
    if (1 == is_main_context) {
      main_context_pos = static_cast<int>(i);
    }
  }

  ORT_RETURN_IF(main_context_pos < 0, "Failed to find the EPContext node with main_context=1");
  return Status::OK();
}

Status GetContextFromOnnxModel(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                               const onnxruntime::PathString& ctx_onnx_model_path,
                               QnnBackendManager* qnn_backend_manager,
                               const logging::Logger& logger,
                               std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models) {
  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    const Node& fused_node = fused_node_and_graph.fused_node;
    qnn_models.emplace(fused_node.Name(),
                       std::make_unique<qnn::QnnModel>(logger, qnn_backend_manager));
  }
  using namespace onnxruntime;
  std::shared_ptr<Model> model;
  ORT_RETURN_IF_ERROR(Model::Load(ctx_onnx_model_path, model, {}, logger));
  const auto& graph = GraphViewer(model->MainGraph());

  for (const auto& ep_context_node : graph.Nodes()) {
    if (EPCONTEXT_OP != ep_context_node.OpType()) {
      continue;
    }
    NodeAttrHelper node_helper(ep_context_node);
    int64_t is_main_context = node_helper.Get(MAIN_CONTEXT, static_cast<int64_t>(0));
    if (1 == is_main_context) {
      return GetEpContextFromMainNode(ep_context_node, ctx_onnx_model_path, qnn_backend_manager, qnn_models);
    }
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Failed to find EPContext node with main_context=1.");
}

Status LoadContextFromOnnxModel(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                                const onnxruntime::PathString& ctx_onnx_model_path,
                                QnnBackendManager* qnn_backend_manager,
                                const logging::Logger& logger,
                                std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models) {
  Status status = GetContextFromOnnxModel(fused_nodes_and_graphs, ctx_onnx_model_path, qnn_backend_manager, logger, qnn_models);

  // This is the protocol with customer that status with INVALID_GRAPH will be generated if failed to load context model
  if (!status.IsOK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Failed to load from EpContextModel. ", status.ErrorMessage());
  }

  return Status::OK();
}

Status CreateNodeArgs(const std::vector<std::string>& names,
                      const std::unordered_map<std::string, OnnxTensorInfo>& tensor_info_table,
                      std::vector<NodeArg*>& node_args,
                      onnxruntime::Graph& graph) {
  using namespace ONNX_NAMESPACE;
  for (size_t i = 0; i < names.size(); ++i) {
    std::string name = names[i];
    ORT_RETURN_IF(tensor_info_table.find(name) == tensor_info_table.end(), "Tensor name: ", name, " not found in tensor_info_table");
    const OnnxTensorInfo& tensor_info = tensor_info_table.at(name);
    TypeProto tensor_type;
    tensor_type.mutable_tensor_type()->set_elem_type(tensor_info.data_type_);
    for (size_t j = 0; j < tensor_info.shape_.size(); ++j) {
      tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(tensor_info.shape_[j]);
    }
    auto& input_arg = graph.GetOrCreateNodeArg(name, &tensor_type);
    node_args.push_back(&input_arg);
  }
  return Status::OK();
}

Status GetEpContextFromMainNode(const onnxruntime::Node& main_context_node,
                                const onnxruntime::PathString& ctx_onnx_model_path,
                                QnnBackendManager* qnn_backend_manager,
                                std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models) {
  ORT_RETURN_IF_NOT(EPCONTEXT_OP == main_context_node.OpType(), "Should only filter in the EPContext node.");
  NodeAttrHelper node_helper(main_context_node);
  bool is_embed_mode = node_helper.Get(EMBED_MODE, true);
  if (is_embed_mode) {
    const std::string& context_binary = node_helper.Get(EP_CACHE_CONTEXT, "");
    return qnn_backend_manager->LoadCachedQnnContextFromBuffer(const_cast<char*>(context_binary.c_str()),
                                                               static_cast<uint64_t>(context_binary.length()),
                                                               qnn_models);
  }

  std::string external_qnn_context_binary_file_name = node_helper.Get(EP_CACHE_CONTEXT, "");
  std::filesystem::path folder_path = std::filesystem::path(ctx_onnx_model_path).parent_path();
  std::filesystem::path context_binary_path = folder_path.append(external_qnn_context_binary_file_name);

  size_t buffer_size{0};
  std::ifstream cache_file(context_binary_path.string().c_str(), std::ifstream::binary);
  ORT_RETURN_IF(!cache_file || !cache_file.good(), "Failed to open cache file.");

  cache_file.seekg(0, cache_file.end);
  buffer_size = static_cast<size_t>(cache_file.tellg());
  ORT_RETURN_IF(0 == buffer_size, "Empty cache file encountered.");

  cache_file.seekg(0, cache_file.beg);
  std::unique_ptr<char[]> buffer = std::make_unique<char[]>(buffer_size);
  ORT_RETURN_IF(nullptr == buffer, "Failed to allocate memory for cache file.");
  // Load file into buffer
  const auto& read_result = cache_file.read(buffer.get(), buffer_size);
  ORT_RETURN_IF(!read_result, "Failed to read contents from cached context file.");
  cache_file.close();
  return qnn_backend_manager->LoadCachedQnnContextFromBuffer(buffer.get(),
                                                             static_cast<uint64_t>(buffer_size),
                                                             qnn_models);
}

Status LoadQnnCtxFromOnnxGraph(const onnxruntime::GraphViewer& graph_viewer,
                               const onnxruntime::PathString& ctx_onnx_model_path,
                               QnnBackendManager* qnn_backend_manager,
                               std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models) {
  Status status = GetEpContextFromMainNode(*graph_viewer.Nodes().begin(), ctx_onnx_model_path, qnn_backend_manager, qnn_models);

  // This is the protocol with customer that status with INVALID_GRAPH will be generated if failed to load context model
  if (!status.IsOK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Failed to load from EpContextModel. ", status.ErrorMessage());
  }

  return Status::OK();
}

Status GetMetadataFromEpContextModel(const onnxruntime::PathString& ctx_onnx_model_path,
                                     std::string& model_name,
                                     std::string& model_description,
                                     std::vector<std::string>& graph_partition_names,
                                     std::string& cache_source,
                                     const logging::Logger& logger) {
  using namespace onnxruntime;
  std::shared_ptr<Model> model;
  ORT_RETURN_IF_ERROR(Model::Load(ctx_onnx_model_path, model, {}, logger));
  const auto& graph = GraphViewer(model->MainGraph());
  model_name = graph.Name();
  model_description = graph.Description();

  for (const auto& ep_context_node : graph.Nodes()) {
    if (EPCONTEXT_OP != ep_context_node.OpType()) {
      continue;
    }
    NodeAttrHelper node_helper(ep_context_node);
    cache_source = node_helper.Get(SOURCE, "");
    graph_partition_names.push_back(node_helper.Get(PARTITION_NAME, ""));
  }

  return Status::OK();
}

bool IsContextCacheFileExists(const std::string& customer_context_cache_path,
                              const onnxruntime::PathString& model_pathstring,
                              onnxruntime::PathString& context_cache_path) {
  // Use user provided context cache file path if exist, otherwise try model_file.onnx_ctx.onnx by default
  if (!customer_context_cache_path.empty()) {
    context_cache_path = ToPathString(customer_context_cache_path);
  } else if (!model_pathstring.empty()) {
    context_cache_path = model_pathstring + ToPathString("_ctx.onnx");
  }

  return std::filesystem::is_regular_file(context_cache_path) && std::filesystem::exists(context_cache_path);
}

Status ValidateWithContextFile(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                               const onnxruntime::PathString& context_cache_path,
                               const std::string& model_name,
                               const std::string& model_description,
                               const logging::Logger& logger) {
  std::string model_name_from_ctx_cache;
  std::string model_description_from_ctx_cache;
  std::vector<std::string> graph_partition_names;
  std::string cache_source;

  auto status = GetMetadataFromEpContextModel(context_cache_path,
                                              model_name_from_ctx_cache,
                                              model_description_from_ctx_cache,
                                              graph_partition_names,
                                              cache_source,
                                              logger);
  if (!status.IsOK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Failed to get metadata from EpContext model.");
  }

  // The source attribute from the skeleton onnx file indicate whether it's generated from QNN toolchain or ORT
  if (cache_source != kQnnExecutionProvider) {
    LOGS(logger, VERBOSE) << "Context binary cache is not generated by Ort.";
    return Status::OK();
  }

  bool partition_names_matched = true;
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    const Node& fused_node = fused_node_graph.fused_node;
    const std::string& graph_meta_id = fused_node.Name();
    bool name_found = false;
    for (auto partition_name_from_ctx : graph_partition_names) {
      if (partition_name_from_ctx == graph_meta_id) {
        name_found = true;
        break;
      }
    }

    if (!name_found) {
      LOGS(logger, ERROR) << "Partition meta_id not found from any EPContext node: " << graph_meta_id;
      partition_names_matched = false;
      break;
    }
  }

  if (model_name != model_name_from_ctx_cache ||
      model_description != model_description_from_ctx_cache ||
      !partition_names_matched) {
    std::string message = onnxruntime::MakeString("Metadata mismatch. onnx: ",
                                                  model_name, " ", model_description,
                                                  " vs epcontext: ",
                                                  model_name_from_ctx_cache, " ",
                                                  model_description_from_ctx_cache,
                                                  " or the partition name not match.");
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, message);
  }

  return Status::OK();
}

Status GenerateCtxCacheOnnxModel(Model* model,
                                 unsigned char* buffer,
                                 uint64_t buffer_size,
                                 const std::string& sdk_build_version,
                                 const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                                 const std::unordered_map<std::string, std::unique_ptr<QnnModel>>& qnn_models,
                                 const onnxruntime::PathString& context_cache_path,
                                 bool qnn_context_embed_mode,
                                 const logging::Logger& logger) {
  auto& graph = model->MainGraph();

  using namespace ONNX_NAMESPACE;
  int index = 0;
  // Still need more work to support multiple partition, it's out of EP's scope.
  // Already have code to make sure it's single partition before this method get invoked.
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    Node& fused_node = fused_node_graph.fused_node;
    auto qnn_model_kv = qnn_models.find(fused_node.Name());
    ORT_RETURN_IF(qnn_model_kv == qnn_models.end(), fused_node.Name(), " not exist in QnnModel table.");

    auto qnn_model = qnn_model_kv->second.get();
    std::vector<NodeArg*> inputs;
    std::vector<NodeArg*> outputs;
    ORT_RETURN_IF_ERROR(CreateNodeArgs(qnn_model->GetInputNames(), qnn_model->GetInputsInfo(), inputs, graph));
    ORT_RETURN_IF_ERROR(CreateNodeArgs(qnn_model->GetOutputNames(), qnn_model->GetOutputsInfo(), outputs, graph));

    const std::string& graph_name = fused_node.Name();
    auto& ep_node = graph.AddNode(graph_name,
                                  EPCONTEXT_OP,
                                  "Onnx Qnn context binary cache for graph partition: " + graph_name,
                                  inputs,
                                  outputs,
                                  nullptr,
                                  kMSDomain);

    // Only dump the context buffer once since all QNN graphs are in one single context
    if (0 == index) {
      if (qnn_context_embed_mode) {
        std::string cache_payload(buffer, buffer + buffer_size);
        ep_node.AddAttribute(EP_CACHE_CONTEXT, cache_payload);
      } else {
        onnxruntime::PathString context_bin_path = context_cache_path + ToPathString("_" + graph_name + ".bin");
        std::string context_cache_name(std::filesystem::path(context_bin_path).filename().string());
        std::ofstream of_stream(context_bin_path.c_str(), std::ofstream::binary);
        if (!of_stream) {
          LOGS(logger, ERROR) << "Failed to open create context file.";
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to open context cache file.");
        }
        of_stream.write(reinterpret_cast<char*>(buffer), buffer_size);
        ep_node.AddAttribute(EP_CACHE_CONTEXT, context_cache_name);
      }
    } else {
      ep_node.AddAttribute(MAIN_CONTEXT, static_cast<int64_t>(0));
    }
    int64_t embed_mode = qnn_context_embed_mode ? static_cast<int64_t>(1) : static_cast<int64_t>(0);
    ep_node.AddAttribute(EMBED_MODE, embed_mode);
    ep_node.AddAttribute(EP_SDK_VER, sdk_build_version);
    ep_node.AddAttribute(PARTITION_NAME, graph_name);
    ep_node.AddAttribute(SOURCE, kQnnExecutionProvider);
    ++index;
  }

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
