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

Status IsFusedGraphHasCtxNode(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                              bool& is_qnn_ctx_model) {
  is_qnn_ctx_model = false;
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    const onnxruntime::GraphViewer& graph_viewer(fused_node_graph.filtered_graph);
    // It's an Onnx model with Qnn context cache binary if it only has a node with EPContext type
    int count = 0;
    for (const auto& node : graph_viewer.Nodes()) {
      if (EPCONTEXT_OP == node.OpType()) {
        is_qnn_ctx_model = true;
      }
      ++count;
    }
    ORT_RETURN_IF(is_qnn_ctx_model && count > 1, "Fused graph should only has 1 single EPContext node.");
  }
  return Status::OK();
}

bool IsQnnCtxModel(const onnxruntime::GraphViewer& graph_viewer) {
  // It's an Onnx model with Qnn context cache binary if it only has a node with EPContext type
  for (const auto& node : graph_viewer.Nodes()) {
    if (EPCONTEXT_OP == node.OpType()) {
      return true;
    }
  }
  return false;
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

Status GetEpContextFromModel(const onnxruntime::PathString& ctx_onnx_model_path,
                             QnnBackendManager* qnn_backend_manager,
                             QnnModel& qnn_model,
                             const logging::Logger& logger) {
  using namespace onnxruntime;
  std::shared_ptr<Model> model;
  ORT_RETURN_IF_ERROR(Model::Load(ToPathString(ctx_onnx_model_path), model, {}, logger));
  const auto& graph = model->MainGraph();
  return GetEpContextFromGraph(GraphViewer(graph),
                               ctx_onnx_model_path,
                               qnn_backend_manager,
                               qnn_model);
}

Status GetEpContextFromGraph(const onnxruntime::GraphViewer& graph_viewer,
                             const onnxruntime::PathString& ctx_onnx_model_path,
                             QnnBackendManager* qnn_backend_manager,
                             QnnModel& qnn_model) {
  const auto& node = graph_viewer.Nodes().begin();
  NodeAttrHelper node_helper(*node);
  bool is_embed_mode = node_helper.Get(EMBED_MODE, true);
  if (is_embed_mode) {
    const std::string& context_binary = node_helper.Get(EP_CACHE_CONTEXT, "");
    return qnn_backend_manager->LoadCachedQnnContextFromBuffer(const_cast<char*>(context_binary.c_str()),
                                                               static_cast<uint64_t>(context_binary.length()),
                                                               qnn_model);
  }

  std::filesystem::path folder_path = std::filesystem::path(ctx_onnx_model_path).parent_path();
  std::string external_qnn_ctx_binary_file_name = node_helper.Get(EP_CACHE_CONTEXT, "");
  ORT_RETURN_IF(external_qnn_ctx_binary_file_name.empty(), "The file path in ep_cache_context should not be empty.");
#ifdef _WIN32
  onnxruntime::PathString external_qnn_context_binary_path = onnxruntime::ToPathString(external_qnn_ctx_binary_file_name);
  auto ctx_file_path = std::filesystem::path(external_qnn_context_binary_path.c_str());
  ORT_RETURN_IF(ctx_file_path.is_absolute(), "External mode should set ep_cache_context field with a relative path, but it is an absolute path: ",
                external_qnn_ctx_binary_file_name);
  auto relative_path = ctx_file_path.lexically_normal().make_preferred().wstring();
  if (relative_path.find(L"..", 0) != std::string::npos) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "The file path in ep_cache_context field has '..'. It's not allowed to point outside the directory.");
  }

  std::filesystem::path context_binary_path = folder_path.append(relative_path);
#else
  ORT_RETURN_IF(external_qnn_ctx_binary_file_name[0] == '/',
                "External mode should set ep_cache_context field with a relative path, but it is an absolute path: ",
                external_qnn_ctx_binary_file_name);
  if (external_qnn_ctx_binary_file_name.find("..", 0) != std::string::npos) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "The file path in ep_cache_context field has '..'. It's not allowed to point outside the directory.");
  }
  std::filesystem::path context_binary_path = folder_path.append(external_qnn_ctx_binary_file_name);
  std::string file_full_path = context_binary_path.string();
#endif
  if (!std::filesystem::is_regular_file(context_binary_path)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "The file path in ep_cache_context does not exist or is not accessible.");
  }

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
                                                             qnn_model);
}

Status LoadQnnCtxFromOnnxModel(const onnxruntime::GraphViewer& graph_viewer,
                               const onnxruntime::PathString& ctx_onnx_model_path,
                               bool is_qnn_ctx_model,
                               bool is_ctx_cache_file_exist,
                               QnnBackendManager* qnn_backend_manager,
                               QnnModel& qnn_model,
                               const logging::Logger& logger) {
  Status status;
  if (is_qnn_ctx_model) {
    status = GetEpContextFromGraph(graph_viewer, ctx_onnx_model_path, qnn_backend_manager, qnn_model);
  } else if (is_ctx_cache_file_exist) {
    status = GetEpContextFromModel(ctx_onnx_model_path, qnn_backend_manager, qnn_model, logger);
  }

  if (!status.IsOK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Failed to load from EpContextModel. ", status.ErrorMessage());
  }

  return Status::OK();
}

Status GetMetadataFromEpContextModel(const onnxruntime::PathString& ctx_onnx_model_path,
                                     std::string& model_name,
                                     std::string& model_description,
                                     std::string& graph_partition_name,
                                     std::string& cache_source,
                                     const logging::Logger& logger) {
  using namespace onnxruntime;
  std::shared_ptr<Model> model;
  ORT_RETURN_IF_ERROR(Model::Load(ctx_onnx_model_path, model, {}, logger));
  const auto& graph = GraphViewer(model->MainGraph());
  const auto& node = graph.Nodes().begin();
  NodeAttrHelper node_helper(*node);
  model_name = graph.Name();
  model_description = graph.Description();
  graph_partition_name = node_helper.Get(PARTITION_NAME, "");
  cache_source = node_helper.Get(SOURCE, "");

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

Status ValidateWithContextFile(const onnxruntime::PathString& context_cache_path,
                               const std::string& model_name,
                               const std::string& model_description,
                               const std::string& graph_partition_name,
                               const logging::Logger& logger) {
  std::string model_name_from_ctx_cache;
  std::string model_description_from_ctx_cache;
  std::string graph_partition_name_from_ctx_cache;
  std::string cache_source;
  auto status = GetMetadataFromEpContextModel(context_cache_path,
                                              model_name_from_ctx_cache,
                                              model_description_from_ctx_cache,
                                              graph_partition_name_from_ctx_cache,
                                              cache_source,
                                              logger);
  if (!status.IsOK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Failed to get metadata from EpContextModel.");
  }

  // The source attribute from the skeleton onnx file indicate whether it's generated from QNN toolchain or ORT
  if (cache_source != kQnnExecutionProvider) {
    LOGS(logger, VERBOSE) << "Context binary cache is not generated by Ort.";
    return Status::OK();
  }

  if (model_name != model_name_from_ctx_cache ||
      model_description != model_description_from_ctx_cache ||
      graph_partition_name != graph_partition_name_from_ctx_cache) {
    std::string message = onnxruntime::MakeString("Metadata mismatch. onnx: ",
                                                  model_name, " ", model_description, " ", graph_partition_name,
                                                  " vs epcontext: ",
                                                  model_name_from_ctx_cache, " ",
                                                  model_description_from_ctx_cache, " ",
                                                  graph_partition_name_from_ctx_cache);
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
