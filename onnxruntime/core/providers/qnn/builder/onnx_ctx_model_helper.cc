// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
#include "core/graph/constants.h"

#include <iostream>
#include <fstream>
#include <filesystem>

namespace onnxruntime {
namespace qnn {

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

Status GenerateCtxCacheOnnxModel(const std::string& model_name, const std::string& graph_name,
                                 const std::vector<std::string>& input_names,
                                 const std::unordered_map<std::string, OnnxTensorInfo>& inputs_info,
                                 const std::vector<std::string>& output_names,
                                 const std::unordered_map<std::string, OnnxTensorInfo>& outputs_info,
                                 const std::string& model_description,
                                 const std::string& sdk_build_version,
                                 const std::string& file_path,
                                 unsigned char* buffer,
                                 uint64_t buffer_size,
                                 bool qnn_context_embed_mode,
                                 const logging::Logger& logger) {
  std::unordered_map<std::string, int> domain_to_version = {{kOnnxDomain, 11}, {kMSDomain, 1}};
  Model model(model_name, false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
              {}, logger);
  auto& graph = model.MainGraph();
  graph.SetDescription(model_description);

  using namespace ONNX_NAMESPACE;
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  ORT_RETURN_IF_ERROR(CreateNodeArgs(input_names, inputs_info, inputs, graph));
  ORT_RETURN_IF_ERROR(CreateNodeArgs(output_names, outputs_info, outputs, graph));

  auto& ep_node = graph.AddNode(graph_name,
                                EPCONTEXT_OP,
                                "Onnx Qnn context binary cache for graph partition: " + graph_name,
                                inputs,
                                outputs,
                                nullptr,
                                kMSDomain);

  if (qnn_context_embed_mode) {
    std::string cache_payload(buffer, buffer + buffer_size);
    ep_node.AddAttribute(EP_CACHE_CONTEXT, cache_payload);
  } else {
    std::string context_cache_path(file_path + "_" + graph_name + ".bin");
    std::string context_cache_name(std::filesystem::path(context_cache_path).filename().string());
    std::ofstream of_stream(context_cache_path.c_str(), std::ofstream::binary);
    if (!of_stream) {
      LOGS(logger, ERROR) << "Failed to open cached context file.";
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to open context cache file.");
    }
    of_stream.write(reinterpret_cast<char*>(buffer), buffer_size);
    ep_node.AddAttribute(EP_CACHE_CONTEXT, context_cache_name);
  }
  int64_t embed_mode = qnn_context_embed_mode ? static_cast<int64_t>(1) : static_cast<int64_t>(0);
  ep_node.AddAttribute(EMBED_MODE, embed_mode);
  ep_node.AddAttribute(EP_SDK_VER, sdk_build_version);
  ep_node.AddAttribute(PARTITION_NAME, graph_name);
  ep_node.AddAttribute(SOURCE, kQnnExecutionProvider);

  ORT_RETURN_IF_ERROR(graph.Resolve());

  ORT_RETURN_IF_ERROR(Model::Save(model, file_path));

  return Status::OK();
}

Status GetEpContextFromModel(const std::string& ctx_onnx_model_path,
                             std::string& ep_cache_context,
                             const logging::Logger& logger) {
  using namespace onnxruntime;
  std::shared_ptr<Model> model;
  ORT_RETURN_IF_ERROR(Model::Load(ToPathString(ctx_onnx_model_path), model, {}, logger));
  const auto& graph = model->MainGraph();
  ORT_RETURN_IF_ERROR(GetEpContextFromGraph(GraphViewer(graph), ctx_onnx_model_path, ep_cache_context));

  return Status::OK();
}

Status GetEpContextFromGraph(const onnxruntime::GraphViewer& graph_viewer,
                             const std::string& ctx_onnx_model_path,
                             std::string& ep_cache_context) {
  const auto& node = graph_viewer.Nodes().begin();
  NodeAttrHelper node_helper(*node);
  bool is_embed_mode = node_helper.Get(EMBED_MODE, true);
  if (is_embed_mode) {
    ep_cache_context = node_helper.Get(EP_CACHE_CONTEXT, "");
  } else {
    std::string external_qnn_context_binary_file_name = node_helper.Get(EP_CACHE_CONTEXT, "");

    std::string context_binary_path(std::filesystem::path(ctx_onnx_model_path).parent_path().string() +
                                    "/" + external_qnn_context_binary_file_name);
    size_t buffer_size{0};
    std::ifstream cache_file(context_binary_path.c_str(), std::ifstream::binary);
    ORT_RETURN_IF(!cache_file || !cache_file.good(), "Failed to open cache file.");
    cache_file.seekg(0, cache_file.end);
    buffer_size = static_cast<size_t>(cache_file.tellg());
    ORT_RETURN_IF(0 == buffer_size, "Empty cache file encountered.");
    cache_file.seekg(0, cache_file.beg);
    ep_cache_context.reserve(buffer_size);
    // Load file into buffer
    ep_cache_context.assign(std::istreambuf_iterator<char>(cache_file), std::istreambuf_iterator<char>());
    cache_file.close();
    ORT_RETURN_IF(ep_cache_context.length() != buffer_size, "Failed to read contents from cached context file.");
  }
  return Status::OK();
}

Status QnnCacheModelHandler::GetMetadataFromEpContextModel(const std::string& ctx_onnx_model_path,
                                                           std::string& model_name,
                                                           std::string& model_description,
                                                           std::string& graph_partition_name,
                                                           std::string& cache_source,
                                                           const logging::Logger& logger) {
  if (!is_metadata_ready_) {
    using namespace onnxruntime;
    std::shared_ptr<Model> model;
    ORT_RETURN_IF_ERROR(Model::Load(ToPathString(ctx_onnx_model_path), model, {}, logger));
    const auto& graph = GraphViewer(model->MainGraph());
    const auto& node = graph.Nodes().begin();
    NodeAttrHelper node_helper(*node);
    model_name_ = graph.Name();
    model_description_ = graph.Description();
    graph_partition_name_ = node_helper.Get(PARTITION_NAME, "");
    cache_source_ = node_helper.Get(SOURCE, "");
    is_metadata_ready_ = true;
  }
  model_name = model_name_;
  model_description = model_description_;
  graph_partition_name = graph_partition_name_;
  cache_source = cache_source_;

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
