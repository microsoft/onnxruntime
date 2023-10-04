// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
#include "core/graph/constants.h"

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

  std::string cache_payload(buffer, buffer + buffer_size);
  ep_node.AddAttribute("ep_cache_context", cache_payload);
  ep_node.AddAttribute("ep_sdk_version", sdk_build_version);
  ep_node.AddAttribute("partition_name", graph_name);
  ep_node.AddAttribute("source", kQnnExecutionProvider);

  ORT_RETURN_IF_ERROR(graph.Resolve());

  ORT_RETURN_IF_ERROR(Model::Save(model, file_path));

  return Status::OK();
}

Status GetEpEngineCacheFromModel(const std::string& onnx_model_path,
                                 std::string& ep_cache_context,
                                 const logging::Logger& logger) {
  using namespace onnxruntime;
  std::shared_ptr<Model> model;
  ORT_RETURN_IF_ERROR(Model::Load(ToPathString(onnx_model_path), model, {}, logger));
  const auto& graph = model->MainGraph();
  ep_cache_context = GetEpEngineCacheFromGraph(GraphViewer(graph));

  return Status::OK();
}

std::string GetEpEngineCacheFromGraph(const onnxruntime::GraphViewer& graph_viewer) {
  const auto& node = graph_viewer.Nodes().begin();
  NodeAttrHelper node_helper(*node);
  return node_helper.Get("ep_cache_context", "");
}

Status QnnCacheModelHandler::GetMetadataFromEpEngineCacheModel(const std::string& onnx_model_path,
                                                               std::string& model_name,
                                                               std::string& model_description,
                                                               std::string& graph_partition_name,
                                                               std::string& cache_source,
                                                               const logging::Logger& logger) {
  if (!is_metadata_ready_) {
    using namespace onnxruntime;
    std::shared_ptr<Model> model;
    ORT_RETURN_IF_ERROR(Model::Load(ToPathString(onnx_model_path), model, {}, logger));
    const auto& graph = GraphViewer(model->MainGraph());
    const auto& node = graph.Nodes().begin();
    NodeAttrHelper node_helper(*node);
    model_name_ = graph.Name();
    model_description_ = graph.Description();
    graph_partition_name_ = node_helper.Get("partition_name", "");
    cache_source_ = node_helper.Get("source", "");
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
