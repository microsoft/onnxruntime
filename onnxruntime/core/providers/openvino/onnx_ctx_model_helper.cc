// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <string>
#include <fstream>
#include <vector>
#include <algorithm>

#include "core/providers/openvino/onnx_ctx_model_helper.h"
#include "core/providers/openvino/backend_utils.h"

namespace onnxruntime {
namespace openvino_ep {

EPCtxHandler::EPCtxHandler(std::string ov_sdk_version, const logging::Logger& logger) : openvino_sdk_version_(std::move(ov_sdk_version)), logger_(logger) {
  epctx_model_ = Model::Create("ovep_context_model", false, logger_);
}

/* Export the serialized blob string embedded onto an EPContext Node
 * along with other metadata necessary to validate the graph on import
 */

Status EPCtxHandler::ExportEPCtxModel(const std::string& model_name) {
  // Serialize modelproto to string
  auto model_proto = epctx_model_->ToProto();
  model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  // Finally, dump the model
  std::ofstream epctx_onnx_model(model_name,
                                 std::ios::out | std::ios::trunc | std::ios::binary);
  if (!epctx_onnx_model) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unable to create epctx onnx model file");
  }

  if (!model_proto->SerializeToOstream(epctx_onnx_model)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to serialize model to file");
  }
  LOGS_DEFAULT(VERBOSE) << "[OpenVINO EP] Export blob as EPContext Node";

  return Status::OK();
}

Status EPCtxHandler::AddOVEPCtxNodeToGraph(const GraphViewer& graph_viewer,
                                           const std::string& graph_name,
                                           const bool embed_mode,
                                           std::string&& model_blob_str) const {
  auto& graph = epctx_model_->MainGraph();

  // Get graph inputs and outputs
  const auto& viewer_inputs = graph_viewer.GetInputs();
  const auto& viewer_outputs = graph_viewer.GetOutputs();
  std::vector<onnxruntime::NodeArg*> inputs(viewer_inputs.size()), outputs(viewer_outputs.size());
  auto transform_f = [&](const onnxruntime::NodeArg* iter) { return &graph.GetOrCreateNodeArg(iter->Name(), iter->TypeAsProto()); };
  auto fill_vectors = [transform_f](auto& src, auto& dst) {
    std::transform(src.begin(), src.end(), dst.begin(), transform_f);
  };
  fill_vectors(viewer_inputs, inputs);
  fill_vectors(viewer_outputs, outputs);

  // Create EP context node attributes
  auto node_attributes = ONNX_NAMESPACE::NodeAttributes::Create();
  node_attributes->reserve(4);
  {
    // Create EP context node attributes

    // embed mode
    auto embed_mode_attr = ONNX_NAMESPACE::AttributeProto::Create();
    embed_mode_attr->set_name(EMBED_MODE);
    embed_mode_attr->set_type(onnx::AttributeProto_AttributeType_INT);
    embed_mode_attr->set_i(embed_mode);
    node_attributes->emplace(EMBED_MODE, std::move(*embed_mode_attr));

    // ep context
    auto ep_cache_context_attr = ONNX_NAMESPACE::AttributeProto::Create();
    ep_cache_context_attr->set_name(EP_CACHE_CONTEXT);
    ep_cache_context_attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    ep_cache_context_attr->set_s(std::move(model_blob_str));
    node_attributes->emplace(EP_CACHE_CONTEXT, std::move(*ep_cache_context_attr));

    // sdk version
    auto sdk_version_attr = ONNX_NAMESPACE::AttributeProto::Create();
    sdk_version_attr->set_name(EP_SDK_VER);
    sdk_version_attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    sdk_version_attr->set_s(openvino_sdk_version_);
    node_attributes->emplace(EP_SDK_VER, std::move(*sdk_version_attr));

    // source
    auto source_attr = ONNX_NAMESPACE::AttributeProto::Create();
    source_attr->set_name(SOURCE);
    source_attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    source_attr->set_s(kOpenVINOExecutionProvider);
    node_attributes->emplace(SOURCE, std::move(*source_attr));
  }

  // Create EP context node
  graph.AddNode(graph_name, EPCONTEXT_OP, "", inputs, outputs, std::move(*node_attributes), kMSDomain);

  ORT_ENFORCE(graph.Resolve().IsOK());

  return Status::OK();
}

std::unique_ptr<std::istream> EPCtxHandler::GetModelBlobStream(const std::filesystem::path& so_context_file_path, const GraphViewer& graph_viewer) const {
  auto first_index = *graph_viewer.GetNodesInTopologicalOrder().begin();
  auto node = graph_viewer.GetNode(first_index);
  ORT_ENFORCE(node != nullptr);
  auto& attrs = node->GetAttributes();

  ORT_ENFORCE(attrs.count(EP_CACHE_CONTEXT) == 1);
  const auto& ep_cache_context = attrs.at(EP_CACHE_CONTEXT).s();

  ORT_ENFORCE(attrs.count(EMBED_MODE) == 1);
  bool embed_mode = static_cast<bool>(attrs.at(EMBED_MODE).i());

  std::unique_ptr<std::istream> result;
  if (embed_mode) {
    result.reset((std::istream*)new std::istringstream(ep_cache_context));
  } else {
    auto blob_filepath = so_context_file_path;
    if (blob_filepath.empty() && !graph_viewer.ModelPath().empty()) {
      blob_filepath = graph_viewer.ModelPath();
    }
    blob_filepath = blob_filepath.parent_path() / ep_cache_context;
    ORT_ENFORCE(std::filesystem::exists(blob_filepath), "Blob file not found: ", blob_filepath.string());
    result.reset((std::istream*)new std::ifstream(blob_filepath, std::ios_base::binary | std::ios_base::in));
  }

  bool isXML = backend_utils::IsModelStreamXML(*result);
  if (!isXML) {
    // If the model stream is not an XML (i.e. precompiled blob), the OpenVINO SDK version that it was
    // exported with must match the version that is currently running.
    ORT_ENFORCE((attrs.count(EP_SDK_VER) == 1) && (attrs.at(EP_SDK_VER).s() == openvino_sdk_version_),
                "EPCtx blob was exported / is compatible with OpenVINO SDK version " + attrs.at(EP_SDK_VER).s() +
                  ", but OpenVINO SDK version currently in use is " + openvino_sdk_version_);
  }

  LOGS_DEFAULT(VERBOSE) << "[OpenVINO EP] Read blob from EPContext Node";
  return result;
}

bool EPCtxHandler::CheckForOVEPCtxNodeInGraph(const GraphViewer& graph_viewer) const {
  if (graph_viewer.NumberOfNodes() == 1) {
    auto first_index = *graph_viewer.GetNodesInTopologicalOrder().begin();
    if (auto node = graph_viewer.GetNode(first_index); (node != nullptr) && CheckForOVEPCtxNode(*node)) {
      return true;
    }
  }
  return false;
}

bool EPCtxHandler::CheckForOVEPCtxNode(const Node& node) const {
  // Check for correct Op Type, EP SOURCE, and SDK version
  if (node.OpType() == EPCONTEXT_OP) {
    auto& attrs = node.GetAttributes();
    bool result = (attrs.count(SOURCE) == 1) && (attrs.at(SOURCE).s() == kOpenVINOExecutionProvider);
    result &= attrs.count(EMBED_MODE) == 1;
    result &= attrs.count(EP_CACHE_CONTEXT) == 1;
    return result;
  }
  return false;
}

InlinedVector<const Node*> EPCtxHandler::GetEPCtxNodes() const {
  const auto& epctx_nodes{epctx_model_->MainGraph().Nodes()};
  return InlinedVector<const Node*>(epctx_nodes.begin(), epctx_nodes.end());
}

// Check if graph's only node is EPContext & EP_CACHE_CONTEXT attribute has target extension.
// @param graph_viewer: The graph to inspect.
// @param target_attr_extn: The string to search for in the EP_CACHE_CONTEXT attribute.
// @return true if the node exists, is of the correct type, and the attribute contains the extension; false otherwise.
bool EPCtxHandler::CheckEPCacheContextAttribute(const GraphViewer& graph_viewer, const std::string& target_attr_extn) const {
  // Only check if the graph has exactly one node
  if (graph_viewer.NumberOfNodes() != 1) {
    return false;
  }
  // Get the first node in topological order
  auto first_index = *graph_viewer.GetNodesInTopologicalOrder().begin();
  const Node* node = graph_viewer.GetNode(first_index);
  if (!node) {
    return false;
  }
  // Check OpType and required attributes
  if (node->OpType() != EPCONTEXT_OP) {
    return false;
  }
  const auto& attrs = node->GetAttributes();
  auto it = attrs.find(EP_CACHE_CONTEXT);
  if (it != attrs.end()) {
    return it->second().s().find(target_attr_extn) != std::string::npos;
  }
  return false;
}

}  // namespace openvino_ep
}  // namespace onnxruntime
