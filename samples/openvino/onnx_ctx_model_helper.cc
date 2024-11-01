// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <string>
#include <fstream>
#include <vector>
#include <cassert>

#include "onnx_ctx_model_helper.h"
#include "openvino_utils.h"

namespace onnxruntime {
namespace openvino_ep {

const OrtGraphApi* EPCtxHandler::graph_api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION)->GetGraphApi(ORT_API_VERSION);

// Utilities to handle EPContext node export and parsing of an EPContext node
// to create the compiled_model object to infer on
static const char EPCONTEXT_OP[] = "EPContext";
static const char EMBED_MODE[] = "embed_mode";
static const char EP_CACHE_CONTEXT[] = "ep_cache_context";
static const char EP_SDK_VER[] = "ep_sdk_version";
static const char SOURCE[] = "source";

/* Export the serialized blob string embedded onto an EPContext Node
 * along with other metadata necessary to validate the graph on import
 */

//Status EPCtxHandler::ExportEPCtxModel(const GraphViewer& graph_viewer,
//                                      const std::string& graph_name,
//                                      const logging::Logger& logger,
//                                      const bool& ep_context_embed_mode,
//                                      const std::string& model_blob_str,
//                                      const std::string& openvino_sdk_version,
//                                      const std::string& device_type) const {
//  auto model_build = graph_viewer.CreateModel(logger);
//  auto& graph_build = model_build->MainGraph();
//
//  // Get graph inputs and outputs
//  std::vector<onnxruntime::NodeArg*> inputs, outputs;
//  for (auto input : graph_viewer.GetInputs()) {
//    auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
//    inputs.push_back(&n_input);
//  }
//  for (auto output : graph_viewer.GetOutputs()) {
//    auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
//    outputs.push_back(&n_output);
//  }
//
//  // Create EP context node attributes
//  auto attr_0 = ONNX_NAMESPACE::AttributeProto::Create();
//  auto attr_1 = ONNX_NAMESPACE::AttributeProto::Create();
//  auto attr_2 = ONNX_NAMESPACE::AttributeProto::Create();
//  auto attr_3 = ONNX_NAMESPACE::AttributeProto::Create();
//
//  // embed mode
//  attr_0->set_name(EMBED_MODE);
//  attr_0->set_type(onnx::AttributeProto_AttributeType_INT);
//  attr_0->set_i(ep_context_embed_mode);
//  // ep context
//  attr_1->set_name(EP_CACHE_CONTEXT);
//  attr_1->set_type(onnx::AttributeProto_AttributeType_STRING);
//  attr_1->set_s(model_blob_str);
//  // sdk version
//  attr_2->set_name(EP_SDK_VER);
//  attr_2->set_type(onnx::AttributeProto_AttributeType_STRING);
//  attr_2->set_s(openvino_sdk_version);
//  // source
//  attr_3->set_name(SOURCE);
//  attr_3->set_type(onnx::AttributeProto_AttributeType_STRING);
//  attr_3->set_s(kOpenVINOExecutionProvider);
//
//  auto node_attributes = ONNX_NAMESPACE::NodeAttributes::Create();
//  node_attributes->reserve(4);
//  node_attributes->emplace(EMBED_MODE, *attr_0);
//  node_attributes->emplace(EP_CACHE_CONTEXT, *attr_1);
//  node_attributes->emplace(EP_SDK_VER, *attr_2);
//  node_attributes->emplace(SOURCE, *attr_3);
//
//  // Create EP context node
//  graph_build.AddNode(graph_name, EPCONTEXT_OP, "", inputs, outputs, node_attributes.get(), kMSDomain);
//  ORT_ENFORCE(graph_build.Resolve().IsOK());
//
//  // Serialize modelproto to string
//  auto new_graph_viewer = graph_build.CreateGraphViewer();
//  auto model = new_graph_viewer->CreateModel(logger);
//  auto model_proto = model->ToProto();
//  new_graph_viewer->ToProto(*model_proto->mutable_graph(), true, true);
//  model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
//
//  // Finally, dump the model
//  std::ofstream dump(graph_name + "-ov_" + device_type + "_blob.onnx",
//                     std::ios::out | std::ios::trunc | std::ios::binary);
//  model_proto->SerializeToOstream(dump);
//
//  LOGS_DEFAULT(VERBOSE) << "[OpenVINO EP] Export blob as EPContext Node";
//
//  return Status::OK();
//}

OrtStatus* EPCtxHandler::ImportBlobFromEPCtxModel(const OrtGraphViewer* graph_viewer) {
  const OrtNode* node = nullptr;
  graph_api_->OrtGraph_GetOrtNode(graph_viewer, 0, &node);
  size_t attr_count = 0;
  graph_api_->OrtNode_GetAttributeKeyCount(node, EP_CACHE_CONTEXT, &attr_count);
  assert(attr_count > 0);

  const char* attr_str = nullptr;
  graph_api_->OrtNode_GetAttributeStr(node, EP_CACHE_CONTEXT, &attr_str);
  model_stream_ = std::make_shared<std::istringstream>(attr_str);

//  LOGS_DEFAULT(VERBOSE) << "[OpenVINO EP] Read blob from EPContext Node";

  is_valid_ep_ctx_graph_ = true;
  return nullptr;
}

bool EPCtxHandler::CheckForOVEPCtxNode(const OrtGraphViewer* graph_viewer, std::string openvino_sdk_version) const {
  int max_node_index = 0;
  graph_api_->OrtGraph_MaxNodeIndex(graph_viewer, &max_node_index);
  for (int i = 0; i < max_node_index; ++i) {
    const OrtNode* node = nullptr;
    graph_api_->OrtGraph_GetOrtNode(graph_viewer, i, &node);
    if (node != nullptr) {
      const char* node_op_type = nullptr;
      graph_api_->OrtNode_GetOpType(node, &node_op_type);
      if (!strcmp(node_op_type, EPCONTEXT_OP)) {
        const char* source_val = nullptr, *ep_sdk_ver_val = nullptr;
        graph_api_->OrtNode_GetAttributeStr(node, SOURCE, &source_val);
        if (!strcmp(source_val, OpenVINOEp.c_str())) {
          graph_api_->OrtNode_GetAttributeStr(node, EP_SDK_VER, &ep_sdk_ver_val);
          if (!strcmp(ep_sdk_ver_val, openvino_sdk_version.c_str())) return true;
          throw std::runtime_error("[Invalid Graph] Versions of OpenVINO used to export blob (" + std::string(ep_sdk_ver_val) +
                    ") and current runtime (" + openvino_sdk_version + ") don't match.");
        }
      }
    }
  }
  return false;
}

}  // namespace openvino_ep
}  // namespace onnxruntime
