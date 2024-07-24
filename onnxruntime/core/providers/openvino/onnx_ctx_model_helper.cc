// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <string>
#include <fstream>
#include <vector>

#include "core/providers/openvino/onnx_ctx_model_helper.h"

namespace onnxruntime {
namespace openvino_ep {

/* Export the serialized blob string embedded onto an EPContext Node
 * along with other metadata necessary to validate the graph on import
 */

Status EPCtxHandler::ExportEPCtxModel(const GraphViewer& graph_viewer,
                                      const std::string& graph_name,
                                      const logging::Logger& logger,
                                      const bool& ep_context_embed_mode,
                                      const std::string& model_blob_str,
                                      const std::string& openvino_sdk_version) const {
  auto model_build = graph_viewer.CreateModel(logger);
  auto& graph_build = model_build->MainGraph();

  // Get graph inputs and outputs
  std::vector<onnxruntime::NodeArg*> inputs, outputs;
  for (auto input : graph_viewer.GetInputs()) {
    auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
    inputs.push_back(&n_input);
  }
  for (auto output : graph_viewer.GetOutputs()) {
    auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
    outputs.push_back(&n_output);
  }

  // Create EP context node attributes
  auto attr_0 = ONNX_NAMESPACE::AttributeProto::Create();
  auto attr_1 = ONNX_NAMESPACE::AttributeProto::Create();
  auto attr_2 = ONNX_NAMESPACE::AttributeProto::Create();
  auto attr_3 = ONNX_NAMESPACE::AttributeProto::Create();

  // embed mode
  attr_0->set_name(EMBED_MODE);
  attr_0->set_type(onnx::AttributeProto_AttributeType_INT);
  attr_0->set_i(ep_context_embed_mode);
  // ep context
  attr_1->set_name(EP_CACHE_CONTEXT);
  attr_1->set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_1->set_s(model_blob_str);
  // sdk version
  attr_2->set_name(EP_SDK_VER);
  attr_2->set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_2->set_s(openvino_sdk_version);
  // source
  attr_3->set_name(SOURCE);
  attr_3->set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_3->set_s(kOpenVINOExecutionProvider);

  auto node_attributes = ONNX_NAMESPACE::NodeAttributes::Create();
  node_attributes->reserve(4);
  node_attributes->emplace(EMBED_MODE, *attr_0);
  node_attributes->emplace(EP_CACHE_CONTEXT, *attr_1);
  node_attributes->emplace(EP_SDK_VER, *attr_2);
  node_attributes->emplace(SOURCE, *attr_3);

  // Create EP context node
  graph_build.AddNode(graph_name, EPCONTEXT_OP, "", inputs, outputs, node_attributes.get(), kMSDomain);
  ORT_ENFORCE(graph_build.Resolve().IsOK());

  // Serialize modelproto to string
  auto new_graph_viewer = graph_build.CreateGraphViewer();
  auto model = new_graph_viewer->CreateModel(logger);
  auto model_proto = model->ToProto();
  new_graph_viewer->ToProto(*model_proto->mutable_graph(), true, true);
  model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  // Finally, dump the model
  std::ofstream epctx_onnx_model(graph_name,
                                 std::ios::out | std::ios::trunc | std::ios::binary);
  if (!epctx_onnx_model) {
    ORT_THROW("Unable to create epctx onnx model file ");
  }
  model_proto->SerializeToOstream(epctx_onnx_model);

  LOGS_DEFAULT(VERBOSE) << "[OpenVINO EP] Export blob as EPContext Node";

  return Status::OK();
}

Status EPCtxHandler::ImportBlobFromEPCtxModel(const GraphViewer& graph_viewer) {
  auto node = graph_viewer.GetNode(0);
  auto& attrs = node->GetAttributes();
  ORT_ENFORCE(attrs.count(EP_CACHE_CONTEXT) > 0);
  model_stream_ = std::make_shared<std::istringstream>(attrs.at(EP_CACHE_CONTEXT).s());
  LOGS_DEFAULT(VERBOSE) << "[OpenVINO EP] Read blob from EPContext Node";

  is_valid_ep_ctx_graph_ = true;
  return Status::OK();
}

bool EPCtxHandler::CheckForOVEPCtxNode(const GraphViewer& graph_viewer, std::string openvino_sdk_version) const {
  for (int i = 0; i < graph_viewer.MaxNodeIndex(); ++i) {
    auto node = graph_viewer.GetNode(i);
    auto& attrs = node->GetAttributes();

    // Check for correct Op Type, EP SOURCE, and SDK version
    if (node != nullptr && node->OpType() == EPCONTEXT_OP) {
      if (attrs.at(SOURCE).s() == kOpenVINOExecutionProvider) {
        if (attrs.at(EP_SDK_VER).s() == openvino_sdk_version) {
          return true;
        } else {
          ORT_THROW("[Invalid Graph] Versions of OpenVINO used to export blob (" + attrs.at(EP_SDK_VER).s() +
                    ") and current runtime (" + openvino_sdk_version + ") don't match.");
        }
      }
    }
  }
  return false;
}

}  // namespace openvino_ep
}  // namespace onnxruntime
