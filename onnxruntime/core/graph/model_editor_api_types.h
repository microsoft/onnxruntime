// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/inlined_containers_fwd.h"
#include "core/framework/ort_value.h"
#include "core/graph/abi_graph_types.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
struct ModelEditorNode : public OrtNode {
  ModelEditorNode() : OrtNode(OrtNode::Type::kEditorNode) {}

  OrtNode* ToExternal() { return static_cast<OrtNode*>(this); }
  const OrtNode* ToExternal() const { return static_cast<const OrtNode*>(this); }
  static ModelEditorNode* ToInternal(OrtNode* ort_node) {
    return ort_node->type == OrtNode::Type::kEditorNode ? static_cast<ModelEditorNode*>(ort_node) : nullptr;
  }
  static const ModelEditorNode* ToInternal(const OrtNode* ort_node) {
    return ort_node->type == OrtNode::Type::kEditorNode ? static_cast<const ModelEditorNode*>(ort_node) : nullptr;
  }

  std::string operator_name;
  std::string domain_name;
  std::string node_name;

  // OrtOpAttr is 1:1 with ONNX_NAMESPACE::AttributeProto currently.
  // https://github.com/microsoft/onnxruntime/blob/bd5a759d0cdbed6e7f611c990d4eb5457a9ecf60/onnxruntime/core/session/standalone_op_invoker.cc#L318
  onnxruntime::InlinedVector<ONNX_NAMESPACE::AttributeProto> attributes;
  onnxruntime::InlinedVector<std::string> input_names;
  onnxruntime::InlinedVector<std::string> output_names;

  // FUTURE if we need control flow nodes
  // std::unordered_map<std::string, OrtGraph> subgraphs;
};

struct ModelEditorGraph : public OrtGraph {
  ModelEditorGraph() : OrtGraph(OrtGraph::Type::kEditorGraph) {}
  OrtGraph* ToExternal() { return static_cast<OrtGraph*>(this); }
  const OrtGraph* ToExternal() const { return static_cast<const OrtGraph*>(this); }
  static ModelEditorGraph* ToInternal(OrtGraph* ort_graph) {
    return ort_graph->type == OrtGraph::Type::kEditorGraph ? static_cast<ModelEditorGraph*>(ort_graph) : nullptr;
  }
  static const ModelEditorGraph* ToInternal(const OrtGraph* ort_graph) {
    return ort_graph->type == OrtGraph::Type::kEditorGraph ? static_cast<const ModelEditorGraph*>(ort_graph) : nullptr;
  }

  onnxruntime::InlinedVector<std::unique_ptr<OrtValueInfo>> inputs;
  onnxruntime::InlinedVector<std::unique_ptr<OrtValueInfo>> outputs;
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> initializers;
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> external_initializers;
  std::vector<std::unique_ptr<onnxruntime::ModelEditorNode>> nodes;
};

}  // namespace onnxruntime

struct OrtModel {
  std::unique_ptr<OrtGraph> graph;
  std::unordered_map<std::string, int> domain_to_version;
};
