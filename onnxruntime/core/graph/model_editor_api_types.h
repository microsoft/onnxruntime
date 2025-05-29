// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_map>
#include <variant>
#include "core/common/inlined_containers_fwd.h"
#include "core/framework/ort_value.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/onnx_protobuf.h"

// ORT C interface types for OrtGraphApi can't be in a namespace.
// We need to define them here so onnxruntime::Model can be created from OrtModel.

struct OrtValueInfo {
  std::string name;
  std::unique_ptr<OrtTypeInfo> type_info;
};

struct OrtOpAttr {
  ONNX_NAMESPACE::AttributeProto attr_proto;
};

struct OrtNode {
  enum class Type {
    kInvalid = 0,
    kEditorNode,
    kEpNode,
  };

  OrtNode() = default;
  explicit OrtNode(OrtNode::Type type) : type(type) {}
  virtual ~OrtNode() = default;
  OrtNode::Type type = OrtNode::Type::kInvalid;
};

namespace onnxruntime {
struct ModelEditorNode : public OrtNode {
  ModelEditorNode() : OrtNode(OrtNode::Type::kEditorNode) {}
  OrtNode* ToExternal() { return static_cast<OrtNode*>(this); }
  const OrtNode* ToExternal() const { return static_cast<const OrtNode*>(this); }

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

struct EpNode : public OrtNode {
  EpNode(const onnxruntime::Node& node) : OrtNode(OrtNode::Type::kEditorNode), node(node) {}
  OrtNode* ToExternal() { return static_cast<OrtNode*>(this); }
  const OrtNode* ToExternal() const { return static_cast<const OrtNode*>(this); }

  const onnxruntime::Node& node;
};
}  // namespace onnxruntime

struct OrtGraph {
  enum class Type {
    kInvalid = 0,
    kEditorGraph,
    kEpGraph,
  };

  OrtGraph() = default;
  explicit OrtGraph(OrtGraph::Type type) : type(type) {}
  virtual ~OrtGraph() = default;
  OrtGraph::Type type = OrtGraph::Type::kInvalid;
};

namespace onnxruntime {
struct ModelEditorGraph : public OrtGraph {
  ModelEditorGraph() : OrtGraph(OrtGraph::Type::kEditorGraph) {}
  OrtGraph* ToExternal() { return static_cast<OrtGraph*>(this); }
  const OrtGraph* ToExternal() const { return static_cast<const OrtGraph*>(this); }

  onnxruntime::InlinedVector<std::unique_ptr<OrtValueInfo>> inputs;
  onnxruntime::InlinedVector<std::unique_ptr<OrtValueInfo>> outputs;
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> initializers;
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> external_initializers;
  std::vector<std::unique_ptr<onnxruntime::ModelEditorNode>> nodes;
};

struct EpGraph : public OrtGraph {
  explicit EpGraph(const GraphViewer& g_viewer) : OrtGraph(OrtGraph::Type::kEpGraph), graph_viewer(g_viewer) {
    nodes.reserve(g_viewer.NumberOfNodes());
    for (const Node& node : g_viewer.Nodes()) {
      nodes.push_back(std::make_unique<EpNode>(node));
      index_to_node[node.Index()] = nodes.back().get();
    }
  }
  OrtGraph* ToExternal() { return static_cast<OrtGraph*>(this); }
  const OrtGraph* ToExternal() const { return static_cast<const OrtGraph*>(this); }

  const onnxruntime::GraphViewer& graph_viewer;
  std::vector<std::unique_ptr<EpNode>> nodes;
  std::unordered_map<NodeIndex, EpNode*> index_to_node;
};
}  // namespace onnxruntime

struct OrtModel {
  std::unique_ptr<OrtGraph> graph;
  std::unordered_map<std::string, int> domain_to_version;
};
