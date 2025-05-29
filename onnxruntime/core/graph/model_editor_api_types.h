// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

namespace onnxruntime {
struct ModelEditorGraph {
  onnxruntime::InlinedVector<std::unique_ptr<OrtValueInfo>> inputs;
  onnxruntime::InlinedVector<std::unique_ptr<OrtValueInfo>> outputs;
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> initializers;
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> external_initializers;
  std::vector<std::unique_ptr<OrtNode>> nodes;
};
}  // namespace onnxruntime

struct OrtGraph {
  explicit OrtGraph(onnxruntime::ModelEditorGraph&& editor_graph) : graph_variant_(std::move(editor_graph)) {}
  explicit OrtGraph(const onnxruntime::GraphViewer& graph_viewer) : graph_variant_(&graph_viewer) {}

  std::variant<std::monostate,
               onnxruntime::ModelEditorGraph,
               gsl::not_null<const onnxruntime::GraphViewer*>>
      graph_variant_;

  const onnxruntime::GraphViewer* TryGetGraphViewer() const {
    const auto* impl_ptr = std::get_if<gsl::not_null<const onnxruntime::GraphViewer*>>(&graph_variant_);
    return (impl_ptr == nullptr) ? nullptr : impl_ptr->get();
  }

  const onnxruntime::ModelEditorGraph* TryGetModelEditorGraph() const {
    return std::get_if<onnxruntime::ModelEditorGraph>(&graph_variant_);
  }

  onnxruntime::ModelEditorGraph* TryGetModelEditorGraph() {
    return std::get_if<onnxruntime::ModelEditorGraph>(&graph_variant_);
  }
};

struct OrtModel {
  std::unique_ptr<OrtGraph> graph;
  std::unordered_map<std::string, int> domain_to_version;
};
