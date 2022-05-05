// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/xnnpack/optimizer/xnnpack_transformer.h"

#include <deque>

#include "core/framework/op_node_proto_helper.h"
#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/graph/schema_registry.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nhwc_transformer.h"
#include "core/optimizer/selectors_actions/helpers.h"
#include "core/optimizer/transpose_optimizer/optimizer_utils.h"
#include "core/optimizer/utils.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/xnnpack/optimizer/common.h"
#include "core/xnnpack/optimizer/conv.h"
#include "core/xnnpack/optimizer/maxpool.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_layout_transformation;
using namespace onnxruntime::xnnpack;

#define DEFINE_PROCESSOR(DOMAIN, OP_TYPE, CLASS_NAME)                                                               \
  processors_[std::make_pair(DOMAIN, OP_TYPE)] = [](const Node& node,                                               \
                                                    const std::unordered_set<const NodeArg*>& graph_const_values) { \
    return new CLASS_NAME##NodeProcessor(node, graph_const_values);                                                 \
  };

namespace onnxruntime {

XNNPackTransformer::XNNPackTransformer(AllocatorPtr cpu_allocator) noexcept
    : GraphTransformer("XNNPackTransformer"), cpu_allocator_(std::move(cpu_allocator)) {
  DEFINE_PROCESSOR(kOnnxDomain, "Conv", Conv);
  DEFINE_PROCESSOR(kOnnxDomain, "MaxPool", MaxPool);
};

Status XNNPackTransformer::ApplyImpl(Graph& main_graph, bool& modified, int graph_level,
                                     const logging::Logger& logger) const {
  IOnnxRuntimeOpSchemaCollectionPtr ptr = main_graph.GetSchemaRegistry();
  if (ptr == nullptr) {
    return Status::OK();
  }
  const ONNX_NAMESPACE::OpSchema* xnnPackMaxPooling2dSchema = ptr->GetSchema("XnnPackMaxPooling2d", 1, "com.microsoft");
  if (xnnPackMaxPooling2dSchema == nullptr) {
    return Status::OK();
  }
  GraphViewer gv(main_graph);
  // Run constant propagation for XNNPack EP. XNNPack expects that weights are constant.
  // Here we expect a constant folding optimizer will be invoked at least once after this NhwcTransformer and
  // XNNPackTransformer. So I can't register XNNPack Optimizer before the constant folding optimizer.
  std::unordered_set<const NodeArg*> graph_const_values;

  for (auto index : gv.GetNodesInTopologicalOrder()) {
    auto& node = *main_graph.GetNode(index);
    if (!node.ContainsSubgraph() && node.OpType() != "DequantizeLinear" && node.OpType() != "QuantizeLinear" &&
        optimizer_utils::IsOperationDeterministic(node.Domain(), node.OpType())) {
      bool is_all_const = true;
      for (const NodeArg* in : node.InputDefs()) {
        if (!in->Exists()) continue;
        if (graph_const_values.find(in) != graph_const_values.end()) continue;
        if (main_graph.GetConstantInitializer(in->Name(), false) != nullptr) {
          graph_const_values.insert(in);
          continue;
        }
        // This input is not const
        is_all_const = false;
      }
      if (is_all_const) {
        for (const NodeArg* out : node.OutputDefs()) {
          graph_const_values.insert(out);
        }
      }
    }
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
  }

  std::unordered_map<Node*, std::unique_ptr<::ONNX_NAMESPACE::GraphProto>> updated_nodes;
  Status st;
  for (auto& nodeRef : gv.Nodes()) {
    auto inputs = nodeRef.InputDefs();
    auto iter_end = nodeRef.InputEdgesEnd();
    if (nodeRef.OpType() == "DequantizeLinear") {
      return Status::OK();
    }
    auto iter =
        processors_.find(std::make_pair<std::string_view, std::string_view>(nodeRef.Domain(), nodeRef.OpType()));
    if (iter != processors_.end()) {
      std::unique_ptr<::ONNX_NAMESPACE::GraphProto> subgraph;
      std::unique_ptr<NodeProcessor> p(iter->second(nodeRef, graph_const_values));
      st = p->Generate(subgraph);
      if (st.IsOK()) {
        if (subgraph) {
          Node* node_p = main_graph.GetNode(nodeRef.Index());
          if (node_p == nullptr) continue;
          updated_nodes[node_p] = std::move(subgraph);
        }
      } else {
        LOGS(logger, INFO) << "Convert node failed: " << st.ErrorMessage();
      }
    }
  }
  for (auto& tvp : updated_nodes) {
    Node* node_p = tvp.first;
    ORT_RETURN_IF_ERROR(main_graph.ReplaceNodeWithSubgraph(*node_p, std::move(tvp.second)));
  }
  modified = !updated_nodes.empty();
  if (modified) {
    ORT_RETURN_IF_ERROR(main_graph.Resolve());
    auto api_graph = MakeApiGraph(main_graph, cpu_allocator_, kCpuExecutionProvider);
    // Ignore the return value.
    onnx_layout_transformation::Optimize(*api_graph, /*allow_extended_ops*/ true);
  }

  return Status::OK();
}

}  // namespace onnxruntime
