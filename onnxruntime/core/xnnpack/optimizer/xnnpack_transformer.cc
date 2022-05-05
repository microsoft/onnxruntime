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
#include "core/xnnpack/optimizer/maxpool.h"
#include "core/xnnpack/optimizer/conv.h"
#include "core/xnnpack/optimizer/trival_subgraph.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_layout_transformation;

namespace onnxruntime {

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

  std::vector<Node*> updated_nodes;
  // Iterate all the nodes first to figure out what can be run by XNNPack. Then we will update the selected nodes one by
  // one. However, there could be chance that in the first pass we thought a node is supported by XNNPack, then we did
  // some updates on the graph which break the assumption. For example, if there is a Maxpool followd by a Conv. At
  // first, the input channel of Conv is known, then we replaced ONNX Maxpool with XNNPack Maxpool and run shape
  // inference again. Assume the XNNPack maxpool shape inference didn't do a great job and lost of information of the
  // output channel dim of the Maxpool, then this transformer would failed to update ONNX Conv to XNNPack Conv because
  // the later one expects the input channel should be known. So the shape inference functions of XNNPack schemas play a
  // key role here.
  for (auto& nodeRef : gv.Nodes()) {
    auto inputs = nodeRef.InputDefs();
    auto iter_end = nodeRef.InputEdgesEnd();
    if (nodeRef.OpType() == "DequantizeLinear") {
      return Status::OK();
    }
    std::unique_ptr<::ONNX_NAMESPACE::GraphProto> subgraph;
    Status st = ReplaceMaxPool(nodeRef, subgraph);
    if (!st.IsOK() || !subgraph) {
      st = ReplaceConv(nodeRef, graph_const_values, subgraph);
    }
    if (st.IsOK() && subgraph.get() != nullptr) {
      Node* node_p = main_graph.GetNode(nodeRef.Index());
      if (node_p == nullptr) continue;
      node_p->SetFunctionBody(std::make_unique<TrivalSubgraph>(main_graph, nodeRef, std::move(subgraph)));
      updated_nodes.push_back(node_p);
    }    
  }
  for (Node* node_p : updated_nodes) {
    ORT_RETURN_IF_ERROR(main_graph.InlineFunction(*node_p));
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
