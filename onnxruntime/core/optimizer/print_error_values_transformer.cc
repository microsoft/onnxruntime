// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef PRINT_ERROR_VALUES

#include "core/optimizer/initializer.h"
#include "core/optimizer/print_error_values_transformer.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status PrintErrorValuesTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                              const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr) {
      continue;
    }

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (graph.NodeProducesGraphOutput(node)) {
      continue;
    }

    if (node.OpType() == "PrintErrorValues") {
      continue;
    }

    std::vector<NodeArg*>& output_args = node.MutableOutputDefs();

    for (auto output_edge_iter = node.OutputEdgesBegin(); output_edge_iter != node.OutputEdgesEnd(); ++output_edge_iter) {
      int output_src_arg_index = output_edge_iter->GetSrcArgIndex();
      int output_dst_arg_index = output_edge_iter->GetDstArgIndex();

      auto output_src_arg = output_args[output_src_arg_index];
      auto proto_type = output_src_arg->TypeAsProto();

      if (!proto_type || !proto_type->tensor_type().has_elem_type()) {
        continue;
      }

      auto data_type = proto_type->tensor_type().elem_type();
      if (data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
          data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        continue;
      }

      Node* output_node = graph.GetNode(output_edge_iter->GetNode().Index());

      if (output_node->OpType() == "PrintErrorValues") {
        continue;
      }

      NodeArg& output_dst_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("PrintErrorValues"), proto_type);

      std::array<NodeArg*, 1> inputs{output_src_arg};
      std::array<NodeArg*, 1> outputs{&output_dst_arg};

      const std::string op_type = "PrintErrorValues";
      Node& print_error_values_node = graph.AddNode(
          graph.GenerateNodeName(op_type),
          op_type,
          "",
          inputs,
          outputs,
          nullptr,
          kMSDomain);
      print_error_values_node.AddAttribute("node_name", node.Name());
      print_error_values_node.AddAttribute("node_type", node.OpType());
      print_error_values_node.AddAttribute("execution_provider", node.GetExecutionProviderType());
      print_error_values_node.AddAttribute("node_output_index", static_cast<int64_t>(output_src_arg_index));
      print_error_values_node.SetExecutionProviderType(kCpuExecutionProvider);

      graph.RemoveEdge(node.Index(), output_node->Index(), output_src_arg_index, output_dst_arg_index);
      graph.AddEdge(node.Index(), print_error_values_node.Index(), output_src_arg_index, 0);
      graph.AddEdge(print_error_values_node.Index(), output_node->Index(), 0, output_dst_arg_index);

      // We modified the node's output nodes, so reset the iterator
      output_edge_iter = node.OutputEdgesBegin();

      modified = true;
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime

#endif
