// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef PRINT_TOLERANCE_ERRORS

#include "core/optimizer/initializer.h"
#include "core/optimizer/print_tolerance_errors_transformer.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

static bool HasFloat16Inputs(const Node& node) {
  auto input_args = node.InputDefs();

  for (const NodeArg* input_arg : input_args) {
    auto proto_type = input_arg->TypeAsProto();

    if (!proto_type || !proto_type->tensor_type().has_elem_type()) {
      continue;
    }

    if (proto_type->tensor_type().elem_type() != TensorProto_DataType_FLOAT16) {
      continue;
    }

    return true;
  }

  return false;
}

static bool HasFloat16Outputs(const Node& node) {
  auto output_args = node.OutputDefs();

  for (const NodeArg* output_arg : output_args) {
    auto proto_type = output_arg->TypeAsProto();

    if (!proto_type || !proto_type->tensor_type().has_elem_type()) {
      continue;
    }

    if (proto_type->tensor_type().elem_type() != TensorProto_DataType_FLOAT16) {
      continue;
    }

    return true;
  }

  return false;
}

static bool HasPrintToleranceErrorsOutput(const Node& node) {
  for (auto output_edge_iter = node.OutputEdgesBegin(); output_edge_iter != node.OutputEdgesEnd(); ++output_edge_iter) {
    if (output_edge_iter->GetNode().OpType() == "PrintToleranceErrors") {
      return true;
    }
  }

  return false;
}

static void CastNodeInputsToFloat32(Graph& graph, Node& node) {
  std::vector<NodeArg*>& input_args = node.MutableInputDefs();

  // Add fp16 -> fp32 cast nodes to all inputs
  for (auto input_edge_iter = node.InputEdgesBegin(); input_edge_iter != node.InputEdgesEnd();) {
    int src_arg_index = input_edge_iter->GetSrcArgIndex();
    int dst_arg_index = input_edge_iter->GetDstArgIndex();
    auto proto_type = input_args[dst_arg_index]->TypeAsProto();

    if (!proto_type || !proto_type->tensor_type().has_elem_type() || proto_type->tensor_type().elem_type() != TensorProto_DataType_FLOAT16) {
      ++input_edge_iter;
      continue;
    }

    const Node& prev_node = input_edge_iter->GetNode();

    // Remove the fp16 input edge
    graph.RemoveEdge(prev_node.Index(), node.Index(), src_arg_index, dst_arg_index);

    TypeProto fp32_proto_type;
    fp32_proto_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    NodeArg& cast_output_node_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("Cast"), &fp32_proto_type);

    std::array<NodeArg*, 1> cast_inputs = {input_args[dst_arg_index]};
    std::array<NodeArg*, 1> cast_outputs = {&cast_output_node_arg};

    Node& cast_node = graph.AddNode(
        graph.GenerateNodeName("Cast"),
        "Cast",
        "",
        cast_inputs,
        cast_outputs);

    cast_node.AddAttribute("to", static_cast<int64_t>(TensorProto_DataType_FLOAT));
    cast_node.SetExecutionProviderType(node.GetExecutionProviderType());

    // Change the datatype of the current input
    input_args[dst_arg_index] = &cast_output_node_arg;

    // Add the Cast edges
    graph.AddEdge(prev_node.Index(), cast_node.Index(), src_arg_index, 0);
    graph.AddEdge(cast_node.Index(), node.Index(), 0, dst_arg_index);

    input_edge_iter = node.InputEdgesBegin();
  }

  // 3. Add fp16 -> fp32 cast nodes for every float16 initializer
  for (int input_arg_index = 0; input_arg_index < input_args.size(); ++input_arg_index) {
    auto proto_type = input_args[input_arg_index]->TypeAsProto();

    if (!proto_type || !proto_type->tensor_type().has_elem_type() || proto_type->tensor_type().elem_type() != TensorProto_DataType_FLOAT16) {
      continue;
    }

    if (!graph.IsInitializedTensor(input_args[input_arg_index]->Name())) {
      continue;
    }

    TypeProto fp32_proto_type;
    fp32_proto_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    NodeArg& cast_output_node_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("Cast"), &fp32_proto_type);

    std::array<NodeArg*, 1> cast_inputs = {input_args[input_arg_index]};
    std::array<NodeArg*, 1> cast_outputs = {&cast_output_node_arg};

    Node& cast_node = graph.AddNode(
        graph.GenerateNodeName("Cast"),
        "Cast",
        "",
        cast_inputs,
        cast_outputs);
    cast_node.AddAttribute("to", static_cast<int64_t>(TensorProto_DataType_FLOAT));
    cast_node.SetExecutionProviderType(node.GetExecutionProviderType());

    // Change the datatype of the current initializer
    input_args[input_arg_index] = &cast_output_node_arg;

    // Add the cast edge
    graph.AddEdge(cast_node.Index(), node.Index(), 0, input_arg_index);
  }
}

static void CastNodeOutputsToFloat16(Graph& graph, Node& node) {
  std::vector<NodeArg*>& output_args = node.MutableOutputDefs();
  std::vector<std::tuple<int, int, const Node*>> removed_edges;

  // Remove all fp16 output edges
  for (auto output_edge_iter = node.OutputEdgesBegin(); output_edge_iter != node.OutputEdgesEnd();) {
    int src_arg_index = output_edge_iter->GetSrcArgIndex();
    int dst_arg_index = output_edge_iter->GetDstArgIndex();
    auto proto_type = output_args[src_arg_index]->TypeAsProto();

    if (!proto_type || !proto_type->tensor_type().has_elem_type() || proto_type->tensor_type().elem_type() != TensorProto_DataType_FLOAT16) {
      ++output_edge_iter;
      continue;
    }

    const Node& next_node = output_edge_iter->GetNode();

    // Remove the fp16 output edge
    graph.RemoveEdge(node.Index(), next_node.Index(), src_arg_index, dst_arg_index);

    removed_edges.emplace_back(src_arg_index, dst_arg_index, &next_node);
    output_edge_iter = node.OutputEdgesBegin();
  }

  // Convert all fp16 output defs to fp32
  for (NodeArg*& output_arg : output_args) {
    auto proto_type = output_arg->TypeAsProto();

    if (proto_type && proto_type->tensor_type().has_elem_type() && proto_type->tensor_type().elem_type() == TensorProto_DataType_FLOAT16) {
      TypeProto fp32_proto_type;
      fp32_proto_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
      output_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.OpType()), &fp32_proto_type);
    }
  }

  // Add the fp32 -> fp16 cast edges
  for (const auto& removed_edge : removed_edges) {
    const auto& [src_arg_index, dst_arg_index, next_node] = removed_edge;

    TypeProto fp16_proto_type;
    fp16_proto_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);
    NodeArg& cast_output_node_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("Cast"), &fp16_proto_type);

    std::array<NodeArg*, 1> cast_inputs = {output_args[src_arg_index]};
    std::array<NodeArg*, 1> cast_outputs = {&cast_output_node_arg};

    Node& cast_node = graph.AddNode(
        graph.GenerateNodeName("Cast"),
        "Cast",
        "",
        cast_inputs,
        cast_outputs);

    cast_node.AddAttribute("to", static_cast<int64_t>(TensorProto_DataType_FLOAT16));
    cast_node.SetExecutionProviderType(node.GetExecutionProviderType());

    // Add the cast nodes
    graph.AddEdge(node.Index(), cast_node.Index(), src_arg_index, 0);
    graph.AddEdge(cast_node.Index(), next_node->Index(), 0, dst_arg_index);
  }
}

Status PrintToleranceErrorsTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                                  const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  // Nodes to force fp32 execution for. When a node has a a huge difference between its fp16 and fp32 results,
  // adding it to this list can help figure out whether keeping the node in fp32 would fix issues in the model.
  std::unordered_set<std::string> force_fp32_ops;
  auto force_fp32_ops_var = Env::Default().GetEnvironmentVar("ORT_DEBUG_FORCE_FP32_OPS");
  std::stringstream ops_ss(force_fp32_ops_var);

  while (ops_ss.good()) {
    std::string operator_name;
    getline(ops_ss, operator_name, ',');
    force_fp32_ops.insert(operator_name);
  }

  std::unordered_set<std::string> force_fp32_nodes;
  auto force_fp32_nodes_var = Env::Default().GetEnvironmentVar("ORT_DEBUG_FORCE_FP32_NODES");
  std::stringstream nodes_ss(force_fp32_nodes_var);

  while (nodes_ss.good()) {
    std::string node_name;
    getline(nodes_ss, node_name, ',');
    force_fp32_nodes.insert(node_name);
  }

  for (auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr) {
      continue;
    }

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (node.OpType() == "PrintToleranceErrors" || node.OpType() == "Cast") {
      continue;
    }

    if (graph.NodeProducesGraphOutput(node)) {
      continue;
    }

    if (force_fp32_ops.count(node.OpType())) {
      CastNodeInputsToFloat32(graph, node);
      CastNodeOutputsToFloat16(graph, node);
      continue;
    }

    if (force_fp32_nodes.count(node.Name())) {
      CastNodeInputsToFloat32(graph, node);
      CastNodeOutputsToFloat16(graph, node);
      continue;
    }

    // Don't re-process the node if it already has the print node as an output
    if (HasPrintToleranceErrorsOutput(node)) {
      continue;
    }

    // Skip nodes that don't have float16 inputs/outputs
    if (!HasFloat16Inputs(node) || !HasFloat16Outputs(node)) {
      continue;
    }

    // Convert the cloned node's fp16 inputs to fp32
    std::vector<NodeArg*>& input_args = node.MutableInputDefs();
    std::vector<NodeArg*> clone_input_args;

    for (NodeArg* input_arg : input_args) {
      auto proto_type = input_arg->TypeAsProto();
      NodeArg* clone_input_arg = nullptr;

      if (proto_type && proto_type->tensor_type().has_elem_type() && proto_type->tensor_type().elem_type() == TensorProto_DataType_FLOAT16) {
        TypeProto clone_proto_type;
        clone_proto_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
        clone_input_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.OpType()), &clone_proto_type);
      } else {
        clone_input_arg = input_arg;
      }

      clone_input_args.push_back(clone_input_arg);
    }

    // Convert the cloned node's fp16 outputs to fp32
    std::vector<NodeArg*>& output_args = node.MutableOutputDefs();
    std::vector<NodeArg*> clone_output_args;

    for (const NodeArg* output_arg : output_args) {
      auto proto_type = output_arg->TypeAsProto();
      NodeArg* clone_output_arg = nullptr;

      if (proto_type && proto_type->tensor_type().has_elem_type() && proto_type->tensor_type().elem_type() == TensorProto_DataType_FLOAT16) {
        TypeProto clone_proto_type;
        clone_proto_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
        clone_output_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.OpType()), &clone_proto_type);
      } else {
        clone_output_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.OpType()), proto_type);
      }

      clone_output_args.push_back(clone_output_arg);
    }

    // Clone the current node
    Node& cloned_node = graph.AddNode(
        graph.GenerateNodeName(node.OpType()),
        node.OpType(),
        "",
        clone_input_args,
        clone_output_args,
        &node.GetAttributes(),
        node.Domain());
    cloned_node.SetExecutionProviderType(node.GetExecutionProviderType());

    // Add fp16 -> fp32 cast edges to all fp32 inputs of the cloned node, and normal edges to other nodes
    for (auto input_edge_iter = node.InputEdgesBegin(); input_edge_iter != node.InputEdgesEnd(); ++input_edge_iter) {
      int src_arg_index = input_edge_iter->GetSrcArgIndex();
      int dst_arg_index = input_edge_iter->GetDstArgIndex();
      auto proto_type = input_args[dst_arg_index]->TypeAsProto();

      if (!proto_type || !proto_type->tensor_type().has_elem_type() || proto_type->tensor_type().elem_type() != TensorProto_DataType_FLOAT16) {
        const Node& prev_node = input_edge_iter->GetNode();
        graph.AddEdge(prev_node.Index(), cloned_node.Index(), src_arg_index, dst_arg_index);
      } else {
        std::array<NodeArg*, 1> cast_inputs = {input_args[dst_arg_index]};
        std::array<NodeArg*, 1> cast_outputs = {clone_input_args[dst_arg_index]};

        Node& cast_node = graph.AddNode(
            graph.GenerateNodeName("Cast"),
            "Cast",
            "",
            cast_inputs,
            cast_outputs);

        cast_node.AddAttribute("to", static_cast<int64_t>(TensorProto_DataType_FLOAT));
        cast_node.SetExecutionProviderType(node.GetExecutionProviderType());

        // Add the Cast edges
        const Node& prev_node = input_edge_iter->GetNode();
        graph.AddEdge(prev_node.Index(), cast_node.Index(), src_arg_index, 0);
        graph.AddEdge(cast_node.Index(), cloned_node.Index(), 0, dst_arg_index);
      }
    }

    // Convert the cloned node's initializers to fp32
    for (int input_arg_index = 0; input_arg_index < input_args.size(); ++input_arg_index) {
      auto proto_type = input_args[input_arg_index]->TypeAsProto();

      if (!proto_type || !proto_type->tensor_type().has_elem_type() || proto_type->tensor_type().elem_type() != TensorProto_DataType_FLOAT16) {
        continue;
      }

      if (!graph.IsInitializedTensor(input_args[input_arg_index]->Name())) {
        continue;
      }

      std::array<NodeArg*, 1> cast_inputs = {input_args[input_arg_index]};
      std::array<NodeArg*, 1> cast_outputs = {clone_input_args[input_arg_index]};

      Node& cast_node = graph.AddNode(
          graph.GenerateNodeName("Cast"),
          "Cast",
          "",
          cast_inputs,
          cast_outputs);
      cast_node.AddAttribute("to", static_cast<int64_t>(TensorProto_DataType_FLOAT));
      cast_node.SetExecutionProviderType(node.GetExecutionProviderType());

      // Add the cast edge
      graph.AddEdge(cast_node.Index(), cloned_node.Index(), 0, input_arg_index);
    }

    // For each float16 output in the original node, we create a PrintToleranceErrors node that takes the float16 and float32 result as inputs
    for (auto output_edge_iter = node.OutputEdgesBegin(); output_edge_iter != node.OutputEdgesEnd();) {
      int output_src_arg_index = output_edge_iter->GetSrcArgIndex();
      int output_dst_arg_index = output_edge_iter->GetDstArgIndex();
      auto proto_type = output_args[output_src_arg_index]->TypeAsProto();

      if (!proto_type || !proto_type->tensor_type().has_elem_type() || proto_type->tensor_type().elem_type() != TensorProto_DataType_FLOAT16) {
        ++output_edge_iter;
        continue;
      }

      Node* output_node = graph.GetNode(output_edge_iter->GetNode().Index());

      if (output_node->OpType() == "PrintToleranceErrors") {
        ++output_edge_iter;
        continue;
      }

      TypeProto float16_proto_type;
      float16_proto_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);

      TypeProto float32_proto_type;
      float32_proto_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

      NodeArg& print_tolerance_errors_output = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("PrintToleranceErrors"), &float16_proto_type);

      std::array<NodeArg*, 2> print_tolerance_errors_inputs = {output_args[output_src_arg_index], clone_output_args[output_src_arg_index]};
      std::array<NodeArg*, 1> print_tolerance_errors_outputs = {&print_tolerance_errors_output};

      const std::string op_type = "PrintToleranceErrors";
      Node& print_tolerance_errors_node = graph.AddNode(
          graph.GenerateNodeName(op_type),
          op_type,
          "",
          print_tolerance_errors_inputs,
          print_tolerance_errors_outputs,
          nullptr,
          kMSDomain);
      print_tolerance_errors_node.AddAttribute("node_name", node.Name());
      print_tolerance_errors_node.AddAttribute("node_type", node.OpType());
      print_tolerance_errors_node.AddAttribute("execution_provider", node.GetExecutionProviderType());
      print_tolerance_errors_node.AddAttribute("node_output_index", static_cast<int64_t>(output_src_arg_index));
      print_tolerance_errors_node.SetExecutionProviderType(kCpuExecutionProvider);

      // Remove the link between the original node and its output node
      graph.RemoveEdge(node.Index(), output_node->Index(), output_src_arg_index, output_dst_arg_index);

      // Link the float16 node to the first input of PrintToleranceErrors
      graph.AddEdge(node.Index(), print_tolerance_errors_node.Index(), output_src_arg_index, 0);

      // Link the float32 node to the second input of PrintToleranceErrors
      graph.AddEdge(cloned_node.Index(), print_tolerance_errors_node.Index(), output_src_arg_index, 1);

      // Link PrintToleranceErrors to the output of the original node
      graph.AddEdge(print_tolerance_errors_node.Index(), output_node->Index(), 0, output_dst_arg_index);

      // We modified the original node's output nodes, so reset the iterator
      output_edge_iter = node.OutputEdgesBegin();
    }

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime

#endif
