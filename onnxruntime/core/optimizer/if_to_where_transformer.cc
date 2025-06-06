// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/if_to_where_transformer.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"

namespace onnxruntime {

static Status InlineSubgraph(
    Graph& main_graph,
    const Graph& subgraph,
    std::unordered_map<std::string, NodeArg*>& name_to_nodearg,
    std::vector<NodeArg*>& subgraph_outputs,
    const logging::Logger& logger) {
  for (const auto* input : subgraph.GetInputs()) {
    NodeArg* outer_arg = main_graph.GetNodeArg(input->Name());
    if (!outer_arg) {
      // If missing, try to add initializer from subgraph to main graph
      const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
      const ONNX_NAMESPACE::TensorProto* initializer_input = nullptr;
      bool has_initializer = subgraph.GetInitializedTensor(input->Name(), initializer);
      if (has_initializer && initializer != nullptr) {
        // If not in main graph, add initializer
        if (!main_graph.GetInitializedTensor(input->Name(), initializer_input)) {
          LOGS(logger, VERBOSE) << "Adding initializer '" << input->Name() << "' to main graph.";
          main_graph.AddInitializedTensor(*initializer);
        }
      }
      // Create NodeArg in main graph for input
      outer_arg = &main_graph.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
    }
    name_to_nodearg[input->Name()] = outer_arg;
  }

  // Copy subgraph initializers to main graph
  for (const auto& pair : subgraph.GetAllInitializedTensors()) {
    const std::string& init_name = pair.first;
    const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
    bool has_initializer = subgraph.GetInitializedTensor(init_name, initializer);
    if (has_initializer && initializer != nullptr) {
      const ONNX_NAMESPACE::TensorProto* initializer_input = nullptr;
      if (!main_graph.GetInitializedTensor(init_name, initializer_input)) {
        LOGS(logger, VERBOSE) << "Creating NodeArg for subgraph initializer '" << init_name << "' in main graph.";
        main_graph.AddInitializedTensor(*initializer);
      }
      main_graph.GetOrCreateNodeArg(init_name, nullptr);
    }
  }

  // Inline nodes
  for (const auto& node : subgraph.Nodes()) {
    std::vector<NodeArg*> inputs;
    for (const auto* input_arg : node.InputDefs()) {
      NodeArg* mapped_input = nullptr;
      auto it = name_to_nodearg.find(input_arg->Name());
      if (it != name_to_nodearg.end()) {
        mapped_input = it->second;
      } else {
        mapped_input = main_graph.GetNodeArg(input_arg->Name());
        if (!mapped_input) {
          // Add NodeArg if missing
          mapped_input = &main_graph.GetOrCreateNodeArg(input_arg->Name(), input_arg->TypeAsProto());
        }
      }
      if (!mapped_input) {
        LOGS(logger, ERROR) << "Node input '" << input_arg->Name() << "' not found in main graph for node '" << node.Name() << "'";
        continue;
      }
      inputs.push_back(mapped_input);
    }

    // Outputs
    std::vector<NodeArg*> outputs;
    for (const auto* output_arg : node.OutputDefs()) {
      // Create a unique NodeArg in main graph for each output
      NodeArg* mapped_output = &main_graph.GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
      name_to_nodearg[output_arg->Name()] = mapped_output;
      outputs.push_back(mapped_output);
    }

    // Add node to main graph
    main_graph.AddNode(
        main_graph.GenerateNodeName(node.OpType()),
        node.OpType(),
        node.Description(),
        inputs,
        outputs,
        &node.GetAttributes(),
        node.Domain());
  }

  // Map subgraph outputs to main graph NodeArgs
  for (const auto* output : subgraph.GetOutputs()) {
    auto it = name_to_nodearg.find(output->Name());
    NodeArg* mapped_output = (it != name_to_nodearg.end()) ? it->second : main_graph.GetNodeArg(output->Name());
    if (!mapped_output) {
      LOGS(logger, VERBOSE) << "Skipping inlining: Missing output: ", output->Name(), " in main graph after inlining.";
      return Status::OK();  // Skip Transformation
    }
    subgraph_outputs.push_back(mapped_output);
  }

  return Status::OK();
}

static bool CanTransformIfToWhere(const Node& if_node) {
  const auto& outputs = if_node.OutputDefs();
  for (const auto* output : outputs) {
    const auto* type_proto = output->TypeAsProto();
    if (!type_proto) return false;

    // If any output is optional, we skip the transformation
    if (type_proto->value_case() == onnx::TypeProto::kOptionalType) {
      return false;
    }

    // Optionally, ensure it's a tensor
    if (type_proto->value_case() != onnx::TypeProto::kTensorType) {
      return false;
    }

    // Sequence type is not supported by Where Op
    if (type_proto->value_case() == onnx::TypeProto::kSequenceType) {
      return false;
    }
  }
  return true;
}

Status IfToWhereTransformer::ApplyImpl(Graph& graph,
                                       bool& modified,
                                       int graph_level,
                                       const logging::Logger& logger) const {
  modified = false;
  (void)graph_level;
  for (auto it = graph.Nodes().begin(), end = graph.Nodes().end(); it != end; ++it) {
    Node& if_node = *it;
    if (if_node.OpType() != "If")
      continue;

    if (!CanTransformIfToWhere(if_node)) {
      LOGS(logger, VERBOSE) << "Skipping IfToWhere transformation for node " << if_node.Name()
                            << " due to optional or unsupported output types.";
      continue;
    }

    const NodeArg* cond_arg = if_node.InputDefs()[0];
    const onnx::TypeProto* type_proto = cond_arg->TypeAsProto();
    if (type_proto && type_proto->has_tensor_type()) {
      if (type_proto->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_BOOL) {
        continue;
      }
    }
    if (graph_utils::IsConstantInitializer(graph, cond_arg->Name(), true))
      continue;

    // Extract the two subgraph branches
    Graph* then_sub = if_node.GetMutableGraphAttribute("then_branch");
    Graph* else_sub = if_node.GetMutableGraphAttribute("else_branch");

    if (then_sub->GetOutputs().size() != else_sub->GetOutputs().size() || then_sub->GetOutputs().size() != if_node.OutputDefs().size()) {
      LOGS(logger, INFO) << "Mismatch in output sizes between then/else branches and If node.";
      continue;
    }
    std::unordered_map<std::string, NodeArg*> then_map, else_map;
    std::vector<NodeArg*> then_outputs, else_outputs;
    ORT_RETURN_IF_ERROR(InlineSubgraph(graph, *then_sub, then_map, then_outputs, logger));
    ORT_RETURN_IF_ERROR(InlineSubgraph(graph, *else_sub, else_map, else_outputs, logger));

    // Build one Where node per original If output
    const auto& if_outputs = if_node.MutableOutputDefs();
    size_t num_outputs = if_outputs.size();

    for (size_t i = 0; i < num_outputs; ++i) {
      NodeArg* out_arg = if_outputs[i];
      graph.AddNode(
          graph.GenerateNodeName("IfToWhere"),
          "Where",
          "Select between then/else outputs",
          {const_cast<NodeArg*>(cond_arg), then_outputs[i], else_outputs[i]},
          {out_arg});
    }

    std::vector<std::tuple<NodeIndex, NodeIndex, int, int>> edges_to_remove;
    for (auto edge_it = if_node.OutputEdgesBegin(), last = if_node.OutputEdgesEnd();
         edge_it != last; ++edge_it) {
      edges_to_remove.emplace_back(
          if_node.Index(),             // src node index
          edge_it->GetNode().Index(),  // dst node index
          edge_it->GetSrcArgIndex(),   // which output slot of 'if_node'
          edge_it->GetDstArgIndex());  // which input slot on consumer
    }

    for (auto& edge : edges_to_remove) {
      graph.RemoveEdge(
          std::get<0>(edge),
          std::get<1>(edge),
          std::get<2>(edge),
          std::get<3>(edge));
    }

    graph.RemoveNode(if_node.Index());
    modified = true;
    break;
  }

  if (modified) {
    ORT_RETURN_IF_ERROR(graph.Resolve());
  }
  return Status::OK();
}

}  // namespace onnxruntime
