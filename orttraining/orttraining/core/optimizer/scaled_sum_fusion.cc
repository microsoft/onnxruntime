// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/scaled_sum_fusion.h"

#include <onnx/defs/attr_proto_util.h>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {

// Supports limited data types.
static constexpr std::array supported_data_types{
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
};

bool IsSupportedDataType(int32_t data_type) {
  return std::find(supported_data_types.cbegin(), supported_data_types.cend(), data_type) !=
         supported_data_types.cend();
}

bool IsShapeEqual(const ONNX_NAMESPACE::TensorShapeProto* lhs_shape,
                  const ONNX_NAMESPACE::TensorShapeProto* rhs_shape) {
  ORT_ENFORCE(lhs_shape != nullptr && rhs_shape != nullptr);

  if (lhs_shape->dim_size() != rhs_shape->dim_size()) {
    return false;
  }

  for (int i = 0; i < lhs_shape->dim_size(); ++i) {
    if (lhs_shape->dim(i).has_dim_value() && rhs_shape->dim(i).has_dim_value()) {
      if (lhs_shape->dim(i).dim_value() != rhs_shape->dim(i).dim_value()) {
        return false;
      }
    } else if (lhs_shape->dim(i).has_dim_param() && rhs_shape->dim(i).has_dim_param()) {
      if (lhs_shape->dim(i).dim_param() != rhs_shape->dim(i).dim_param()) {
        return false;
      }
    } else {
      return false;
    }
  }

  return true;
}

bool IsScaleOperator(Graph& graph, Node& node,
                     const ONNX_NAMESPACE::TensorShapeProto* output_shape,
                     float& scale_value) {
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Div", {7, 13, 14})) {
    // If node is Div, check:
    // 1. The first input has the same shape as the given output_shape.
    // 2. The second input is a constant initializer (containing a scalar or 1-D 1-element tensor).
    bool first_input_check = (node.InputDefs()[0]->Shape() &&
                              IsShapeEqual(node.InputDefs()[0]->Shape(), output_shape));

    if (first_input_check) {
      const Node* div_input_2 = graph_utils::GetInputNode(node, 1);
      auto div_input_2_shape = node.InputDefs()[1]->Shape();
      bool second_input_check = div_input_2 == nullptr && div_input_2_shape &&
                                graph_utils::IsConstantInitializer(graph, node.InputDefs()[1]->Name(), false) &&
                                (div_input_2_shape->dim_size() == 0  // scalar
                                 || (div_input_2_shape->dim_size() == 1 &&
                                     div_input_2_shape->dim(0).has_dim_value() &&
                                     div_input_2_shape->dim(0).dim_value() == 1) /* 1d with 1 element */);

      if (second_input_check) {
        const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
        if (!graph.GetInitializedTensor(node.InputDefs()[1]->Name(), tensor_proto)) {
          return false;
        }

        Initializer init_const{*tensor_proto, graph.ModelPath()};
        const auto data_type = tensor_proto->data_type();
        if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
          const MLFloat16* val = init_const.data<MLFloat16>();
          scale_value = 1.0f / math::halfToFloat(val[0].val);
        } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
          scale_value = 1.0f / *init_const.data<float>();
        } else {
          return false;
        }

        return true;
      }
    }
  }
  return false;
}

}  // namespace

Status ScaledSumFusion::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                                  const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);

  [[maybe_unused]] size_t handled_scaled_sum_count = 0;  // For summary
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr)
      // node was removed.
      continue;

    auto& node = *node_ptr;
    // Find an Add that takes two inputs from other nodes' outputs (instead of any graph inputs or initializers).
    // We also don't allow Add is generating graph outputs.
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Add", {6, 7, 13, 14}) ||
        node.GetInputEdgesCount() != 2 /* two input MUST come from other nodes' outputs */ ||
        graph.IsOutput(node.OutputDefs()[0]) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    const ONNX_NAMESPACE::TensorShapeProto* output_shape = node.OutputDefs()[0]->Shape();
    const ONNX_NAMESPACE::TypeProto* output_type = node.OutputDefs()[0]->TypeAsProto();
    if (!output_shape || !output_type) {
      continue;
    }

    int elem_type = output_type->tensor_type().elem_type();
    if (!IsSupportedDataType(elem_type)) {
      continue;
    }

    InlinedVector<NodeArg*> data_input_args;
    InlinedVector<float> scales;
    data_input_args.reserve(3);
    scales.reserve(3);
    InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;

    // Be noted: it is possible the two input nodes are from the same node.
    const Node* add_input_0 = graph_utils::GetInputNode(node, 0);
    const Node* add_input_1 = graph_utils::GetInputNode(node, 1);
    if (add_input_0 == nullptr || add_input_1 == nullptr) {
      continue;
    }

    // Check the two inputs nodes of Add, if they are scaled operators, add them to the node list to remove.
    auto check_add_input = [&graph, &output_shape, &nodes_to_remove,
                            &data_input_args, &scales](Node* add_input_node) -> bool {
      float scale_value = 1.0f;
      if (!IsScaleOperator(graph, *add_input_node, output_shape, scale_value)) {
        return false;
      }

      // If node is not in nodes_to_remove, add it.
      auto it = std::find_if(nodes_to_remove.begin(), nodes_to_remove.end(),
                             [&add_input_node](std::reference_wrapper<Node> n) {
                               return ((Node&)n).Index() == add_input_node->Index();
                             });
      if (it == nodes_to_remove.end()) {
        nodes_to_remove.push_back(*add_input_node);
      }

      data_input_args.push_back(add_input_node->MutableInputDefs()[0]);
      scales.push_back(scale_value);

      return true;
    };

    Node* add_input_node_0 = graph.GetNode(add_input_0->Index());
    Node* add_input_node_1 = graph.GetNode(add_input_1->Index());
    if (!check_add_input(add_input_node_0) || !check_add_input(add_input_node_1)) {
      continue;
    }

    Node* last_node = &node;
    // Handle three inputs only when Add node has one single consumer; and be noted we already check earlier
    // the output is not in graph outputs.
    if (node.GetOutputEdgesCount() == 1) {
      Node& output_node = *graph.GetNode(node.OutputEdgesBegin()->GetNode().Index());
      int output_node_port = node.OutputEdgesBegin()->GetDstArgIndex();
      // Find the next Add node that use the output of current Add node as one of its inputs.
      if (graph_utils::IsSupportedOptypeVersionAndDomain(output_node, "Add", {6, 7, 13, 14}) &&
          !graph.IsOutput(output_node.OutputDefs()[0]) /* this Add cannot generate graph output */
      ) {
        int the_other_input_port = 1 - output_node_port;
        NodeArg* the_other_input_arg = output_node.MutableInputDefs()[the_other_input_port];
        const Node* the_other_input_node = graph.GetProducerNode(the_other_input_arg->Name());
        Node* mutable_the_other_input_node = the_other_input_node
                                                 ? graph.GetNode(the_other_input_node->Index())
                                                 : nullptr;

        bool the_other_node_output_edge_check = mutable_the_other_input_node == nullptr ||
                                                mutable_the_other_input_node->GetOutputEdgesCount() == 1;

        // Also make sure the other input arg has Shape equal to output_shape, we don't want to
        // handle broadcast cases now.
        if (the_other_node_output_edge_check &&
            the_other_input_arg->Shape() && IsShapeEqual(the_other_input_arg->Shape(), output_shape)) {
          last_node = &output_node;
          nodes_to_remove.push_back(node);

          float scale_value = 1.0f;
          if (mutable_the_other_input_node && IsScaleOperator(graph, *mutable_the_other_input_node,
                                                              output_shape, scale_value)) {
            data_input_args.push_back(mutable_the_other_input_node->MutableInputDefs()[0]);
            nodes_to_remove.push_back(*mutable_the_other_input_node);
            scales.push_back(scale_value);
          } else {
            // The other input is 1). a constant initializer or graph input, OR 2). it is not a scale operator:
            // then we only add node arg into data input args, NOT need add any mode into nodes_to_remove.
            data_input_args.push_back(mutable_the_other_input_node->MutableInputDefs()[0]);
            scales.push_back(scale_value);
          }
        }
      }
    }

    if (data_input_args.size() != scales.size() || data_input_args.size() < 2) {
      continue;
    }

    auto type_info = *output_type;
    InlinedVector<NodeArg*> output_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("ScaledSum"), &type_info)};
    Node& scaled_sum_node = graph.AddNode(graph.GenerateNodeName("ScaledSum"),
                                          "ScaledSum",
                                          "FusedScaledSum",
                                          data_input_args,
                                          output_args,
                                          nullptr,
                                          kMSDomain);
    ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(scaled_sum_node),
                "Failed to set op schema for " + scaled_sum_node.Name());
    scaled_sum_node.SetExecutionProviderType(last_node->GetExecutionProviderType());

    for (size_t scale_index = 0; scale_index < scales.size(); ++scale_index) {
      scaled_sum_node.AddAttribute("scale_" + std::to_string(scale_index), scales[scale_index]);
    }

    graph_utils::ReplaceDownstreamNodeInput(graph, *last_node, 0, scaled_sum_node, 0);

    // Firstly remove the node itself.
    graph_utils::RemoveNodeOutputEdges(graph, *last_node);
    graph.RemoveNode(last_node->Index());

    // Then remove the parent nodes that may not be used by other nodes.
    for (auto it = nodes_to_remove.rbegin(); it != nodes_to_remove.rend(); ++it) {
      Node& n = *it;
      if (n.GetOutputEdgesCount() != 0) {
        continue;
      }

      graph_utils::RemoveNodeOutputEdges(graph, n);
      graph.RemoveNode(n.Index());
    }

    modified = true;
    handled_scaled_sum_count += 1;
  }

  LOGS(logger, INFO) << "Total fused ScaledSum node count:  " << handled_scaled_sum_count;

  return Status::OK();
}

}  // namespace onnxruntime
