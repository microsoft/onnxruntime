// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/defs/attr_proto_util.h>

#include "orttraining/core/optimizer/scaled_sum_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/framework/random_seed.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {

// Supports limited data types.
static constexpr std::array supported_data_types{
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
    ONNX_NAMESPACE::TensorProto_DataType_DOUBLE,
    ONNX_NAMESPACE::TensorProto_DataType_INT32,
    ONNX_NAMESPACE::TensorProto_DataType_INT64,
};

bool IsSupportedDataType(int32_t data_type) {
  return std::find(supported_data_types.cbegin(), supported_data_types.cend(), data_type) !=
         supported_data_types.cend();
}

using SupportedTypeList = boost::mp11::mp_list<MLFloat16, float, double, int32_t, int64_t>;

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
                     NodeArg*& scale_input_arg) {
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Div", {7, 13, 14})) {
    const Node* div_input_1 = graph_utils::GetInputNode(node, 0);
    bool first_input_check = (div_input_1 && node.InputDefs()[0]->Shape() &&
                              IsShapeEqual(node.InputDefs()[0]->Shape(), output_shape));

    if (first_input_check) {
      const Node* div_input_2 = graph_utils::GetInputNode(node, 1);
      auto div_input_2_shape = node.InputDefs()[1]->Shape();
      bool second_input_check = div_input_2 == nullptr && div_input_2_shape &&
                                graph_utils::IsConstantInitializer(graph, node.InputDefs()[1]->Name(), false) &&
                                (div_input_2_shape->dim_size() == 0  //scalar
                                 || (div_input_2_shape->dim_size() == 1 &&
                                     div_input_2_shape->dim(0).has_dim_value() &&
                                     div_input_2_shape->dim(0).dim_value() == 1)  // 1d with 1 element
                                );

      if (second_input_check) {
        scale_input_arg = node.MutableInputDefs()[1];
        return true;
      }
    }
  }
  return false;
}

template <typename T>
struct CreateOneScalarFunctor {
  void operator()(Graph& graph,
                  int data_type,
                  const std::string& name,
                  NodeArg*& initializer_node_arg) const {
    ONNX_NAMESPACE::TensorProto const_tensor;
    const_tensor.set_name(name);
    const_tensor.set_data_type(data_type);
    static const InlinedVector<int64_t> dims = {};
    InlinedVector<T> values{static_cast<T>(1.0f)};

    int64_t total_count = 1;
    for (const int64_t dim : dims) {
      const_tensor.add_dims(dim);
      total_count *= dim;
    }
    ORT_ENFORCE(total_count == static_cast<int64_t>(values.size()));
    const_tensor.set_raw_data(values.data(), values.size() * sizeof(T));
    initializer_node_arg = &graph_utils::AddInitializer(graph, const_tensor);
  }
};

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
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Add", {6, 7, 13, 14}) ||
        node.GetInputEdgesCount() != 2 ||
        graph_utils::IsGraphInput(graph, node.InputDefs()[0]) ||
        graph_utils::IsGraphInput(graph, node.InputDefs()[1]) ||
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
    InlinedVector<NodeArg*> scale_input_args;
    data_input_args.reserve(2);
    scale_input_args.reserve(2);

    InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;

    // Be noted: it is possible the two input nodes are from the same node.
    const Node* add_input_0 = graph_utils::GetInputNode(node, 0);
    const Node* add_input_1 = graph_utils::GetInputNode(node, 1);
    if (add_input_0 == nullptr || add_input_1 == nullptr) {
      continue;
    }

    auto check_add_input = [&graph, &output_shape, &nodes_to_remove,
                            &data_input_args, &scale_input_args](Node* add_input_node) -> bool {
      NodeArg* scale_input_arg = nullptr;
      if (!IsScaleOperator(graph, *add_input_node, output_shape, scale_input_arg)) {
        return false;
      }

      auto it = std::find_if(nodes_to_remove.begin(), nodes_to_remove.end(),
                             [&add_input_node](std::reference_wrapper<Node> n) {
                               return ((Node&)n).Index() == add_input_node->Index();
                             });
      if (it == nodes_to_remove.end()) {
        nodes_to_remove.push_back(*add_input_node);
      }

      data_input_args.push_back(add_input_node->MutableInputDefs()[0]);
      scale_input_args.push_back(scale_input_arg);
      return true;
    };

    Node* add_input_node_0 = graph.GetNode(add_input_0->Index());
    Node* add_input_node_1 = graph.GetNode(add_input_1->Index());
    if (!check_add_input(add_input_node_0) || !check_add_input(add_input_node_1)) {
      continue;
    }

    Node* last_node = &node;
    // Handle three pairs of inputs.
    if (node.GetOutputEdgesCount() == 1) {
      Node& output_node = *graph.GetNode(node.OutputEdgesBegin()->GetNode().Index());
      int output_node_port = node.OutputEdgesBegin()->GetDstArgIndex();
      if (graph_utils::IsSupportedOptypeVersionAndDomain(output_node, "Add", {6, 7, 13, 14})) {
        int the_other_input_port = 1 - output_node_port;
        NodeArg* the_other_input_arg = output_node.MutableInputDefs()[the_other_input_port];
        const Node* the_other_input_node = graph.GetProducerNode(the_other_input_arg->Name());
        Node* mutable_the_other_input_node = the_other_input_node ? graph.GetNode(the_other_input_node->Index()) : nullptr;

        bool the_ther_node_output_edge_check = mutable_the_other_input_node == nullptr || mutable_the_other_input_node->GetOutputEdgesCount() == 1;

        if (the_ther_node_output_edge_check &&
            the_other_input_arg->Shape() && IsShapeEqual(the_other_input_arg->Shape(), output_shape)) {
          last_node = &output_node;
          nodes_to_remove.push_back(node);

          NodeArg* scale_input_arg = nullptr;
          if (mutable_the_other_input_node &&
              IsScaleOperator(graph, *mutable_the_other_input_node, output_shape, scale_input_arg)) {
            data_input_args.push_back(mutable_the_other_input_node->MutableInputDefs()[0]);
            scale_input_args.push_back(scale_input_arg);
            nodes_to_remove.push_back(*mutable_the_other_input_node);
          } else {
            utils::MLTypeCallDispatcherFromTypeList<SupportedTypeList> t_disp(elem_type);
            NodeArg* one_arg = nullptr;
            t_disp.Invoke<CreateOneScalarFunctor>(graph, elem_type,
                                                  graph.GenerateNodeArgName(the_other_input_arg->Name() + "_scale"),
                                                  one_arg);
            data_input_args.push_back(the_other_input_arg);
            scale_input_args.push_back(one_arg);
          }
        }
      }
    }

    InlinedVector<NodeArg*> input_args;
    input_args.reserve(data_input_args.size() + scale_input_args.size());
    for (size_t pair_index = 0; pair_index < data_input_args.size(); ++pair_index) {
      input_args.push_back(data_input_args[pair_index]);
      input_args.push_back(scale_input_args[pair_index]);
    }

    auto type_info = *output_type;
    InlinedVector<NodeArg*> output_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("ScaledSum"), &type_info)};
    Node& scaled_sum_node = graph.AddNode(graph.GenerateNodeName("ScaledSum"),
                                          "ScaledSum",
                                          "FusedScaledSum",
                                          input_args,
                                          output_args,
                                          nullptr,
                                          kMSDomain);
    ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(scaled_sum_node),
                "Failed to set op schema for " + scaled_sum_node.Name());
    scaled_sum_node.SetExecutionProviderType(last_node->GetExecutionProviderType());

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
