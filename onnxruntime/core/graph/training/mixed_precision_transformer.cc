// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph_utils.h"
#include "core/graph/training/attr_proto_util.h"
#include "core/graph/graph_viewer.h"
#include "mixed_precision_transformer.h"
#include "core/graph/training/gradient_builder_base.h"

namespace onnxruntime {
namespace training {

static bool IsFP32Node(const Node* node) {
  return node->OpType() == "ReduceMean" ||
         node->OpType() == "ReduceSum" ||
         node->OpType() == "LayerNormalization" ||
         node->OpType() == "LayerNormalizationGrad" ||
         node->OpType() == "SparseSoftmaxCrossEntropy" ||
         node->OpType() == "SparseSoftmaxCrossEntropyGrad" ||
         node->OpType() == "AdamOptimizer";
}

static void SplitNodes(const std::vector<Node*>& nodes, std::vector<Node*>& fp32_nodes, std::vector<Node*>& fp16_nodes) {
  for (Node* node : nodes) {
    if (IsFP32Node(node)) {
      fp32_nodes.push_back(node);
    } else {
      fp16_nodes.push_back(node);
    }
  }
}

static void CastNodeArgHelper(onnxruntime::Graph& graph,
                              const NodeArg* arg,
                              const Node* producer_node,
                              int producer_node_arg_index,
                              Node* cast_node,
                              Node* consumer_node) {
  auto& consumer_inputs = consumer_node->MutableInputDefs();
  for (int i = 0; i < static_cast<int>(consumer_inputs.size()); i++) {
    if (arg == consumer_inputs[i]) {
      if (producer_node != nullptr) {
        graph.RemoveEdge(producer_node->Index(), consumer_node->Index(), producer_node_arg_index, i);
      }

      consumer_inputs[i] = cast_node->MutableOutputDefs()[0];
      graph.AddEdge(cast_node->Index(), consumer_node->Index(), 0, i);
      break;
    }
  }
}

static Status CastNodeArg(onnxruntime::Graph& graph, NodeArg* arg, ONNX_NAMESPACE::TensorProto_DataType elem_type) {
  if (arg == nullptr) {
    return Status::OK();
  }

  // Check consumer nodes
  std::vector<Node*> consumer_nodes = graph.GetMutableConsumerNodes(arg->Name());
  std::vector<Node*> fp32_nodes, fp16_nodes;
  SplitNodes(consumer_nodes, fp32_nodes, fp16_nodes);
  if ((elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 && fp16_nodes.empty()) ||
      (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT && fp32_nodes.empty())) {
    return Status::OK();
  }

  // Create output arg of Cast
  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(elem_type);
  std::string output_name = graph.GenerateNodeArgName(arg->Name());
  const std::string cast_node_name = graph.GenerateNodeName("cast_" + output_name);

  output_name += (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ? "_f16" : "_f32");
  NodeArg& output = graph.GetOrCreateNodeArg(output_name, &type_proto);

  // Create Cast node
  NodeAttributes attrs;
  attrs["to"] = MakeAttribute("to", int64_t(elem_type));
  Node& cast_node = graph.AddNode(cast_node_name, "Cast", "", {arg}, {&output}, &attrs);

  // Find node arg index in producer
  Node* producer_node = graph.GetMutableProducerNode(arg->Name());
  int producer_node_arg_index = 0;
  if (producer_node != nullptr) {
    while (producer_node_arg_index < static_cast<int>(producer_node->OutputDefs().size()) &&
           producer_node->OutputDefs()[producer_node_arg_index] != arg) {
      producer_node_arg_index++;
    }
    ORT_RETURN_IF_NOT(producer_node_arg_index != static_cast<int>(producer_node->OutputDefs().size()));
  }

  // Update consumer
  if (!consumer_nodes.empty()) {
    if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      for (Node* fp16_node : fp16_nodes) {
        CastNodeArgHelper(graph, arg, producer_node, producer_node_arg_index, &cast_node, fp16_node);
      }

      fp32_nodes.push_back(&cast_node);
      graph.UpdateConsumerNodes(arg->Name(), fp32_nodes);
      graph.UpdateConsumerNodes(output_name, fp16_nodes);
    } else {
      for (Node* fp32_node : fp32_nodes) {
        CastNodeArgHelper(graph, arg, producer_node, producer_node_arg_index, &cast_node, fp32_node);
      }
      fp16_nodes.push_back(&cast_node);
      graph.UpdateConsumerNodes(arg->Name(), fp16_nodes);
      graph.UpdateConsumerNodes(output_name, fp32_nodes);
    }
  } else {
    // Make sure it is not one of graph outputs, otherwise, graph outputs need to be updated.
    ORT_RETURN_IF_NOT(std::find(graph.GetOutputs().cbegin(), graph.GetOutputs().cend(), arg) == graph.GetOutputs().cend());
  }

  // Update producer
  if (producer_node != nullptr) {
    graph.AddEdge(producer_node->Index(), cast_node.Index(), producer_node_arg_index, 0);
  }
  graph.UpdateProducerNode(output_name, cast_node.Index());

  return Status::OK();
}

// TODO we only consider one level of subgraph now, handle this more generally
static Status HandleFunctionBody(Graph& graph) {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();
  for (auto index : order) {
    Node* node = graph.GetNode(index);
    const Function* node_func = node->GetFunctionBody();
    if (nullptr == node_func) {
      continue;
    }
    const Graph& subgraph = node_func->Body();
    GraphViewer subgraph_viewer(subgraph);
    const auto& subgraph_order = subgraph_viewer.GetNodesInTopologicalOrder();
    for (auto subgraph_index : subgraph_order) {
      // TODO can we directly deal with a mutable subgraph instead (no const_cast)? may need an ORT API change
      Node* subgraph_node = const_cast<Node*>(subgraph.GetNode(subgraph_index));
      if (subgraph_node->OpType() == "Constant") {
        for (NodeArg* output : subgraph_node->MutableOutputDefs()) {
          if (output->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
            ORT_RETURN_IF_ERROR(
                CastNodeArg(const_cast<Graph&>(subgraph), output, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
          }
        }
      } else if (IsFP32Node(subgraph_node)) {
        for (NodeArg* input : subgraph_node->MutableInputDefs()) {
          if (input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
            ORT_RETURN_IF_ERROR(CastNodeArg(graph, input, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
          }
        }

        for (NodeArg* output : subgraph_node->MutableOutputDefs()) {
          ORT_RETURN_IF_ERROR(CastNodeArg(graph, output, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
        }
      }
    }
  }
  return Status::OK();
}

Status TransformGraphForMixedPrecision(Graph& graph,
                                       const std::unordered_set<std::string>& weights_to_train) {
  ORT_RETURN_IF_ERROR(graph.Resolve());
  GraphViewer graph_viewer(graph);

  const auto& initialized_tensors = graph.GetAllInitializedTensors();

  std::unordered_set<const NodeArg*> node_args_to_exclude_from_initializers;

  // 1. Add cast node for Nodes which will be computed in FP32
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();
  for (auto index : order) {
    Node* node = graph.GetNode(index);
    if (IsFP32Node(node) && node->OpType() != "AdamOptimizer") {
      for (NodeArg* input : node->MutableInputDefs()) {
        if (input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
          if (initialized_tensors.find(input->Name()) != initialized_tensors.cend()) {
            node_args_to_exclude_from_initializers.insert(input);
          } else {
            ORT_RETURN_IF_ERROR(CastNodeArg(graph, input, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
          }
        }
      }

      for (NodeArg* output : node->MutableOutputDefs()) {
        ORT_RETURN_IF_ERROR(CastNodeArg(graph, output, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
      }
    } else if (node->OpType() == "TrainableDropout" ||
               node->OpType() == "TrainableDropoutGrad") {
      // The ratio of dropout can't be casted to FP16
      node_args_to_exclude_from_initializers.insert(node->InputDefs()[1]);
    } else if (node->OpType() == "Cast" ||
               node->OpType() == "ConstantOfShape") {
      // Handle implicit data type casting nodes such as Cast, ConstantOfShape
      if (node->MutableOutputDefs()[0]->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        ORT_RETURN_IF_ERROR(CastNodeArg(graph, node->MutableOutputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
      }
    }
  }

  // 2. Insert Cast node to convert inputs from FP32 to FP16
  for (const NodeArg* input : graph.GetInputs()) {
    if (input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      ORT_RETURN_IF_ERROR(
          CastNodeArg(graph, graph.GetNodeArg(input->Name()), ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
    }
  }

  // 3. Insert Cast node to convert all initializers including trainable weights from FP32 to FP16
  for (const auto& it : initialized_tensors) {
    NodeArg* input = graph.GetNodeArg(it.first);
    if (node_args_to_exclude_from_initializers.find(input) == node_args_to_exclude_from_initializers.cend() &&
        input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      ORT_RETURN_IF_ERROR(CastNodeArg(graph, input, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
    }
  }

  ORT_RETURN_IF_ERROR(graph.Resolve());

  // 5. Handle function body
  ORT_RETURN_IF_ERROR(HandleFunctionBody(graph));

  //Insert Cast node to convert gradients of trainable weights from FP16 to FP32
  for (const auto& weight_to_train : weights_to_train) {
    std::string gradient_name = GradientBuilderBase::GradientName(weight_to_train);
    auto nodes = graph.GetMutableConsumerNodes(gradient_name);
    NodeArg* input = graph.GetNodeArg(gradient_name);
    if (input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      ORT_RETURN_IF_ERROR(CastNodeArg(graph, input, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    }
  }

  ORT_RETURN_IF_ERROR(graph.Resolve());

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
