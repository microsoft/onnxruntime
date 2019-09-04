// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/training/mixed_precision_transformer.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph_utils.h"
#include "core/graph/training/attr_proto_util.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/training/gradient_builder_base.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {
namespace training {

static const std::unordered_set<std::string> FP32_Nodes = {
    "ReduceSum",
    "SparseSoftmaxCrossEntropy",
    "SparseSoftmaxCrossEntropyGrad"};

static bool IsFP32Node(const Node* node) {
  return FP32_Nodes.find(node->OpType()) != FP32_Nodes.cend();
}

// At present, we use these table to identify which input needs to be keep in FP32
static const std::unordered_map<std::string, std::vector<int>> stage1_fp32_node_args = {
    {"TrainableDropout", {1}},
    {"TrainableDropoutGrad", {2}},
};

static const std::unordered_map<std::string, std::vector<int>> stage2_fp32_node_args = {
    {"TrainableDropout", {1}},
    {"TrainableDropoutGrad", {2}},
    {"ReduceSum", {0}},
    {"SparseSoftmaxCrossEntropy", {0, 2}},
    {"SparseSoftmaxCrossEntropyGrad", {0, 1, 3}},
};

// Seperate the consumer nodes of `arg` into two groups: FP32 vs FP16
// The argument `fp32_node_args` specifies the cases where the `arg` should be 32-bit float.
static void GetConsumerNodeInputs(onnxruntime::Graph& graph,
                                  const std::unordered_map<std::string, std::vector<int>>& fp32_node_args,
                                  const NodeArg* arg,
                                  std::vector<std::pair<Node*, int>>& fp16_inputs,
                                  std::vector<std::pair<Node*, int>>& fp32_inputs) {
  std::vector<Node*> consumer_nodes = graph.GetMutableConsumerNodes(arg->Name());
  for (Node* node : consumer_nodes) {
    int node_arg_slot = -1;
    for (int i = 0; i < static_cast<int>(node->InputDefs().size()); i++) {
      if (node->InputDefs()[i] == arg) {
        node_arg_slot = i;
        break;
      }
    }

    if (node_arg_slot == -1) {
      continue;
    }

    auto it = fp32_node_args.find(node->OpType());
    if (it == fp32_node_args.cend()) {
      fp16_inputs.push_back({node, node_arg_slot});
    } else {
      const auto index_it = std::find(it->second.cbegin(), it->second.cend(), node_arg_slot);
      if (index_it == it->second.cend()) {
        fp16_inputs.push_back({node, node_arg_slot});
      } else {
        fp32_inputs.push_back({node, node_arg_slot});
      }
    }
  }
}

static void RewireCastedNodeArg(onnxruntime::Graph& graph,
                                Node* cast_node,
                                const Node* producer_node,
                                int producer_node_arg_index,
                                Node* consumer_node,
                                int consumer_node_arg_index) {
  auto& consumer_inputs = consumer_node->MutableInputDefs();
  if (producer_node != nullptr) {
    graph.RemoveEdge(producer_node->Index(), consumer_node->Index(), producer_node_arg_index, consumer_node_arg_index);
  }
  consumer_inputs[consumer_node_arg_index] = cast_node->MutableOutputDefs()[0];
  graph.AddEdge(cast_node->Index(), consumer_node->Index(), 0, consumer_node_arg_index);
}

// This function tries casting `arg` to `element_type`.
// The argument `fp32_node_args` specifies the cases where the `arg` should be 32-bit float.
static Status CastNodeArg(onnxruntime::Graph& graph,
                          const std::unordered_map<std::string, std::vector<int>>& fp32_node_args,
                          NodeArg* arg,
                          ONNX_NAMESPACE::TensorProto_DataType elem_type) {
  if (arg == nullptr) {
    return Status::OK();
  }
  ORT_ENFORCE(elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
                  elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              "elem_type should be float or float16");

  // Get consumer nodes of the input `arg`
  std::vector<std::pair<Node*, int>> fp16_inputs;
  std::vector<std::pair<Node*, int>> fp32_inputs;
  GetConsumerNodeInputs(graph, fp32_node_args, arg, fp16_inputs, fp32_inputs);
  if ((elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 && fp16_inputs.empty()) ||
      (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT && fp32_inputs.empty())) {
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
  if (!fp16_inputs.empty() || !fp32_inputs.empty()) {
    if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      std::vector<Node*> fp16_nodes;
      fp16_nodes.reserve(fp16_inputs.size());
      for (const auto& kv : fp16_inputs) {
        RewireCastedNodeArg(graph, &cast_node, producer_node, producer_node_arg_index, kv.first, kv.second);
        fp16_nodes.emplace_back(kv.first);
      }

      std::vector<Node*> fp32_nodes;
      fp32_nodes.reserve(fp32_inputs.size() + 1);
      fp32_nodes.emplace_back(&cast_node);
      for (const auto& kv : fp32_inputs) {
        fp32_nodes.emplace_back(kv.first);
      }
      graph.UpdateConsumerNodes(arg->Name(), fp32_nodes);
      graph.UpdateConsumerNodes(output_name, fp16_nodes);
    } else {
      std::vector<Node*> fp32_nodes;
      fp32_nodes.reserve(fp32_inputs.size());
      for (const auto& kv : fp32_inputs) {
        RewireCastedNodeArg(graph, &cast_node, producer_node, producer_node_arg_index, kv.first, kv.second);
        fp32_nodes.emplace_back(kv.first);
      }

      std::vector<Node*> fp16_nodes;
      fp16_nodes.reserve(fp16_inputs.size() + 1);
      fp16_nodes.push_back(&cast_node);
      for (const auto& kv : fp16_inputs) {
        fp16_nodes.emplace_back(kv.first);
      }

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

static Status HandleFunctionBody(Graph& graph, const Function& node_func,
                                 const std::unordered_map<std::string, std::vector<int>>& fp32_node_args) {
  const Graph& subgraph = node_func.Body();
  // TODO: eliminate use of const_casts
  Graph& sg = const_cast<Graph&>(subgraph);
  // TODO: The resolve below is likely unnecessary.
  ORT_RETURN_IF_ERROR(sg.Resolve());
  GraphViewer subgraph_viewer(subgraph);
  const auto& subgraph_order = subgraph_viewer.GetNodesInTopologicalOrder();
  for (auto subgraph_index : subgraph_order) {
    // TODO can we directly deal with a mutable subgraph instead (no const_cast)? may need an ORT API change
    Node* subgraph_node = const_cast<Node*>(subgraph.GetNode(subgraph_index));
    if (subgraph_node->OpType() == "Constant") {
      for (NodeArg* output : subgraph_node->MutableOutputDefs()) {
        if (output->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
          ORT_RETURN_IF_ERROR(
              CastNodeArg(const_cast<Graph&>(subgraph), fp32_node_args, output, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
        }
      }
    } else if (IsFP32Node(subgraph_node)) {
      for (NodeArg* input : subgraph_node->MutableInputDefs()) {
        if (input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
          ORT_RETURN_IF_ERROR(CastNodeArg(graph, fp32_node_args, input, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
        }
      }

      for (NodeArg* output : subgraph_node->MutableOutputDefs()) {
        ORT_RETURN_IF_ERROR(CastNodeArg(graph, fp32_node_args, output, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
      }
    }
    const Function* nested = subgraph_node->GetFunctionBody();
    if (nested != nullptr) {
      ORT_RETURN_IF_ERROR(HandleFunctionBody(sg, *nested, fp32_node_args));
    }
  }
  return Status::OK();
}

static Status HandleFunctionCalls(Graph& graph, const std::unordered_map<std::string, std::vector<int>>& fp32_node_args) {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();
  for (auto index : order) {
    Node* node = graph.GetNode(index);
    const Function* node_func = node->GetFunctionBody();
    if (nullptr != node_func) {
      ORT_RETURN_IF_ERROR(HandleFunctionBody(graph, *node_func, fp32_node_args));
    }
  }
  return Status::OK();
}

// Create FP16 weights based on FP32 weights for mixed precision.
// And update the inputs of consumer with FP16 weights.
static NodeArg* CreateFP16WeightsAndUpdateComsumers(Graph& graph,
                                                    const std::unordered_map<std::string, std::vector<int>>& fp32_node_args,
                                                    const NodeArg* arg,
                                                    const ONNX_NAMESPACE::TensorProto* tensor_proto) {
  ORT_ENFORCE(arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              "data type is not float");
  // Create FP16 Node Arg
  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  type_proto.mutable_tensor_type()->mutable_shape()->CopyFrom(*arg->Shape());
  std::string arg_name = arg->Name() + "_fp16";
  NodeArg& new_arg = graph.GetOrCreateNodeArg(arg_name, &type_proto);

  // Check consumer nodes
  std::vector<std::pair<Node*, int>> fp16_inputs;
  std::vector<std::pair<Node*, int>> fp32_inputs;
  GetConsumerNodeInputs(graph, fp32_node_args, arg, fp16_inputs, fp32_inputs);
  if (fp16_inputs.empty()) {
    return nullptr;
  }

  for (auto kv : fp16_inputs) {
    kv.first->MutableInputDefs()[kv.second] = &new_arg;
  }

  // copy weights and put them into the graph
  Initializer initializer(tensor_proto);
  ONNX_NAMESPACE::TensorProto weight_tensor_proto = initializer.ToFP16(arg_name);
  graph.AddInitializedTensor(weight_tensor_proto);

  return &new_arg;
}

// fp16_weights_map stores the map from the name of the original FP32 weight to the coresponding fp16 NodeArg.
Status TransformGraphForMixedPrecision(Graph& graph,
                                       const std::unordered_set<std::string>& weights_to_train,
                                       bool use_fp16_initializer,
                                       std::unordered_map<std::string, NodeArg*>& fp16_weights_map) {
  // Stag 1: Convert whole graph including forward and backward to FP16
  // Insert Cast node to convert inputs from FP32 to FP16
  for (const NodeArg* input : graph.GetInputs()) {
    if (input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      ORT_RETURN_IF_ERROR(
          CastNodeArg(graph, stage1_fp32_node_args, graph.GetNodeArg(input->Name()), ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
    }
  }

  // Convert initializers including trainable weights from FP32 to FP16
  const auto& initialized_tensors = graph.GetAllInitializedTensors();
  for (const auto& kv : initialized_tensors) {
    NodeArg* input = graph.GetNodeArg(kv.first);
    if (input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      if (use_fp16_initializer) {
        NodeArg* fp16_weight_arg = CreateFP16WeightsAndUpdateComsumers(graph, stage1_fp32_node_args, input, kv.second);
        const auto it = weights_to_train.find(kv.first);
        if (it != weights_to_train.cend()) {
          fp16_weights_map[kv.first] = fp16_weight_arg;
        }
      } else {
        ORT_RETURN_IF_ERROR(CastNodeArg(graph, stage1_fp32_node_args, input, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
      }
    }
  }

  // Handle implicit data type casting nodes such as Cast, ConstantOfShape
  GraphViewer graph_viewer(graph);
  const auto& nodes_order = graph_viewer.GetNodesInTopologicalOrder();
  for (auto index : nodes_order) {
    Node* node = graph.GetNode(index);
    if (node->OpType() == "Cast" ||
        node->OpType() == "ConstantOfShape") {
      if (node->MutableOutputDefs()[0]->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        ORT_RETURN_IF_ERROR(CastNodeArg(graph, stage1_fp32_node_args, node->MutableOutputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
      }
    }
  }

  // Handle function body
  ORT_RETURN_IF_ERROR(HandleFunctionCalls(graph, stage1_fp32_node_args));

  // At this point, the model has been transformed to a valid FP16 model.
  ORT_RETURN_IF_ERROR(graph.Resolve(&weights_to_train));

  // Stage 2: Keep nodes such as ReduceSum in FP32
  // Add cast node for nodes which need to be computed in FP32
  // Convert fp16 tensor --> Op --> fp16 tensor to
  // fp16 tensor --> Cast --> fp32 tensor --> Op --> fp32 tensor --> Cast --> fp16 tensor
  GraphViewer graph_viewer1(graph);
  const auto& nodes_order1 = graph_viewer.GetNodesInTopologicalOrder();
  for (auto index : nodes_order1) {
    Node* node = graph.GetNode(index);
    if (IsFP32Node(node)) {
      for (NodeArg* input : node->MutableInputDefs()) {
        if (input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
          ORT_RETURN_IF_ERROR(CastNodeArg(graph, stage2_fp32_node_args, input, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
        }
      }

      for (NodeArg* output : node->MutableOutputDefs()) {
        if (output->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
          ORT_RETURN_IF_ERROR(CastNodeArg(graph, stage2_fp32_node_args, output, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
        }
      }
    }
  }
  ORT_RETURN_IF_ERROR(graph.Resolve(&weights_to_train));

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
