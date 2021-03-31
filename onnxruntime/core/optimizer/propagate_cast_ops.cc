// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/propagate_cast_ops.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

std::vector<std::string> fp16_allow = {"Transpose", "Reshape", "Gather", "Split", "Relu", "Where", "Dropout"};
std::vector<std::string> fp16_safe = { "LayerNorm", "Gelu", "FastGelu", "Tanh", "MatMul", "MatAdd", "Add",
                                       "Sub", "Mul", "Div", "Neg", "Gemm", "FusedMatMul", "FusedGemm"};

// Insert a Cast node after each NodeArg
static Status InsertCastNodes(Graph& graph, const std::set<NodeArg*>& require_cast, bool is_fp16, std::deque<onnxruntime::NodeIndex>& removed_nodes)
{
  TensorProto_DataType data_type = is_fp16 ? TensorProto_DataType_FLOAT16 : TensorProto_DataType_FLOAT;
  TypeProto fp16_type_proto;
  fp16_type_proto.mutable_tensor_type()->set_elem_type(data_type);
  std::vector<std::pair<NodeArg*, NodeArg*>> substitute_list; // List of cast_input and cast_output nodes
  std::set<Node*> consumer_nodes; // Nodes that need to be modified due to inserting Cast nodes

  //Create requirred new Cast nodes.
  for (NodeArg* node_arg : require_cast) {
    NodeArg& cast_input = *node_arg;
    NodeArg& cast_output = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node_arg->Name()), &fp16_type_proto);
    const std::vector<NodeArg*> new_cast_input = {&cast_input};
    const std::vector<NodeArg*> new_cast_output = {&cast_output};
    ONNX_NAMESPACE::AttributeProto to_attribute;
    to_attribute.set_name("to");
    to_attribute.set_type(ONNX_NAMESPACE::AttributeProto::INT);
    to_attribute.set_i(static_cast<int64_t>(data_type));
    NodeAttributes attributes({{"to", to_attribute}});

    (void) graph.AddNode(graph.GenerateNodeName(node_arg->Name() + "_cast"),
                        "Cast",
                        "Created a new Cast node",
                        new_cast_input,
                        new_cast_output,
                        &attributes);
    substitute_list.push_back(std::make_pair(&cast_input, &cast_output));
    std::vector<Node*> consumers = graph.GetMutableConsumerNodes(node_arg->Name());
    std::copy(consumers.begin(), consumers.end(), std::inserter(consumer_nodes, consumer_nodes.end()));
  }
  for (Node* consumer : consumer_nodes) {
    if (consumer) {
      std::vector<NodeArg*> consumer_inputs = consumer->MutableInputDefs();
      auto consumer_outputs = consumer->MutableOutputDefs();
      for (std::pair<NodeArg*, NodeArg*> p : substitute_list) {
        std::replace(consumer_inputs.begin(), consumer_inputs.end(), p.first, p.second);
      }
      Node& new_consumer =  graph.AddNode(graph.GenerateNodeName(consumer->Name() + "_replace"),
                                          consumer->OpType(),
                                          "Created to replace a node",
                                          consumer_inputs,
                                          consumer_outputs,
                                          &consumer->GetAttributes(),
                                          consumer->Domain());
      (void) new_consumer;
      removed_nodes.push_front(consumer->Index());
    }
  }
  return Status::OK();
}

static Status RemoveCastNode(Graph& graph, Node* cast, std::deque<onnxruntime::NodeIndex>& removed_nodes)
{
  NodeArg* cast_input = cast->MutableInputDefs()[0];
  auto parent = graph.GetMutableProducerNode(cast_input->Name());
  if (parent) {
    graph_utils::FinalizeNodeFusion(graph, *parent, *cast);
  } else {
    removed_nodes.push_front(cast->Index());
  }
  return Status::OK();
}

// SearchUpstream:
// Recursively traverse the graph upstream collecting all the NodeArgs that require a cast
// inorder to remove an FP16 Cast operation down the graph.
static void SearchUpstream(Graph& graph, NodeArg* node_arg, std::set<NodeArg*>& require_cast)
{
  Node* node = graph.GetMutableProducerNode(node_arg->Name());
  if (node == nullptr) {
    // The graph inputs don't have the producer nodes
    require_cast.insert(node_arg);
  } else {
    std::string op_type = node->OpType();
    if (std::find(fp16_allow.begin(), fp16_allow.end(), op_type) == fp16_allow.end() &&
        std::find(fp16_safe.begin(), fp16_safe.end(), op_type) == fp16_safe.end()) {
      require_cast.insert(node_arg);
    } else {
      for (NodeArg* node_input : node->MutableInputDefs()) {
        SearchUpstream(graph, node_input, require_cast);
      }
    }
  }
}

// SearchDownstream:
// Recursively traverse the graph downstream collecting all the NodeArgs that require a cast
// inorder to remove an FP32 Cast operation up the graph.
static void SearchDownstream(Graph& graph, NodeArg* node_arg, std::set<NodeArg*>& require_cast)
{
  for (Node* node : graph.GetMutableConsumerNodes(node_arg->Name())) {
    if (node) {
      std::string op_type = node->OpType();
      if (std::find(fp16_allow.begin(), fp16_allow.end(), op_type) == fp16_allow.end()) {
        require_cast.insert(node_arg);
      } else {
        for (NodeArg* node_output : node->MutableOutputDefs()) {
          SearchDownstream(graph, node_output, require_cast);
        }
      }
    }
  }
}

static bool PropagateForwards(Graph& graph, Node* node, std::deque<onnxruntime::NodeIndex>& removed_nodes)
{
  bool modified = false;
  if (node == nullptr) {
    return false;
  }
  if (node->OpType() == "Cast") {
    const NodeAttributes& attributes = node->GetAttributes();
    ORT_ENFORCE(attributes.find("to") != attributes.end());
    if (attributes.at("to").i() == static_cast<int64_t> (TensorProto::FLOAT)) {
      std::set<NodeArg*> require_cast;
      NodeArg* cast_output = node->MutableOutputDefs()[0];
      SearchDownstream(graph, cast_output, require_cast);
      if (require_cast.find(cast_output) == require_cast.end()) {
        // Remove Cast operation
        RemoveCastNode(graph, node, removed_nodes);
        InsertCastNodes(graph, require_cast, false, removed_nodes);
        modified = true;
      }
    }
  } else if (std::find(fp16_safe.begin(), fp16_safe.end(), node->OpType()) == fp16_safe.end()) {
    bool all_inputs_have_casts = true;
    for (NodeArg* input : node->MutableInputDefs()) {
      Node* producer = graph.GetMutableProducerNode(input->Name());
      if (producer->OpType() == "Cast") {
        const NodeAttributes& attributes = producer->GetAttributes();
        ORT_ENFORCE(attributes.find("to") != attributes.end());
        if (attributes.at("to").i() == static_cast<int64_t> (TensorProto::FLOAT)) {
          continue;
        }
      }
      all_inputs_have_casts = false;
      break;
    }
    if (all_inputs_have_casts) {
      for (NodeArg* input : node->MutableInputDefs()) {
        Node* producer = graph.GetMutableProducerNode(input->Name());
        RemoveCastNode(graph, producer, removed_nodes);
      }
    }
    InsertCastNodes(graph, {node->MutableOutputDefs()[0]}, false, removed_nodes);
  } else {
    for (NodeArg* output: node->MutableOutputDefs()) {
      for (Node* consumer : graph.GetMutableConsumerNodes(output->Name())) {
        modified |= PropagateForwards(graph, consumer, removed_nodes);
      }
    }
  }
  return modified;
}

static bool PropagateBackwards(Graph& graph, Node* node, std::deque<onnxruntime::NodeIndex>& removed_nodes)
{
  bool modified = false;
  if (node == nullptr) {
    return false;
  }
  if (node->OpType() == "Cast") {
    const NodeAttributes& attributes = node->GetAttributes();
    ORT_ENFORCE(attributes.find("to") != attributes.end());
    if (attributes.at("to").i() == static_cast<int64_t> (TensorProto::FLOAT16)) {
      std::set<NodeArg*> require_cast;
      NodeArg* cast_input = node->MutableInputDefs()[0];
      SearchUpstream(graph, cast_input, require_cast);
      if (require_cast.find(cast_input) == require_cast.end()) {
        // Remove Cast operation
        RemoveCastNode(graph, node, removed_nodes);
        InsertCastNodes(graph, require_cast, true, removed_nodes);
        modified = true;
      }
    }
  } else {
    for (NodeArg* input: node->MutableInputDefs()) {
      Node* producer = graph.GetMutableProducerNode(input->Name());
      modified |= PropagateBackwards(graph, producer, removed_nodes);
    }
  }
  return modified;
}

// Fuse all nodes, replace with a single node.
// Assumptions:
// 1. all nodes are Cast ops and are of the same Cast type
// 2. all the nodes have the same input
static void FuseNodes(Graph& graph, NodeArg* input, std::vector<Node*> nodes)
{
  std::vector<NodeArg*> outputs;
  for (Node* node : nodes) {
    std::vector<NodeArg*> node_outputs = node->MutableOutputDefs();
    outputs.insert(outputs.end(), node_outputs.begin(), node_outputs.end());
  }
  Node* node = nodes[0];
  (void) graph.AddNode(graph.GenerateNodeName(node->Name() + "_replace"),
                       node->OpType(),
                       "Created to replace a node",
                       {input},
                       outputs,
                       &node->GetAttributes(),
                       node->Domain());
  for (Node* n : nodes) {
    graph_utils::RemoveNodeOutputEdges(graph, *n);
    graph.RemoveNode(n->Index());    
  }
}
// Traverse the graph recursively searching/collecting sibling Cast op nodes to fuse and call FuseNodes.
static bool FuseSubgraphs(Graph& graph, Node* parent)
{
  bool modified = false;
  for (NodeArg* output : parent->MutableOutputDefs()) {
    std::vector<Node*> cast_fp16_siblings;
    std::vector<Node*> cast_fp_siblings;
    for (Node* node : graph.GetMutableConsumerNodes(output->Name())) {
      if (node == nullptr) {
        continue;
      }
      if (node->OpType() == "Cast") {
        const NodeAttributes& attributes = node->GetAttributes();
        ORT_ENFORCE(attributes.find("to") != attributes.end());
        if (attributes.at("to").i() == static_cast<int64_t> (TensorProto::FLOAT16)) {
          cast_fp16_siblings.push_back(node);
        } else if (attributes.at("to").i() == static_cast<int64_t> (TensorProto::FLOAT)) {
          cast_fp_siblings.push_back(node);
        }
      }
    }
    if (cast_fp16_siblings.size() > 1) {
      modified = true;
      FuseNodes(graph, output, cast_fp16_siblings);
    }
    if (cast_fp_siblings.size() > 1) {
      modified = true;
      FuseNodes(graph, output, cast_fp_siblings);
    }
  }
  return modified;
}

Status PropagateCastOps::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  std::deque<onnxruntime::NodeIndex> removed_nodes;
  (void) graph_level;
  (void) logger;
  modified = false;
  // Propagate FP32 Casts forward
  for (const NodeArg* input: graph.GetInputs()) {
    for (Node* node : graph.GetMutableConsumerNodes(input->Name())) {
      modified |= PropagateForwards(graph, node, removed_nodes);
    }
  }

  // Propagate FP16 Casts backward
  for (const NodeArg* output: graph.GetOutputs()) {
    Node* node = graph.GetMutableProducerNode(output->Name());
    modified |= PropagateBackwards(graph, node, removed_nodes);
  }

  // Fuse subgraphs, sibling Cast nodes with same input
  for (auto& node: graph.Nodes()) {
    modified |= FuseSubgraphs(graph, &node);
  }

  for (onnxruntime::NodeIndex removed_node : removed_nodes) {
    graph.RemoveNode(removed_node);
  }

  return Status::OK();
}

} // namespace onnxruntime