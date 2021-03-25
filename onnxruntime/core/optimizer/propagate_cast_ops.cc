// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/propagate_cast_ops.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

std::vector<std::string> fp16_allow = {"Transpose", "Reshape", "Gather", "Split", "Relu", "Where", "Dropout"};
std::vector<std::string> fp16_safe = { "LayerNorm", "Gelu", "FastGelu", "Tanh", "Add", "Sub", "Mul", "Div", "Neg", "Gemm", "FusedMatMul", "FusedGemm"};
#if 0
static size_t UpdateConsumerCount(Graph& graph, NodeArg* target, std::unordered_map<NodeArg*, size_t>& count_map) {
  const auto& node_consumers = graph.GetConsumerNodes(target->Name());
  ORT_ENFORCE(!node_consumers.empty());
  auto it = count_map.find(target);
  if (it == count_map.end()) {
    count_map.insert({target, node_consumers.size() - 1});
    return node_consumers.size() - 1;
  } else {
    count_map[target] -= 1;
    return count_map[target];
  }
}

static Status RemoveCastNode(Graph& graph, Node* cast,
                      std::deque<onnxruntime::NodeIndex>& removed_nodes,
                      std::unordered_map<NodeArg*, size_t>& count_map)
{
  auto parent = graph.GetMutableProducerNode(*cast->MutableInputDefs()[0]);
  ORT_ENFORCE(parent != nullptr);
  NodeArg* cast_output = cast->MutableOutputDefs();
  (void) graph.AddNode(graph.GenerateNodeName(parent->Name() + "_transformed"),
                       parent->OpType(),
                       "Created a new Cast node to interchange Cast and Transpose nodes",
                       parent->MutableOutDefs(),
                       cast->MutableOutputDefs(),
                       &parent->GetAttributes(),
                       parent->Domain());
  size_t consumers = UpdateConsumerCount(graph, parent->MutableOutputDefs()[0], consumer_count);
  graph_utils::RemoveNodeOutputEdges(graph, *cast);
  graph.RemoveNode(cast->Index());
  if (consumers == 0) {
    removed_nodes.push_front(parent->Index());
  }
  return Status::OK();
}
#endif
// InsertCastFP16Before
// Check if the node op type is safe to move Cast node up and recursively call on
// every input node. Insert Cast before every failed input, if any of the inputs succeeds
// if all the inputs fail the insert a FP16 Cast node after the present node if it is not
// an FP16 Cast node.

static bool InsertCastFP16Upstream(Graph& graph, Node* node)
{
  std::vector<NodeArg*> require_cast;
  bool failed_propagation = true;
  std::string op_type = node->OpType();
  if (find(fp16_allow.begin(), fp16_allow.end(), op_type) == fp16_allow.end() &&
      find(fp16_allow.begin(), fp16_allow.end(), op_type) == fp16_allow.end()) {
    return false;
  }
  // If the present node op is Cast the return true if cast is to FP16 and false otherwise.
  if (op_type == "Cast") {
    const NodeAttributes& attributes = node->GetAttributes();
    ORT_ENFORCE(attributes.find("to") != attributes.end());
    return RetrieveValues<int64_t>(attributes.at("to"))[0] == static_cast<int64_t> (TensorProto::FLOAT16);
  }
  for (NodeArg* node_input : node->MutableInputDefs()) {
    Node* parent = graph.GetMutableProducerNode(node_input->Name());
    if (InsertCastFP16Upstream(graph, parent) == false) {
      require_cast.push_back(node_input);
    } else {
      failed_propagation = false;
    }
  }
  if (failed_propagation) {
    return false;
  } else {
    if (node->OpType() != "Cast") {
      for (NodeArg* node_input : require_cast) {
        NodeArg& cast_output = graph_utils::CreateNodeArg(graph, *node_input);
        const std::vector<NodeArg*> new_cast_input = {node_input};
        const std::vector<NodeArg*> new_cast_output = {&cast_output};
        ONNX_NAMESPACE::AttributeProto to_attribute;
        to_attribute.set_name("to");
        to_attribute.set_type(ONNX_NAMESPACE::AttributeProto::INT);
        to_attribute.set_i(static_cast<int64_t>(TensorProto::FLOAT16));
        NodeAttributes attributes({{"to", to_attribute}});

        (void) graph.AddNode(graph.GenerateNodeName(node->Name() + "_cast"),
                            "Cast",
                            "Created a new Cast node to interchange Cast and Transpose nodes",
                            new_cast_input,
                            new_cast_output,
                            &attributes);
        std::replace(node->MutableInputDefs().begin(), node->MutableInputDefs().end(), node_input, &cast_output);
      }
    } else {
      // Propagation is successful and the current node is a Cast Node.
      // Remove this node.
      NodeArg* node_input = node->MutableInputDefs()[0];
      for (NodeArg* node_output: node->MutableOutputDefs()) {
         for (Node* child : graph.GetMutableConsumerNodes(node_output->Name())) {
           std::replace(child->MutableInputDefs().begin(), child->MutableInputDefs().end(), node_output, node_input);
         }
      }
      graph_utils::RemoveNodeOutputEdges(graph, *node);
      graph.RemoveNode(node->Index());
    }
  }
  return true;
}
#if 0
static bool InsertCastFP32After(Graph& graph, Node* node)
{

}
#endif
Status PropagateCastOps::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  // std::deque<onnxruntime::NodeIndex> removed_nodes;
  (void) modified;
  (void) graph_level;
  (void) logger;
  for (const NodeArg* output: graph.GetOutputs()) {
    InsertCastFP16Upstream(graph, graph.GetMutableProducerNode(output->Name()));
  }
  return Status::OK();
}

} // namespace onnxruntime