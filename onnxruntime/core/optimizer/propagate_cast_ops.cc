// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/propagate_cast_ops.h"

#include "core/common/span_utils.h"
#include "core/optimizer/initializer.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
/*********************************************************************************************************************************
 * PropagateCastOps transformation tries to expand float16 computation regions in the graph by converting float operations to float16.
 * In order to perform the transformation, certain operations are considered FP16 safe, i.e. the computation may be performed
 * without effecting the numerical result, ex. transpose, shape, etc. The transformation supports three levels of optimization, 0, 1
 * and 2. Level 2 being the most aggressive, may consider moving float operations to float16 which may result in different numerical results
 * due to loss of precision. The user may choose level 0, whereby the user chooses the opcodes which are "FP16 Safe" instead of a list
 * predetermined opcodes as in levels 1 and 2.
 * Currently three strategies are available, None, InsertAndReduce and FloodFill.
 * None:
 *   Although no new cast operations are inserted or propagated using this strategy some optimizations are performed
 *   1. Remove back-to-back casts
 *   2. Fuse subgraphs
 *   3. Remove unnecessary casts
 * InsertAndReduce :
 *   This transformation converts all FP16 operations to float16. The transformation first
 *   1. Inserts float16 cast operation on all the float inputs
 *   2. Changes all float outputs to float16
 *   3. Inserts float cast operations on all float outputs as expected
 *   After inserting the FP16 and FP32 cast operation nodes around all the nodes with FP16 Safe opcodes, the transformation reduces the
 *   cast operations, using the following operations on the graph, iteratively until no more reduction is possible.
 *   1. Remove back-to-back casts
 *   2. Remove redundant casts
 *   3. Fuse sibling subgraphs with same cast operation
 * FloodFill:
 *   Operations are converted from float to float16 by propagating float16 cast operations up the graph or float cast operations down the
 *   graph. Using this strategy, for each pre-existing float/float16 cast operations the transformation first finds the possible expansion of
 *   float16 region up/down the graph using DFS/ReverseDFS and (TODO) identifies loss/gain by performing such expansion, considering
 *   the gain by reducing float operations lower precision and loss due to newly inserted cast operations (/TODO).
 *   In addition to propagating cast operations up/down the graph, in this strategy, the above mentioned cast op reduction functions
 *   are also used.
 * InsertAndReduce exhaustively inserts cast operations before and after all the nodes with the allowed opcodes whereas FloodFill only
 * propagates existing casts up or down the graph, inserting new casts as it expands the float16 regions.
 *********************************************************************************************************************************/
namespace onnxruntime {
// NodeArg to Select consumer node map.
using NodeArgToConsumerMap = InlinedHashMap<NodeArg*, InlinedVector<Node*>>;

/*
 * ConcatNames
 * Collects all the names from the pointers of the objects stores in the container class C
 * the class should have a member functions returning a string (or a ref).
 */
template <typename C, typename T = typename C::value_type>
static std::string ConcatNames(
    C const& items, std::string (*f)(const T& n) = [](const T& n) { return n->Name(); }) {
  std::vector<std::string> names;
  std::transform(items.begin(), items.end(), back_inserter(names), f);
  return std::accumulate(names.begin(), names.end(), std::string(), [](const std::string& a, const std::string& b) { return a + ", " + b; });
}

// GetName
// Collect the node-arg names and the corresponding consumers name for reporting
static std::string GetName(const std::pair<const NodeArg*, InlinedVector<Node*>>& p) {
  return p.first->Name() + " feeding " + ConcatNames(p.second) + "; ";
};

using NodeIndices = InlinedHashSet<NodeIndex>;
using FP16AllowOps = PropagateCastOps::FP16AllowOps;

/*
 *  Check if the input is relevant to consider for cast propagation for the given node.
 *  Return true if the opcode is not found in the opcode_to_input map.
 */
static bool IsRelevantInput(const Node* node, const NodeArg* input) {
  /*
   *  The following map specifies the opcode to input mapping to list the inputs to consider while propagating
   *  cast operations. All other inputs not listed in this table are not relevant for deciding whether an operation
   *  performed in float or float16. If an opcode is not listed in these tables, the code will look at all the inputs to validate
   *  transformation.
   */
  static const InlinedHashMap<std::string_view, std::array<int, 3>> opcode_to_input_map = {
      {"Gather", {0}},
      {"Reshape", {0}},
      {"Dropout", {0}},
      {"Expand", {0}},
      {"LayerNormalization", {0, 1, 2}},
      {"Squeeze", {0}},
      {"Unsqueeze", {0}}};

  auto it = opcode_to_input_map.find(node->OpType());
  if (it != opcode_to_input_map.cend()) {
    const auto& selected_inputs = it->second;
    int input_index = optimizer_utils::IndexOfNodeInput(*node, *input);
    return std::find(selected_inputs.begin(), selected_inputs.end(), input_index) != selected_inputs.end();
  }
  return true;
}

/*
 *  Check if the output is relevant to consider for cast propagation for the given node.
 *  Return true if the opcode is not found in the opcode_to_output map.
 */
static bool IsRelevantOutput(const Node* node, const NodeArg* output) {
  /*
   *  The following map specifies the opcode to output mapping to list the outputs to consider while propagating
   *  cast operations. All other outputs not listed in this table are not relevant for deciding whether an operation
   *  performed in float or float16. If an opcode is not listed in these tables, the code will look at all the outputs to validate
   *  transformation.
   */
  static const InlinedHashMap<std::string_view, std::array<int, 1>> opcode_to_output_map = {
      {"Gather", {0}},
      {"Reshape", {0}},
      {"Dropout", {0}},
      {"Expand", {0}},
      {"LayerNormalization", {0}},
      {"Squeeze", {0}},
      {"Unsqueeze", {0}}};

  auto it = opcode_to_output_map.find(node->OpType());
  if (it != opcode_to_output_map.cend()) {
    const auto& selected_outputs = it->second;
    int input_index = optimizer_utils::IndexOfNodeOutput(*node, *output);
    return std::find(selected_outputs.begin(), selected_outputs.end(), input_index) != selected_outputs.end();
  }
  return true;
}

// Check whether the node is a cast operation from float16/float to the specified data_type.
static bool IsCastTo(const Node* node, TensorProto_DataType data_type) {
  if (node->OpType() == "Cast") {
    const NodeAttributes& attributes = node->GetAttributes();
    const auto attr_hit = attributes.find("to");
    ORT_ENFORCE(attr_hit != attributes.end(), "Node: ", node->Name(),
                " is a Cast node and it must have 'to' attribute set.");
    const NodeArg* input = node->InputDefs()[0];
    auto input_data_type = static_cast<TensorProto_DataType>(input->TypeAsProto()->tensor_type().elem_type());
    // Allow cast nodes with same input and output type float/float16 to eliminate such casts.
    return (input_data_type == TensorProto::FLOAT16 || input_data_type == TensorProto::FLOAT) &&
           attr_hit->second.i() == static_cast<int64_t>(data_type);
  }
  return false;
}

// when node is softmax, and its input comes from a cast-to-fp32 node, and its output is only consumed by one cast-to-fp16 node, then we can treat it as a fp16-allowed op,
// as ort's softmax implementation already does the necessary cast logic, for example do reduce sum at fp32
static bool SoftmaxCanBeFP16(const Node& node) {
  if (node.OpType() != "Softmax")
    return false;
  // 1. input comes from a cast-to-fp32 node
  const Node* input_node = graph_utils::GetInputNode(node, 0);
  if (!(input_node && IsCastTo(input_node, TensorProto::FLOAT)))
    return false;
  // 2. output is consumed by a cast-to-fp16 node ONLY
  if (node.GetOutputEdgesCount() != 1)
    return false;
  const Node* output_node = &(*node.OutputNodesBegin());
  if (!(output_node && IsCastTo(output_node, TensorProto::FLOAT16)))
    return false;

  return true;
}

// Check whether the given opcode is fp16 allowed for the given level of optimization.
static bool IsFP16Allow(const Node* node, size_t level, const FP16AllowOps& fp16_allow_level0_ops) {
  if (!node)
    return false;
  // XXX: Shall we add a check for unsupported level or just ignore it as the current code does?
  constexpr size_t MaxSupportedCastPropagationLevel = 2;

  using OpsSetType = InlinedHashSet<std::string_view>;
  static const OpsSetType level1_fp16_allow_set =
      {"Expand", "Transpose", "Relu", "Reshape", "Split", "Tanh", "Squeeze", "Unsqueeze", "Gelu"};
  static const OpsSetType level2_fp16_allow_set = {
      "Add", "BiasGelu", "Dropout", "FastGelu", "Gather", "LayerNormalization", "Where"};

  // To support new optimization levels, you need to extend the below array with a set ops for the new level
  static const std::array<std::reference_wrapper<const OpsSetType>, MaxSupportedCastPropagationLevel> allowed_ops =
      {level1_fp16_allow_set, level2_fp16_allow_set};

  bool fp16_allow = Contains(fp16_allow_level0_ops, node->OpType());
  for (size_t i = 1, limit = std::min(level, MaxSupportedCastPropagationLevel); i <= limit && !fp16_allow; ++i) {
    fp16_allow = Contains(allowed_ops[i - 1].get(), node->OpType());
  }
  return fp16_allow || SoftmaxCanBeFP16(*node);
}

// Check whether the node-arg element type is same the specified data type
static inline bool IsType(const NodeArg& node_arg, TensorProto_DataType data_type) {
  return node_arg.TypeAsProto()->tensor_type().elem_type() == data_type;
}

/* InsertCastNodes
 * Insert a new Cast node after each NodeArg in the require_cast map, feeding the nodes (consumer) in the vector mapped to
 * the NodeArg. The other consumers of the NodeArg will not be changed. The cast node is FLOAT16 if is_fp16 is True
 * and FLOAT otherwise. This function fixes the graph edges in addition to inserting the cast nodes.
 *
 * In the following example only the first two consumers, Opcode0 and Opcode1 get casted and the third consumer Opcode2 does not.
 *
 *                Input0/NodeArg                                  Input/NodeArg
 *               ___ ____|________________                           |____________________
 *               |              |        |                           |                   |
 *               |              |        |                      _____V______             |
 *               |              |        |                      | CastFP16 |             |
 *               |              |   _____V_____                 |_or_FP32__|       ______V___
 *               |              |   | Opcode2 |                      |             | Opcode2|
 *               |              |   |_________|               _______|___________  |________|
 *               |              |                             |                 |
 *               |              |                ---\    _____V______      _____V_____
 *               |              |                ---/    | Opcode0  |      | Opcode1 |
 *          _____V_____    _____V______                  |__________|      |_________|
 *          | Opcode0 |    | Opcode1  |                       |                 |
 *          |_________|    |__________|
 *               |              |
 */
static Status InsertCastNodes(Graph& graph,
                              const NodeArgToConsumerMap& require_cast,
                              bool is_fp16,
                              const NodeIndices& removed_nodes,
                              NodeIndices& inserted_nodes) {
  // Create required new Cast nodes.
  for (const auto& [node_arg, nodes] : require_cast) {
    if (!node_arg->Exists()) {
      continue;
    }
    // data_type is the data type of the Cast output.
    TensorProto_DataType data_type = is_fp16 ? TensorProto_DataType_FLOAT16 : TensorProto_DataType_FLOAT;
    TypeProto type_proto;
    bool is_node_arg_cast_output = IsType(*node_arg, data_type);  // true if the producer node_arg is being replaced
    TensorProto_DataType new_node_arg_data_type = data_type;

    if (is_node_arg_cast_output) {
      new_node_arg_data_type = (data_type == TensorProto_DataType_FLOAT) ? TensorProto_DataType_FLOAT16 : TensorProto_DataType_FLOAT;
    }
    // The below code assumes that the node is a tensor
    type_proto.mutable_tensor_type()->set_elem_type(new_node_arg_data_type);
    NodeArg& new_node_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node_arg->Name()), &type_proto);
    // Expect that a NodeArg is not both a graph input as well as a graph output
    ORT_ENFORCE(!(graph.IsInputsIncludingInitializers(node_arg) && graph.IsOutput(node_arg)),
                "Expect that a NodeArg is not both a graph input as well as a graph output");
    NodeArg& cast_input = !is_node_arg_cast_output ? *node_arg : new_node_arg;
    NodeArg& cast_output = is_node_arg_cast_output ? *node_arg : new_node_arg;
    const std::array cast_inputs = {&cast_input};
    const std::array cast_outputs = {&cast_output};

    Node& cast = graph.AddNode(graph.GenerateNodeName(node_arg->Name() + "_cast"),
                               "Cast",
                               "Created a new Cast node",
                               cast_inputs,
                               cast_outputs);
    cast.AddAttribute("to", static_cast<int64_t>(data_type));

    inserted_nodes.insert(cast.Index());
    Node* producer = graph.GetMutableProducerNode(node_arg->Name());
    auto consumers = graph.GetMutableConsumerNodes(node_arg->Name());
    InlinedVector<Node*> other_nodes;
    if (nullptr != producer) {
      int output_index = optimizer_utils::IndexOfNodeOutput(*producer, *node_arg);
      for (Node* consumer : consumers) {
        // Removed the edges to the consumers, getting casted.
        if (std::find(nodes.begin(), nodes.end(), consumer) != nodes.end()) {
          int input_index = optimizer_utils::IndexOfNodeInput(*consumer, *node_arg);
          graph.RemoveEdge(producer->Index(), consumer->Index(), output_index, input_index);
        } else {
          other_nodes.push_back(consumer);
        }
      }
      if (is_node_arg_cast_output) {
        // Replace the node_arg with the new_node_arg in the producer outputs
        auto& producer_outputs = producer->MutableOutputDefs();
        std::replace(producer_outputs.begin(), producer_outputs.end(), &cast_output, &cast_input);
        graph.UpdateProducerNode(cast_input.Name(), producer->Index());
      }
    }
    // Update consumers of node_arg to use the output of the cast node
    int cast_output_index = optimizer_utils::IndexOfNodeOutput(cast, cast_output);
    for (Node* consumer : consumers) {
      if (!Contains(removed_nodes, consumer->Index())) {
        if (std::find(nodes.begin(), nodes.end(), consumer) == nodes.end()) {
          // Consumers not getting casted need to replace input-def if the producer's output-def is changed
          if (is_node_arg_cast_output) {
            auto& consumer_inputs = consumer->MutableInputDefs();
            std::replace(consumer_inputs.begin(), consumer_inputs.end(), &cast_output, &cast_input);
          }
        } else {
          // Consumers getting casted need to get new edges from the new cast node..
          int input_index = optimizer_utils::IndexOfNodeInput(*consumer, *node_arg);
          if (!is_node_arg_cast_output) {
            auto& consumer_inputs = consumer->MutableInputDefs();
            std::replace(consumer_inputs.begin(), consumer_inputs.end(), &cast_input, &cast_output);
          }
          graph.AddEdge(cast.Index(), consumer->Index(), cast_output_index, input_index);
        }
      }
    }
    // Complete the input/output connections to the new cast node, and update the graph.
    other_nodes.push_back(&cast);
    graph.UpdateProducerNode(cast_output.Name(), cast.Index());
    graph.UpdateConsumerNodes(cast_output.Name(), nodes);
    graph.UpdateConsumerNodes(cast_input.Name(), other_nodes);
    if (nullptr != producer) {
      int cast_input_index = optimizer_utils::IndexOfNodeInput(cast, cast_input);
      int output_index = optimizer_utils::IndexOfNodeOutput(*producer, cast_input);
      graph.AddEdge(producer->Index(), cast.Index(), output_index, cast_input_index);
      if (is_node_arg_cast_output) {
        graph.UpdateProducerNode(cast_input.Name(), producer->Index());
      }
    }
  }
  return Status::OK();
}

/* RemoveCastNodesChain
 * Remove the cast nodes specified in casts vector and fix the graph edges accordingly. If the output of the cast
 * is also a graph output then insert an Identity node if the input of the cast node feeds other nodes. In the
 * trivial case the chain has only one cast node. The caller is responsible for the validity of removing casts.
 *
 *                         _____|______
 *                         | Opcode1  |
 *                         |__________|
 *                              |
 *                         _____V______                         ____________
 *                         |  Cast    |                         | Opcode 1 |
 *                         |__________|                         |__________|
 *                              |                                    |
 *                              .                                    |
 *                              .                  ---\         _____V______
 *                              .                  ---/         | Opcode2  |
 *                         _____V______                         |__________|
 *                         |  Cast    |                              |
 *                         |__________|                              V
 *                              |
 *                         _____V______
 *                         |  Opcode2 |
 *                         |__________|
 *                              |
 *                              V
 *
 *    OR
 *                         _____|______                         _____|_____
 *                         | Opcode1  |                         | Opcode1 |
 *                         |__________|                         |_________|
 *                              |                                    |
 *                         _____V______            ---\         _____V______
 *                         |  Cast    |            ---/         | Opcode 2 |
 *                         |__________|                         |__________|
 *                              |                                    |
 *                         _____V______                              V
 *                         |  Opcode2 |
 *                         |__________|
 *                              |
 *                              V
 */

static Status RemoveCastNodesChain(Graph& graph, gsl::span<Node* const> casts, NodeIndices& removed_nodes) {
  ORT_ENFORCE(!casts.empty(), "Casts must not be empty");
  Node* lead_cast = *casts.begin();
  Node* trail_cast = casts.back();
  NodeArg* cast_input = lead_cast->MutableInputDefs()[0];
  NodeArg* cast_output = trail_cast->MutableOutputDefs()[0];
  // Update producer node
  Node* producer = graph.GetMutableProducerNode(cast_input->Name());
  auto consumers = graph.GetMutableConsumerNodes(cast_output->Name());
  int output_index = (nullptr != producer) ? optimizer_utils::IndexOfNodeOutput(*producer, *cast_input) : -1;
  if (producer) {
    int input_index = optimizer_utils::IndexOfNodeInput(*lead_cast, *cast_input);
    graph.RemoveEdge(producer->Index(), lead_cast->Index(), output_index, input_index);
    if (graph.IsOutput(cast_output)) {
      // cast_output is a graph output. Replace the cast node with an Identity operator unless node
      // has no other outputs.
      if (producer->GetOutputEdgesCount() == 0) {
        graph.RemoveEdge(producer->Index(), lead_cast->Index(), output_index, input_index);
        auto& outputs = producer->MutableOutputDefs();
        std::replace(outputs.begin(), outputs.end(), cast_input, cast_output);
        graph.UpdateProducerNode(cast_output->Name(), producer->Index());
      } else {
        Node& identity = graph.AddNode(graph.GenerateNodeName(producer->Name() + "_identity"),
                                       "Identity",
                                       "Created as a place-holder for a graph output",
                                       {cast_input},
                                       {cast_output});
        graph.AddEdge(producer->Index(), identity.Index(), output_index, 0);
        // Add identity to the producer's consumer nodes.
        graph.AddConsumerNode(cast_input->Name(), &identity);
      }
    }
  }
  // Update consumer nodes
  if (!consumers.empty()) {
    int cast_output_index = optimizer_utils::IndexOfNodeOutput(*trail_cast, *cast_output);
    for (Node* consumer : consumers) {
      if (nullptr != consumer &&
          !Contains(removed_nodes, consumer->Index())) {
        auto& consumer_inputs = consumer->MutableInputDefs();
        int input_index = optimizer_utils::IndexOfNodeInput(*consumer, *cast_output);
        graph.RemoveEdge(trail_cast->Index(), consumer->Index(), cast_output_index, input_index);
        std::replace(consumer_inputs.begin(), consumer_inputs.end(), cast_output, cast_input);
        if (producer) {
          graph.AddEdge(producer->Index(), consumer->Index(), output_index, input_index);
        }
      }
    }
    graph.UpdateConsumerNodes(cast_input->Name(), consumers);
  }
  for (auto cast : casts) {
    graph_utils::RemoveNodeOutputEdges(graph, *cast);
    removed_nodes.insert(cast->Index());
  }
  return Status::OK();
}

/*
 * RemoveBackToBackCasts
 * Remove FLOAT and FLOAT16 casts back-to-back, only if the parent cast is from FLOAT16 to FLOAT
 * and the child cast is from FLOAT to FLOAT16 or from FLOAT to FLOAT.
 * The trivial case is parent cast has only one output. A non-trivial case handled by this function
 * is, parent has multiple children and one or more child node output is also a graph output.
 * In the non-trivial case, when a child-cast nullifies the parent cast, the consumer nodes are moved
 * from the child-cast to the producer of the parent-cast input. The code handles cornercases such as
 * the child-cast output is also a graph output, in addition to possibly other nodes, and the
 * parent-cast has other output nodes. The first check CanRemoveNode rules-out the possibility of
 * parent-cast output also being the graph output.
 * The inputs is Cast to FLOAT16
 * With the possibility of child/parent cast feeding other nodes/graph-output, the transformation is either
 *                         _____|______
 *                         | Opcode1  |
 *                         |__________|
 *                              |
 *                         _____V______                         ____________
 *                         | Cast     |                         | Opcode 1 |
 *                         |FP16->FP32|                         |__________|
 *                         |__________|                              |
 *                              |                                    |
 *                              |                  ---\         _____V______
 *                              |                  ---/         | Opcode2  |
 *                         _____V______                         |__________|
 *                         | Cast     |
 *                         |FP32->Fp16|
 *                         |__________|
 *                              |
 *                         _____V______
 *                         |  Opcode2 |
 *                         |__________|
 *
 *   or
 *
 *                         _____|______
 *                         | Opcode1  |
 *                         |__________|
 *                              |
 *                         _____V______                         ____________
 *                         | Cast     |                         | Opcode 1 |
 *                         |FP16->FP32|                         |__________|
 *                         |__________|                              |
 *                              |                                    |
 *                              |                  ---\         _____V______
 *                              |                  ---/         | Cast FP32|
 *                         _____V______                         |__________|
 *                         | Cast     |                              |
 *                         |FP32->FP32|                              |
 *                         |__________|                         _____V______
 *                              |                               | Opcode2  |
 *                         _____V______                         |__________|
 *                         |  Opcode2 |                              |
 *                         |__________|
 *                              |
 */
static bool RemoveBackToBackCasts(Graph& graph, Node* parent,
                                  NodeIndices& removed_nodes,
                                  const logging::Logger& logger) {
  ORT_ENFORCE(IsCastTo(parent, TensorProto::FLOAT),
              "Expecting parent node: ", parent->Name(), " to be a cast to FLOAT");
  bool modified = false;
  if (graph_utils::CanRemoveNode(graph, *parent, logger)) {
    NodeArg* cast_output = parent->MutableOutputDefs()[0];
    auto children = graph.GetMutableConsumerNodes(cast_output->Name());
    if (children.size() == 1) {
      Node* child = children[0];
      if (!Contains(removed_nodes, child->Index())) {
        if (IsCastTo(child, TensorProto::FLOAT16)) {
          // The parent and child cancel out
          LOGS(logger, VERBOSE) << "RemoveBackToBackCasts: Removed Cast nodes  " << parent->Name() << " and " << child->Name();
          ORT_THROW_IF_ERROR(RemoveCastNodesChain(graph, AsSpan({parent, child}), removed_nodes));
          modified = true;
        } else if (IsCastTo(child, TensorProto::FLOAT)) {
          // Child is a duplicate of parent
          LOGS(logger, VERBOSE) << "RemoveBackToBackCasts: Removed Cast node  " << child->Name();
          ORT_THROW_IF_ERROR(RemoveCastNodesChain(graph, AsSpan({child}), removed_nodes));
          modified = true;
        }
      }
    } else {
      NodeArg* parent_input = parent->MutableInputDefs()[0];
      const Node* producer = graph.GetProducerNode(parent_input->Name());
      int producer_output_index = producer ? optimizer_utils::IndexOfNodeOutput(*producer, *parent_input) : -1;
      InlinedVector<Node*> new_consumers;
      size_t children_count = children.size();
      for (Node* child : children) {
        if (!Contains(removed_nodes, child->Index())) {
          if (IsCastTo(child, TensorProto::FLOAT16)) {
            // The parent and child cancell out
            // Remove the child node without effecting the other nodes.
            // move all the consumers to the producer.
            LOGS(logger, VERBOSE) << "RemoveBackToBackCasts: Removed Cast node  " << child->Name();
            NodeArg* child_output = child->MutableOutputDefs()[0];
            for (Node* consumer : graph.GetMutableConsumerNodes(child_output->Name())) {
              auto& consumer_inputs = consumer->MutableInputDefs();
              int output_index = optimizer_utils::IndexOfNodeOutput(*child, *child_output);
              int input_index = optimizer_utils::IndexOfNodeInput(*consumer, *child_output);
              graph.RemoveEdge(child->Index(), consumer->Index(), output_index, input_index);
              std::replace(consumer_inputs.begin(), consumer_inputs.end(), child_output, parent_input);
              if (nullptr != producer) {
                graph.AddEdge(producer->Index(), consumer->Index(), producer_output_index, input_index);
              }
              new_consumers.push_back(consumer);
            }
            if (graph.IsOutput(child_output) && nullptr != producer) {
              Node& identity = graph.AddNode(graph.GenerateNodeName(producer->Name() + "_identity"),
                                             "Identity",
                                             "Created as a place-holder for a graph output",
                                             {parent_input},
                                             {child_output});
              graph.AddEdge(producer->Index(), identity.Index(), producer_output_index, 0);
              graph.AddConsumerNode(parent_input->Name(), &identity);
              graph.UpdateProducerNode(child_output->Name(), identity.Index());
            }
            modified = true;
            graph.RemoveEdge(parent->Index(), child->Index(), 0, 0);
            removed_nodes.insert(child->Index());
            children_count--;
          } else if (IsCastTo(child, TensorProto::FLOAT)) {
            // Child is a duplicate of parent
            LOGS(logger, VERBOSE) << "RemoveBackToBackCasts: Removed Cast node  " << child->Name();
            graph.RemoveEdge(parent->Index(), child->Index(), 0, 0);
            ORT_THROW_IF_ERROR(RemoveCastNodesChain(graph, AsSpan({child}), removed_nodes));
            modified = true;
          }
        }
      }
      if (children_count == 0) {
        // No more children nodes exists, and the parent-cast output is not a graph output. Remove it!
        ORT_THROW_IF_ERROR(RemoveCastNodesChain(graph, AsSpan({parent}), removed_nodes));
      }
      if (!new_consumers.empty()) {
        auto consumers = graph.GetMutableConsumerNodes(parent_input->Name());
        std::copy(new_consumers.begin(), new_consumers.end(), back_inserter(consumers));
        graph.UpdateConsumerNodes(parent_input->Name(), consumers);
      }
    }
  }
  return modified;
}

/*
 *  SearchUpstream:
 *  ReverseDFS, traverse bottom-up, the graph upstream collecting all the NodeArgs that require a cast
 *  in order to move an FP16 Cast operation up the graph.
 *  Visited float NodeArgs are either in require_cast or require_type_change so that the same
 *  nodearg is traversed not more than once.
 *  If the level is 2, the functions traverses up the graph identifying required FP32 casts even if
 *  multiple consumers for the node outputs are found while level 0 or 1 quit traversing up.
 */
static void SearchUpstream(Graph& graph, NodeArg* node_arg, Node* dst_node,
                           NodeArgToConsumerMap& require_cast,
                           NodeArgToConsumerMap& require_cast_fp32,
                           InlinedHashSet<NodeArg*>& require_type_change,
                           const NodeIndices& removed_nodes,
                           size_t level,
                           const FP16AllowOps& fp16_allow_ops) {
  Node* node = graph.GetMutableProducerNode(node_arg->Name());
  // If the Cast input feeds more than one node or the cast node feeds a graph output and at least one
  // node then it cannot propagate.
  size_t consumer_node_count = graph.GetConsumerNodes(node_arg->Name()).size();
  // Do not traverse up the graph if the node produces a graph output as well as feeds other nodes
  // or if the node has more than one consumers.
  if (level < 2 && (consumer_node_count > 1 ||
                    (nullptr != node &&
                     consumer_node_count > 0 &&
                     graph.IsOutput(node_arg)))) {
    require_cast[node_arg].push_back(dst_node);
  } else if (node == nullptr) {
    // The graph inputs don't have the producer nodes
    if (IsType(*node_arg, TensorProto_DataType_FLOAT)) {
      require_cast[node_arg].push_back(dst_node);
    }
  } else if (!Contains(removed_nodes, node->Index())) {
    if (IsCastTo(node, TensorProto_DataType_FLOAT)) {
      // This Cast node and the Cast node that will be created later will cancel out
      require_cast[node_arg].push_back(dst_node);
    } else {
      if (!IsFP16Allow(node, level, fp16_allow_ops)) {
        // Cannot traverse-up beyond this point
        if (node_arg->Exists() && IsType(*node_arg, TensorProto_DataType_FLOAT)) {
          require_cast[node_arg].push_back(dst_node);
        }
      } else {
        // If the node has other float32 output(s) then stop the search.
        for (const auto* output_def : node->OutputDefs()) {
          // TODO: If the specified optimization is greater than 1 then insert a Cast to the
          // other output_def and still propagate FP16 cast up the graph.
          if (output_def != node_arg) {
            if (IsRelevantOutput(node, output_def) &&
                IsType(*output_def, TensorProto_DataType_FLOAT) &&
                graph.GetConsumerNodes(output_def->Name()).size() > 0) {
              require_cast[node_arg].push_back(dst_node);
              return;
            }
          }
        }
        if (level >= 2) {
          for (Node* consumer : graph.GetMutableConsumerNodes(node_arg->Name())) {
            if (nullptr != consumer && consumer != dst_node && consumer->OpType() != "Cast" &&
                !Contains(removed_nodes, consumer->Index())) {
              require_cast_fp32[node_arg].push_back(consumer);
            }
          }
          if (graph.IsOutput(node_arg)) {
            require_cast_fp32[node_arg].clear();
          }
        }
        for (NodeArg* node_input : node->MutableInputDefs()) {
          if (IsRelevantInput(node, node_input) &&
              IsType(*node_input, TensorProto_DataType_FLOAT) &&
              !Contains(require_cast, node_input) &&
              !Contains(require_type_change, node_input)) {
            SearchUpstream(graph, node_input, node, require_cast, require_cast_fp32, require_type_change, removed_nodes, level, fp16_allow_ops);
            if (!Contains(require_cast, node_input) &&
                !Contains(require_cast_fp32, node_input)) {
              require_type_change.insert(node_input);
            }
          }
        }
      }
    }
  }
}

/*
 *  SearchDownstream:
 *  Recursively DFS traverse the graph downstream collecting all the NodeArgs that require a cast
 *  in order to remove an FP32 Cast operation up the graph. Also collect the NodeArgs that need to
 *  be converted from float to float16 along the way.
 */
static void SearchDownstream(Graph& graph, NodeArg* node_arg,
                             NodeArgToConsumerMap& require_cast,
                             NodeArgToConsumerMap& require_cast_fp16,
                             InlinedHashSet<NodeArg*>& require_type_change,
                             size_t level,
                             const FP16AllowOps& fp16_allow_ops) {
  for (Node* node : graph.GetMutableConsumerNodes(node_arg->Name())) {
    if (node) {
      if (IsCastTo(node, TensorProto_DataType_FLOAT)) {
        // This Cast node and the Cast node that will be created later will cancel out
        require_cast[node_arg].push_back(node);
      } else {
        if (!IsFP16Allow(node, level, fp16_allow_ops)) {
          if (node_arg->Exists() &&
              IsType(*node_arg, TensorProto_DataType_FLOAT)) {
            require_cast[node_arg].push_back(node);
          }
        } else {
          // If the node has other float32 inputs then stop the search
          for (NodeArg* input_def : node->MutableInputDefs()) {
            // TODO: If the specified level of the optimization is greater than 1 then
            // convert initializers if any from float to float16.
            if (input_def != node_arg && IsRelevantInput(node, input_def) &&
                IsType(*input_def, TensorProto_DataType_FLOAT)) {
              if (level < 2) {
                require_cast[node_arg].push_back(node);
                return;
              } else {
                require_cast_fp16[input_def].push_back(node);
              }
            }
          }
          for (NodeArg* node_output : node->MutableOutputDefs()) {
            if (IsRelevantOutput(node, node_output) && IsType(*node_output, TensorProto_DataType_FLOAT) &&
                !Contains(require_cast, node_output) &&
                !Contains(require_type_change, node_output)) {
              SearchDownstream(graph, node_output, require_cast, require_cast_fp16, require_type_change, level, fp16_allow_ops);
              if (!Contains(require_cast, node_output)) {
                require_type_change.insert(node_output);
              }
            }
          }
        }
      }
    }
  }

  if (graph.IsOutput(node_arg)) {
    require_cast.emplace(node_arg, InlinedVector<Node*>{});
  }
}

// Change the elem_type of the given NodeArgs from FLOAT to FLOAT16.
static void ChangeTypeToFP16(Graph& graph, InlinedHashSet<NodeArg*>& require_type_change, bool is_forward,
                             NodeIndices& converted_nodes,
                             const NodeIndices& inserted_nodes,
                             const logging::Logger& logger) {
  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT16);
  for (NodeArg* node_arg : require_type_change) {
    if (IsType(*node_arg, TensorProto::FLOAT)) {
      ORT_THROW_IF_ERROR(node_arg->UpdateTypeAndShape(type_proto, true, true, logger));
      LOGS(logger, VERBOSE) << "Changed " << node_arg->Name() << " from float to float16" << std::endl;
      if (is_forward) {
        // Propagating forwards. Count consumers.
        for (const Node* node : graph.GetConsumerNodes(node_arg->Name())) {
          if (!Contains(inserted_nodes, node->Index())) {
            converted_nodes.insert(node->Index());
          }
        }
      } else {
        // Propagating backwards. Count producers.
        const Node* producer = graph.GetProducerNode(node_arg->Name());
        if (nullptr != producer && !Contains(inserted_nodes, producer->Index())) {
          converted_nodes.insert(producer->Index());
        }
      }
    }
  }
}

/*
 * PropagateForwards
 * Propagate FP32 Cast operations forwards (downstream)
 * Using SearchDownStream search the graph for Cast FP16 safe/allowed operations to expand
 * the float16 computation region.
 * The required_cast vector is the collection of nodes that require float cast.
 * All nodeargs on a path down to any of the
 * frontier nodes require type change from FLOAT to FLOAT16.
 * require_type_change consists of such nodes.  All the frontier nodes require fp32 cast
 * The input node is expected to be non-nullptr
 */
static bool PropagateForwards(Graph& graph, Node* node,
                              NodeIndices& removed_nodes,
                              size_t level,
                              const FP16AllowOps& fp16_allow_ops,
                              NodeIndices& converted_nodes,
                              NodeIndices& inserted_nodes,
                              const logging::Logger& logger) {
  ORT_ENFORCE(nullptr != node, "Invalid argument, node must not be nullptr");
  bool modified = false;
  NodeArgToConsumerMap require_cast;
  NodeArgToConsumerMap require_cast_fp16;
  InlinedHashSet<NodeArg*> require_type_change;
  NodeArg* cast_output = node->MutableOutputDefs()[0];
  SearchDownstream(graph, cast_output, require_cast, require_cast_fp16, require_type_change, level, fp16_allow_ops);
  if (!require_cast.empty() && !Contains(require_cast, cast_output)) {
    if (!require_cast_fp16.empty()) {
      ORT_THROW_IF_ERROR(InsertCastNodes(graph, require_cast_fp16, true, removed_nodes, inserted_nodes));
      LOGS(logger, VERBOSE) << "PropagateForwards: Inserted FP16 Cast nodes "
                            << ConcatNames(require_cast_fp16, GetName);
    }
    // Remove Cast operation
    LOGS(logger, VERBOSE) << "PropagateForwards: Removed Cast node  " << node->Name();
    ORT_THROW_IF_ERROR(RemoveCastNodesChain(graph, AsSpan({node}), removed_nodes));
    ORT_THROW_IF_ERROR(InsertCastNodes(graph, require_cast, false, removed_nodes, inserted_nodes));
    LOGS(logger, VERBOSE) << "PropagateForwards: Inserted Cast FP32 nodes "
                          << ConcatNames(require_cast, GetName);
    ChangeTypeToFP16(graph, require_type_change, true, converted_nodes, inserted_nodes, logger);
    modified = true;
  }
  return modified;
}
/*
 *  PropagateBackwards
 *  Propagate FP16 Cast operations backwards (upstream)
 *  Using SearchUpstream search the graph for Cast FP16 safe/allowed operations and expand
 *  float16 computation regsion and
 *  find the frontiers of the float16 computation region.
 *  The required_cast or require_cast_fp32 vector is a collection of
 *  FP16-cast-frontiers of the cast node. All node-args on the path from any of the
 *  frontier nodes to the cast node require type change from  FLOAT to FLOAT16.
 *  Each of the frontier nodes requires an fp16 cast or fp32 cast.
 *  The input node is expected be non-nullptr.
 */
static bool PropagateBackwards(Graph& graph, Node* node,
                               NodeIndices& removed_nodes,
                               size_t level,
                               const FP16AllowOps& fp16_allow_ops,
                               NodeIndices& converted_nodes,
                               NodeIndices& inserted_nodes,
                               const logging::Logger& logger) {
  ORT_ENFORCE(nullptr != node, "Invalid argument node must not be nullptr");
  bool modified = false;
  NodeArgToConsumerMap require_cast;
  NodeArgToConsumerMap require_cast_fp32;
  NodeArg* cast_input = node->MutableInputDefs()[0];
  InlinedHashSet<NodeArg*> require_type_change;
  SearchUpstream(graph, cast_input, node, require_cast, require_cast_fp32, require_type_change, removed_nodes, level, fp16_allow_ops);
  if (require_cast_fp32.empty()) {
    require_type_change.insert(cast_input);
  }
  // TODO need a heuristic when to insert FP32 Cast
  if (!require_cast.empty() && !Contains(require_cast, cast_input) /* && require_cast.size() >= require_cast_fp32.size() */) {
    if (!require_cast_fp32.empty()) {
      ORT_THROW_IF_ERROR(InsertCastNodes(graph, require_cast_fp32, false, removed_nodes, inserted_nodes));
      LOGS(logger, VERBOSE) << "PropagateBackwards: Inserted FP32 Cast nodes "
                            << ConcatNames(require_cast_fp32, GetName);
    }
    // Remove Cast operations
    ORT_THROW_IF_ERROR(RemoveCastNodesChain(graph, AsSpan({node}), removed_nodes));
    LOGS(logger, VERBOSE) << "PropagateBackwards: Removed Cast node  " << node->Name();
    ORT_THROW_IF_ERROR(InsertCastNodes(graph, require_cast, true, removed_nodes, inserted_nodes));
    LOGS(logger, VERBOSE) << "PropagateBackwards: Inserted Cast nodes "
                          << ConcatNames(require_cast, GetName);
    ChangeTypeToFP16(graph, require_type_change, false, converted_nodes, inserted_nodes, logger);
    LOGS(logger, VERBOSE) << "PropagateBackwards: Changed the type from float to float16 : "
                          << ConcatNames(require_type_change);
    modified = true;
  }
  return modified;
}

/*
 * FuseNodes
 * Fuse all (cast) nodes, and replace with a single (cast) node.
 * Assumptions:
 * 1. all nodes are Cast ops and are of the same Cast type
 * 2. all the nodes have the same input
 *                Input0/NodeArg
 *               ___ ____|______                               Input0/NodeArg
 *               |             |                                     |
 *          _____V____    _____V______                          _____V______
 *          |Cast FP16|    | Cast FP16|                         | CastFP16 |
 *          |or_FP32__|    |_or_FP32 _|                         |_or_FP32__|
 *               |              |                                    |
 *               |              |                             _______|___________
 *               |              |                             |                 |
 *               |              |                ---\    _____V______      _____V_____
 *               |              |                ---/    | Opcode0  |      | Opcode1 |
 *          _____V_____    _____V______                  |__________|      |_________|
 *          | Opcode0 |    | Opcode1  |                       |                 |
 *          |_________|    |__________|
 *               |              |
 */
static void FuseNodes(Graph& graph, const NodeArg* input, gsl::span<Node* const> nodes,
                      NodeIndices& removed_nodes,
                      NodeIndices& inserted_nodes) {
  ORT_ENFORCE(!nodes.empty(), "Nodes to fuse must not be empty");
  Node* node = nodes[0];
  const Node* producer = graph.GetProducerNode(input->Name());
  int output_index = nullptr != producer ? optimizer_utils::IndexOfNodeOutput(*producer, *input) : -1;
  NodeArg* node_arg = node->MutableOutputDefs()[0];
  NodeArg& new_output = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(input->Name()), node_arg->TypeAsProto());
  Node& new_cast = graph.AddNode(graph.GenerateNodeName(node->Name() + "_replace"),
                                 node->OpType(),
                                 "Created to replace a node",
                                 {graph.GetNodeArg(input->Name())},
                                 {&new_output},
                                 &node->GetAttributes(),
                                 node->Domain());
  if (nullptr != producer) {
    graph.AddEdge(producer->Index(), new_cast.Index(), output_index, 0);
  }
  inserted_nodes.insert(new_cast.Index());
  InlinedVector<Node*> consumers;
  for (Node* cast : nodes) {
    for (NodeArg* output : cast->MutableOutputDefs()) {
      for (Node* consumer : graph.GetMutableConsumerNodes(output->Name())) {
        int input_index = optimizer_utils::IndexOfNodeInput(*consumer, *output);
        auto& inputs = consumer->MutableInputDefs();
        graph.RemoveEdge(cast->Index(), consumer->Index(), 0, input_index);
        std::replace(inputs.begin(), inputs.end(), output, &new_output);
        graph.AddEdge(new_cast.Index(), consumer->Index(), 0, input_index);
        consumers.push_back(consumer);
      }
    }
    if (nullptr != producer) {
      int input_index = optimizer_utils::IndexOfNodeInput(*cast, *input);
      graph.RemoveEdge(producer->Index(), cast->Index(), output_index, input_index);
    }
    graph.RemoveConsumerNode(input->Name(), cast);
  }
  graph.AddConsumerNode(input->Name(), &new_cast);
  graph.UpdateConsumerNodes(new_output.Name(), consumers);
  for (Node* n : nodes) {
    removed_nodes.insert(n->Index());
    graph_utils::RemoveNodeOutputEdges(graph, *n);
  }
}

// Traverse the graph recursively searching/collecting sibling Cast op nodes to fuse and call FuseNodes.
static bool FuseSiblingCasts(Graph& graph, const NodeArg* node_arg,
                             NodeIndices& removed_nodes,
                             NodeIndices& inserted_nodes,
                             const logging::Logger& logger) {
  bool modified = false;
  InlinedVector<Node*> cast_fp16_siblings;
  InlinedVector<Node*> cast_fp32_siblings;
  for (Node* node : graph.GetMutableConsumerNodes(node_arg->Name())) {
    // If a cast node feeds a graph output then it is not a candidate for fusion.
    if (nullptr == node || node->OpType() != "Cast" ||
        Contains(removed_nodes, node->Index()) ||
        graph.IsOutput(node->OutputDefs()[0])) {
      continue;
    }
    if (IsCastTo(node, TensorProto::FLOAT16)) {
      cast_fp16_siblings.push_back(node);
    } else if (IsCastTo(node, TensorProto::FLOAT)) {
      cast_fp32_siblings.push_back(node);
    }
  }
  if (cast_fp16_siblings.size() > 1) {
    modified = true;
    FuseNodes(graph, node_arg, cast_fp16_siblings, removed_nodes, inserted_nodes);
    LOGS(logger, VERBOSE) << "FusedSubgraphs: Fused Cast nodes : " << ConcatNames(cast_fp16_siblings);
  }
  if (cast_fp32_siblings.size() > 1) {
    modified = true;
    FuseNodes(graph, node_arg, cast_fp32_siblings, removed_nodes, inserted_nodes);
    LOGS(logger, VERBOSE) << "FusedSubgraphs: Fused Cast nodes : " << ConcatNames(cast_fp32_siblings);
  }
  return modified;
}

// Overloaded function which goes through all the output args of a node and calls FuseSiblingCasts
static bool FuseSiblingCasts(Graph& graph, const Node* parent,
                             NodeIndices& removed_nodes,
                             NodeIndices& inserted_nodes,
                             const logging::Logger& logger) {
  bool modified = false;
  for (const NodeArg* output : parent->OutputDefs()) {
    modified |= FuseSiblingCasts(graph, output, removed_nodes, inserted_nodes, logger);
  }
  return modified;
}

// RemoveUnnecessaryCasts
// Remove a cast if the input elem_type is same the required cast type.
static bool RemoveUnnecessaryCasts(Graph& graph, Node* node,
                                   NodeIndices& removed_nodes,
                                   const logging::Logger& logger) {
  bool modified = false;
  if (node->InputDefs().size() == 1) {
    const NodeArg* node_arg = node->InputDefs()[0];
    auto elem_type = node_arg->TypeAsProto()->tensor_type().elem_type();
    TensorProto_DataType data_type = static_cast<TensorProto_DataType>(elem_type);
    if (IsCastTo(node, data_type)) {
      LOGS(logger, VERBOSE) << "Removed unnecessary cast " << node->Name();
      ORT_THROW_IF_ERROR(RemoveCastNodesChain(graph, AsSpan({node}), removed_nodes));
      modified = true;
    }
  }
  return modified;
}

/*
 * PropagateFP32CastsFromInputsToOutputs
 * This non-recursive fusion, checks whether the given node is fp16 safe op and
 * whether all floatingpoint inputs are cast to fp32
 * and propagates cast op to the floatingpoint outputs.
 * Convert the following graph
 *
 *        Input0/NodeArg  Input1/NodeArg
 *               |             |
 *          _____V____    _____V______
 *          |Cast FP32|   | Cast FP32|
 *          |_________|   |__________|
 *               |              |
 *             __V______________V___
 *            |       Opcode        |
 *            |(operation performed |
 *            |    in float32)      |
 *            |_____________________|
 *              |             |
 *              |             |
 *
 *  and produce the following output
 *
 *          Input0/NodeArg  Input1/NodeArg
 *                |               |
 *              __V_______________V___
 *             |       Opcode        |
 *             |(operation performed |
 *             |    in float16)      |
 *             |_____________________|
 *               |             |
 *          _____V____    _____V______
 *          |Cast FP32|   | Cast FP32|
 *          |_________|   |__________|
 *               |              |
 *
 *
 */
static bool PropagateFP32CastsFromInputsToOutputs(Graph& graph, Node* node,
                                                  NodeIndices& removed_nodes,
                                                  size_t level,
                                                  const FP16AllowOps& fp16_allow_ops,
                                                  NodeIndices& converted_nodes,
                                                  NodeIndices& inserted_nodes,
                                                  const logging::Logger& logger) {
  bool modified = false;
  if (IsFP16Allow(node, level, fp16_allow_ops)) {
    bool has_float_inputs = false;
    bool all_float_inputs_have_casts = true;
    InlinedVector<Node*> casts;
    auto& inputs = node->MutableInputDefs();
    InlinedHashSet<NodeArg*> require_type_change;
    // TODO Here we require the all floating point inputs are generated by an immediate
    // parent cast node and all casts feed only one output node. Remove or reduce these restrictions.
    NodeArgToConsumerMap non_cast_producers_map;
    for (auto iter = inputs.begin(); iter != inputs.end() && (level >= 2 || all_float_inputs_have_casts); ++iter) {
      NodeArg* input = *iter;
      if (!IsType(*input, TensorProto::FLOAT) || !IsRelevantInput(node, input)) {
        continue;
      }
      has_float_inputs = true;
      Node* producer = graph.GetMutableProducerNode(input->Name());
      if (nullptr != producer) {
        if (!Contains(removed_nodes, producer->Index())) {
          if (IsCastTo(producer, TensorProto::FLOAT) &&
              producer->GetOutputEdgesCount() == 1 &&
              !graph.IsOutput(input)) {
            casts.push_back(producer);
            require_type_change.insert(input);
          } else {
            non_cast_producers_map[input].push_back(node);
            all_float_inputs_have_casts = false;
          }
        } else {
          // Ignore removed nodes.
        }
      } else if (graph_utils::IsGraphInput(graph, input)) {
        non_cast_producers_map[input].push_back(node);
        all_float_inputs_have_casts = false;
      }
    }
    if (has_float_inputs && (level >= 2 || all_float_inputs_have_casts) && casts.size() > 1) {
      if (!non_cast_producers_map.empty()) {
        ORT_THROW_IF_ERROR(InsertCastNodes(graph, non_cast_producers_map, true, removed_nodes, inserted_nodes));
        LOGS(logger, VERBOSE) << "PropagateFP32CastsFromInputsToOutputs: Inserted FP16 Cast node to "
                              << ConcatNames(non_cast_producers_map, GetName);
      }
      for (Node* cast : casts) {
        ORT_THROW_IF_ERROR(RemoveCastNodesChain(graph, AsSpan({cast}), removed_nodes));
      }
      LOGS(logger, VERBOSE) << "PropagateFP32CastsFromInputsToOutputs: Removed Cast nodes "
                            << ConcatNames(casts)
                            << " feeding the same compute node " << node->Name();
      NodeArgToConsumerMap node_args_map;
      for (NodeArg* output : node->MutableOutputDefs()) {
        if (output->Exists() && IsRelevantOutput(node, output) && IsType(*output, TensorProto::FLOAT)) {
          auto mutable_consumer_nodes = graph.GetMutableConsumerNodes(output->Name());
          node_args_map.emplace(output, InlinedVector<Node*>(mutable_consumer_nodes.cbegin(), mutable_consumer_nodes.cend()));
        }
      }
      ORT_THROW_IF_ERROR(InsertCastNodes(graph, node_args_map, false, removed_nodes, inserted_nodes));
      LOGS(logger, VERBOSE) << "PropagateFP32CastsFromInputsToOutputs: Inserted FP32 Cast node to "
                            << ConcatNames(node_args_map, GetName);
      ChangeTypeToFP16(graph, require_type_change, true, converted_nodes, inserted_nodes, logger);
      modified = true;
    }
  }
  return modified;
}
/*
 * PropagateFP16CastsFromOutputsToInputs
 * This non-recursive fusion, checks whether the given node is fp16 safe op and
 * whether all floating point outputs are cast to fp16
 * and propagates cast op to the floating point inputs.
 * Convert the following graph
 *
 *        Input0/NodeArg  Input1/NodeArg
 *               |             |
 *             __V______________V___
 *            |       Opcode        |
 *            |(operation performed |
 *            |    in float32)      |
 *            |_____________________|
 *               |             |
 *          _____V____    _____V______
 *          |Cast FP16|   | Cast FP16|
 *          |_________|   |__________|
 *               |              |
 *               |              |
 *               V              V
 *
 *  and produce the following output
 *
 *        Input0/NodeArg   Input1/NodeArg
 *               |               |
 *          _____V____      _____V______
 *          |Cast FP16|     | Cast FP16|
 *          |_________|     |__________|
 *                |              |
 *              __V______________V___
 *             |       Opcode        |
 *             |(operation performed |
 *             |    in float16)      |
 *             |_____________________|
 *                 |             |
 *                 V             V
 */
static bool PropagateFP16CastsFromOutputsToInputs(Graph& graph, Node* node,
                                                  NodeIndices& removed_nodes,
                                                  size_t level,
                                                  const FP16AllowOps& fp16_allow_ops,
                                                  NodeIndices& converted_nodes,
                                                  NodeIndices& inserted_nodes,
                                                  const logging::Logger& logger) {
  bool modified = false;
  if (IsFP16Allow(node, level, fp16_allow_ops)) {
    bool has_float_outputs = false;
    bool all_float_outputs_have_casts = true;
    InlinedVector<Node*> casts;  // Cast nodes to propagate.
    auto& outputs = node->MutableOutputDefs();
    InlinedHashSet<NodeArg*> require_type_change;
    NodeArgToConsumerMap non_cast_consumers_map;
    for (auto iter = outputs.begin(); iter != outputs.end() && (level >= 2 || all_float_outputs_have_casts); ++iter) {
      NodeArg* output = *iter;
      if (!IsType(*output, TensorProto::FLOAT) || !IsRelevantOutput(node, output)) {
        continue;
      }
      has_float_outputs = true;
      auto consumers = graph.GetMutableConsumerNodes(output->Name());
      for (auto node_iter = consumers.begin(); node_iter != consumers.end() && (level >= 2 || all_float_outputs_have_casts); ++node_iter) {
        Node* consumer = *node_iter;
        if (nullptr != consumer &&
            !Contains(removed_nodes, consumer->Index())) {
          if (IsCastTo(consumer, TensorProto::FLOAT16)) {
            casts.push_back(consumer);
          } else {
            non_cast_consumers_map[output].push_back(consumer);
            all_float_outputs_have_casts = false;
          }
        }
      }
      if (graph.IsOutput(output)) {
        non_cast_consumers_map.emplace(output, InlinedVector<Node*>());
      }
      if (non_cast_consumers_map.empty()) {
        require_type_change.insert(output);
      }
    }
    if (has_float_outputs && (level >= 2 || all_float_outputs_have_casts) && casts.size() > 1) {
      if (!non_cast_consumers_map.empty()) {
        ORT_THROW_IF_ERROR(InsertCastNodes(graph, non_cast_consumers_map, false, removed_nodes, inserted_nodes));
        LOGS(logger, VERBOSE) << "PropagateFP16CastsFromOutputsToInputs: Inserted FP32 Cast node to "
                              << ConcatNames(non_cast_consumers_map, GetName);
      }
      for (Node* cast : casts) {
        ORT_THROW_IF_ERROR(RemoveCastNodesChain(graph, AsSpan({cast}), removed_nodes));
      }
      LOGS(logger, VERBOSE) << "PropagateFP16CastsFromOutputsToInputs: Removed Cast nodes "
                            << ConcatNames(casts)
                            << " feeding from the same compute node " << node->Name();
      NodeArgToConsumerMap node_args_map;
      for (NodeArg* input : node->MutableInputDefs()) {
        if (IsRelevantInput(node, input) && IsType(*input, TensorProto::FLOAT)) {
          node_args_map.emplace(input, InlinedVector<Node*>{node});
        }
      }
      ORT_THROW_IF_ERROR(InsertCastNodes(graph, node_args_map, true, removed_nodes, inserted_nodes));
      LOGS(logger, VERBOSE) << "PropagateFP16CastsFromOutputsToInputs: Inserted FP16 Cast node to "
                            << ConcatNames(node_args_map, GetName);
      ChangeTypeToFP16(graph, require_type_change, false, converted_nodes, inserted_nodes, logger);
      modified = true;
    }
  }
  return modified;
}

/*
 *  CreateCast
 *  Create a cast node based on the node_arg for the given data type. If the node_arg is a graph output is_graph_outut is set.
 *  If the node_arg is not a graph output then the node_arg is the input of the new cast node. Otherwise the node_arg is the output
 *  of the new cast node. This function is used by InsertFP16Cast or InsertFP32Casts.
 */
static Node& CreateCast(Graph& graph, NodeArg* node_arg, TensorProto_DataType data_type,
                        NodeIndices& inserted_nodes, bool is_graph_output = false) {
  TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(is_graph_output ? (data_type == TensorProto::FLOAT ? TensorProto::FLOAT16 : TensorProto::FLOAT) : data_type);
  NodeArg& new_node_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node_arg->Name()), &type_proto);

  NodeArg& cast_input = is_graph_output ? new_node_arg : *node_arg;
  NodeArg& cast_output = is_graph_output ? *node_arg : new_node_arg;
  const std::array inputs = {&cast_input};
  const std::array outputs = {&cast_output};
  Node& node = graph.AddNode(graph.GenerateNodeName(node_arg->Name() + "_cast"),
                             "Cast",
                             "Created a new Cast node",
                             inputs,
                             outputs);
  node.AddAttribute("to", static_cast<int64_t>(data_type));
  inserted_nodes.insert(node.Index());
  return node;
}
/* InsertFP16Cast
 * Insert a new cast input for the given float input for the give node. This function should be used with InsertFP32Casts
 * in order to compute FP16 allowed operations in 16 bit precision instead of 32 bit precision.
 *
 *        Input0/NodeArg  Input1/NodeArg
 *               |              |
 *             __V______________V___
 *            |       Opcode        |
 *            |(operation performed |
 *            |    in float32)      |
 *            |_____________________|
 *               |             |
 *               V             V
 *
 * Is eventually translated to the following, after all inputs are casted
 *
 *        Input0/NodeArg   Input1/NodeArg
 *               |               |
 *          _____V____      _____V______
 *          |Cast FP16|     | Cast FP16|
 *          |_________|     |__________|
 *                |              |
 *              __V______________V___
 *             |       Opcode        |
 *             |(operation performed |
 *             |    in float16)      |
 *             |_____________________|
 *                 |             |
 *                 V             V
 *
 */
static void InsertFP16Cast(Graph& graph, NodeArg* input_arg, Node* node,
                           NodeIndices& inserted_nodes, const logging::Logger& logger) {
  Node& cast = CreateCast(graph, input_arg, TensorProto::FLOAT16, inserted_nodes);
  NodeArg* new_input_arg = cast.MutableOutputDefs()[0];
  auto& inputs = node->MutableInputDefs();
  Node* producer = graph.GetMutableProducerNode(input_arg->Name());
  if (nullptr != producer) {
    int output_index = optimizer_utils::IndexOfNodeOutput(*producer, *input_arg);
    int input_index = optimizer_utils::IndexOfNodeInput(*node, *input_arg);
    graph.RemoveEdge(producer->Index(), node->Index(), output_index, input_index);
    std::replace(inputs.begin(), inputs.end(), input_arg, new_input_arg);
    graph.AddEdge(cast.Index(), node->Index(), 0, input_index);
  } else {
    std::replace(inputs.begin(), inputs.end(), input_arg, new_input_arg);
  }
  graph.RemoveConsumerNode(input_arg->Name(), node);
  graph.AddConsumerNode(input_arg->Name(), &cast);
  graph.UpdateConsumerNodes(new_input_arg->Name(), {node});
  LOGS(logger, VERBOSE) << "Inserted FP16 Cast " << cast.Name() << " for the node arg " << input_arg->Name() << " feeding " << node->Name();
}

/* InsertFP32Casts
 *  Insert float casts on the given output for each consumer.  This function should be used with InsertFP16Casts
 * in order to compute FP16 allowed operations in 16 bit precision instead of 32 bit precision.
 *
 *               |               |
 *             __V_______________V___
 *            |       Opcode        |
 *            |(operation performed |
 *            |    in float16)      |
 *            |_____________________|
 *               _______|_______
 *               |             |
 *          _____V____    _____V______
 *          |Consumer0|   |Consumer1 |
 *          |_________|   |__________|
 *               |              |
 *
 *  will be translated to the following
 *
 *               |               |
 *             __V_______________V___
 *            |       Opcode        |
 *            |(operation performed |
 *            |    in float16)      |
 *            |_____________________|
 *               _______|_______
 *               |             |
 *          _____V____    _____V______
 *          |Cast FP32|   | Cast FP32|
 *          |_________|   |__________|
 *               |             |
 *          _____V____    _____V______
 *          |Consumer0|   |Consumer1 |
 *          |_________|   |__________|
 *               |              |
 */
static void InsertFP32Casts(Graph& graph, NodeArg* output_arg,
                            NodeIndices& inserted_nodes, const logging::Logger& logger) {
  NodeArg* orig_output_arg = output_arg;
  Node* producer = graph.GetMutableProducerNode(output_arg->Name());
  auto consumers = graph.GetMutableConsumerNodes(output_arg->Name());
  // First remove all the edges from the producer to the consumers.
  InlinedVector<Node*> new_consumers;
  for (Node* consumer : consumers) {
    int output_index = optimizer_utils::IndexOfNodeOutput(*producer, *output_arg);
    int input_index = optimizer_utils::IndexOfNodeInput(*consumer, *output_arg);
    graph.RemoveEdge(producer->Index(), consumer->Index(), output_index, input_index);
  }
  // Create a new output_arg on the producer if the output_arg is a graph output
  if (graph.IsOutput(output_arg)) {
    Node& cast = CreateCast(graph, output_arg, TensorProto::FLOAT, inserted_nodes, true);
    graph.UpdateProducerNode(output_arg->Name(), cast.Index());
    NodeArg* new_output_arg = cast.MutableInputDefs()[0];
    graph.UpdateProducerNode(new_output_arg->Name(), producer->Index());
    new_consumers.push_back(&cast);
    auto& outputs = producer->MutableOutputDefs();
    std::replace(outputs.begin(), outputs.end(), output_arg, new_output_arg);
    output_arg = new_output_arg;
    LOGS(logger, VERBOSE) << "Inserted FP32 Cast " << cast.Name() << " for the node arg " << output_arg->Name();
  } else {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT16);
    ORT_THROW_IF_ERROR(output_arg->UpdateTypeAndShape(type_proto, true, true, logger));
  }
  // Create a new cast node for each consumer
  for (Node* consumer : consumers) {
    Node& cast = CreateCast(graph, output_arg, TensorProto::FLOAT, inserted_nodes);
    NodeArg* new_output_arg = cast.MutableOutputDefs()[0];
    auto& inputs = consumer->MutableInputDefs();
    int output_index = optimizer_utils::IndexOfNodeOutput(cast, *new_output_arg);
    std::replace(inputs.begin(), inputs.end(), orig_output_arg, new_output_arg);
    int input_index = optimizer_utils::IndexOfNodeInput(*consumer, *new_output_arg);
    graph.AddEdge(cast.Index(), consumer->Index(), output_index, input_index);
    new_consumers.push_back(&cast);
    graph.UpdateConsumerNodes(new_output_arg->Name(), {consumer});
    LOGS(logger, VERBOSE) << "Inserted FP32 Cast " << cast.Name() << " for the node arg " << output_arg->Name() << " feeding " << consumer->Name();
  }
  // Update the consumers of the original output_arg
  graph.UpdateConsumerNodes(output_arg->Name(), new_consumers);
}
/*
 *  Expand FP16 compute regions on the graph by example float16 compute nodes,
 *  propagating float32 Cast operation down the graph and propagating float16
 *  Cast operations up the graph. The following functions are performed
 *  1. Fuse subgraphs
 *  2. Propagate fp32 casts forwards
 *  3. Propagate fp16 casts back
 *  4. Insert casts before and after allowed operations (InsertAndReduce strategy)
 *  5. Remove back to back casts
 *  6. Remove redundant casts
 *  7. Move FP32 casts from inputs to outputs
 *  8. Move FP16 casts from outputs to inputs
 */
Status PropagateCastOps::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  bool local_modified = false;
  NodeIndices inserted_nodes;   // Names of the nodes inserted
  NodeIndices converted_nodes;  // Names of the nodes converted to FP16
  // First apply the transformation to the subgraphs.
  {
    GraphViewer graph_viewer(graph);
    const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

    for (auto node_index : node_topology_list) {
      auto* node_ptr = graph.GetNode(node_index);
      if (nullptr == node_ptr)
        continue;  // node was removed

      auto& node = *node_ptr;

      ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
      if (strategy_ == GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::InsertAndReduce) {
        // Using InsertFP16Cast and InsertFP32Casts insert float16 casts on all inputs and float casts on all outputs.
        // Each consumer of each output gets a separate float cast inserted. Doing so will convert the computation of
        // current node from 32 bit float to 16 bit float operation. These cast operations will be eventually reduced.
        if (IsFP16Allow(node_ptr, level_, fp16_allow_ops_0_)) {
          // Insert FP16 Cast on all float inputs
          converted_nodes.insert(node.Index());
          for (NodeArg* input_arg : node.MutableInputDefs()) {
            if (IsRelevantInput(&node, input_arg) && IsType(*input_arg, TensorProto::FLOAT)) {
              InsertFP16Cast(graph, input_arg, node_ptr, inserted_nodes, logger);
              local_modified = true;
            }
          }
          // Convert all output args to FP16 and insert FP32 cast for all consumers
          for (NodeArg* output_arg : node.MutableOutputDefs()) {
            if (IsRelevantOutput(&node, output_arg) && IsType(*output_arg, TensorProto::FLOAT)) {
              InsertFP32Casts(graph, output_arg, inserted_nodes, logger);
              local_modified = true;
            }
          }
        }
      }
    }
  }
  InlinedVector<std::string> removed_node_names;
  int pass = 0;
  do {
    LOGS(logger, VERBOSE) << "Propagate Cast Operations Pass " << pass << ":";
    NodeIndices removed_nodes;

    if (local_modified) {
      ORT_RETURN_IF_ERROR(graph.Resolve());
      local_modified = false;
    }

    GraphViewer graph_viewer(graph);
    const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

    // Remove unnecessary casts
    for (auto node_index : node_topology_list) {
      Node* node = graph.GetNode(node_index);
      if (nullptr != node &&
          !Contains(removed_nodes, node->Index()) &&
          node->OpType() == "Cast") {
        local_modified |= RemoveUnnecessaryCasts(graph, node, removed_nodes, logger);
      }
    }

    // Fuse subgraphs, sibling Cast nodes with same input
    for (auto node_index : node_topology_list) {
      Node* node = graph.GetNode(node_index);
      if (nullptr != node &&
          !Contains(removed_nodes, node->Index())) {
        local_modified |= FuseSiblingCasts(graph, node, removed_nodes, inserted_nodes, logger);
      }
    }
    for (const NodeArg* node_arg : graph.GetInputs()) {
      local_modified |= FuseSiblingCasts(graph, node_arg, removed_nodes, inserted_nodes, logger);
    }

    // Remove back to back Casts, with FLOAT->FLOAT16 followed by FLOAT16->FLOAT, but not the other way.
    for (auto node_index : node_topology_list) {
      Node* node = graph.GetNode(node_index);
      if (nullptr != node &&
          !Contains(removed_nodes, node->Index()) &&
          IsCastTo(node, TensorProto::FLOAT)) {
        local_modified |= RemoveBackToBackCasts(graph, node, removed_nodes, logger);
      }
    }

    if ((strategy_ & GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::FloodFill) !=
        GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::None) {
      // Propagate FP16 Casts from outputs to inputs
      for (auto node_index : node_topology_list) {
        Node* node = graph.GetNode(node_index);
        if (nullptr != node &&
            !Contains(removed_nodes, node->Index())) {
          local_modified |= PropagateFP16CastsFromOutputsToInputs(graph, node, removed_nodes, level_, fp16_allow_ops_0_, converted_nodes, inserted_nodes, logger);
        }
      }

      // Propagate FP32 Casts from inputs to outputs
      for (auto node_index : node_topology_list) {
        Node* node = graph.GetNode(node_index);
        if (nullptr != node &&
            !Contains(removed_nodes, node->Index())) {
          local_modified |= PropagateFP32CastsFromInputsToOutputs(graph, node, removed_nodes, level_, fp16_allow_ops_0_, converted_nodes, inserted_nodes, logger);
        }
      }

      // Propagate FP32 Casts forward
      for (auto node_index : node_topology_list) {
        Node* node = graph.GetNode(node_index);
        if (nullptr != node &&
            !Contains(removed_nodes, node->Index()) &&
            IsCastTo(node, TensorProto::FLOAT)) {
          local_modified |= PropagateForwards(graph, node, removed_nodes, level_, fp16_allow_ops_0_, converted_nodes, inserted_nodes, logger);
        }
      }

      // Propagate FP16 Casts backward
      for (auto node_index : node_topology_list) {
        Node* node = graph.GetNode(node_index);
        if (nullptr != node &&
            !Contains(removed_nodes, node->Index()) &&
            IsCastTo(node, TensorProto::FLOAT16)) {
          local_modified |= PropagateBackwards(graph, node, removed_nodes, level_, fp16_allow_ops_0_, converted_nodes, inserted_nodes, logger);
        }
      }
    }
    // In order to generate summary collect only removed node names found in the input graph
    // and remove all removed nodes from inserted_nodes and converted nodes collections.
    for (NodeIndex removed_node : removed_nodes) {
      auto it = inserted_nodes.find(removed_node);
      if (it == inserted_nodes.end()) {
        removed_node_names.push_back(graph.GetNode(removed_node)->Name());
      } else {
        inserted_nodes.erase(it);
      }
      converted_nodes.erase(removed_node);
      graph.RemoveNode(removed_node);
    }
    modified |= local_modified;
    pass++;
  } while (local_modified);

  // Generate summary if the graph is modified
  if (modified) {
    LOGS(logger, INFO) << "Propagate Cast operations summary:";
    LOGS(logger, INFO) << "Number of passes = " << pass;
    LOGS(logger, INFO) << "Nodes Inserted:";
    std::for_each(inserted_nodes.begin(), inserted_nodes.end(), [&](NodeIndex idx) { LOGS(logger, INFO) << graph.GetNode(idx)->Name(); });

    LOGS(logger, INFO) << "Nodes Removed:";
    std::for_each(removed_node_names.begin(), removed_node_names.end(), [&](const std::string& name) { LOGS(logger, INFO) << name; });

    LOGS(logger, INFO) << "Nodes Converted to FP16:";
    std::for_each(converted_nodes.begin(), converted_nodes.end(), [&](NodeIndex idx) { LOGS(logger, INFO) << graph.GetNode(idx)->Name(); });
  }
  return Status::OK();
}

PropagateCastOps::PropagateCastOps(GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy strategy,
                                   size_t level, gsl::span<const std::string> allow_list,
                                   const InlinedHashSet<std::string_view>& compatible_execution_providers)
    : GraphTransformer("PropagateCastOps", compatible_execution_providers), level_(level), fp16_allow_ops_0_(), strategy_(strategy) {
  if (!allow_list.empty()) {
    fp16_allow_ops_0_.reserve(allow_list.size());
    fp16_allow_ops_0_.insert(allow_list.begin(), allow_list.end());
  }
}

}  // namespace onnxruntime
