// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/propagate_cast_ops.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
/*********************************************************************************************************************************
* PropagateCastOps transformation tries to exapand float16 computation regions in the graph by converting float operations to float16.
* In order to perform the the transformation, certain operations are considered FP16 safe, i.e. the computation may be performed
* without effecting the numerical result, ex. transpose, shape, etc. The transformation supports three levels of optimization, 0, 1
* and 2. Level 2 being the most agressive, may consider moving float operations to float16 which may result in different numerical results 
* due to loss of precision. The user may choose level 0, whereby the user choosed the opcodes which are "FP16 Safe" instead of a list
* predetermined opcodes as in levels 1 and 2.
* Currently two strategies are available, InsertAndReduce and FloodFill.
* InsertAndReduce :
* The transformation first 
*   1. Inserts float16 cast operation on all the float inputs
*   2. Change all float outputs to float16
*   3. Insert float cast operations on all float outputs as expected
*   After inserting the FP16 and FP32 cast operation nodes around all the nodes with FP16 Safe opcodes, the transformation reduces the
*   cast operations, using the following operations on the graph, iteratively until no more reduction is possible.
*   1. Remove back-to-back casts
*   2. Remove redundant casts
*   3. Fuse sibling subgraphs with same cast operation
* FloodFill:
    Operations are converted from float to float16 by propagating float16 cast operation up the graph or float cast perations down the
*   graph. Using this strategy, for eatch pre-existing float/float16 cast operations the transformation first finds the possible expansion of 
*   float16 region up/down the graph using DFS/ReverseDFS and (TODO) identifies loss/gain by performing such expansion, considering 
*   the gain by reducing float operations lower precision and loss due to newly inserted cast operations (TODO).
*   In addition to propagating cast operations up/down the graph, in this strategy, the above mentioned cast op reduction functions
*   are also used.
*********************************************************************************************************************************/
namespace onnxruntime {
// NodeArg to Select consumer node map.
typedef std::unordered_map<NodeArg*, std::vector<Node*>> NodeArgToConsumerMap;

// ConcatNames
// Collects all the names from the pointers of the objects stores in the container class C
// the class should have a member functions returning a string (or a ref).
template <typename C, typename T = typename C::value_type>
static std::string ConcatNames(
    C const& items, std::string (*f)(const T& n) = [](const T& n) { return n->Name(); }) {
  std::vector<std::string> names;
  std::transform(items.begin(), items.end(), back_inserter(names), f);
  return std::accumulate(names.begin(), names.end(), std::string(), [](const std::string& a, const std::string& b) { return a + ", " + b; });
}

// GetName
// Collect the node-arg names and the correspnding consumers name for reporting
static std::string GetName(const std::pair<const NodeArg*, std::vector<Node*>>& p) {
  return p.first->Name() + " feeding " + ConcatNames(p.second) + "; ";
};

// The collection fp16_allow_ops, specifies for a given propagate_cast_ops level, a vector of node op_types that
// the code is allowed to propage Cast operations across. The user may specify a custom list of optypes using level 0.
// The opcodes are split into multiple levels. Cast propagation is done based on the level. Level 2 op code
// list includes Level 1 list also.
static std::vector<std::unordered_set<std::string>> fp16_allow_ops = {
    /* Level 0 */ {},
    /* Level 1 */ {"Transpose", "Relu", "Reshape", "Split", "Tanh"},
    /* Level 2 */ {"BiasGelu", "Dropout", "FastGelu", "Gather", "Gelu", "LayerNormalization", "Where"}};
static std::unordered_set<std::string> inserted_node_names;   // Names of the nodes inserted
static std::unordered_set<std::string> converted_node_names;  // Names of the nodes converted to FP16

// Check whether the given opcode is fp16 allowed for the given level of optimization.
static bool IsFP16Allow(const std::string& op_type, size_t level) {
  bool fp16_allow = false;
  for (size_t i = 0; i <= level && i < fp16_allow_ops.size() && !fp16_allow; ++i) {
    fp16_allow = std::find(fp16_allow_ops[i].begin(), fp16_allow_ops[i].end(), op_type) != fp16_allow_ops[i].end();
  }
  return fp16_allow;
}

// Check whether the node is cast operation to the specified data type
static bool IsCastTo(const Node* node, TensorProto_DataType data_type) {
  if (node->OpType() == "Cast") {
    const NodeAttributes& attributes = node->GetAttributes();
    ORT_ENFORCE(attributes.find("to") != attributes.end());
    return attributes.at("to").i() == static_cast<int64_t>(data_type);
  }
  return false;
}

// Check whether the node-arg element type is same the specified data type
static bool IsType(const NodeArg& node_arg, TensorProto_DataType data_type) {
  return node_arg.TypeAsProto()->tensor_type().elem_type() == data_type;
}

// InsertCastNodes
// Insert a new Cast node after each NodeArg in the require_cast map, feeding the nodes (consumer) in the vector mapped to
// the NodeArg. The other consumers of the NodeArg will not be changed. The cast node is FLOAT16 if is_fp16 is True
// and FLOAT otherwise. This funtion fixes the graph edges in addition to inserting the cast nodes.
static Status InsertCastNodes(Graph& graph,
                              const NodeArgToConsumerMap& require_cast,
                              bool is_fp16,
                              std::deque<NodeIndex>& removed_nodes) {
  //Create requirred new Cast nodes.
  for (std::pair<NodeArg*, std::vector<Node*>> element : require_cast) {
    NodeArg* node_arg = element.first;
    std::vector<Node*> nodes = element.second;
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
    type_proto.mutable_tensor_type()->set_elem_type(new_node_arg_data_type);
    NodeArg& new_node_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node_arg->Name()), &type_proto);
    // Expect that a NodeArg is not both a graph input as well as a graph output
    ORT_ENFORCE(!(graph.IsInputsIncludingInitializers(node_arg) && graph.IsOutput(node_arg)));
    NodeArg& cast_input = !is_node_arg_cast_output ? *node_arg : new_node_arg;
    NodeArg& cast_output = is_node_arg_cast_output ? *node_arg : new_node_arg;
    const std::vector<NodeArg*> cast_inputs = {&cast_input};
    const std::vector<NodeArg*> cast_outputs = {&cast_output};
    ONNX_NAMESPACE::AttributeProto to_attribute;
    to_attribute.set_name("to");
    to_attribute.set_type(ONNX_NAMESPACE::AttributeProto::INT);
    to_attribute.set_i(static_cast<int64_t>(data_type));
    NodeAttributes attributes({{"to", to_attribute}});

    Node& cast = graph.AddNode(graph.GenerateNodeName(node_arg->Name() + "_cast"),
                               "Cast",
                               "Created a new Cast node",
                               cast_inputs,
                               cast_outputs,
                               &attributes);
    inserted_node_names.insert(cast.Name());
    Node* producer = graph.GetMutableProducerNode(node_arg->Name());
    std::vector<Node*> consumers = graph.GetMutableConsumerNodes(node_arg->Name());
    std::vector<Node*> other_nodes;
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
      if (std::find(removed_nodes.begin(), removed_nodes.end(), consumer->Index()) == removed_nodes.end()) {
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
*                         |  Cast    |
*                         |__________|
*                              |
*                         _____V______
*                         |  Opcode2 |
*                         |__________|
*                              |
*                              V
*/

static Status RemoveCastNodesChain(Graph& graph, std::vector<Node*> casts, std::deque<NodeIndex>& removed_nodes) {
  ORT_ENFORCE(casts.size() > 0);
  Node* lead_cast = casts.front();
  Node* trail_cast = casts.back();
  NodeArg* cast_input = lead_cast->MutableInputDefs()[0];
  NodeArg* cast_output = trail_cast->MutableOutputDefs()[0];
  // Update producer node
  Node* producer = graph.GetMutableProducerNode(cast_input->Name());
  auto consumers = graph.GetMutableConsumerNodes(cast_output->Name());
  int output_index = (nullptr != producer) ? optimizer_utils::IndexOfNodeOutput(*producer, *cast_input) : -1;
  if (producer) {
    if (graph.IsOutput(cast_output)) {
      // cast_output is a graph output. Replace the cast node with an Identity operator unless node
      // has no other outputs.
      if (producer->GetOutputEdgesCount() == 1) {
        int input_index = optimizer_utils::IndexOfNodeInput(*lead_cast, *cast_input);
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
  if (consumers.size() > 0) {
    int cast_output_index = optimizer_utils::IndexOfNodeOutput(*trail_cast, *cast_output);
    for (Node* consumer : consumers) {
      if (nullptr != consumer &&
          std::find(removed_nodes.begin(), removed_nodes.end(), consumer->Index()) == removed_nodes.end()) {
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
    removed_nodes.push_front(cast->Index());
  }
  return Status::OK();
}

/*
* RemoveBackToBackCasts
* Remove FLOAT and FLOAT16 casts back-to-back, only if the parent cast is from FLOAT16 to FLOAT
* and the child cast is from FLOAT to FLOAT16 or from FLOAT to FLOAT.
* The tirivial case is parent cast has only one output. A non-trivial case handled by this function
* is, parent has multiple children and one or more child node output is also a graph output.
* In the non-trivial case, when a child-cast nullfies the parent cast, the consumer nodes are moved
* from the child-cast to the producer of the parent-cast input. The code handles cornercases suchas
* the child-cast output is also a graph output, in addition to possibly other nodes, and the
* parent-cast has other output nodes. The first check CanRemoveNode rules-out the possibility of
* parent-cast output also being the graph output.
* The inputs is Cast to FLOAT16
* With the possiblility of child/parent cast feeding other nodes/graph-output, the transformation is either
*                         _____|______
*                         | Opcode1  |
*                         |__________|
*                              |
*                         _____V______                         ____________
*                         | Cast FP16|                         | Opcode 1 |  
*                         |__________|                         |__________|
*                              |                                    |
*                              |                                    |
*                              |                  ---\         _____V______
*                              |                  ---/         | Opcode2  |
*                         _____V______                         |__________|
*                         | Cast FP32|
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
*                         | Cast FP32|                         | Opcode 1 |  
*                         |__________|                         |__________|
*                              |                                    |
*                              |                                    |
*                              |                  ---\         _____V______
*                              |                  ---/         | Cast FP32|
*                         _____V______                         |__________|
*                         | Cast FP32|                              |
*                         |__________|                         _____V______
*                              |                               | Opcode2  |
*                         _____V______                         |___________|
*                         |  Opcode2 |
*                         |__________|

*/
static bool RemoveBackToBackCasts(Graph& graph, Node* parent,
                                  std::deque<NodeIndex>& removed_nodes,
                                  const logging::Logger& logger) {
  ORT_ENFORCE(IsCastTo(parent, TensorProto::FLOAT));
  bool modified = false;
  if (graph_utils::CanRemoveNode(graph, *parent, logger)) {
    NodeArg* cast_output = parent->MutableOutputDefs()[0];
    std::vector<Node*> children = graph.GetMutableConsumerNodes(cast_output->Name());
    if (children.size() == 1) {
      Node* child = children[0];
      if (std::find(removed_nodes.begin(), removed_nodes.end(), child->Index()) == removed_nodes.end()) {
        if (IsCastTo(child, TensorProto::FLOAT16)) {
          // The parent and child cancell out
          LOGS(logger, VERBOSE) << "RemoveBackToBackCasts: Removed Cast nodes  " << parent->Name() << " and " << child->Name();
          RemoveCastNodesChain(graph, {parent, child}, removed_nodes);
          modified = true;
        } else if (IsCastTo(child, TensorProto::FLOAT)) {
          // Child is a duplicate of parent
          LOGS(logger, VERBOSE) << "RemoveBackToBackCasts: Removed Cast node  " << child->Name();
          RemoveCastNodesChain(graph, {child}, removed_nodes);
          modified = true;
        }
      }
    } else {
      NodeArg* parent_input = parent->MutableInputDefs()[0];
      const Node* producer = graph.GetProducerNode(parent_input->Name());
      int producer_output_index = producer ? optimizer_utils::IndexOfNodeOutput(*producer, *parent_input) : -1;
      std::vector<Node*> new_consumers;
      size_t children_count = children.size();
      for (Node* child : children) {
        if (std::find(removed_nodes.begin(), removed_nodes.end(), child->Index()) == removed_nodes.end()) {
          if (IsCastTo(child, TensorProto::FLOAT16)) {
            // The parent and child cancell out
            // Remove the child node without effecting the other nodes.
            // move all the consumers to the producer.
            LOGS(logger, VERBOSE) << "RemoveBackToBackCasts: Removed Cast node  " << child->Name();
            NodeArg* child_output = child->MutableOutputDefs()[0];
            for (Node* consumer : graph.GetMutableConsumerNodes(child_output->Name())) {
              std::vector<NodeArg*>& consumer_inputs = consumer->MutableInputDefs();
              int output_index = optimizer_utils::IndexOfNodeOutput(*child, *child_output);
              int input_index = optimizer_utils::IndexOfNodeInput(*consumer, *child_output);
              graph.RemoveEdge(child->Index(), consumer->Index(), output_index, input_index);
              std::replace(consumer_inputs.begin(), consumer_inputs.end(), child_output, parent_input);
              if (nullptr != producer) {
                graph.AddEdge(producer->Index(), consumer->Index(), producer_output_index, input_index);
              }
              new_consumers.push_back(consumer);
            }
            if (graph.IsOutput(child_output)) {
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
            removed_nodes.push_front(child->Index());
            children_count--;
          } else if (IsCastTo(child, TensorProto::FLOAT)) {
            // Child is a duplicate of parent
            LOGS(logger, VERBOSE) << "RemoveBackToBackCasts: Removed Cast node  " << child->Name();
            graph.RemoveEdge(parent->Index(), child->Index(), 0, 0);
            RemoveCastNodesChain(graph, {child}, removed_nodes);
            modified = true;
          }
        }
      }
      if (children_count == 0) {
        // No more children nodes exists, and the parent-cast output is not a graph output. Remove it!
        RemoveCastNodesChain(graph, {parent}, removed_nodes);
      }
      if (new_consumers.size() > 0) {
        std::vector<Node*> consumers = graph.GetMutableConsumerNodes(parent_input->Name());
        std::copy(new_consumers.begin(), new_consumers.end(), back_inserter(consumers));
        graph.UpdateConsumerNodes(parent_input->Name(), consumers);
      }
    }
  }
  return modified;
}

// SearchUpstream:
// ReverseDFS, traverse bottom-up, the graph upstream collecting all the NodeArgs that require a cast
// inorder to move an FP16 Cast operation up the graph.
// Visited float NodeArgs are either in require_cast or require_type_change so that the same
// nodearg is traversed not more than once.
// If the level is 2, the functions traverses up the graph identifying required FP32 casts even if 
// multiple consumers for the node outputs are found while level 0 or 1 quit traversing up.
static void SearchUpstream(Graph& graph, NodeArg* node_arg, Node* dst_node,
                           NodeArgToConsumerMap& require_cast,
                           NodeArgToConsumerMap& require_cast_fp32,
                           std::unordered_set<NodeArg*>& require_type_change,
                           std::deque<NodeIndex>& removed_nodes,
                           size_t level) {
  Node* node = graph.GetMutableProducerNode(node_arg->Name());
  // If the Cast input feeds more than one node or the cast node feeds a graph output and at least one
  // node then it cannot propagate.
  size_t consumer_node_count = graph.GetConsumerNodes(node_arg->Name()).size();
  // Do not traverse up the graph if the node produces a graph output as well as feeds other nodes
  // or if the node has more than one consumers.
  if (level < 2 && (consumer_node_count > 1 ||
                    nullptr != node &&
                        consumer_node_count > 0 &&
                        graph.IsOutput(node_arg))) {
    require_cast[node_arg].push_back(dst_node);
  } else if (node == nullptr) {
    // The graph inputs don't have the producer nodes
    if (IsType(*node_arg, TensorProto_DataType_FLOAT)) {
      require_cast[node_arg].push_back(dst_node);
    }
  } else if (std::find(removed_nodes.begin(), removed_nodes.end(), node->Index()) == removed_nodes.end()) {
    if (IsCastTo(node, TensorProto_DataType_FLOAT)) {
      // This Cast node and the Cast node that will be created later will cancel out
      require_cast[node_arg].push_back(dst_node);
    } else {
      std::string op_type = node->OpType();
      if (!IsFP16Allow(op_type, level)) {
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
            if (IsType(*output_def, TensorProto_DataType_FLOAT) && graph.GetConsumerNodes(output_def->Name()).size() > 0) {
              require_cast[node_arg].push_back(dst_node);
              return;
            }
          }
        }
        if (level >= 2) {
          for (Node* consumer : graph.GetMutableConsumerNodes(node_arg->Name())) {
            if (nullptr != consumer && consumer != dst_node && consumer->OpType() != "Cast" &&
                std::find(removed_nodes.begin(), removed_nodes.end(), consumer->Index()) == removed_nodes.end()) {
              require_cast_fp32[node_arg].push_back(consumer);
            }
          }
          if (graph.IsOutput(node_arg)) {
            require_cast_fp32[node_arg] = std::vector<Node*>();
          }
        }
        for (NodeArg* node_input : node->MutableInputDefs()) {
          if (IsType(*node_input, TensorProto_DataType_FLOAT) &&
              require_cast.find(node_input) == require_cast.end() &&
              require_type_change.find(node_input) == require_type_change.end()) {
            SearchUpstream(graph, node_input, node, require_cast, require_cast_fp32, require_type_change, removed_nodes, level);
            if (require_cast.find(node_input) == require_cast.end() &&
                require_cast_fp32.find(node_input) == require_cast_fp32.end()) {
              require_type_change.insert(node_input);
            }
          }
        }
      }
    }
  }
}

// SearchDownstream:
// Recursively DFS traverse the graph downstream collecting all the NodeArgs that require a cast
// inorder to remove an FP32 Cast operation up the graph. Also collect the NodeArgs that need to
// be converted from float to float16 along the way.
// The recursion only traverses an
// TODO same as SearchUpstream impelement level 2, travesing down the tree identifying FP32 casts 
// required instead of quitting, as in the case of level 0 or 1.
static void SearchDownstream(Graph& graph, NodeArg* node_arg,
                             NodeArgToConsumerMap& require_cast,
                             std::unordered_set<NodeArg*>& require_type_change,
                             std::deque<NodeIndex>& removed_nodes,
                             size_t level) {
  for (Node* node : graph.GetMutableConsumerNodes(node_arg->Name())) {
    if (node) {
      std::string op_type = node->OpType();
      if (IsCastTo(node, TensorProto_DataType_FLOAT)) {
        // This Cast node and the Cast node that will be created later will cancel out
        require_cast[node_arg].push_back(node);
      } else {
        if (!IsFP16Allow(op_type, level)) {
          if (node_arg->Exists() &&
              IsType(*node_arg, TensorProto_DataType_FLOAT)) {
            require_cast[node_arg].push_back(node);
          }
        } else {
          // If the node has other float32 inputs then stop the search
          for (const auto* input_def : node->InputDefs()) {
            // TODO: If the specified level of the optimization is greater than 1 then
            // convert initializers if any from float to float16.
            if (input_def != node_arg) {
              if (IsType(*input_def, TensorProto_DataType_FLOAT)) {
                require_cast[node_arg].push_back(node);
                return;
              }
            }
          }
          for (NodeArg* node_output : node->MutableOutputDefs()) {
            if (IsType(*node_output, TensorProto_DataType_FLOAT) &&
                require_cast.find(node_output) == require_cast.end() &&
                require_type_change.find(node_output) == require_type_change.end()) {
              SearchDownstream(graph, node_output, require_cast, require_type_change, removed_nodes, level);
              if (require_cast.find(node_output) == require_cast.end()) {
                require_type_change.insert(node_output);
              }
            }
          }
        }
      }
    }
  }
  if (graph.IsOutput(node_arg) && require_cast.find(node_arg) == require_cast.end()) {
    require_cast.insert(std::make_pair(node_arg, std::vector<Node*>()));
  }
}

// Change the elem_type of the given NodeArgs from FLOAT to FLOAT16.
static void ChangeTypeToFP16(Graph& graph, std::unordered_set<NodeArg*>& require_type_change, bool is_forward, const logging::Logger& logger) {
  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT16);
  for (NodeArg* node_arg : require_type_change) {
    if (IsType(*node_arg, TensorProto::FLOAT)) {
      node_arg->UpdateTypeAndShape(type_proto, true, true, logger);
      if (is_forward) {
        // Propagating forwards. Count consumers.
        for (const Node* node : graph.GetConsumerNodes(node_arg->Name())) {
          converted_node_names.insert(node->Name());
        }
      } else {
        // Propagating backwards. Count producers.
        const Node* producer = graph.GetProducerNode(node_arg->Name());
        if (nullptr != producer) {
          converted_node_names.insert(producer->Name());
        }
      }
    }
  }
}

// PropagateForwards
// Propagate FP32 Cast operations forwards (downstream)
// Using SearchDownStream search the graph for Cast FP16 safe/allowed operations to expand
// the float16 computation region.
// The required_cast vector is the collection of nodes that require float cast.
// All nodeargs on a path down to any of the
// frontier nodes require type change from FLOAT to FLOAT16.
// require_type_change consists of such nodes.  All the frontier nodes require fp32 cast
// The input node is expected to be non-nullptr
static bool PropagateForwards(Graph& graph, Node* node,
                              std::deque<NodeIndex>& removed_nodes,
                              size_t level,
                              const logging::Logger& logger) {
  ORT_ENFORCE(nullptr != node);
  bool modified = false;
  NodeArgToConsumerMap require_cast;
  std::unordered_set<NodeArg*> require_type_change;
  NodeArg* cast_output = node->MutableOutputDefs()[0];
  SearchDownstream(graph, cast_output, require_cast, require_type_change, removed_nodes, level);
  if (require_cast.size() > 0 && require_cast.find(cast_output) == require_cast.end()) {
    // Remove Cast operation
    LOGS(logger, VERBOSE) << "PropagateForwards: Removed Cast node  " << node->Name();
    RemoveCastNodesChain(graph, {node}, removed_nodes);
    InsertCastNodes(graph, require_cast, false, removed_nodes);
    ChangeTypeToFP16(graph, require_type_change, true, logger);
    LOGS(logger, VERBOSE) << "PropagateForwwards: Inserted Cast nodes "
                          << ConcatNames<NodeArgToConsumerMap>(require_cast, GetName);
    modified = true;
  }
  return modified;
}

// PropagateBackwards
// Propagate FP16 Cast operations backwards (upstream)
// Using SearchUpstream search the graph for Cast FP16 safe/allowed operations and expand
// float16 computation regsion and
// find the frontiers of the float16 computation region.
// The required_cast or require_cast_fp32 vector is a collection of
// FP16-cast-frontiers of the cast node. All node-args on the path from any of the
// frontier nodes to the cast node require type change from  FLOAT to FLOAT16.
// Each of the frontier nodes requires an fp16 cast or fp32 cast.
// The input node is expected be non-nullptr.
static bool PropagateBackwards(Graph& graph, Node* node,
                               std::deque<NodeIndex>& removed_nodes,
                               size_t level,
                               const logging::Logger& logger) {
  bool modified = false;
  ORT_ENFORCE(nullptr != node);
  NodeArgToConsumerMap require_cast;
  NodeArgToConsumerMap require_cast_fp32;
  NodeArg* cast_input = node->MutableInputDefs()[0];
  std::unordered_set<NodeArg*> require_type_change;
  SearchUpstream(graph, cast_input, node, require_cast, require_cast_fp32, require_type_change, removed_nodes, level);
  if (require_cast_fp32.empty()) {
    require_type_change.insert(cast_input);
  }
  // TODO need a huristic when to insert FP32 Cast
  if (require_cast.size() > 0 && require_cast.find(cast_input) == require_cast.end() /* && require_cast.size() >= require_cast_fp32.size() */) {
    // Remove Cast operation
    if (require_cast_fp32.size() > 0) {
      InsertCastNodes(graph, require_cast_fp32, false, removed_nodes);
      LOGS(logger, VERBOSE) << "PropagateBackwards: Inserted FP32 Cast nodes "
                            << ConcatNames<NodeArgToConsumerMap>(require_cast_fp32, GetName);
    }
    RemoveCastNodesChain(graph, {node}, removed_nodes);
    LOGS(logger, VERBOSE) << "PropagateBackwards: Removed Cast node  " << node->Name();
    InsertCastNodes(graph, require_cast, true, removed_nodes);
    LOGS(logger, VERBOSE) << "PropagateBackwards: Inserted Cast nodes "
                          << ConcatNames<NodeArgToConsumerMap>(require_cast, GetName);
    ChangeTypeToFP16(graph, require_type_change, false, logger);
    LOGS(logger, VERBOSE) << "PropagateBackwards: Changed the type from float to float16 : "
                          << ConcatNames<std::unordered_set<NodeArg*>>(require_type_change);
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
static void FuseNodes(Graph& graph, const NodeArg* input, std::vector<Node*> nodes,
                      std::deque<NodeIndex>& removed_nodes) {
  ORT_ENFORCE(nodes.size() > 0);
  Node* node = nodes[0];
  NodeArg* node_arg = node->MutableOutputDefs()[0];
  NodeArg& new_output = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(input->Name()), node_arg->TypeAsProto());
  Node& new_cast = graph.AddNode(graph.GenerateNodeName(node->Name() + "_replace"),
                                 node->OpType(),
                                 "Created to replace a node",
                                 {graph.GetNodeArg(input->Name())},
                                 {&new_output},
                                 &node->GetAttributes(),
                                 node->Domain());
  inserted_node_names.insert(new_cast.Name());
  std::vector<Node*> consumers;
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
    graph.RemoveConsumerNode(input->Name(), cast);
  }
  graph.AddConsumerNode(input->Name(), &new_cast);
  graph.UpdateConsumerNodes(new_output.Name(), consumers);
  for (Node* n : nodes) {
    removed_nodes.push_front(n->Index());
    graph_utils::RemoveNodeOutputEdges(graph, *n);
  }
}

// Traverse the graph recursively searching/collecting sibling Cast op nodes to fuse and call FuseNodes.
static bool FuseSiblingCasts(Graph& graph, const NodeArg* node_arg,
                             std::deque<NodeIndex>& removed_nodes,
                             const logging::Logger& logger) {
  bool modified = false;
  std::vector<Node*> cast_fp16_siblings;
  std::vector<Node*> cast_fp32_siblings;
  for (Node* node : graph.GetMutableConsumerNodes(node_arg->Name())) {
    // If a cast node feeds a graph output then it is not a candidate for fusion.
    if (nullptr == node || node->OpType() != "Cast" ||
        std::find(removed_nodes.begin(), removed_nodes.end(), node->Index()) != removed_nodes.end() ||
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
    FuseNodes(graph, node_arg, cast_fp16_siblings, removed_nodes);
    LOGS(logger, VERBOSE) << "FusedSubgraphs: Fused Cast nodes : " << ConcatNames<std::vector<Node*>>(cast_fp16_siblings);
  }
  if (cast_fp32_siblings.size() > 1) {
    modified = true;
    FuseNodes(graph, node_arg, cast_fp32_siblings, removed_nodes);
    LOGS(logger, VERBOSE) << "FusedSubgraphs: Fused Cast nodes : " << ConcatNames<std::vector<Node*>>(cast_fp32_siblings);
  }
  return modified;
}

// Overloaded function which goes through all the output args of a node and calls FuseSiblingCasts
static bool FuseSiblingCasts(Graph& graph, const Node* parent,
                             std::deque<NodeIndex>& removed_nodes,
                             const logging::Logger& logger) {
  bool modified = false;
  for (const NodeArg* output : parent->OutputDefs()) {
    modified |= FuseSiblingCasts(graph, output, removed_nodes, logger);
  }
  return modified;
}

// RemoveUnnecessaryCasts
// Remove a cast if the input elem_type is same the required cast type.
static bool RemoveUnnecessaryCasts(Graph& graph, Node* node,
                                   std::deque<NodeIndex>& removed_nodes,
                                   const logging::Logger& logger) {
  bool modified = false;
  if (node->InputDefs().size() == 1) {
    const NodeArg* node_arg = node->InputDefs()[0];
    auto elem_type = node_arg->TypeAsProto()->tensor_type().elem_type();
    TensorProto_DataType data_type = static_cast<TensorProto_DataType>(elem_type);
    if (IsCastTo(node, data_type)) {
      LOGS(logger, VERBOSE) << "Removed unnecessary cast " << node->Name();
      RemoveCastNodesChain(graph, {node}, removed_nodes);
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
                                                  std::deque<NodeIndex>& removed_nodes,
                                                  size_t level,
                                                  const logging::Logger& logger) {
  bool modified = false;
  if (IsFP16Allow(node->OpType(), level)) {
    bool has_float_inputs = false;
    bool all_float_inputs_have_casts = true;
    std::vector<Node*> casts;
    std::unordered_set<NodeArg*> require_type_change;
    // TODO Here we require the all floating point inputs are generated by an immediate
    // parent cast node and all casts feed only one output node.
    for (NodeArg* input : node->MutableInputDefs()) {
      if (!IsType(*input, TensorProto::FLOAT)) {
        continue;
      }
      has_float_inputs = true;
      Node* producer = graph.GetMutableProducerNode(input->Name());
      if (nullptr != producer &&
          std::find(removed_nodes.begin(), removed_nodes.end(), producer->Index()) == removed_nodes.end() &&
          IsCastTo(producer, TensorProto::FLOAT) &&
          producer->GetOutputEdgesCount() == 1 &&
          !graph.IsOutput(input)) {
        casts.push_back(producer);
        require_type_change.insert(input);
        continue;
      }
      all_float_inputs_have_casts = false;
      break;
    }
    if (has_float_inputs && all_float_inputs_have_casts && casts.size() > 1) {
      LOGS(logger, VERBOSE) << "PropagateFP32CastsFromInputsToOutputs: Removed Cast nodes "
                            << ConcatNames<std::vector<Node*>>(casts)
                            << " feeding the same compute node " << node->Name();
      for (Node* cast : casts) {
        RemoveCastNodesChain(graph, {cast}, removed_nodes);
      }
      NodeArgToConsumerMap node_args_map;
      for (NodeArg* output : node->MutableOutputDefs()) {
        if (output->Exists() && IsType(*output, TensorProto::FLOAT)) {
          node_args_map.insert(std::make_pair(output, graph.GetMutableConsumerNodes(output->Name())));
        }
      }
      InsertCastNodes(graph, node_args_map, false, removed_nodes);
      ChangeTypeToFP16(graph, require_type_change, true, logger);

      LOGS(logger, VERBOSE) << "PropagateFP32CastsFromInputsToOutputs: Inserted Cast node to "
                            << ConcatNames<NodeArgToConsumerMap>(node_args_map, GetName);
      modified = true;
    }
  }
  return modified;
}
/*
* PropagateFP16CastsFromOutputsToInputs
* This non-recursive fusion, checks whether the given node is fp16 safe op and
* whether all floatingpoint outputs are cast to fp16
* and propagates cast op to the floatingpoint inputs.
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
                                                  std::deque<NodeIndex>& removed_nodes,
                                                  size_t level,
                                                  const logging::Logger& logger) {
  bool modified = false;
  if (IsFP16Allow(node->OpType(), level)) {
    bool has_float_outputs = false;
    bool all_float_outputs_have_casts = true;
    std::vector<Node*> casts;  // Cast nodes to propagate.
    std::vector<NodeArg*>& outputs = node->MutableOutputDefs();
    std::unordered_set<NodeArg*> require_type_change;
    NodeArgToConsumerMap non_cast_consumers_map;
    // TODO Here we require the all floating point outputs are consumer by an immediate
    // child cast node.
    for (auto iter = outputs.begin(); iter != outputs.end() && all_float_outputs_have_casts; ++iter) {
      NodeArg* output = *iter;
      if (!IsType(*output, TensorProto::FLOAT)) {
        continue;
      }
      has_float_outputs = true;
      std::vector<Node*> consumers = graph.GetMutableConsumerNodes(output->Name());
      for (auto node_iter = consumers.begin(); node_iter != consumers.end() && (level >= 2 || all_float_outputs_have_casts); ++node_iter) {
        Node* consumer = *node_iter;
        if (nullptr != consumer &&
            std::find(removed_nodes.begin(), removed_nodes.end(), consumer->Index()) == removed_nodes.end()) {
          if (IsCastTo(consumer, TensorProto::FLOAT16)) {
            casts.push_back(consumer);
            continue;
          } else {
            non_cast_consumers_map[output].push_back(consumer);
          }
        }
        all_float_outputs_have_casts = false;
      }
      if (non_cast_consumers_map.empty()) {
        require_type_change.insert(output);
      }
    }
    if (has_float_outputs && (level >= 2 || all_float_outputs_have_casts) && casts.size() > 1) {
      LOGS(logger, VERBOSE) << "PropagateFP16CastsFromOutputsToInputs: Removed Cast nodes "
                            << ConcatNames<std::vector<Node*>>(casts)
                            << " feeding from the same compute node " << node->Name();
      InsertCastNodes(graph, non_cast_consumers_map, false, removed_nodes);
      for (Node* cast : casts) {
        RemoveCastNodesChain(graph, {cast}, removed_nodes);
      }
      NodeArgToConsumerMap node_args_map;
      for (NodeArg* input : node->MutableInputDefs()) {
        if (IsType(*input, TensorProto::FLOAT)) {
          node_args_map.insert(std::make_pair(input, std::vector<Node*>({node})));
        }
      }
      InsertCastNodes(graph, node_args_map, true, removed_nodes);
      ChangeTypeToFP16(graph, require_type_change, false, logger);
      LOGS(logger, VERBOSE) << "PropagateFP16CastsFromOutputsToInputs: Inserted Cast node to "
                            << ConcatNames<NodeArgToConsumerMap>(node_args_map, GetName);
      LOGS(logger, VERBOSE) << "PropagateFP16CastsFromOutputsToInputs: Inserted FP32 Cast node to "
                            << ConcatNames<NodeArgToConsumerMap>(non_cast_consumers_map, GetName);
      modified = true;
    }
  }
  return modified;
}
// CreateCast
// Create a cast node based on the node_arg for the given data tpye. If the node_arg is a graph output is_graph_outut is set.
// If the node_arg is not a graph output then the node_arg is the input of the new cast node. Otherwise the node_arg is the output
// of the new cast node.

static Node& CreateCast(Graph& graph, NodeArg* node_arg, TensorProto_DataType data_type, bool is_graph_output = false) {
  TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(is_graph_output ? (data_type == TensorProto::FLOAT ? TensorProto::FLOAT16 : TensorProto::FLOAT) : data_type);
  NodeArg& new_node_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node_arg->Name()), &type_proto);
  ONNX_NAMESPACE::AttributeProto to_attribute;
  to_attribute.set_name("to");
  to_attribute.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  to_attribute.set_i(static_cast<int64_t>(data_type));
  NodeAttributes attributes({{"to", to_attribute}});
  NodeArg& cast_input = is_graph_output ? new_node_arg : *node_arg;
  NodeArg& cast_output = is_graph_output ? *node_arg : new_node_arg;
  std::vector<NodeArg*> inputs = {&cast_input};
  std::vector<NodeArg*> outputs = {&cast_output};
  Node& node = graph.AddNode(graph.GenerateNodeName(node_arg->Name() + "_cast"),
                             "Cast",
                             "Created a new Cast node",
                             inputs,
                             outputs,
                             &attributes);
  inserted_node_names.insert(node.Name());
  return node;
}
/* InsertFP16Cast
* Insert a new cast input for the given float input for the give node.
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
static void InsertFP16Cast(Graph& graph, NodeArg* input_arg, Node* node, const logging::Logger& logger) {
  Node& cast = CreateCast(graph, input_arg, TensorProto::FLOAT16);
  NodeArg* new_input_arg = cast.MutableOutputDefs()[0];
  std::vector<NodeArg*>& inputs = node->MutableInputDefs();
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
*  Insert float casts on the given output for each consumer.
*
*                |               |
*              __V_______________V___                         
*             |       Opcode        |
*             |(operation performed |
*             |    in float16)      |
*             |_____________________|                                  
*               _______|_______             
*               |             |
*          _____V____    _____V______                         
*          |Consumer0|   |Consumer1 | 
*          |_________|   |__________|                        
*               |              |             
*
*  will be translated to the following
*
*                |               |
*              __V_______________V___                         
*             |       Opcode        |
*             |(operation performed |
*             |    in float16)      |
*             |_____________________|                                  
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
static void InsertFP32Casts(Graph& graph, NodeArg* output_arg, const logging::Logger& logger) {
  NodeArg* orig_output_arg = output_arg;
  Node* producer = graph.GetMutableProducerNode(output_arg->Name());
  std::vector<Node*> consumers = graph.GetMutableConsumerNodes(output_arg->Name());
  // First remove all the edges from the producer to the consumers.
  std::vector<Node*> new_consumers;
  for (Node* consumer : consumers) {
    int output_index = optimizer_utils::IndexOfNodeOutput(*producer, *output_arg);
    int input_index = optimizer_utils::IndexOfNodeInput(*consumer, *output_arg);
    graph.RemoveEdge(producer->Index(), consumer->Index(), output_index, input_index);
  }
  // Create a new output_arg on the producer if the output_arg is a graph output
  if (graph.IsOutput(output_arg)) {
    Node& cast = CreateCast(graph, output_arg, TensorProto::FLOAT, true);
    graph.UpdateProducerNode(output_arg->Name(), cast.Index());
    NodeArg* new_output_arg = cast.MutableInputDefs()[0];
    graph.UpdateProducerNode(new_output_arg->Name(), producer->Index());
    new_consumers.push_back(&cast);
    std::vector<NodeArg*>& outputs = producer->MutableOutputDefs();
    std::replace(outputs.begin(), outputs.end(), output_arg, new_output_arg);
    output_arg = new_output_arg;
    LOGS(logger, VERBOSE) << "Inserted FP32 Cast " << cast.Name() << " for the node arg " << output_arg->Name();
  } else {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT16);
    output_arg->UpdateTypeAndShape(type_proto, true, true, logger);
  }
  // Create a new cast node for each consumer
  for (Node* consumer : consumers) {
    Node& cast = CreateCast(graph, output_arg, TensorProto::FLOAT);
    NodeArg* new_output_arg = cast.MutableOutputDefs()[0];
    std::vector<NodeArg*>& inputs = consumer->MutableInputDefs();
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

// Expand FP16 compute regions on the graph by example float16 compute nodes,
// propagating float32 Cast operation down the graph and propagating float16
// Cast operations up the graph. The following functions are performed
// 1. Fuse subgraphs
// 2. Propagate fp32 casts forwards
// 3. Propagate fp16 casts back
Status PropagateCastOps::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
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
        // Each consumer of each output gets a seperate float cast inserted. Doing so will convert the computation of 
        // curren node from 32 bit float to 16 bit float operation. These cast operations will be eventually reduced.
        if (IsFP16Allow(node.OpType(), level_)) {
          // Insert FP16 Cast on all float inputs
          converted_node_names.insert(node.Name());
          for (NodeArg* input_arg : node.MutableInputDefs()) {
            if (IsType(*input_arg, TensorProto::FLOAT)) {
              InsertFP16Cast(graph, input_arg, node_ptr, logger);
            }
          }
          // Convert all output args to FP16 and insert FP32 cast for all consumers
          for (NodeArg* output_arg : node.MutableOutputDefs()) {
            if (IsType(*output_arg, TensorProto::FLOAT)) {
              InsertFP32Casts(graph, output_arg, logger);
            }
          }
        }
      }
    }
  }
  std::unordered_set<std::string> removed_node_names;
  int pass = 0;
  bool local_modified = strategy_ == GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::InsertAndReduce;
  do {
    LOGS(logger, VERBOSE) << "Propagate Cast Operations Pass " << pass << ":";
    std::deque<NodeIndex> removed_nodes;

    if (local_modified) {
      graph.Resolve();
      local_modified = false;
    }

    GraphViewer graph_viewer(graph);
    const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

    // Remove unnecessary casts
    for (auto node_index : node_topology_list) {
      Node* node = graph.GetNode(node_index);
      if (nullptr != node &&
          std::find(removed_nodes.begin(), removed_nodes.end(), node->Index()) == removed_nodes.end() &&
          node->OpType() == "Cast") {
        local_modified |= RemoveUnnecessaryCasts(graph, node, removed_nodes, logger);
      }
    }

    // Fuse subgraphs, sibling Cast nodes with same input
    for (auto node_index : node_topology_list) {
      Node* node = graph.GetNode(node_index);
      if (nullptr != node &&
          std::find(removed_nodes.begin(), removed_nodes.end(), node->Index()) == removed_nodes.end()) {
        local_modified |= FuseSiblingCasts(graph, node, removed_nodes, logger);
      }
    }
    for (const NodeArg* node_arg : graph.GetInputs()) {
      local_modified |= FuseSiblingCasts(graph, node_arg, removed_nodes, logger);
    }

    // Remove back to back Casts, with FLOAT->FLOAT16 followed by FLOAT16->FLOAT, but not the other way.
    for (auto node_index : node_topology_list) {
      Node* node = graph.GetNode(node_index);
      if (nullptr != node &&
          std::find(removed_nodes.begin(), removed_nodes.end(), node->Index()) == removed_nodes.end() &&
          IsCastTo(node, TensorProto::FLOAT)) {
        local_modified |= RemoveBackToBackCasts(graph, node, removed_nodes, logger);
      }
    }

    if (strategy_ == GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::FloodFill) {
      // Propagate FP16 Casts from outputs to inputs
      for (auto node_index : node_topology_list) {
        Node* node = graph.GetNode(node_index);
        if (nullptr != node &&
            std::find(removed_nodes.begin(), removed_nodes.end(), node->Index()) == removed_nodes.end()) {
          local_modified |= PropagateFP16CastsFromOutputsToInputs(graph, node, removed_nodes, level_, logger);
        }
      }

      // Propagate FP32 Casts forward
      for (auto node_index : node_topology_list) {
        Node* node = graph.GetNode(node_index);
        if (nullptr != node &&
            std::find(removed_nodes.begin(), removed_nodes.end(), node->Index()) == removed_nodes.end() &&
            IsCastTo(node, TensorProto::FLOAT)) {
          local_modified |= PropagateForwards(graph, node, removed_nodes, level_, logger);
        }
      }

      // Propagate FP16 Casts
      for (auto node_index : node_topology_list) {
        Node* node = graph.GetNode(node_index);
        if (nullptr != node &&
            std::find(removed_nodes.begin(), removed_nodes.end(), node->Index()) == removed_nodes.end() &&
            IsCastTo(node, TensorProto::FLOAT16)) {
          local_modified |= PropagateBackwards(graph, node, removed_nodes, level_, logger);
        }
      }

      // Propagate FP32 Casts from inputs to outputs
      for (auto node_index : node_topology_list) {
        Node* node = graph.GetNode(node_index);
        if (nullptr != node &&
            std::find(removed_nodes.begin(), removed_nodes.end(), node->Index()) == removed_nodes.end()) {
          local_modified |= PropagateFP32CastsFromInputsToOutputs(graph, node, removed_nodes, level_, logger);
        }
      }
    }
    for (NodeIndex removed_node : removed_nodes) {
      removed_node_names.insert(graph.GetNode(removed_node)->Name());
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
    std::for_each(inserted_node_names.begin(), inserted_node_names.end(), [removed_node_names, logger](std::string name) {
      if (removed_node_names.find(name) == removed_node_names.end()) { LOGS(logger, INFO) << name; } });

    LOGS(logger, INFO) << "Nodes Removed:";
    std::for_each(removed_node_names.begin(), removed_node_names.end(), [logger](std::string name) {
      if (inserted_node_names.find(name) == inserted_node_names.end()) { LOGS(logger, INFO) << name; } });

    LOGS(logger, INFO) << "Nodes Converted to FP16:";
    std::for_each(converted_node_names.begin(), converted_node_names.end(), [removed_node_names, logger](std::string name) {
      if (removed_node_names.find(name) == removed_node_names.end() && inserted_node_names.find(name) == inserted_node_names.end()) {
        LOGS(logger, INFO) << name; } });
  }
  inserted_node_names.clear();
  converted_node_names.clear();
  return Status::OK();
}

PropagateCastOps::PropagateCastOps(GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy strategy,
                                   size_t level, const std::vector<std::string>& _allow_list,
                                   const std::unordered_set<std::string>& compatible_execution_providers) noexcept
    : GraphTransformer("PropagateCastOps", compatible_execution_providers), level_(level), strategy_(strategy) {
  fp16_allow_ops[0].clear();  // Remove previously added op types if any.
  std::copy(_allow_list.begin(), _allow_list.end(), std::inserter(fp16_allow_ops[0], fp16_allow_ops[0].begin()));
}

}  // namespace onnxruntime