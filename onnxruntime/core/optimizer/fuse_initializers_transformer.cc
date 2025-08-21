// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <set>
#include <algorithm>
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/fuse_initializers_transformer.h"

namespace onnxruntime {

/**
 * @brief   Check if the input node is cast node with no inputs and single output.
 *
 * @param graph_viewer  GraphViewer object
 * @param node          Node object
 *
 * @return  True, if input node is Cast node with no inputs and single output,
 *          else, False.
 */
static bool IsCastNodeWithConstraints(const GraphViewer& graph_viewer, const Node& node) {
  // Node must be cast node
  if (!("Cast" == node.OpType())) return false;

  // Node must have no input edges
  if (!(0 == node.GetInputEdgesCount())) return false;

  // Node must have only one output edge
  if (!(1 == node.GetOutputEdgesCount())) return false;

  // Node output/s must not be part of graph output
  // This check is added as GetOutputEdgesCount()
  // don't count the output edges which are part
  // of graph output.
  if (graph_viewer.NodeProducesGraphOutput(node)) return false;

  return true;
}

/**
 * @brief   Check if for the current node has an initialized tensor of a specific type with specific type output.
 *
 * As the node to be checked is a Cast node with zero input, one initializer and one output, the input_defs_index
 * and the output_defs_index is assumed to be 0 in all cases for all the checks.
 *
 * @param graph            Graph object.
 * @param node             Node object.
 * @param tensor_type      The type of initialized tensor to be found in the given node.
 * @param output_type      The type of output for the given node.
 *
 * @return  True, if for the given node an initialized tensor of "tensor_type" with specific "output_type" is found,
 *          else, False.
 */
static bool IsNodeValidForFusion(const Graph& graph,
                                 const Node& node,
                                 const onnxruntime::MLDataType tensor_type,
                                 const onnxruntime::MLDataType output_type) {
  // Node must have initialized tensor
  if (!(graph.IsInitializedTensor(node.InputDefs()[0]->Name()))) return false;

  // Initialzed tensor must be of tensor_type
  if (!(DataTypeImpl::TypeFromProto(*(node.InputDefs()[0]->TypeAsProto())) == tensor_type)) return false;

  // Node output must be of output_type
  if (!(DataTypeImpl::TypeFromProto(*(node.OutputDefs()[0]->TypeAsProto())) == output_type)) return false;

  return true;
}

/**
 * @brief   Make a new name from the old node arg name.
 *
 * It replaces "InsertedPrecisionFreeCast_" prefix in a node name with "FusedBack_" prefix.
 *
 * @param   old_node_arg_name Old arg name.
 *
 * @return  New arg name.
 */
static const std::string NewNodeArgName(const std::string& old_node_arg_name) {
  static thread_local const std::string pattern_to_be_replaced = "InsertedPrecisionFreeCast_";
  std::string new_node_arg_name = old_node_arg_name;
  auto pos = new_node_arg_name.find(pattern_to_be_replaced);
  if (std::string::npos != pos) new_node_arg_name.replace(pos, pattern_to_be_replaced.size(), "");
  new_node_arg_name = "FusedBack_" + new_node_arg_name;
  return new_node_arg_name;
}

/**
 * @brief   It fuses the initializer in the current node to its next node / output node, and then
 *          remove the link/edge between current node and next node / output node.
 *
 * The node input_defs_index and output_defs_index is assumed to be always 0, as the node which encapsulates
 * a single Initializer have just zero input, one initializer and one output.
 *
 * @param graph                 Graph object.
 * @param node                  Current Node to be fused with its next node.
 * @param next_node_arg_type    The "type" of initializer expected by the next-node to the current node.
 * @param thread_pool           Thread pool for multi-threaded conversion of the initializer
 *                              from an unsupported to supported type tensor.
 */
static void FuseInitializerWithNode(Graph& graph,
                                    Node& node,
                                    const onnxruntime::MLDataType next_node_arg_type,
                                    onnxruntime::concurrency::ThreadPool* thread_pool) {
  // Get next node
  Node& next_node = *graph.GetNode(node.OutputNodesBegin()->Index());

  // Get the index in next node at which the initializer must be replaced
  NodeIndex next_node_arg_index = 0;
  for (; next_node_arg_index < next_node.InputDefs().size(); ++next_node_arg_index) {
    if (node.Name() == next_node.InputDefs()[next_node_arg_index]->Name()) {
      break;
    }
  }

  // Get the src initialized tensor at input def index 0
  const auto* constant_initializer_tensor = graph_utils::GetConstantInitializer(graph, node.InputDefs()[0]->Name());
  Initializer src_init{*constant_initializer_tensor, graph.ModelPath()};

  // Convert to dst tensor
  std::string new_arg_name = graph.GenerateNodeArgName(NewNodeArgName(
      next_node.InputDefs()[next_node_arg_index]->Name()));

  OrtValue new_data;
  if (next_node_arg_type == DataTypeImpl::GetTensorType<float>())
    new_data = src_init.ToFloat32(thread_pool);
  else if (next_node_arg_type == DataTypeImpl::GetTensorType<MLFloat16>())
    new_data = src_init.ToFP16();
  else if (next_node_arg_type == DataTypeImpl::GetTensorType<BFloat16>())
    new_data = src_init.ToBFloat16();
  else
    return;

  // Remove the edge between the current node output def at index 0 and next node arg at relative arg index.
  graph.RemoveEdge(node.Index(), next_node.Index(), 0, static_cast<int>(next_node_arg_index));

  // Add the new converted Tensor in next node as initializer potentially with external data
  ONNX_NAMESPACE::TensorProto dst_tensor = utils::TensorToTensorProto(new_data.Get<Tensor>(), new_arg_name, false);
  auto& new_arg = graph_utils::AddInitializer(graph, dst_tensor);
  graph_utils::ReplaceNodeInput(next_node, static_cast<int>(next_node_arg_index), new_arg);
}

Status FuseInitializersTransformer::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const logging::Logger& /*logger*/) const {
  // Init
  std::set<NodeIndex> nodes_to_be_fused_and_removed_from_graph;

  // Get nodes in topological order
  const GraphViewer graph_viewer(graph);
  auto nodes_indexes_in_topological_order = graph_viewer.GetNodesInTopologicalOrder();

  // For each Node
  for (auto node_index : nodes_indexes_in_topological_order) {
    // Get Node
    auto node = graph.GetNode(node_index);

    // Check if the current node is a Cast node with all constraints and valid for fusion
    if (node && IsCastNodeWithConstraints(graph_viewer, *node) && IsNodeValidForFusion(graph, *node, init_type_, cvt_type_)) {
      // Add node to the set of nodes to be fused and removed
      nodes_to_be_fused_and_removed_from_graph.insert(node_index);
    }
  }

  // Fuse all Cast Node with src type Initializer casted to dst type to Next Node
  for (auto node_index : nodes_to_be_fused_and_removed_from_graph) {
    auto node = graph.GetNode(node_index);
    FuseInitializerWithNode(graph, *node, cvt_type_, thread_pool_);
  }

  // Remove all nodes considered during fusion
  for (auto node_index : nodes_to_be_fused_and_removed_from_graph) {
    graph.RemoveNode(node_index);
  }

  // set flag to true indicating the graph is changed
  if (!nodes_to_be_fused_and_removed_from_graph.empty()) modified = true;

  return Status::OK();
}

}  // namespace onnxruntime
