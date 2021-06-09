// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>

#include "core/common/common.h"
#include "core/common/const_pointer_container.h"
#include "core/common/status.h"
#include "core/graph/basic_types.h"
#include "core/graph/constants.h"
#include "core/graph/graph_nodes.h"
#include "core/graph/node_arg.h"
#include "orttraining/core/graph/training_op_defs.h"
#include "orttraining/core/graph/gradient_builder_base.h"
#include "core/optimizer/graph_transformer_mgr.h"

namespace onnxruntime {
namespace training {

typedef std::set<const Node*, NodeCompare> NodeSet;

static std::unordered_map<std::string, std::unordered_set<size_t>>
    STOP_GRADIENT_EDGES = {
        {"Not", {0}},
        {"And", {0, 1}},
        {"BatchNormalization", {3, 4}},
        {"BatchNormInternal", {3, 4}},
        {"Or", {0, 1}},
        {"Xor", {0, 1}},
        {"Equal", {0, 1}},
        {"Less", {0, 1}},
        {"LessOrEqual", {0, 1}},
        {"Greater", {0, 1}},
        {"GreaterOrEqual", {0, 1}},
        {"IsInf", {0}},
        {"IsNaN", {0}},
        {"NonZero", {0}},
        {"Pow", {1}},  // TODO: Pow's input_1 is differentiable, but gradient not yet implemented
        {"Gather", {1}},
        {"GatherElements", {1}},
        {"GatherND", {1}},
        {"Shape", {0}},
        {"Size", {0}},
        {"Reshape", {1}},
        {"Expand", {1}},
        {"Dropout", {1, 2}},
        {"Slice", {1, 2, 3, 4}},
        {"SparseSoftmaxCrossEntropy", {1, 2}},
        {"SoftmaxCrossEntropyLoss", {1, 2}},
        {"SoftmaxCrossEntropyLossInternal", {1, 2, 3}},
        {"ConstantOfShape", {0}},
        {"Scatter", {1}},
        {"ScatterElements", {1}},
        {"ScatterND", {1}},
        {"OneHot", {0, 1, 2}},
        {"Where", {0}},
        {"Range", {0, 1, 2}},
        {"Tile", {1}},
        {"BroadcastGradientArgs", {0, 1}},
        {"TopK", {1}},
        {"Squeeze", {1}},
        {"Unsqueeze", {1}},
        {"ReduceSum", {1}},
        {"Split", {1}},
        {"Clip", {1, 2}}};

class GradientGraphBuilder {
 public:
  /**
    This builder class constructs the gradient graph on top of the existing graph

    @param graph The forward computation graph
    @param y_node_arg_names_ Set of name for NodeArgs whose initial gradients will be provided
    @param x_node_arg_names_ Set of name for NodeArgs that need the gradients

    @remarks Given initial gradients at 'y_node_args' w.r.t some loss function L,
    the backward graph computes the partial derivative of 'L' w.r.t the 'x_node_args'
    **/
  GradientGraphBuilder(Graph* graph,
                       const std::unordered_set<std::string>& y_node_arg_names,
                       const std::unordered_set<std::string>& x_node_arg_names,
                       const std::string& loss_node_arg_name,
                       const GradientGraphConfiguration& gradient_graph_config,
                       const logging::Logger& logger);

  Status Build(const std::unordered_set<std::string>* p_initializer_names_to_preserve = nullptr);

  const std::unordered_set<std::string>& GetNonDifferentiableYNodeArgNames() const {
    return non_differentiable_y_node_arg_names_;
  }

 private:
  std::unordered_set<const NodeArg*> y_node_args_;
  std::unordered_set<const NodeArg*> x_node_args_;

  NodeSet y_nodes_;
  NodeSet x_nodes_;
  NodeSet reachable_nodes_;

  std::unordered_set<std::string> non_differentiable_y_node_arg_names_;

  Graph* graph_;

  std::string loss_node_arg_name_;

  const GradientGraphConfiguration& gradient_graph_config_;

  const logging::Logger& logger_;

  onnxruntime::GraphTransformerManager graph_transformation_mgr_{5};

  // key: ArgDef for the gradient after accumulation
  // value: ArgDef for the gradients to be accumulated
  struct ArgDefHasher {
    std::size_t operator()(const ArgDef& arg) const {
      return std::hash<std::string>()(arg.name);
    }
  };
  std::unordered_map<ArgDef, std::vector<ArgDef>, ArgDefHasher> gradients_to_accumulate_;

  // key: name of the gradient, value: num of gradients pending
  std::unordered_map<std::string, int> pending_;

  /**
  Performs a BFS on the graph with STOP_GRADIENT_EDGES constrain
  It will skip traversing over the edges defined in STOP_GRADIENT_EDGES map.
  The resulting node set contains all the nodes that are differentiable wrt the x_node_args
  @param Starting nodes arg name for BFS
  @returns All the nodes visited during BFS
  */
  NodeSet BFSWithStopGradient(const std::unordered_set<std::string>& x_node_arg_names) const;

  /**
  Performs a ReverseBFS on the graph with STOP_GRADIENT_EDGES constrain
  It will skip traversing over the edges defined in STOP_GRADIENT_EDGES map.
  The resulting node set contains all the nodes that are differentiable wrt the input nodes
  @param Starting nodes for ReverseBFS
  @returns All the nodes visited during ReverseBFS
  */
  NodeSet ReverseBFSWithStopGradient(const NodeSet& nodes) const;

  /**
  Check if 'x_node_args_' are reachable from 'y_node_args_' for computing the partial derivative
  @param reachable_nodes All the nodes reachable from the 'y_node_args_'
  @returns OK if all 'x_node_args_' are reachable, else an ONNXRUNTIME INVALID_ARGUMENT status
  */
  Status CheckNodeArgsReachable() const;

  /** 
  Check if node is reachable from the 'y_node_args_'
   **/
  bool IsReachable(const Node* node) const {
    return reachable_nodes_.find(node) != reachable_nodes_.end();
  }
};

}  // namespace training
}  // namespace onnxruntime
