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
        {"Pow", {1}},
        {"Gather", {1}},
        {"Reshape", {1}},
        {"Expand", {1}},
        {"TrainableDropout", {1}},
        {"Dropout", {1, 2}},
        {"Slice", {1, 2, 3, 4}},
        {"SparseSoftmaxCrossEntropy", {1, 2}},
        {"SoftmaxCrossEntropyLoss", {1, 2}},
        {"ConstantOfShape", {0}},
        {"Scatter", {1}},
        {"OneHot", {0, 1, 2}},
        {"Where", {0}},
        {"Range", {0, 1, 2}},
        {"BroadcastGradientArgs", {0, 1}}};

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

  Status Build();

 private:
  std::unordered_set<const NodeArg*> y_node_args_;
  std::unordered_set<const NodeArg*> x_node_args_;

  NodeSet y_nodes_;
  NodeSet x_nodes_;

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
  Perferms a ReverseBFS on the graph
  @param nodes Starting nodes for ReverseBFS
  @returns All the nodes visited during ReverseBFS
  */
  NodeSet ReverseBFS(const NodeSet& nodes);

  /**
  Check if 'x_node_args_' are reachable from 'y_node_args_' for computing the partial derivative
  @param reachable_nodes All the nodes reachable from the 'y_node_args_'
  @returns OK if all 'x_node_args_' are reachable, else an ONNXRUNTIME INVALID_ARGUMENT status
  */
  Status CheckNodeArgsReachable(const NodeSet& reachable_nodes);
};

}  // namespace training
}  // namespace onnxruntime
