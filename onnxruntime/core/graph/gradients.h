// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "core/common/common.h"
#include "core/common/const_pointer_container.h"
#include "core/common/status.h"
#include "core/graph/basic_types.h"
#include "core/graph/contrib_ops/gradient_schema_defs.h"
#include "core/graph/constants.h"
#include "core/graph/graph_nodes.h"
#include "core/graph/node_arg.h"
#include "core/graph/onnx_protobuf.h"
#include "core/training/gradient_builder_base.h"

namespace onnxruntime {

class GradientGraphBuilder {
 public:
  /**
    This builder class constructs the gradient graph on top of the existing graph

    @param graph The forward computation graph
    @param y_node_arg_names_ List of name for NodeArgs whose initial gradients will be provided
    @param x_node_arg_names_ List of name for NodeArgs that need the gradients

    @remarks Given initial graidents at 'y_node_args' w.r.t some loss function L,
    the backward graph computes the partial derivative of 'L' w.r.t the 'x_node_args'
    **/
  GradientGraphBuilder(Graph* graph,
                       const std::vector<std::string>& y_node_arg_names_,
                       const std::vector<std::string>& x_node_arg_names_,
                       std::string loss_node_arg_name);

  Status Build();

 private:
  std::vector<std::string> y_node_arg_names_;
  std::vector<std::string> x_node_arg_names_;

  std::vector<const NodeArg*> y_node_args_;
  std::vector<const NodeArg*> x_node_args_;

  Graph* graph_;

  std::string loss_node_arg_name_;

  // key: name of the gradient, value: names of gardients to accumulate
  std::unordered_map<std::string, std::vector<std::string>> gradients_to_accumulate_;

  // key: name of the gradient, value: num of gradients pending
  std::unordered_map<std::string, int> pending_;

  void AddLossGradient();

  /**
  Perferms a BFS on the graph
  @param starting_node_args Starting node args
  @returns All the nodes visited during BFS
  */
  std::unordered_set<const Node*> BFS(const std::vector<const NodeArg*>& starting_node_args);

  /**
  Adds gradient nodes to the graph according to the gradient op definitions
  @param op_defs The deinitions of gradient nodes
  */
  void AddGradientNodes(const std::vector<OpDef>& op_defs);
};

}  // namespace onnxruntime
