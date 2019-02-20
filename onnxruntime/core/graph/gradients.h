// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>
#include <memory>
#include <string>
#include <type_traits>
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
#include "core/graph/function.h"
#include "gsl/gsl_util"
#include "gsl/pointers"

namespace onnxruntime {

class GradientGraphBuilder {
 public:
  /**
  This builder class consturct the backward gradient graph

  @param fw_graph The forward computation graph
  @param y_node_args List of NodeArgs whoes initial gradients will be provided
  @param x_node_args List of NodeArgs that need the gradients

  @remarks Given initial graidents at 'y_node_args' w.r.t some loss function L,
  the backward graph computes the partial derivative of 'L' w.r.t the 'x_node_args'
  **/
  GradientGraphBuilder(Graph* fw_graph,
                       Graph* bw_graph,
                       std::vector<NodeArg*> y_node_args,
                       std::vector<NodeArg*> x_node_args,
                       std::string loss_node_arg_name);

  GradientGraphBuilder(Graph* fw_graph,
                       Graph* bw_graph,
                       const std::vector<std::string>& y_node_args,
                       const std::vector<std::string>& x_node_args,
                       std::string loss_node_arg_name);

  /*
  @param bw_graph Construted backward graph
  */
  Status Build();

 private:
  std::vector<NodeArg*> y_node_args_;
  std::vector<NodeArg*> x_node_args_;

  Graph* fw_graph_;
  Graph* bw_graph_;

  std::string loss_node_arg_name_;

  std::unordered_map<std::string, std::vector<std::string>>
      gradients_to_accumulate_;

  std::unordered_map<std::string, int> pending_;

  void AddLossGradient();

  void CopyInitializedTensor(const std::string& tensor_name);

  NodeArg& GetOrCreateNodeArg(const Node* node, GradientOps::DefsMapping mapping,
                              const std::unordered_set<const NodeArg*>& visited_node_arg = std::unordered_set<const NodeArg*>());

  std::string GradientName(const std::string& name) {
    return name + "_grad";
  }

  std::string GradientOpType(const std::string& op_type) {
    return op_type + "Grad";
  }
};

}  // namespace onnxruntime
