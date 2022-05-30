// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "node_support_checker.h"

#include <unordered_map>

#include "core/common/common.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/graph_utils.h"
#include "core/providers/common.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/providers/xnnpack/detail/utils.h"

// each operator provides a helper to check if supported
#include "core/providers/xnnpack/nn/conv.h"
#include "core/providers/xnnpack/nn/max_pool.h"

namespace onnxruntime {
namespace xnnpack {

namespace {
// function to check if a node is supported. kernel must have been matched previously to check type constraints.
using CheckerFn = std::function<bool(const Node& node,
                                     const GraphViewer& graph)>;

// function to check if we can fuse a node with a previously selected one.
// returns node to fuse with, or nullptr.
using FuseCheckerFn = std::function<const Node*(const Node& node,
                                                const GraphViewer& graph,
                                                const std::unordered_set<const Node*>& supported_nodes)>;

const Node* ClipReluChecker(const Node& node,
                            const GraphViewer& graph,
                            const std::unordered_set<const Node*>& supported_nodes) {
  const Node* fuse_with{nullptr};

  do {
    // input 0 must come from a node we support
    const Node::EdgeEnd* input0_edge = graph_utils::GetInputEdge(node, 0);
    if (!input0_edge) {
      break;
    }

    // must be NHWC Conv or MaxPool in the supported nodes
    const Node& input0 = input0_edge->GetNode();
    if (supported_nodes.count(&input0) == 0 ||
        input0.Domain() != kMSInternalNHWCDomain ||
        (input0.OpType() != "Conv" && input0.OpType() != "MaxPool")) {
      break;
    }

    // if Clip check the min/max are constant.
    if (node.OpType() == "Clip") {
      const auto& input_args = node.InputDefs();
      const auto num_inputs = input_args.size();
      if (num_inputs >= 2) {
        // check 'min' is constant
        if (!graph.IsConstantInitializer(input_args[1]->Name(), true)) {
          break;
        }
      }

      if (num_inputs == 3) {
        // check 'max' is constant
        if (!graph.IsConstantInitializer(input_args[2]->Name(), true)) {
          break;
        }
      }
    }

    fuse_with = &input0;

  } while (false);

  return fuse_with;
}

}  // namespace

bool NodeSupportChecker::IsNodeSupported(const Node& node) {
  static std::unordered_map<std::string, CheckerFn> checkers{
      {"Conv", Conv::IsOnnxNodeSupported},
      {"MaxPool", MaxPool::IsOnnxNodeSupported},
  };

  bool supported = false;

  if (node.Domain() == onnxruntime::kOnnxDomain) {
    const auto entry = checkers.find(node.OpType());
    if (entry != checkers.cend()) {
      supported = entry->second(node, graph_);
    }
  }

  return supported;
}

const Node* NodeSupportChecker::IsNodeSupportedWithFusion(const Node& node) {
  static std::unordered_map<std::string, FuseCheckerFn> checkers{
      {"Clip", ClipReluChecker},  // fusion of Conv+Clip or MaxPool+Clip
      {"Relu", ClipReluChecker},  // fusion of Conv+Relu or MaxPool+Relu
  };

  const Node* fuse_with{nullptr};

  if (node.Domain() == onnxruntime::kOnnxDomain) {
    const auto entry = checkers.find(node.OpType());
    if (entry != checkers.cend()) {
      fuse_with = entry->second(node, graph_, supported_nodes_);
    }
  }

  return fuse_with;
}
}  // namespace xnnpack
}  // namespace onnxruntime
