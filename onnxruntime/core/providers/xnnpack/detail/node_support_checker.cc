// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "node_support_checker.h"

#include <unordered_map>

#include "core/common/common.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/xnnpack/detail/utils.h"

// each operator provides a helper to check if supported
#include "core/providers/xnnpack/math/gemm.h"
#include "core/providers/xnnpack/math/matmul.h"
#include "core/providers/xnnpack/math/softmax.h"
#include "core/providers/xnnpack/nn/average_pool.h"
#include "core/providers/xnnpack/nn/conv.h"
#include "core/providers/xnnpack/nn/conv_transpose.h"
#include "core/providers/xnnpack/nn/max_pool.h"
#include "core/providers/xnnpack/tensor/resize.h"

namespace onnxruntime {
namespace xnnpack {

namespace {
// function to check if a node is supported. kernel must have been matched previously to check type constraints.
using CheckerFn = std::function<bool(const NodeUnit& node,
                                     const GraphViewer& graph)>;

// function to check if we can fuse a node with a previously selected one.
// returns node to fuse with, or nullptr.
using FuseCheckerFn = std::function<const NodeUnit*(
    const NodeUnit& node_unit,
    const GraphViewer& graph,
    const std::unordered_map<const Node*, const NodeUnit*>& supported_node_unit_map)>;

const NodeUnit* ClipReluChecker(const NodeUnit& node_unit,
                                const GraphViewer& graph,
                                const std::unordered_map<const Node*, const NodeUnit*>& supported_node_unit_map) {
  const NodeUnit* fuse_with{nullptr};
  static const std::unordered_set<std::string> node_to_be_fuse = {"Conv", "MaxPool", "AveragePool"};
  const Node& node = node_unit.GetNode();
  do {
    // input 0 must come from a node we support
    const Node::EdgeEnd* input0_edge = graph_utils::GetInputEdge(node, 0);
    if (!input0_edge) {
      break;
    }

    // must be NHWC Conv or MaxPool in the supported nodes
    const Node& input0 = input0_edge->GetNode();
    if (supported_node_unit_map.count(&input0) == 0 ||
        input0.Domain() != kMSInternalNHWCDomain ||
        (node_to_be_fuse.count(input0.OpType()) == 0) ||
        supported_node_unit_map.at(&input0)->UnitType() == NodeUnit::Type::QDQGroup) {
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

    fuse_with = supported_node_unit_map.at(&input0);

  } while (false);

  return fuse_with;
}

}  // namespace

bool NodeSupportChecker::IsNodeSupported(const NodeUnit& nodeunit) {
  static std::unordered_map<std::string, CheckerFn> checkers{
      {"Conv", Conv::IsOnnxNodeSupported},
      {"ConvTranspose", ConvTranspose::IsOnnxNodeSupported},
      {"QLinearConv", Conv::IsOnnxNodeSupported},
      {"MaxPool", MaxPool::IsOnnxNodeSupported},
      {"AveragePool", AveragePool::IsOnnxNodeSupported},
      {"Softmax", Softmax::IsOnnxNodeSupported},
      {"Resize", Resize::IsOnnxNodeSupported},
      {"Gemm", Gemm::IsOnnxNodeSupported},
      {"MatMul", MatMul::IsOnnxNodeSupported},
  };

  bool supported = false;

  if (nodeunit.Domain() == onnxruntime::kOnnxDomain) {
    const auto entry = checkers.find(nodeunit.OpType());
    if (entry != checkers.cend()) {
      supported = entry->second(nodeunit, graph_);
    }
  }

  return supported;
}

const NodeUnit* NodeSupportChecker::IsNodeSupportedWithFusion(const NodeUnit& node_unit) {
  static std::unordered_map<std::string, FuseCheckerFn> checkers{
      {"Clip", ClipReluChecker},  // fusion of Conv+Clip or MaxPool+Clip
      {"Relu", ClipReluChecker},  // fusion of Conv+Relu or MaxPool+Relu
  };

  const NodeUnit* fuse_with{nullptr};
  // There is not the case to fuse QDQGroup and QDQGroup
  if (node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
    return fuse_with;
  }

  if (node_unit.Domain() == onnxruntime::kOnnxDomain) {
    const auto entry = checkers.find(node_unit.OpType());
    if (entry != checkers.cend()) {
      fuse_with = entry->second(node_unit, graph_, supported_node_unit_map_);
    }
  }

  return fuse_with;
}
}  // namespace xnnpack
}  // namespace onnxruntime
