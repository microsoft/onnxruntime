// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_support_checker.h"

#include <unordered_map>

#include "core/common/common.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/graph_utils.h"
#include "core/providers/common.h"
#include "core/providers/cpu/nn/pool_attributes.h"

namespace onnxruntime {
namespace xnnpack {

namespace {
bool IsPaddingTypeSupported(AutoPadType auto_pad) {
  return auto_pad == AutoPadType::NOTSET ||
         auto_pad == AutoPadType::VALID ||
         auto_pad == AutoPadType::SAME_UPPER;
}

// function to check if a node is supported. kernel must have been matched previously to check type constraints.
using CheckerFn = std::function<bool(const Node& node,
                                     const GraphViewer& graph)>;

// function to check if we can fuse a node with a previously selected one.
// returns node to fuse with, or nullptr.
using FuseCheckerFn = std::function<const Node*(const Node& node,
                                                const GraphViewer& graph,
                                                const std::unordered_set<const Node*>& supported_nodes)>;

// check if the details of Conv are supported.
//
// we already validated type constraints via the kernel lookup in GetCapability so we know we're dealing with
// float input.
bool ConvChecker(const Node& node, const GraphViewer& graph) {
  bool supported = false;

  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    // Conv has at least 2 inputs.
    auto input_defs = node.InputDefs();
    const auto& x_arg = *input_defs[0];
    const auto& weight_arg = *input_defs[1];

    // we only support 2D (4 dims with batch and channel)
    const auto* x_shape = x_arg.Shape();
    if (!x_shape || x_shape->dim_size() != 4) {
      break;
    }

    // require C, H, W to be known so we can construct the xnnpack kernel prior to Compute
    if (!x_shape->dim(1).has_dim_value() ||
        !x_shape->dim(2).has_dim_value() ||
        !x_shape->dim(3).has_dim_value()) {
      break;
    }

    // weight must be constant and also rank 4
    const auto* weight = graph.GetConstantInitializer(weight_arg.Name(), true);
    if (weight == nullptr || weight->dims_size() != 4) {
      break;
    }

    // if there's a bias input it must be constant
    if (input_defs.size() == 3) {
      const auto& bias_arg = *input_defs[2];
      if (bias_arg.Exists() && !graph.IsConstantInitializer(bias_arg.Name(), true)) {
        break;
      }
    }

    ProtoHelperNodeContext nc(node);
    OpNodeProtoHelper info(&nc);

    // 'group' value needs to be 1 or C.
    // the second dim of weight is C/group, so if that == 1, group == C
    int64_t group = 0;
    info.GetAttrOrDefault<int64_t>("group", &group, 1);
    if (group != 1 && weight->dims(1) != 1) {
      break;
    }

    // if 'pads' is not specified we use 'auto_pad'
    if (graph_utils::GetNodeAttribute(node, "pads") == nullptr) {
      AutoPadType auto_pad = AutoPadType::NOTSET;

      std::string auto_pad_str;
      if (info.GetAttr<std::string>("auto_pad", &auto_pad_str).IsOK()) {
        // auto_pad was set
        //
        // The "auto_pad_str" string must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID
        // tf2onnx converter doesn't use SAME_LOWER.
        // SAME_UPPER maps to TF SAME padding.
        // TODO: What does PT converter use? We need to support models from PT in mobile.
        auto_pad = StringToAutoPadType(auto_pad_str);
        if (!IsPaddingTypeSupported(auto_pad)) {
          break;
        }
      }
    }

    supported = true;
  } while (false);

  return supported;
}

bool MaxPoolChecker(const Node& node, const GraphViewer& /*graph*/) {
  bool supported = false;

  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    // MaxPool has 1 input.
    auto input_defs = node.InputDefs();
    const auto& x_arg = *input_defs[0];

    // we only support 2D (4 dims with batch and channel)
    const auto* x_shape = x_arg.Shape();
    if (!x_shape || x_shape->dim_size() != 4) {
      break;
    }

    // require C, H, W to be known so we can construct the xnnpack kernel prior to Compute
    if (!x_shape->dim(1).has_dim_value() ||
        !x_shape->dim(2).has_dim_value() ||
        !x_shape->dim(3).has_dim_value()) {
      break;
    }

    // we don't support creating the optional 'I' output
    const auto& output_defs = node.OutputDefs();
    if (output_defs.size() == 2 && output_defs[1]->Exists()) {
      break;
    }

    ProtoHelperNodeContext nc(node);
    OpNodeProtoHelper info(&nc);
    PoolAttributes pool_attrs(info, "MaxPool", node.SinceVersion());

    // xnnpack doesn't appear to support using 'ceil' to calculate the output shape
    // https://github.com/google/XNNPACK/blob/3caa8b9de973839afa1e2a1462ff356e6927a66b/src/operators/max-pooling-nhwc.c#L256
    // calls compute_output_dimension but there's no ability to specify rounding that value up.
    if (pool_attrs.ceil_mode != 0) {
      break;
    }

    if (!IsPaddingTypeSupported(pool_attrs.auto_pad)) {
      break;
    }

    if ((pool_attrs.kernel_shape.size() != 2) ||
        (pool_attrs.kernel_shape[0] == 1 && pool_attrs.kernel_shape[1] == 1)) {
      // XNNPack doesn't support 1x1 maxpool.
      break;
    }

    supported = true;
  } while (false);

  return supported;
}

const Node* ClipReluChecker(const Node& node,
                            const GraphViewer& graph,
                            const std::unordered_set<const Node*>& supported_nodes) {
  const Node* fuse_with{nullptr};

  do {
    if (node.Domain() != kOnnxDomain) {
      break;
    }

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
      {"Conv", ConvChecker},
      {"MaxPool", MaxPoolChecker},
  };

  const auto entry = checkers.find(node.OpType());
  bool supported = false;
  if (entry != checkers.cend()) {
    supported = entry->second(node, graph_);
  }

  return supported;
}

const Node* NodeSupportChecker::IsNodeSupportedWithFusion(const Node& node) {
  static std::unordered_map<std::string, FuseCheckerFn> checkers{
      {"Clip", ClipReluChecker},  // fusion of Conv+Clip or MaxPool+Clip
      {"Relu", ClipReluChecker},  // fusion of Conv+Relu or MaxPool+Relu
  };

  const Node* fuse_with{nullptr};

  const auto entry = checkers.find(node.OpType());
  if (entry != checkers.cend()) {
    fuse_with = entry->second(node, graph_, supported_nodes_);
  }

  return fuse_with;
}
}  // namespace xnnpack
}  // namespace onnxruntime
