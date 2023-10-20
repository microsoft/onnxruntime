// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/transpose_optimization/ort_transpose_optimization.h"

#include <algorithm>
#include "core/graph/constants.h"
#include "core/framework/utils.h"
#include "core/optimizer/transpose_optimization/ort_optimizer_utils.h"

using namespace onnx_transpose_optimization;

namespace onnxruntime {

static bool EPAwareHandleResize(HandlerArgs& args) {
  // Whilst Resize is not technically layout sensitive, execution providers typically implement handling for only one
  // layout. Due to that, only push a Transpose through a Resize once it is assigned and we know it's being handled
  // by an EP that supports multiple layouts. Currently that's the CPU and XNNPACK EPs.
  const auto ep_type = args.node.GetExecutionProviderType();
  if (ep_type == kCpuExecutionProvider /*|| ep_type == kXnnpackExecutionProvider*/) {
    // allow NCHW <-> NHWC for now. not clear any other sort of transpose has a valid usage in a real model
    int64_t rank_int = gsl::narrow_cast<int64_t>(args.perm.size());
    if (rank_int == 4) {
      static const std::vector<int64_t> nchw_to_nhwc_perm{0, 2, 3, 1};
      static const std::vector<int64_t> nhwc_to_nchw_perm{0, 3, 1, 2};
      if (args.perm == nchw_to_nhwc_perm || args.perm == nhwc_to_nchw_perm) {
        return HandleResize(args);
      }
    }
  }

  return false;
}

constexpr HandlerInfo ep_aware_resize_handler = {&FirstInput, &EPAwareHandleResize};

static bool HandleQLinearConcat(HandlerArgs& args) {
  return HandleSimpleNodeWithAxis(args);
}

std::vector<size_t> QLinearConcatInputs(OptimizerCtx& ctx, api::NodeRef& node) {
  (void)ctx;
  std::vector<size_t> indices;
  size_t num_inputs = node.Inputs().size();
  for (size_t i = 2; i < num_inputs; i += 3) {
    indices.push_back(i);
  }
  return indices;
}

constexpr HandlerInfo q_linear_concat_handler = {&QLinearConcatInputs, &HandleQLinearConcat};

static bool HandleQLinearBinaryOp(HandlerArgs& args) {
  return HandleSimpleNodeBroadcast(args);
}

std::vector<size_t> QLinearBinaryOpInputs(OptimizerCtx&, api::NodeRef&) {
  // Inputs are: [A, A_scale, A_zero_point, B, B_scale, B_zero_point, C_scale, C_zero_point],
  // we want [A, B].
  return {0, 3};
}

constexpr HandlerInfo q_linear_binary_op_handler = {&QLinearBinaryOpInputs, &HandleQLinearBinaryOp};

static bool HandleQLinearPoolOp(HandlerArgs& args) {
  // Swap between channel first/last variants. Only works for applicable values of perm.
  int64_t channels_last = args.node.GetAttributeIntDefault("channels_last", 0);
  size_t rank = args.perm.size();
  if (rank < 2) return false;
  auto p = ChannelLastToFirstPerm(rank);
  if ((!channels_last && args.perm == p) || (channels_last && args.perm_inv == p)) {
    args.node.SetAttributeInt("channels_last", 1 - channels_last);
    TransposeFirstInput(args.ctx, args.node, args.perm_inv);
    TransposeOutputs(args.ctx, args.node, args.perm);
    return true;
  }
  return false;
}

constexpr HandlerInfo q_linear_pool_op_handler = {&FirstInput, &HandleQLinearPoolOp};

static bool HandleMaxPool(HandlerArgs& args) {
#if defined(DISABLE_CONTRIB_OPS)
  // Cannot convert MaxPool to com.microsoft.NhwcMaxPool if contrib ops are disabled in this build.
  ORT_UNUSED_PARAMETER(args);
  return false;
#else
  if (args.node.GetExecutionProviderType() != kCpuExecutionProvider) {
    return false;
  }

  auto outputs = args.node.Outputs();
  if (outputs.size() == 2 && outputs[1] != "") {
    // Can't optimize if optional "indices" output is provided
    return false;
  }

  auto info = args.ctx.graph.GetValueInfo(outputs[0]);
  api::DataType dtype = info->DType();
  if (dtype != api::DataType::UINT8 && dtype != api::DataType::INT8) {
    return false;
  }

  size_t rank = args.perm.size();
  if (args.perm != ChannelLastToFirstPerm(rank)) {
    return false;
  }

  auto new_node = SwapNodeOpTypeDomainAndSinceVersion(args.ctx.graph, args.node, "NhwcMaxPool", kMSDomain, 1);
  new_node->ClearAttribute("storage_order");  // Only relevant for indices output. Prohibited for NhwcMaxPool.
  TransposeFirstInput(args.ctx, *new_node, args.perm_inv);
  TransposeOutputs(args.ctx, *new_node, args.perm);
  return true;
#endif  // defined(DISABLE_CONTRIB_OPS)
}

static bool HandleContribQuantizeDequantizeLinear(HandlerArgs& args) {
  if (!TransposeQuantizeDequantizeAxis(args.ctx.graph, args.perm, args.node)) {
    return false;
  }

  TransposeFirstInput(args.ctx, args.node, args.perm_inv);
  TransposeOutputs(args.ctx, args.node, args.perm);

  return true;
}

constexpr HandlerInfo max_pool_op_handler = {&FirstInput, &HandleMaxPool};

constexpr HandlerInfo node_1_inp_handler = {&FirstInput, &HandleSimpleNode};
constexpr HandlerInfo reduce_op_handler = {&FirstInput, &HandleReduceOps};
constexpr HandlerInfo contrib_quantize_dequantize_linear_handler = {&FirstInput,
                                                                    &HandleContribQuantizeDequantizeLinear};

// ORT contrib ops and special cased ONNX ops where we have EP specific handling
const HandlerMap& OrtExtendedHandlers() {
  static const HandlerMap extended_handler_map = []() {
    HandlerMap map = {
        {"MaxPool", max_pool_op_handler},
        {"Resize", ep_aware_resize_handler},
        {"com.microsoft.QuantizeLinear", contrib_quantize_dequantize_linear_handler},
        {"com.microsoft.DequantizeLinear", contrib_quantize_dequantize_linear_handler},
        {"com.microsoft.QLinearAdd", q_linear_binary_op_handler},
        {"com.microsoft.QLinearAveragePool", q_linear_pool_op_handler},
        {"com.microsoft.QLinearConcat", q_linear_concat_handler},
        {"com.microsoft.QLinearGlobalAveragePool", q_linear_pool_op_handler},
        {"com.microsoft.QLinearLeakyRelu", node_1_inp_handler},
        {"com.microsoft.QLinearMul", q_linear_binary_op_handler},
        {"com.microsoft.QLinearReduceMean", reduce_op_handler},
        {"com.microsoft.QLinearSigmoid", node_1_inp_handler},
    };

    return map;
  }();

  return extended_handler_map;
}

CostCheckResult OrtEPCostCheck(const api::GraphRef& graph, const api::NodeRef& node,
                               const std::vector<int64_t>& /*perm*/,
                               const std::unordered_set<std::string>& /*outputs_leading_to_transpose*/) {
  // special case some kernels based on the ORT implementation details
  if (node.GetExecutionProviderType() == kCpuExecutionProvider) {
    if (node.IsOp("MaxPool")) {
      // MaxPool has higher perf in the NHWC variant when supported. HandleMaxPool does the support checks.
      return CostCheckResult::kPushTranspose;
    }

    if (node.IsOp("Resize")) {
      // Resize is included because it has higher perf in the NHWC variant when
      // the input X is 4D int8 tensor and the mode is linear
      auto X_value_info = graph.GetValueInfo(node.Inputs()[0]);
      auto X_shape = X_value_info->Shape();
      auto X_dtype = X_value_info->DType();
      auto mode = node.GetAttributeString("mode");
      if (X_shape && X_shape->size() == 4 &&
          (X_dtype == api::DataType::UINT8 || X_dtype == api::DataType::INT8) &&
          mode && *mode == "linear") {
        return CostCheckResult::kPushTranspose;
      }
    }
  }

  return CostCheckResult::kFallThrough;
}

static std::unique_ptr<api::NodeRef> SwapNodeImpl(api::GraphRef& graph, api::NodeRef& node,
                                                  std::string_view op_type, std::string_view domain,
                                                  std::optional<int> since_version) {
  auto outputs = node.Outputs();
  auto new_node = graph.CopyNode(node, op_type, domain, since_version);

  for (size_t j = 0; j < outputs.size(); ++j) {
    if (outputs[j] != "") {
      graph.MoveOutput(node, j, *new_node, j);
    }
  }
  graph.RemoveNode(node);
  return new_node;
}

std::unique_ptr<api::NodeRef> SwapNodeOpTypeAndDomain(api::GraphRef& graph, api::NodeRef& node,
                                                      std::string_view op_type, std::string_view domain) {
  return SwapNodeImpl(graph, node, op_type, domain, std::nullopt);
}

std::unique_ptr<api::NodeRef> SwapNodeOpTypeDomainAndSinceVersion(api::GraphRef& graph, api::NodeRef& node,
                                                                  std::string_view op_type, std::string_view domain,
                                                                  int since_version) {
  return SwapNodeImpl(graph, node, op_type, domain, since_version);
}
}  // namespace onnxruntime
