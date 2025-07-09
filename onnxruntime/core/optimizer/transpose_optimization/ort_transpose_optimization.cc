// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/transpose_optimization/ort_transpose_optimization.h"

#include <algorithm>

#include <gsl/gsl>

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
  if (ep_type == kCpuExecutionProvider) {
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

static bool EPAwareHandleReshape(HandlerArgs& args) {
  const auto ep_type = args.node.GetExecutionProviderType();
  if (ep_type == kQnnExecutionProvider) {
    // In some cases, the pattern of Transpose-Reshape-Transpose can be optimized to a single Reshape.
    // For example, [N,H,W,C] - [N,C,H,W] - [N,C,HxW] - [N,HxW,C] is functionally equivalent to [N,H,W,C] - [N,HxW,C].
    // In this optimization, we attempt to handle those "channel" Transpose and "spatial" Reshape cases, like the
    // example above.

    // Only attempts to push through if Transpose is possibly canceled with following one.
    const std::string output_name = std::string(args.node.Outputs()[0]);
    if (args.outputs_leading_to_transpose.find(output_name) == args.outputs_leading_to_transpose.end()) {
      return HandleReshape(args);
    }

    // Get input/output shapes.
    auto reshape_input_shape = args.ctx.graph.GetValueInfo(args.node.Inputs()[0])->Shape();
    auto reshape_output_shape = args.ctx.graph.GetValueInfo(args.node.Outputs()[0])->Shape();
    if (!reshape_input_shape.has_value() || !reshape_output_shape.has_value()) {
      return HandleReshape(args);
    }

    const std::vector<int64_t>& input_shape = *reshape_input_shape;
    const std::vector<int64_t>& output_shape = *reshape_output_shape;
    const size_t input_rank = input_shape.size();
    const size_t output_rank = output_shape.size();

    std::vector<int64_t> output_perm;

    // Determine "channel" Transpose by checking perm being channel-first to channel-last or vice versa.
    const std::vector<int64_t> perm_3d{0, 2, 1};
    const std::vector<int64_t> perm_4d_cl{0, 2, 3, 1};
    const std::vector<int64_t> perm_4d_cf{0, 3, 1, 2};

    // Determine "spatial" Reshape by checking the batch and channel dimensions untouched.
    const bool batch_preserved = (input_shape[0] == output_shape[0]);
    const bool cf_preserved = (input_shape[1] == output_shape[1]);
    const bool cl_preserved = (input_shape[input_rank - 1] == output_shape[output_rank - 1]);

    if (args.perm == perm_3d) {
      // There is ambiguity to determine the direction solely from this Transpose perm.
      // The implementation may result in non-fully optimized pattern since the perm info from the output Transpose is
      // mandatory for determination. Leave it as future work as such info is inaccessible in current infra.
      if (batch_preserved && cf_preserved) {
        output_perm = ChannelFirstToLastPerm(output_rank);
      } else if (batch_preserved && cl_preserved) {
        output_perm = ChannelLastToFirstPerm(output_rank);
      } else {
        return HandleReshape(args);
      }
    } else if (args.perm == perm_4d_cl && batch_preserved && cl_preserved) {
      output_perm = ChannelLastToFirstPerm(output_rank);
    } else if (args.perm == perm_4d_cf && batch_preserved && cf_preserved) {
      output_perm = ChannelFirstToLastPerm(output_rank);
    } else {
      return HandleReshape(args);
    }

    TransposeFirstInput(args.ctx, args.node, args.perm_inv);

    std::vector<int64_t> new_shape;
    new_shape.reserve(output_rank);
    for (size_t axis = 0; axis < output_rank; ++axis) {
      new_shape.push_back(output_shape[static_cast<size_t>(output_perm[axis])]);
    }

    const uint8_t* new_shape_data = reinterpret_cast<const uint8_t*>(new_shape.data());
    const std::string_view new_shape_name = args.ctx.graph.AddInitializer(
        api::DataType::INT64,
        {gsl::narrow_cast<int64_t>(new_shape.size())},
        std::vector<uint8_t>(new_shape_data, new_shape_data + new_shape.size() * sizeof(int64_t)));

    const std::string_view old_shape_name = args.node.Inputs()[1];
    args.node.SetInput(1, new_shape_name);
    if (!args.ctx.graph.HasValueConsumers(old_shape_name)) {
      args.ctx.graph.RemoveInitializer(old_shape_name);
    }

    TransposeOutputs(args.ctx, args.node, InvertPerm(output_perm));
    return true;
  }

  // Fallback to default handler.
  return HandleReshape(args);
}

constexpr HandlerInfo ep_aware_reshape_handler = {&FirstInput, &EPAwareHandleReshape, /*transposes_outputs*/ false};

std::vector<size_t> QLinearConcatInputs(OptimizerCtx& ctx, api::NodeRef& node) {
  (void)ctx;
  std::vector<size_t> indices;
  size_t num_inputs = node.Inputs().size();
  for (size_t i = 2; i < num_inputs; i += 3) {
    indices.push_back(i);
  }
  return indices;
}

constexpr HandlerInfo q_linear_concat_handler = {&QLinearConcatInputs, &HandleConcat};

std::vector<size_t> QLinearBinaryOpInputs(OptimizerCtx&, api::NodeRef&) {
  // Inputs are: [A, A_scale, A_zero_point, B, B_scale, B_zero_point, C_scale, C_zero_point],
  // we want [A, B].
  return {0, 3};
}

constexpr HandlerInfo q_linear_binary_op_handler = {&QLinearBinaryOpInputs, &HandleSimpleNodeBroadcast};

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
constexpr HandlerInfo soft_hard_max_handler = {&FirstInput, &HandleSoftHardMax};
constexpr HandlerInfo contrib_quantize_dequantize_linear_handler = {&FirstInput,
                                                                    &HandleContribQuantizeDequantizeLinear};

// ORT contrib ops and special cased ONNX ops where we have EP specific handling
const HandlerMap& OrtExtendedHandlers() {
  static const HandlerMap extended_handler_map = []() {
    HandlerMap map = {
        {"MaxPool", max_pool_op_handler},
        {"Resize", ep_aware_resize_handler},
        {"Reshape", ep_aware_reshape_handler},
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
        {"com.microsoft.QLinearSoftmax", soft_hard_max_handler},
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
