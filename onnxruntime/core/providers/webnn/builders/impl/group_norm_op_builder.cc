// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

// Builds ai.onnx GroupNormalization (opset 18/21) and com.microsoft GroupNorm / SkipGroupNorm.
//
// WebNN has no dedicated group-normalization ops, so we decompose it into the canonical
// instanceNormalization pattern. This is deliberately shaped so that a backend able to
// recognize the mean-variance-normalization subgraph can re-fuse it back into a single native
// group normalization. WebNN instanceNormalization is 4D-only, so we emit the 4D variant:
//
//   input(NCHW) -> reshape{N,G,(C/G)*H*W,1} -> instanceNormalization(no scale/bias, eps)
//               -> reshape{N,C,H,W} -> mul(gamma[1,C,1,1]) -> add(beta[1,C,1,1])
//
// The reshape -> instanceNormalization(normalizes over {2,3}) -> reshape -> mul(gamma) ->
// add(beta) chain is the exact subgraph a backend would match. Everything op-specific (NHWC
// transposes, the SkipGroupNorm residual add, the optional S output, and SiLU activation)
// lands outside that subgraph and therefore does not block the re-fusion.
class GroupNormOpBuilder : public BaseOpBuilder {
 public:
  // SkipGroupNorm's residual bias (input 4) is optional.
  GroupNormOpBuilder() : BaseOpBuilder(/*allow_empty_tensor_as_input=*/true) {}

  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer&, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

Status GroupNormOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                 const Node& node,
                                                 const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();
  emscripten::val wnn_builder = model_builder.GetBuilder();

  const bool is_contrib = op_type == "GroupNorm" || op_type == "SkipGroupNorm";
  const bool is_skip = op_type == "SkipGroupNorm";

  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");

  NodeAttrHelper helper(node);
  const float epsilon = helper.Get("epsilon", 1e-05f);
  // Standard GroupNormalization is NCHW; com.microsoft GroupNorm/SkipGroupNorm default to NHWC.
  const bool channels_last = is_contrib && helper.Get("channels_last", static_cast<int64_t>(1)) != 0;
  const int64_t groups = helper.Get(is_contrib ? "groups" : "num_groups", static_cast<int64_t>(0));

  // WebNN instanceNormalization is 4D-only, but group normalization is rank-agnostic: we always
  // normalize on a 4D [N, G, merged, 1] tensor, so any input rank >= 3 is supported by reshaping
  // in and out (mirrors how instanceNormalization handles non-4D input in normalization_op_builder).
  const size_t rank = input_shape.size();
  const size_t channel_axis = channels_last ? rank - 1 : 1;  // NHWC: last; NCHW: 1.
  const uint32_t batch = SafeInt<uint32_t>(input_shape[0]);
  const uint32_t channels = SafeInt<uint32_t>(input_shape[channel_axis]);
  const uint32_t group_count = SafeInt<uint32_t>(groups);

  // merged = (C / G) * product(spatial dims). Spatial = every dim except batch and channel.
  uint32_t spatial = 1;
  for (size_t i = 1; i < rank; ++i) {
    if (i != channel_axis) {
      spatial *= SafeInt<uint32_t>(input_shape[i]);
    }
  }
  const uint32_t merged = (channels / group_count) * spatial;

  // NCHW shape the normalization runs on (channel at index 1). We reshape/affine/reshape in this
  // layout, then transpose back for channels_last. Spatial dims keep their original order.
  std::vector<uint32_t> nchw_shape{batch, channels};
  nchw_shape.reserve(rank);
  for (size_t i = 1; i < rank; ++i) {
    if (i != channel_axis) {
      nchw_shape.push_back(SafeInt<uint32_t>(input_shape[i]));
    }
  }

  // Per-channel affine parameters broadcast as [1, C, 1, ..., 1] over the NCHW tensor.
  std::vector<uint32_t> channel_broadcast_shape(rank, 1);
  channel_broadcast_shape[1] = channels;

  // Reused by every decomposed op that only needs a "label". Ops with extra options
  // (instanceNormalization's epsilon, transpose's permutation) use their own option object.
  emscripten::val common_options = emscripten::val::object();

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());

  // SkipGroupNorm: s = x + skip + bias, then group-normalize s. Optional output S = s.
  if (is_skip) {
    emscripten::val skip = model_builder.GetOperand(input_defs[3]->Name());
    std::vector<int64_t> skip_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[3], skip_shape, logger), "Cannot get skip shape");
    // Skip may be (N, C); reshape it to broadcast over the spatial dims in the input's layout.
    if (skip_shape.size() == 2) {
      std::vector<uint32_t> skip_broadcast_shape(rank, 1);
      skip_broadcast_shape[0] = batch;
      skip_broadcast_shape[channel_axis] = channels;
      common_options.set("label", node.Name() + "_reshape_skip");
      skip = wnn_builder.call<emscripten::val>("reshape", skip, emscripten::val::array(skip_broadcast_shape),
                                               common_options);
    }
    common_options.set("label", node.Name() + "_add_skip");
    input = wnn_builder.call<emscripten::val>("add", input, skip, common_options);

    // Optional residual bias (input 4), shape (C).
    if (TensorExists(input_defs, 4)) {
      emscripten::val bias = model_builder.GetOperand(input_defs[4]->Name());
      // (C) broadcasts over the last axis in NHWC; reshape to [1, C, 1, ..., 1] for NCHW.
      if (!channels_last) {
        common_options.set("label", node.Name() + "_reshape_residual_bias");
        bias = wnn_builder.call<emscripten::val>("reshape", bias, emscripten::val::array(channel_broadcast_shape),
                                                 common_options);
      }
      common_options.set("label", node.Name() + "_add_residual_bias");
      input = wnn_builder.call<emscripten::val>("add", input, bias, common_options);
    }

    // Optional output S (the residual sum) — output index 1.
    if (TensorExists(output_defs, 1)) {
      emscripten::val skip_sum = input;
      model_builder.AddOperand(output_defs[1]->Name(), skip_sum);
    }
  }

  // Bring the node input into NCHW so the channel dimension is at index 1 (re-fusion input).
  // NHWC -> NCHW permutation: [0, rank-1, 1, 2, ..., rank-2].
  if (channels_last) {
    std::vector<uint32_t> to_nchw_perm{0, SafeInt<uint32_t>(rank - 1)};
    to_nchw_perm.reserve(rank);
    for (size_t i = 1; i + 1 < rank; ++i) {
      to_nchw_perm.push_back(SafeInt<uint32_t>(i));
    }
    emscripten::val transpose_options = emscripten::val::object();
    transpose_options.set("label", node.Name() + "_transpose_nhwc_to_nchw");
    transpose_options.set("permutation", emscripten::val::array(to_nchw_perm));
    input = wnn_builder.call<emscripten::val>("transpose", input, transpose_options);
  }

  // Reshape to [N, G, (C/G)*spatial, 1] (4D MVN variant with trailing unit dimension).
  std::vector<uint32_t> pre_mvn_shape{batch, group_count, merged, 1};
  common_options.set("label", node.Name() + "_reshape_pre_mvn");
  emscripten::val grouped = wnn_builder.call<emscripten::val>(
      "reshape", input, emscripten::val::array(pre_mvn_shape), common_options);

  // instanceNormalization normalizes over spatial dims {2,3} per group — MVN with no scale/bias.
  // Do NOT set scale/bias: a spurious mul/add would break a backend's optional instance-scale match.
  emscripten::val inorm_options = emscripten::val::object();
  inorm_options.set("label", node.Name() + "_instanceNormalization");
  inorm_options.set("epsilon", epsilon);
  emscripten::val normalized =
      wnn_builder.call<emscripten::val>("instanceNormalization", grouped, inorm_options);

  // Reshape back to the NCHW input shape (must equal the re-fusion input shape).
  common_options.set("label", node.Name() + "_reshape_post_mvn");
  emscripten::val restored = wnn_builder.call<emscripten::val>(
      "reshape", normalized, emscripten::val::array(nchw_shape), common_options);

  // Reshape a per-channel (C) or per-group (num_groups) affine parameter to [1, C, 1, ..., 1] so it
  // broadcasts over the NCHW tensor. Per-group params (ai.onnx GroupNormalization opset 18) are
  // expanded to per-channel by repeating each group value C/groups times.
  const auto to_channel_broadcast = [&](emscripten::val param, const std::vector<int64_t>& param_shape,
                                        const std::string& tag) -> emscripten::val {
    if (param_shape.size() == 1 && SafeInt<uint32_t>(param_shape[0]) == channels) {
      common_options.set("label", node.Name() + tag + "_reshape");
      return wnn_builder.call<emscripten::val>("reshape", param, emscripten::val::array(channel_broadcast_shape),
                                               common_options);
    }
    // Per-group: (G) -> (G, 1) -> expand (G, C/G) -> [1, C, 1, ..., 1].
    common_options.set("label", node.Name() + tag + "_reshape_group");
    param = wnn_builder.call<emscripten::val>(
        "reshape", param, emscripten::val::array(std::vector<uint32_t>{group_count, 1}), common_options);

    common_options.set("label", node.Name() + tag + "_expand_group");
    param = wnn_builder.call<emscripten::val>(
        "expand", param, emscripten::val::array(std::vector<uint32_t>{group_count, channels / group_count}),
        common_options);

    common_options.set("label", node.Name() + tag + "_reshape_channel");
    return wnn_builder.call<emscripten::val>("reshape", param, emscripten::val::array(channel_broadcast_shape),
                                             common_options);
  };

  // Per-channel affine: y = normalized * gamma + beta.
  std::vector<int64_t> gamma_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], gamma_shape, logger), "Cannot get gamma shape");
  emscripten::val gamma =
      to_channel_broadcast(model_builder.GetOperand(input_defs[1]->Name()), gamma_shape, "_gamma");
  common_options.set("label", node.Name() + "_mul_gamma");
  emscripten::val scaled = wnn_builder.call<emscripten::val>("mul", restored, gamma, common_options);

  std::vector<int64_t> beta_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[2], beta_shape, logger), "Cannot get beta shape");
  emscripten::val beta = to_channel_broadcast(model_builder.GetOperand(input_defs[2]->Name()), beta_shape, "_beta");
  common_options.set("label", node.Name() + "_add_beta");
  emscripten::val output = wnn_builder.call<emscripten::val>("add", scaled, beta, common_options);

  // Optional SiLU activation (com.microsoft, activation == 1): y = y * sigmoid(y).
  if (is_contrib && helper.Get("activation", static_cast<int64_t>(0)) == 1) {
    common_options.set("label", node.Name() + "_sigmoid");
    emscripten::val sigmoid = wnn_builder.call<emscripten::val>("sigmoid", output, common_options);
    common_options.set("label", node.Name() + "_silu_mul");
    output = wnn_builder.call<emscripten::val>("mul", output, sigmoid, common_options);
  }

  // Transpose back to NHWC. NCHW -> NHWC permutation: [0, 2, 3, ..., rank-1, 1].
  if (channels_last) {
    std::vector<uint32_t> to_nhwc_perm{0};
    to_nhwc_perm.reserve(rank);
    for (size_t i = 2; i < rank; ++i) {
      to_nhwc_perm.push_back(SafeInt<uint32_t>(i));
    }
    to_nhwc_perm.push_back(1);
    emscripten::val transpose_options = emscripten::val::object();
    transpose_options.set("label", node.Name() + "_transpose_nchw_to_nhwc");
    transpose_options.set("permutation", emscripten::val::array(to_nhwc_perm));
    output = wnn_builder.call<emscripten::val>("transpose", output, transpose_options);
  }

  model_builder.AddOperand(output_defs[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool GroupNormOpBuilder::IsOpSupportedImpl(const GraphViewer&,
                                           const Node& node,
                                           const WebnnDeviceType /* device_type */,
                                           const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();
  const bool is_contrib = op_type == "GroupNorm" || op_type == "SkipGroupNorm";
  const bool is_skip = op_type == "SkipGroupNorm";
  NodeAttrHelper helper(node);

  if (input_defs.size() < 3) {
    LOGS(logger, VERBOSE) << op_type << " requires at least three inputs (X, scale/gamma, bias/beta).";
    return false;
  }
  if (is_skip && !TensorExists(input_defs, 3)) {
    LOGS(logger, VERBOSE) << op_type << " requires the skip input.";
    return false;
  }

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return false;
  }
  // Group normalization needs a batch, a channel and at least one spatial dim. It is normalized on
  // a 4D [N, G, merged, 1] tensor internally, so any rank >= 3 is supported (see AddToModelBuilderImpl).
  if (input_shape.size() < 3) {
    LOGS(logger, VERBOSE) << op_type << " requires input rank >= 3.";
    return false;
  }

  const bool channels_last = is_contrib && helper.Get("channels_last", static_cast<int64_t>(1)) != 0;
  const int64_t channels = channels_last ? input_shape.back() : input_shape[1];
  const int64_t groups = helper.Get(is_contrib ? "groups" : "num_groups", static_cast<int64_t>(0));
  if (groups <= 0 || channels % groups != 0) {
    LOGS(logger, VERBOSE) << op_type << " requires groups > 0 and channels divisible by groups.";
    return false;
  }

  // scale/gamma and bias/beta must be 1D, either per-channel (C) or per-group (num_groups).
  for (size_t i = 1; i <= 2; ++i) {
    std::vector<int64_t> param_shape;
    if (!GetShape(*input_defs[i], param_shape, logger)) {
      return false;
    }
    if (param_shape.size() != 1 || (param_shape[0] != channels && param_shape[0] != groups)) {
      LOGS(logger, VERBOSE) << op_type << " scale/bias must be 1D of size channels or groups.";
      return false;
    }
  }

  if (is_contrib) {
    const int64_t activation = helper.Get("activation", static_cast<int64_t>(0));
    if (activation != 0 && activation != 1) {
      LOGS(logger, VERBOSE) << op_type << " only supports activation 0 (None) or 1 (SiLU).";
      return false;
    }
  }

  if (is_skip) {
    if (TensorExists(input_defs, 4)) {
      std::vector<int64_t> residual_bias_shape;
      if (!GetShape(*input_defs[4], residual_bias_shape, logger) ||
          residual_bias_shape.size() != 1 || residual_bias_shape[0] != channels) {
        LOGS(logger, VERBOSE) << op_type << " residual bias must be 1D of size channels.";
        return false;
      }
    }
    if (output_defs.size() > 2) {
      LOGS(logger, VERBOSE) << op_type << " output count must not exceed 2.";
      return false;
    }
    std::vector<int64_t> skip_shape;
    if (!GetShape(*input_defs[3], skip_shape, logger)) {
      return false;
    }
    // Skip is either (N, C) or the full input shape.
    if (skip_shape.size() != 2 && skip_shape.size() != input_shape.size()) {
      LOGS(logger, VERBOSE) << op_type << " skip must be 2D or match the input rank.";
      return false;
    }
  } else if (output_defs.size() != 1) {
    LOGS(logger, VERBOSE) << op_type << " output count must be one.";
    return false;
  }

  return true;
}

bool GroupNormOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                                const emscripten::val& wnn_limits,
                                                const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string_view op_type = node.OpType();

  std::vector<int32_t> input_types;
  for (size_t i = 0; i < input_defs.size(); ++i) {
    if (TensorExists(input_defs, i)) {
      int32_t input_type;
      if (!GetType(*input_defs[i], input_type, logger)) {
        return false;
      }
      input_types.push_back(input_type);
    }
  }
  if (!AreDataTypesSame(op_type, input_types, logger)) {
    return false;
  }

  // Check the input data type is supported by every decomposed WebNN op (see decomposed_op_map).
  for (const std::string_view decomposed_op_type : decomposed_op_map.at(op_type)) {
    const std::string_view webnn_op_type = GetWebNNOpType(decomposed_op_type);
    const std::string_view webnn_input_name = GetWebNNOpFirstInputName(decomposed_op_type);
    if (!IsDataTypeSupportedByWebNNOp(
            op_type, webnn_op_type, input_types[0], wnn_limits, webnn_input_name, "input", logger)) {
      return false;
    }
  }

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return false;
  }
  // instanceNormalization always runs on the 4D [N, G, merged, 1] tensor, while the per-channel
  // mul/add run at the input rank.
  return IsRankSupportedByWebNNOp(wnn_limits, "instanceNormalization", "input", 4, node.Name(), logger) &&
         IsRankSupportedByWebNNOp(wnn_limits, "mul", "a", input_shape.size(), node.Name(), logger) &&
         IsRankSupportedByWebNNOp(wnn_limits, "add", "a", input_shape.size(), node.Name(), logger);
}

bool GroupNormOpBuilder::HasSupportedOutputsImpl(const Node& node,
                                                 const emscripten::val& wnn_limits,
                                                 const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();
  const std::string_view op_type = node.OpType();
  int32_t output_type = 0;
  if (!GetType(*output_defs[0], output_type, logger)) {
    return false;
  }

  for (const std::string_view decomposed_op_type : decomposed_op_map.at(op_type)) {
    const std::string_view webnn_op_type = GetWebNNOpType(decomposed_op_type);
    if (!IsDataTypeSupportedByWebNNOp(op_type, webnn_op_type, output_type, wnn_limits, "output", "output", logger)) {
      return false;
    }
  }
  return true;
}

void CreateGroupNormOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  constexpr static std::string_view op_types[] =
      {
          "GroupNormalization",
          "GroupNorm",
          "SkipGroupNorm",
      };

  op_registrations.builders.push_back(std::make_unique<GroupNormOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
