// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "core/graph/constants.h"
#include <limits>
#include <utility>

namespace onnxruntime::contrib {

inline constexpr const char* kMamba2VerifyDoc = R"DOC(
Fused causal convolution, selective state-space recurrence and gated grouped RMS normalization.
This Bolt-specific operator processes one sequence per invocation, not a batch of independent sequences.
Currently requires head_dim=64, d_state=128, 1<=R<=16.
projection[R, 2*di+2*n_groups*d_state+heads] packs [gate, xBC, dt], di=heads*head_dim.
state_row is a device int32[1] selecting the starting snapshot (0<=state_row<slots).
conv_state and ssm_state must have the same slots dimension, with slots>=R.
Convolution has four taps, with BF16 SiLU output. The FP32 recurrence is
dt=max(softplus(projection_dt+dt_bias),time_step_min),
S=exp(A_negative*dt)*S+(dt*x)*B, readout=D*x+sum(C*S).
The time-step clamp propagates NaN. Softplus uses raw directly when raw>20.
Readout sums four ordered FP32 partitions of d_state, except when R=1 and slots=1,
which uses two partitions. Single-row replay with slots>1 retains the multi-row readout order.
Readout is gated by FP32 SiLU(gate), normalized per group, scaled by norm_weight and rounded to BF16.
Every processed token writes its post-token convolution and SSM snapshot to the corresponding output row.
The final/accepted state is chosen by the caller through state_row on the next invocation.
Input states may alias their corresponding outputs. Slots R..slots-1 are preserved.
xbc and readout are independently optional debug outputs; omitted intermediates use internal workspace.
If state_row is outside [0,slots), rows 0..R-1 of conv_snapshots and ssm_snapshots
and all of Y, xbc and readout are NaN; no error is raised.
No accepted-length truncation, hidden mutable state, host reads or synchronization is performed.
)DOC";

inline constexpr const char* kMamba2PrefillDoc = R"DOC(
Bulk causal prefill companion of Mamba2Verify for one sequence chunk of 1<=R<=2048 tokens.
Inputs, attributes and state layouts are identical to Mamba2Verify, and every token uses
Mamba2Verify's multi-row arithmetic: the same four-tap convolution, FP32 recurrence,
four ordered FP32 readout partitions, gating and grouped RMS normalization. Processing a chunk
therefore matches successive Mamba2Verify calls that each continue from their last row.
Unlike Mamba2Verify, no per-token snapshots are produced and slots>=1 is sufficient.
The post-chunk convolution history and SSM state are written to slot 0 of conv_state_out and
ssm_state_out; slots 1..slots-1 are preserved. A caller continues from this state with state_row=0.
The incoming state is read from the slot selected by state_row, which may be slot 0.
Input states may alias their corresponding outputs.
xbc and readout are independently optional debug outputs; omitted intermediates use internal workspace.
If state_row is outside [0,slots), slot 0 of both state outputs and all of Y, xbc and readout
are NaN; no error is raised. No host reads or synchronization are performed.
)DOC";

inline ONNX_NAMESPACE::OpSchema Mamba2StateSpaceSchema(ONNX_NAMESPACE::OpSchema schema, bool prefill) {
  using namespace ONNX_NAMESPACE;
  return schema.SetDomain(kMSDomain).SinceVersion(1).SetDoc(prefill ? kMamba2PrefillDoc : kMamba2VerifyDoc)
      .Attr("heads", "Number of heads.", AttributeProto::INT)
      .Attr("n_groups", "Number of B/C and normalization groups, dividing heads.", AttributeProto::INT)
      .Attr("head_dim", "Currently 64.", AttributeProto::INT, int64_t{64})
      .Attr("d_state", "Currently 128.", AttributeProto::INT, int64_t{128})
      .Attr("epsilon", "Grouped RMS epsilon.", AttributeProto::FLOAT, 1e-5f)
      .Attr("time_step_min", "Lower bound for softplus time step.", AttributeProto::FLOAT, 0.001f)
      .Input(0, "projection", "Packed projection [R, width].", "T")
      .Input(1, "conv_state", "Snapshots [slots, di+2*n_groups*d_state, 3].", "T")
      .Input(2, "ssm_state", "Snapshots [slots, heads, d_state, head_dim].", "F")
      .Input(3, "conv_weight", "Depthwise weights [channels,4] or [channels,1,4].", "T")
      .Input(4, "conv_bias", "Depthwise convolution bias [channels].", "T")
      .Input(5, "dt_bias", "Time-step bias [heads].", "F")
      .Input(6, "A_negative", "Negative decay coefficient [heads], not A_log.", "F")
      .Input(7, "D", "Skip coefficient [heads].", "F")
      .Input(8, "norm_weight", "Normalization weight [di].", "F")
      .Input(9, "state_row", "Starting snapshot index int32[1], on device.", "I")
      .Output(0, "Y", "Normalized mixer result [R, di].", "T")
      .Output(1, prefill ? "conv_state_out" : "conv_snapshots",
              prefill ? "Convolution states, same shape as conv_state; slot 0 receives the post-chunk state."
                      : "Post-token convolution snapshots, same shape as conv_state.",
              "T")
      .Output(2, prefill ? "ssm_state_out" : "ssm_snapshots",
              prefill ? "Recurrence states, same shape as ssm_state; slot 0 receives the post-chunk state."
                      : "Post-token recurrence snapshots, same shape as ssm_state.",
              "F")
      .Output(3, "xbc", "Optional BF16 convolution SiLU result [R, channels].", "T", OpSchema::Optional)
      .Output(4, "readout", "Optional FP32 recurrence readout before gating [R, di].", "F", OpSchema::Optional)
      .TypeConstraint("T", {"tensor(bfloat16)"}, "BF16 activation and convolution parameters.")
      .TypeConstraint("F", {"tensor(float)"}, "FP32 recurrence and normalization.")
      .TypeConstraint("I", {"tensor(int32)"}, "Device snapshot selector.")
      .TypeAndShapeInferenceFunction([prefill](InferenceContext& ctx) {
        const char* op = prefill ? "Mamba2Prefill" : "Mamba2Verify";
        const int64_t max_rows = prefill ? 2048 : 16;
        const auto heads = getAttribute(ctx, "heads", 0);
        const auto groups = getAttribute(ctx, "n_groups", 0);
        if (heads <= 0 || heads > std::numeric_limits<int>::max() / 1024 ||
            groups <= 0 || heads % groups != 0) {
          fail_shape_inference(
              op, " requires heads > 0, n_groups > 0, heads % n_groups == 0 "
                  "and heads <= INT_MAX / 1024, got heads=",
              heads, ", n_groups=", groups);
        }
        if (getAttribute(ctx, "head_dim", 64) != 64 || getAttribute(ctx, "d_state", 128) != 128) {
          fail_shape_inference("Invalid ", op, " attributes");
        }
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        for (size_t input = 0; input < ctx.getNumInputs(); ++input) {
          if (!hasInputShape(ctx, input)) continue;
          const auto& shape = getInputShape(ctx, input);
          const int rank = shape.dim_size();
          const bool valid = input == 0 ? rank == 2 : input == 1 ? rank == 3
                                                  : input == 2   ? rank == 4
                                                  : input == 3   ? (rank == 2 || rank == 3)
                                                                 : rank == 1;
          if (!valid) fail_shape_inference("Invalid input rank at ", input);
        }
        const int64_t di = heads * 64, channels = di + 2 * groups * 128;
        const auto check_dim = [&ctx](size_t input, int axis, int64_t expected) {
          if (!hasInputShape(ctx, input)) return;
          const auto& dim = getInputShape(ctx, input).dim(axis);
          if (dim.has_dim_value() && dim.dim_value() != expected) {
            fail_shape_inference("Invalid input dimension at ", input, " axis ", axis, ": expected ", expected);
          }
        };
        check_dim(0, 1, di + channels + heads);
        check_dim(1, 1, channels);
        check_dim(1, 2, 3);
        check_dim(2, 1, heads);
        check_dim(2, 2, 128);
        check_dim(2, 3, 64);
        if (hasInputShape(ctx, 3)) {
          const int rank = getInputShape(ctx, 3).dim_size();
          check_dim(3, 0, channels);
          if (rank == 3) check_dim(3, 1, 1);
          check_dim(3, rank - 1, 4);
        }
        check_dim(4, 0, channels);
        for (size_t input : {size_t{5}, size_t{6}, size_t{7}}) check_dim(input, 0, heads);
        check_dim(8, 0, di);
        check_dim(9, 0, 1);
        for (size_t input : {size_t{1}, size_t{2}}) {
          if (!hasInputShape(ctx, input)) continue;
          const auto& slots = getInputShape(ctx, input).dim(0);
          if (!slots.has_dim_value()) continue;
          if (slots.dim_value() < 1 ||
              (!prefill && hasInputShape(ctx, 0) && getInputShape(ctx, 0).dim(0).has_dim_value() &&
               slots.dim_value() < getInputShape(ctx, 0).dim(0).dim_value())) {
            fail_shape_inference(prefill ? "Snapshot slots must be >= 1" : "Snapshot slots must be >= R");
          }
        }
        if (hasInputShape(ctx, 1) && hasInputShape(ctx, 2)) {
          const auto& conv_slots = getInputShape(ctx, 1).dim(0);
          const auto& ssm_slots = getInputShape(ctx, 2).dim(0);
          if (conv_slots.has_dim_value() && ssm_slots.has_dim_value() &&
              conv_slots.dim_value() != ssm_slots.dim_value()) {
            fail_shape_inference("Snapshot slot dimensions must match");
          }
        }
        if (hasInputShape(ctx, 0)) {
          const auto& shape = getInputShape(ctx, 0);
          if (shape.dim(0).has_dim_value() &&
              (shape.dim(0).dim_value() < 1 || shape.dim(0).dim_value() > max_rows)) {
            fail_shape_inference("Rows must be in [1,", max_rows, "]");
          }
          appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, 0);
          appendDim(getOutputShape(ctx, 0), heads * 64);
        }
        for (size_t output : {size_t{1}, size_t{2}}) {
          propagateElemTypeFromInputToOutput(ctx, output, output);
          if (hasInputShape(ctx, output)) propagateShapeFromInputToOutput(ctx, output, output);
        }
        for (size_t output : {size_t{3}, size_t{4}}) {
          if (ctx.getNumOutputs() <= output || ctx.getOutputType(output) == nullptr) continue;
          propagateElemTypeFromInputToOutput(ctx, output == 3 ? 0 : 2, output);
          if (hasInputShape(ctx, 0)) {
            appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, output, 0);
            appendDim(getOutputShape(ctx, output), output == 3 ? channels : di);
          }
        }
      });
}

inline ONNX_NAMESPACE::OpSchema Mamba2VerifySchema(ONNX_NAMESPACE::OpSchema schema) {
  return Mamba2StateSpaceSchema(std::move(schema), false);
}

inline ONNX_NAMESPACE::OpSchema Mamba2PrefillSchema(ONNX_NAMESPACE::OpSchema schema) {
  return Mamba2StateSpaceSchema(std::move(schema), true);
}

}  // namespace onnxruntime::contrib
