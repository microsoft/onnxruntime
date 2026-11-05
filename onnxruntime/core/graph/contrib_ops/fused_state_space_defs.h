// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "core/graph/constants.h"
#include <limits>

namespace onnxruntime::contrib {

inline ONNX_NAMESPACE::OpSchema Mamba2VerifySchema(ONNX_NAMESPACE::OpSchema schema) {
  using namespace ONNX_NAMESPACE;
  return schema.SetDomain(kMSDomain).SinceVersion(1).SetDoc(R"DOC(
Fused causal convolution, selective state-space recurrence and gated grouped RMS normalization.
Local provisional contract: head_dim=64, d_state=128, 1<=R<=16.
projection[R, 2*di+2*n_groups*d_state+heads] packs [gate, xBC, dt], di=heads*head_dim.
state_row is a device int32[1] selecting the starting snapshot (0<=state_row<slots).
Convolution has four taps, with BF16 SiLU output. The FP32 recurrence is
dt=max(softplus(projection_dt+dt_bias),time_step_min),
S=exp(A_negative*dt)*S+(dt*x)*B, readout=D*x+sum(C*S).
Readout is gated by FP32 SiLU(gate), normalized per group, scaled by norm_weight and rounded to BF16.
Every processed token writes its post-token convolution and SSM snapshot to the corresponding output row.
The final/accepted state is chosen by the caller through state_row on the next invocation.
Input states may alias their corresponding outputs. Slots R..slots-1 are preserved.
No accepted-length truncation, hidden mutable state, host reads or synchronization is performed.
)DOC")
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
      .Output(1, "conv_snapshots", "Post-token convolution snapshots, same shape as conv_state.", "T")
      .Output(2, "ssm_snapshots", "Post-token recurrence snapshots, same shape as ssm_state.", "F")
      .Output(3, "xbc", "BF16 convolution SiLU result [R, channels].", "T")
      .Output(4, "readout", "FP32 recurrence readout before gating [R, di].", "F")
      .TypeConstraint("T", {"tensor(bfloat16)"}, "BF16 activation and convolution parameters.")
      .TypeConstraint("F", {"tensor(float)"}, "FP32 recurrence and normalization.")
      .TypeConstraint("I", {"tensor(int32)"}, "Device snapshot selector.")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        const auto heads = getAttribute(ctx, "heads", 0);
        const auto groups = getAttribute(ctx, "n_groups", 0);
        if (heads <= 0 || heads > std::numeric_limits<int>::max() / 1024 ||
            groups <= 0 || heads % groups != 0 ||
            getAttribute(ctx, "head_dim", 64) != 64 || getAttribute(ctx, "d_state", 128) != 128) {
          fail_shape_inference("Invalid Mamba2Verify attributes");
        }
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        for (size_t input = 0; input < ctx.getNumInputs(); ++input) {
          if (!hasInputShape(ctx, input)) continue;
          const auto& shape = getInputShape(ctx, input);
          if (input == 3 && (shape.dim_size() == 2 || shape.dim_size() == 3)) continue;
          const int rank = input == 0 ? 2 : input == 1 ? 3
                                        : input == 2   ? 4
                                                       : 1;
          if (shape.dim_size() != rank) fail_shape_inference("Invalid input rank at ", input);
        }
        if (hasInputShape(ctx, 0)) {
          const auto& shape = getInputShape(ctx, 0);
          if (shape.dim(0).has_dim_value() && (shape.dim(0).dim_value() < 1 || shape.dim(0).dim_value() > 16)) {
            fail_shape_inference("Rows must be in [1,16]");
          }
          appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, 0);
          appendDim(getOutputShape(ctx, 0), heads * 64);
        }
        for (size_t output : {size_t{1}, size_t{2}}) {
          propagateElemTypeFromInputToOutput(ctx, output, output);
          if (hasInputShape(ctx, output)) propagateShapeFromInputToOutput(ctx, output, output);
        }
        propagateElemTypeFromInputToOutput(ctx, 0, 3);
        propagateElemTypeFromInputToOutput(ctx, 2, 4);
        if (hasInputShape(ctx, 0)) {
          appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 3, 0);
          appendDim(getOutputShape(ctx, 3), heads * 64 + 2 * groups * 128);
          appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 4, 0);
          appendDim(getOutputShape(ctx, 4), heads * 64);
        }
      });
}

inline ONNX_NAMESPACE::OpSchema BiasCorrectedMoERouteSchema(ONNX_NAMESPACE::OpSchema schema) {
  using namespace ONNX_NAMESPACE;
  return schema.SetDomain(kMSDomain).SinceVersion(1).SetDoc(
                                                        "FP32 sigmoid(X*W^T); top-k by score+bias (ties prefer lower index). "
                                                        "Weights use unbiased scores normalized by sum+1e-20 and multiplied by scale. "
                                                        "BF16 X[R,K], W[E,K]; K positive and divisible by 256, topk<=min(E,32), E<=512.")
      .Attr("topk", "Number of selected experts.", AttributeProto::INT)
      .Attr("scale", "Multiplier after normalization.", AttributeProto::FLOAT, 1.0f)
      .Input(0, "X", "BF16 input [R,K].", "T")
      .Input(1, "W", "BF16 router weight [E,K].", "T")
      .Input(2, "bias", "FP32 correction [E], used only for selection.", "F")
      .Output(0, "indices", "Ordered selected experts [R,topk].", "I")
      .Output(1, "weights", "Normalized unbiased scores [R,topk].", "F")
      .Output(2, "scores", "Unbiased sigmoid scores [R,E].", "F")
      .TypeConstraint("T", {"tensor(bfloat16)"}, "BF16 input.")
      .TypeConstraint("F", {"tensor(float)"}, "FP32 scores.")
      .TypeConstraint("I", {"tensor(int32)"}, "Expert indices.")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        const auto topk = getAttribute(ctx, "topk", 0);
        if (topk <= 0 || topk > 32) fail_shape_inference("topk must be in [1,32]");
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto_DataType_INT32);
        propagateElemTypeFromInputToOutput(ctx, 2, 1);
        propagateElemTypeFromInputToOutput(ctx, 2, 2);
        if (!hasInputShape(ctx, 0)) return;
        if (getInputShape(ctx, 0).dim_size() != 2) fail_shape_inference("X must have rank 2");
        for (size_t output : {size_t{0}, size_t{1}}) {
          appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, output, 0);
          appendDim(getOutputShape(ctx, output), topk);
        }
        if (hasInputShape(ctx, 2)) {
          if (getInputShape(ctx, 2).dim_size() != 1) fail_shape_inference("bias must have rank 1");
          updateOutputShape(ctx, 2, {getInputShape(ctx, 0).dim(0), getInputShape(ctx, 2).dim(0)});
        }
      });
}

inline ONNX_NAMESPACE::OpSchema ReluSquaredSchema(ONNX_NAMESPACE::OpSchema schema) {
  using namespace ONNX_NAMESPACE;
  return schema.SetDomain(kMSDomain).SinceVersion(1).SetDoc("Y=BF16(fmax(float(X),0)^2), with one final BF16 rounding. fmax maps NaN to zero.").Input(0, "X", "BF16 tensor.", "T").Output(0, "Y", "Same shape as X.", "T").TypeConstraint("T", {"tensor(bfloat16)"}, "BF16 input and output.").TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput);
}
}  // namespace onnxruntime::contrib
