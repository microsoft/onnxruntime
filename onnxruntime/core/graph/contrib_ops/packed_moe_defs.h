// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "core/graph/constants.h"

namespace onnxruntime::contrib {
inline ONNX_NAMESPACE::OpSchema MoEPackedIQ2XXSSchema(ONNX_NAMESPACE::OpSchema schema) {
  using namespace ONNX_NAMESPACE;
  return schema.SetDomain(kMSDomain).SinceVersion(1).SetDoc(R"DOC(
Fused routed two-layer IQ2_XXS expert MLP, with expert sharing across up to eight rows.
X[R,1024]; P=R*topk<=128. Expert indices must be in [0,E) and distinct within each token row.
FC1 packed bytes [E,I,32,8], FC2 [E,1024,I/32,8], I positive and divisible by 256.
Each eight-byte record contains four grid indices, four parity-extended seven-bit signs and a four-bit scale.
Scales [E,I,4] and [E,1024,I/256] contain raw IEEE FP16 bits, including when tensor storage is typed BF16.
Each complete signed grid * FP16-super-scale * (0.5+record-scale)/4 operand is rounded to BF16, not the scale alone.
BF16 MMA with FP32 accumulation uses fixed two-tile, four/eight-warp FC1/FC2 reductions.
H=BF16(relu(BF16(X*W1^T))^2); expert_output=float(BF16(H*W2^T)).
Y=BF16(sum_k(weights_k*expert_output_k)), accumulated in routing order.
metadata[0]=unique expert count U, metadata[1..U]=first-occurrence expert IDs.
metadata[1+P+u*8+r] is routed pair index or -1. Unused unique-ID slots are -1.
No native-GGUF activation quantization, host routing, environment tuning or hidden mutable state is used.
)DOC")
      .Attr("topk", "Number of experts selected per row.", AttributeProto::INT)
      .Input(0, "X", "BF16 activations [R,1024].", "T")
      .Input(1, "indices", "Unique-per-row expert indices [R,topk].", "I")
      .Input(2, "weights", "FP32 routing weights [R,topk], need not sum to one.", "F")
      .Input(3, "fc1", "Packed FC1 records [E,I,32,8].", "Q")
      .Input(4, "scale1", "FP16 scale bits [E,I,4].", "S")
      .Input(5, "fc2", "Packed FC2 records [E,1024,I/32,8].", "Q")
      .Input(6, "scale2", "FP16 scale bits [E,1024,I/256].", "S")
      .Output(0, "Y", "Weighted BF16 expert sum [R,1024].", "T")
      .Output(1, "hidden", "BF16 squared-ReLU intermediate [P,I].", "T")
      .Output(2, "expert_output", "BF16-rounded expert projections held in FP32 [P,1024].", "F")
      .Output(3, "metadata", "Device expert grouping buffer [1+9*P].", "I")
      .TypeConstraint("T", {"tensor(bfloat16)"}, "BF16 activations.")
      .TypeConstraint("S", {"tensor(float16)", "tensor(bfloat16)"},
                      "Raw FP16 bits; BF16 storage is compatibility-only.")
      .TypeConstraint("F", {"tensor(float)"}, "FP32 routing and partial results.")
      .TypeConstraint("I", {"tensor(int32)"}, "Device routing metadata.")
      .TypeConstraint("Q", {"tensor(uint8)"}, "Packed IQ2 records.")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        const auto topk = getAttribute(ctx, "topk", 0);
        if (topk <= 0 || topk > 32) fail_shape_inference("topk must be in [1,32]");
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        propagateElemTypeFromInputToOutput(ctx, 0, 1);
        propagateElemTypeFromInputToOutput(ctx, 2, 2);
        propagateElemTypeFromInputToOutput(ctx, 1, 3);
        if (!hasInputShape(ctx, 0)) return;
        const auto& x = getInputShape(ctx, 0);
        if (x.dim_size() != 2) fail_shape_inference("X must have rank 2");
        appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, 0);
        appendDim(getOutputShape(ctx, 0), 1024);
        if (hasInputShape(ctx, 3) && getInputShape(ctx, 3).dim_size() != 4) {
          fail_shape_inference("FC1 must have rank 4");
        }
        if (x.dim(0).has_dim_value()) {
          const int64_t rows = x.dim(0).dim_value();
          if (rows < 1 || rows > 8) fail_shape_inference("Invalid row count");
          if (rows * topk > 128) fail_shape_inference("Invalid pair count");
        }
        const auto pairs = x.dim(0) * topk;
        if (hasInputShape(ctx, 3)) {
          updateOutputShape(ctx, 1, {pairs, getInputShape(ctx, 3).dim(1)});
        }
        updateOutputShape(ctx, 2, {pairs});
        appendDim(getOutputShape(ctx, 2), 1024);
        updateOutputShape(ctx, 3, {pairs * 9 + 1});
      });
}
}  // namespace onnxruntime::contrib
