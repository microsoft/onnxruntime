// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/constants.h"

namespace onnxruntime {
namespace contrib {

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::InferenceContext;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;
using ONNX_NAMESPACE::TypeProto;

void RegisterCollectiveOps() {
  ONNX_CONTRIB_OPERATOR_SCHEMA(AllReduce)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "input", "tensors to be reduced", "T", OpSchema::Variadic)
      .Output(0, "output", "reduced tensors", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(AllGather)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("axis",
            "the axis to gather on.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
      .Attr("group_size",
            "total size in the group that need to be gathered.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
      .Input(0, "input", "tensors to be sent", "T", OpSchema::Variadic)
      .Output(0, "output", "gathered tensors", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto group_size = getAttribute(ctx, "group_size", 1);
        auto axis = getAttribute(ctx, "axis", 0);
        assert(group_size >= static_cast<int64_t>(1));
        // propagate type for output
        propagateElemTypeFromInputToOutput(ctx, 0, 0);

        // propagate shape for output.
        // output shape is [group_size * input_shape[0], ...]
        auto output_type = ctx.getOutputType(0);
        auto input_type = ctx.getInputType(0);
        if (hasShape(*input_type)) {
          auto shape = input_type->tensor_type().shape();
          auto dim = shape.dim(static_cast<int>(axis)) * group_size;
          *shape.mutable_dim(static_cast<int>(axis)) = dim;
          *output_type->mutable_tensor_type()->mutable_shape() = shape;
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(AllToAll)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("group_size",
            "total size in the group that need to participate.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
      .Input(0, "input", "tensors to be sent", "T", OpSchema::Variadic)
      .Output(0, "output", "collected tensors", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });
}

}  // namespace contrib
}  // namespace onnxruntime
