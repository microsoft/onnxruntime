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

#define ONNX_DISTRIBUTED_OPERATOR_SCHEMA(name)                                               \
  ONNX_CONTRIB_OPERATOR_SCHEMA(name)                                                         \
      .Attr("input_device_mesh_elements",                                                    \
            "device_mesh_elements[i] defines the device mesh's value for the i-th input. "   \
            "E.g., device_mesh_elements=[\"[0, 1]\", \"[0, 1]\"] means the 1st and the 2nd " \
            " inputs are stored on the 0-th and the 1st devices, respectively.",             \
            AttributeProto::STRINGS)                                                         \
      .Attr("input_device_mesh_shapes",                                                      \
            "device_mesh_shape[i] defines the device mesh's shape for the i-th input.",      \
            AttributeProto::STRINGS)                                                         \
      .Attr("input_shard_specs",                                                             \
            "The sharding spec of inputs. "                                                  \
            "E.g., if input_shard_specs[i] is \"RRR\", the i-th input is a 3-D tensor "      \
            "replicated across all devices.",                                                \
            AttributeProto::STRINGS)                                                         \
      .Attr("output_device_mesh_elements",                                                   \
            "Similar to input_device_mesh_elments but for outputs.",                         \
            AttributeProto::STRINGS)                                                         \
      .Attr("output_device_mesh_shapes",                                                     \
            "Similar to input_device_mesh_shapes but for outputs.",                          \
            AttributeProto::STRINGS)                                                         \
      .Attr("output_shard_specs",                                                            \
            "Similar to input_shard_specs but for outputs.",                                 \
            AttributeProto::STRINGS)

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
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(bool)"},
          "Constrain to bool, float, float16 and double tensors.")
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
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(bool)"},
          "Constrain to bool, float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_DISTRIBUTED_OPERATOR_SCHEMA(DistributedMatMul)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "A", "N-dimensional matrix A", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(1, "B", "N-dimensional matrix B", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Output(0, "Y", "Matrix multiply results from A * B", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .TypeConstraint(
          "T",
          {
              "tensor(float16)",
              "tensor(float)",
          },
          "Constrain input and output types to float tensors.");

  ONNX_DISTRIBUTED_OPERATOR_SCHEMA(DistributedSlice)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(
          0,
          "data",
          "Tensor of data to extract slices from.",
          "T",
          OpSchema::Single,
          true,
          1,
          OpSchema::Differentiable)
      .Input(
          1,
          "starts",
          "1-D tensor of starting indices of corresponding axis in `axes`",
          "Tind",
          OpSchema::Single,
          true,
          1,
          OpSchema::NonDifferentiable)
      .Input(
          2,
          "ends",
          "1-D tensor of ending indices (exclusive) of corresponding axis in `axes`",
          "Tind",
          OpSchema::Single,
          true,
          1,
          OpSchema::NonDifferentiable)
      .Input(
          3,
          "axes",
          "1-D tensor of axes that `starts` and `ends` apply to. Negative value means counting dimensions "
          "from the back. Accepted range is [-r, r-1] where r = rank(data). Behavior is undefined if an "
          "axis is repeated.",
          "Tind",
          OpSchema::Optional,
          true,
          1,
          OpSchema::NonDifferentiable)
      .Input(
          4,
          "steps",
          "1-D tensor of slice step of corresponding axis in `axes`. "
          "Negative value means slicing backward. 'steps' cannot be 0. "
          "Defaults to 1s.",
          "Tind",
          OpSchema::Optional,
          true,
          1,
          OpSchema::NonDifferentiable)
      .Output(0, "output", "Sliced data tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
      .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types");

  ONNX_DISTRIBUTED_OPERATOR_SCHEMA(DistributedReshape)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr(
          "allowzero",
          "(Optional) By default, when any value in the 'shape' input is equal to zero "
          "the corresponding dimension value is copied from the input tensor dynamically. "
          "allowzero=1 indicates that if any value in the 'shape' input is set to zero, "
          "the zero value is honored, similar to NumPy.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(
          1,
          "shape",
          "Specified shape for output.",
          "tensor(int64)",
          OpSchema::Single,
          true,
          1,
          OpSchema::NonDifferentiable)
      .Output(0, "reshaped", "Reshaped data.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.");

  ONNX_DISTRIBUTED_OPERATOR_SCHEMA(DistributedExpand)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(
          1,
          "shape",
          "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule",
          "tensor(int64)",
          OpSchema::Single,
          true,
          1,
          OpSchema::NonDifferentiable)
      .Output(0, "output", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensors.");

  ONNX_DISTRIBUTED_OPERATOR_SCHEMA(DistributedReduceSum)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("keepdims",
            "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
      .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(
          1,
          "shape",
          "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule",
          "tensor(int64)",
          OpSchema::Single,
          true,
          1,
          OpSchema::NonDifferentiable)
      .Output(0, "output", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensors.");

  ONNX_DISTRIBUTED_OPERATOR_SCHEMA(DistributedReduceMax)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("keepdims",
            "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
      .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(
          1,
          "shape",
          "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule",
          "tensor(int64)",
          OpSchema::Single,
          true,
          1,
          OpSchema::NonDifferentiable)
      .Output(0, "output", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensors.");

  ONNX_DISTRIBUTED_OPERATOR_SCHEMA(DistributedReduceMean)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("keepdims",
            "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
      .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(
          1,
          "shape",
          "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule",
          "tensor(int64)",
          OpSchema::Single,
          true,
          1,
          OpSchema::NonDifferentiable)
      .Output(0, "output", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensors.");

  ONNX_DISTRIBUTED_OPERATOR_SCHEMA(DistributedUnsqueeze)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(
          1,
          "axes",
          "A 1-D tensor indicates the axes to add.",
          "tensor(int64)",
          OpSchema::Single,
          true,
          1,
          OpSchema::NonDifferentiable)
      .Output(0, "output", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensors.");

  ONNX_DISTRIBUTED_OPERATOR_SCHEMA(DistributedSqueeze)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(
          1,
          "axes",
          "A 1-D tensor indicates the axes to add.",
          "tensor(int64)",
          OpSchema::Single,
          true,
          1,
          OpSchema::NonDifferentiable)
      .Output(0, "output", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensors.");
}

}  // namespace contrib
}  // namespace onnxruntime
