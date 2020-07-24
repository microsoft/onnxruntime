// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/gradient_builder.h"

#include <cmath>
#include <numeric>

#include "onnx/defs/attr_proto_util.h"

#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/core/graph/gradient_builder_registry.h"
#include "orttraining/core/graph/graph_augmenter.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

#define IMPLEMENT_GRADIENT_BUILDER(name) \
  std::vector<NodeDef> name::GetGradientDefsImpl() const

IMPLEMENT_GRADIENT_BUILDER(GetCastGradient) {
  // TODO: handle invalid conversion cases
  const auto data_type = I(0).type_proto->tensor_type().elem_type();
  return std::vector<NodeDef>{
      NodeDef("Cast",
              {GO(0)},
              {GI(0)},
              {MakeAttribute("to", int64_t(data_type))})};
}

IMPLEMENT_GRADIENT_BUILDER(GetSinGradient) {
  return std::vector<NodeDef>{
      NodeDef("SinGrad",
              {GO(0), I(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetTanhGradient) {
  return std::vector<NodeDef>{
      NodeDef("TanhGrad",
              {O(0), GO(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetSqrtGradient) {
  return std::vector<NodeDef>{
      NodeDef("SqrtGrad",
              {O(0), GO(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetErfGradient) {
  return std::vector<NodeDef>{
      NodeDef("ErfGrad",
              {I(0), GO(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetMatMulGradient) {
  std::vector<NodeDef> result;

  ArgDef A = I(0), B = I(1), Y = O(0);
  std::vector<Dimension> A_shape = GetShape(A);
  std::vector<Dimension> B_shape = GetShape(B);
  std::vector<Dimension> Y_shape = GetShape(Y);

  std::vector<AttributeProto> shared_attributes;
  shared_attributes.push_back(MakeAttribute("beta", float(0)));
  AttributeProto transpose_first_input = MakeAttribute("transA", int64_t(1));
  AttributeProto transpose_second_input = MakeAttribute("transB", int64_t(1));

  if (A_shape.size() == 2 && B_shape.size() == 2) {
    NodeDef zero_constant_node = ZeroConstantNode();
    ArgDef ZERO = zero_constant_node.output_args[0];
    result.push_back(zero_constant_node);

    // is GI(0) required
    if (IsGradientRequiredForSrcNodeInput(0)) {
      // dA = dY * B'
      std::vector<AttributeProto> attrs(shared_attributes);
      attrs.push_back(transpose_second_input);
      result.push_back(
          NodeDef("Gemm",
                  {GO(0), B, ZERO},
                  {GI(0)},
                  attrs));
    }

    // is GI(1) required
    if (IsGradientRequiredForSrcNodeInput(1)) {
      // dB = A' * dY
      std::vector<AttributeProto> attrs(shared_attributes);
      attrs.push_back(transpose_first_input);
      result.push_back(
          NodeDef("Gemm",
                  {A, GO(0), ZERO},
                  {GI(1)},
                  attrs));
    }
  } else if (A_shape.size() > 2 || B_shape.size() > 2) {
    if (IsGradientRequiredForSrcNodeInput(0)) {
      // If B_shape.size() == 2, dA is computed through 2 ops: transpose and matmul.
      // It can be replaced with Gemm(dY_reshape, B_transpose) and reshape.
      // However, there is a performance degradation.
      // Thus this implementation is not implemented.
      int64_t B_rank = B_shape.size();
      std::vector<int64_t> B_perm(B_rank);
      std::iota(B_perm.begin(), B_perm.end(), 0);
      std::swap(B_perm[B_rank - 1], B_perm[B_rank - 2]);

      std::vector<Dimension> output_shape;
      for (size_t i = 0; i < Y_shape.size() - 1; i++) {
        output_shape.push_back(Y_shape[i]);
      }
      output_shape.push_back(B_shape[B_shape.size() - 2]);

      std::vector<int64_t> A_axes;
      ComputeBroadcastBackwardAxes(A_shape, output_shape, &A_axes, nullptr);

      result.push_back(
          NodeDef("Transpose",
                  {B},
                  {IA("B_t")},
                  {MakeAttribute("perm", B_perm)}));

      ArgDef matmul_out = A_axes.size() > 0 ? IA("PreReduceGrad0") : GI(0);

      result.push_back(
          NodeDef("MatMul",
                  {GO(0), IA("B_t")},
                  {matmul_out}));

      if (A_axes.size() > 0) {
        result.push_back(
            NodeDef("ReduceSum",
                    {IA("PreReduceGrad0")},
                    {IA("ReduceGrad0")},
                    {{"keepdims", MakeAttribute("keepdims", int64_t(1))},
                     {"axes", MakeAttribute("axes", A_axes)}}));

        result.push_back(
            NodeDef("Shape",
                    {A},
                    {IA("A_shape")}));

        result.push_back(
            NodeDef("Reshape",
                    {IA("ReduceGrad0"), IA("A_shape")},
                    {GI(0)}));
      }
    }
    if (IsGradientRequiredForSrcNodeInput(1)) {
      if (B_shape.size() == 2 &&
          (B_shape[0].has_dim_value() || A_shape[A_shape.size() - 1].has_dim_value()) &&
          (B_shape[1].has_dim_value() || Y_shape[Y_shape.size() - 1].has_dim_value())) {
        // A[M, K], B[K, N], Y[M, N]
        int64_t K, N;
        if (B_shape[0].has_dim_value()) {
          K = B_shape[0].dim_value();
        } else {
          K = A_shape[A_shape.size() - 1].dim_value();
        }
        if (B_shape[1].has_dim_value()) {
          N = B_shape[1].dim_value();
        } else {
          N = Y_shape[Y_shape.size() - 1].dim_value();
        }

        std::vector<int64_t> A_shape_2d{-1, K};
        NodeDef A_shape_2d_node = ConstantValueNode(A_shape_2d, Name("A_shape_2d"));
        ArgDef A_shape_2d_arg = A_shape_2d_node.output_args[0];
        result.push_back(A_shape_2d_node);

        std::vector<int64_t> dY_shape_2d{-1, N};
        NodeDef dY_shape_2d_node = ConstantValueNode(dY_shape_2d, Name("dY_shape_2d"));
        ArgDef dY_shape_2d_arg = dY_shape_2d_node.output_args[0];
        result.push_back(dY_shape_2d_node);

        NodeDef zero_constant_node = ZeroConstantNode();
        ArgDef ZERO = zero_constant_node.output_args[0];
        result.push_back(zero_constant_node);

        result.push_back(
            NodeDef("Reshape",
                    {A, A_shape_2d_arg},
                    {IA("A_reshape_2d")}));
        result.push_back(
            NodeDef("Reshape",
                    {GO(0), dY_shape_2d_arg},
                    {IA("dY_reshape_2d")}));

        // dB = A' * dY
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_first_input);
        result.push_back(
            NodeDef("Gemm",
                    {IA("A_reshape_2d"), IA("dY_reshape_2d"), ZERO},
                    {GI(1)},
                    attrs));
      } else {
        int64_t A_rank = A_shape.size();
        std::vector<int64_t> A_perm(A_rank);
        std::iota(A_perm.begin(), A_perm.end(), 0);
        std::swap(A_perm[A_rank - 1], A_perm[A_rank - 2]);

        std::vector<Dimension> output_shape;
        for (size_t i = 0; i < Y_shape.size() - 2; i++) {
          output_shape.push_back(Y_shape[i]);
        }
        output_shape.push_back(A_shape[A_shape.size() - 1]);
        output_shape.push_back(Y_shape[Y_shape.size() - 1]);

        std::vector<int64_t> B_axes;
        ComputeBroadcastBackwardAxes(B_shape, output_shape, &B_axes, nullptr);

        result.push_back(
            NodeDef("Transpose",
                    {A},
                    {IA("A_t")},
                    {MakeAttribute("perm", A_perm)}));

        ArgDef matmul_out = B_axes.size() > 0 ? IA("PreReduceGrad1") : GI(1);

        result.push_back(
            NodeDef("MatMul",
                    {IA("A_t"), GO(0)},
                    {matmul_out}));

        if (B_axes.size() > 0) {
          result.push_back(
              NodeDef("ReduceSum",
                      {IA("PreReduceGrad1")},
                      {IA("ReduceGrad1")},
                      {{"keepdims", MakeAttribute("keepdims", int64_t(0))},
                       {"axes", MakeAttribute("axes", B_axes)}}));
          result.push_back(
              NodeDef("Shape",
                      {B},
                      {IA("B_shape")}));
          result.push_back(
              NodeDef("Reshape",
                      {IA("ReduceGrad1"), IA("B_shape")},
                      {GI(1)}));
        }
      }
    }
  } else {
    ORT_THROW("Matmul Gradient Builder shouldn't reach here. ");
  }

  return result;
};

IMPLEMENT_GRADIENT_BUILDER(GetGemmGradient) {
  auto attributes = SrcNodeAttributes();

  bool has_alpha = attributes.at("alpha").has_f();
  float alpha = attributes.at("alpha").f();
  bool transA = static_cast<bool>(attributes.at("transA").i());
  bool transB = static_cast<bool>(attributes.at("transB").i());

  ArgDef A = I(0), B = I(1), C = I(2), dY = GO(0),
         dA = GI(0), dB = GI(1), dC = GI(2);
  AttributeProto transpose_first_input = MakeAttribute("transA", int64_t(1));
  AttributeProto transpose_second_input = MakeAttribute("transB", int64_t(1));

  NodeDef zero_contant_node = ZeroConstantNode();
  ArgDef ZERO = zero_contant_node.output_args[0];

  std::vector<NodeDef> result;
  result.push_back(zero_contant_node);

  std::vector<AttributeProto> shared_attributes;
  shared_attributes.push_back(MakeAttribute("beta", float(0)));
  if (has_alpha && alpha != 1.0f) {
    ORT_ENFORCE(alpha != 0.0f);
    AttributeProto alpha_attr = MakeAttribute("alpha", alpha);
    shared_attributes.push_back(alpha_attr);
  }

  if (transA) {
    if (transB) {
      // Y = alpha * A' * B'
      // dA = alpha * B' * dY', dB = alpha *  dY' * A'
      if (IsGradientRequiredForSrcNodeInput(0)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_first_input);
        attrs.push_back(transpose_second_input);
        result.push_back(NodeDef("Gemm", {B, dY, ZERO}, {dA}, attrs));
      }

      if (IsGradientRequiredForSrcNodeInput(1)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_first_input);
        attrs.push_back(transpose_second_input);
        result.push_back(NodeDef("Gemm", {dY, A, ZERO}, {dB}, attrs));
      }
    } else {
      // Y = alpha * A' * B
      // dA = alpha * B * dY', dB = alpha * A * dY
      if (IsGradientRequiredForSrcNodeInput(0)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_second_input);
        result.push_back(NodeDef("Gemm", {B, dY, ZERO}, {dA}, attrs));
      }

      if (IsGradientRequiredForSrcNodeInput(1)) {
        result.push_back(NodeDef("Gemm", {A, dY, ZERO}, {dB}, shared_attributes));
      }
    }
  } else {
    if (transB) {
      // Y = alpha * A * B'
      // dA = alpha * dY * B, dB = alpha * dY' * A
      if (IsGradientRequiredForSrcNodeInput(0)) {
        result.push_back(NodeDef("Gemm", {dY, B, ZERO}, {dA}, shared_attributes));
      }

      if (IsGradientRequiredForSrcNodeInput(1)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_first_input);
        result.push_back(NodeDef("Gemm", {dY, A, ZERO}, {dB}, attrs));
      }
    } else {
      // Y = alpha * A * B
      // dA = alpha * dY * B', dB = alpha * A' * dY
      if (IsGradientRequiredForSrcNodeInput(0)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_second_input);
        result.push_back(NodeDef("Gemm", {dY, B, ZERO}, {dA}, attrs));
      }

      if (IsGradientRequiredForSrcNodeInput(1)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_first_input);
        result.push_back(NodeDef("Gemm", {A, dY, ZERO}, {dB}, attrs));
      }
    }
  }

  if (IsGradientRequiredForSrcNodeInput(2)) {
    // Y = beta * C
    // dC = beta * dY
    bool has_beta = attributes.at("beta").has_f();
    float beta = attributes.at("beta").f();
    ORT_ENFORCE(beta != 0.0f);

    std::vector<Dimension> C_shape = GetShape(C);
    std::vector<Dimension> dY_shape = GetShape(dY);

    std::vector<int64_t> C_axes, dY_axes;
    ComputeBroadcastBackwardAxes(C_shape, dY_shape, &C_axes, &dY_axes);

    if (C_axes.size() > 0) {
      HandleBroadcasting(dY, C, IA("dC_reduced"), C_axes, result);

      if (has_beta && beta != 1.0f) {
        NodeDef scale_node = ConstantValueNode(beta, Name("Scale"));
        ArgDef SCALE = scale_node.output_args[0];
        result.push_back(scale_node);
        result.push_back(
            NodeDef("Mul",
                    {IA("dC_reduced"), SCALE},
                    {dC}));
      } else {
        result.push_back(
            NodeDef("Identity", {IA("dC_reduced")}, {dC}));
      }
    } else {
      if (has_beta && beta != 1.0f) {
        NodeDef scale_node = ConstantValueNode(beta, Name("Scale"));
        ArgDef SCALE = scale_node.output_args[0];
        result.push_back(scale_node);
        result.push_back(
            NodeDef("Mul",
                    {dY, SCALE},
                    {dC}));
      } else {
        result.push_back(
            NodeDef("Identity",
                    {dY},
                    {dC}));
      }
    }
  }
  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetSplitGradient) {
  std::vector<NodeDef> result = {};
  std::vector<ArgDef> input_args;

  for (int i = 0; i < GetSrcNodeOutputSize(); i++) {
    if (IsGradientAvailableForSrcNodeOutput(i)) {
      input_args.push_back(GO(i));
    }
  }

  if (!input_args.empty()) {
    auto attributes = SrcNodeAttributes();
    ORT_ENFORCE(attributes.at("axis").has_i());
    auto axis = attributes.at("axis").i();
    result.push_back(
        NodeDef("Concat",
                input_args,
                {GI(0)},
                {MakeAttribute("axis", axis)}));
  }
  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetConcatGradient) {
  auto attributes = SrcNodeAttributes();
  ORT_ENFORCE(attributes.at("axis").has_i());
  auto axis = attributes.at("axis").i();

  std::vector<int64_t> split_attribute(GetSrcNodeInputSize());
  std::vector<ArgDef> outputs;
  for (int i = 0; i < GetSrcNodeInputSize(); ++i) {
    std::vector<Dimension> data_shape = GetShape(I(i));
    int64_t axis_index = axis < 0 ? static_cast<int64_t>(data_shape.size()) + axis : axis;
    if (axis_index >= 0 && axis_index < static_cast<int64_t>(data_shape.size()) && data_shape[axis_index].has_dim_value()) {
      split_attribute[i] = data_shape[axis_index].dim_value();
    } else {
      ORT_THROW("Error: can't infer split attribute value for ConcatGrad");
    }
    outputs.push_back(GI(i));
  }

  std::vector<AttributeProto> new_attributes;
  new_attributes.push_back(MakeAttribute("axis", axis));
  new_attributes.push_back(MakeAttribute("split", split_attribute));

  return std::vector<NodeDef>{
      NodeDef("Split",
              {GO(0)},
              outputs,
              new_attributes)};
}

IMPLEMENT_GRADIENT_BUILDER(GetGatherNDGradient) {
  auto attributes = SrcNodeAttributes();
  ORT_ENFORCE(attributes.at("batch_dims").has_i());
  auto batch_dims = attributes.at("batch_dims").i();
  return std::vector<NodeDef>{
      NodeDef("Shape",
              {I(0)},
              {IA("x_shape")}),
      NodeDef(OpDef{"GatherNDGrad", kMSDomain, 1},
              {IA("x_shape"), I(1), GO(0)},
              {GI(0)},
              {MakeAttribute("batch_dims", batch_dims)})};
};

IMPLEMENT_GRADIENT_BUILDER(GetReshapeGradient) {
  return std::vector<NodeDef>{
      NodeDef("ReshapeGrad",
              {I(0), GO(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetTransposeGradient) {
  std::vector<int64_t> bw_perm;
  auto attributes = SrcNodeAttributes();
  if (attributes.empty()) {
    const TensorShapeProto& input_shape = I(0).type_proto->tensor_type().shape();
    for (int i = input_shape.dim_size() - 1; i >= 0; --i) {
      bw_perm.push_back(i);
    }
  } else {
    auto fw_perm = RetrieveValues<int64_t>(attributes.at("perm"));
    auto size = fw_perm.size();
    bw_perm.resize(size);
    for (int i = 0; i < static_cast<int>(size); ++i) {
      bw_perm[fw_perm[i]] = i;
    }
  }

  return std::vector<NodeDef>{
      NodeDef("Transpose",
              {GO(0)},
              {GI(0)},
              {MakeAttribute("perm", bw_perm)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetAveragePoolGradient) {
  return std::vector<NodeDef>{
      NodeDef("AveragePoolGrad",
              {GO(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetMaxPoolGradient) {
  return std::vector<NodeDef>{
      NodeDef("MaxPoolGrad",
              {GO(0), O(1)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetPoolGradient) {
  return std::vector<NodeDef>{
      NodeDef(SrcNodeOpType() + "Grad",
              {GO(0), I(0), O(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetDropoutGradient) {
  std::vector<ArgDef> inputs{GO(0), O(1)};
  for (int i = 1; i < GetSrcNodeInputSize(); i++) {
    inputs.push_back(I(i));
  }
  return std::vector<NodeDef>{
      NodeDef(OpDef{"DropoutGrad", kMSDomain, 1},
              inputs,
              {GI(0)},
              {SrcNodeAttributes()})};
}

IMPLEMENT_GRADIENT_BUILDER(GetTrainableDropoutGradient) {
  std::vector<ArgDef> inputs{GO(0), O(1)};
  for (int i = 1; i < GetSrcNodeInputSize(); i++) {
    inputs.push_back(I(i));
  }
  return std::vector<NodeDef>{
      NodeDef(OpDef{"TrainableDropoutGrad", kMSDomain, 1},
              inputs,
              {GI(0)},
              {SrcNodeAttributes()})};
}

IMPLEMENT_GRADIENT_BUILDER(GetConvGradient) {
  std::vector<ArgDef> outputs;
  for (int i = 0; i < 3; i++) {
    if (IsGradientRequiredForSrcNodeInput(i)) {
      outputs.push_back(GI(i));
    } else {
      outputs.push_back(ArgDef("", nullptr));
    }
  }

  return std::vector<NodeDef>{
      NodeDef("ConvGrad",
              {GO(0), I(0), I(1)},
              outputs,
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetSoftmaxGradient) {
  return std::vector<NodeDef>{
      NodeDef(OpDef{"SoftmaxGrad", kMSDomain, 1},
              {GO(0), O(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetUnsqueezeGradient) {
  return std::vector<NodeDef>{
      NodeDef("Squeeze",
              {GO(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetGatherGradient) {
  return std::vector<NodeDef>{
      NodeDef("Shape",
              {I(0)},
              {IA("I0_shape")}),
      NodeDef(OpDef{"GatherGrad", kMSDomain, 1},
              {IA("I0_shape"), I(1), GO(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetGatherElementsGradient) {
  return std::vector<NodeDef>{
      NodeDef("Shape",
              {I(0)},
              {IA("x_shape")}),
      NodeDef(OpDef{"GatherElementsGrad", kMSDomain, 1},
              {GO(0), IA("x_shape"), I(1)},
              {GI(0)},
              SrcNodeAttributes())};
};

IMPLEMENT_GRADIENT_BUILDER(GetReluGradient) {
  return std::vector<NodeDef>{
      NodeDef("ReluGrad",
              {GO(0), O(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetSqueezeGradient) {
  std::vector<NodeDef> result;
  auto attributes = SrcNodeAttributes();
  std::vector<int64_t> axes_values;
  if (attributes.find("axes") != attributes.end()) {
    axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
    result.push_back(
        NodeDef("Unsqueeze",
                {GO(0)},
                {GI(0)},
                {MakeAttribute("axes", axes_values)}));
    // if axes attribute not provided for squeeze
  } else {
    result.push_back(
        NodeDef("Shape",
                {I(0)},
                {IA("I0_shape")}));
    result.push_back(
        NodeDef("Reshape",
                {GO(0), IA("I0_shape")},
                {GI(0)}));
  }
  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetAddSubGradient) {
  bool is_sub = (SrcNodeOpType() == "Sub");

  const ArgDef a = I(0), b = I(1);

  std::vector<Dimension> a_shape = GetShape(a);
  std::vector<Dimension> b_shape = GetShape(b);

  std::vector<int64_t> a_axes, b_axes;
  ComputeBroadcastBackwardAxes(a_shape, b_shape, &a_axes, &b_axes);

  std::vector<NodeDef> output;

  if (IsGradientRequiredForSrcNodeInput(0)) {
    if (a_axes.size() > 0) {
      HandleBroadcasting(GO(0), a, GI(0), a_axes, output);
    } else {
      output.push_back(
          NodeDef("Identity",
                  {GO(0)},
                  {GI(0)}));
    }
  }

  if (IsGradientRequiredForSrcNodeInput(1)) {
    if (b_axes.size() > 0) {
      ArgDef reshape_output = is_sub ? IA("ReshapeReduceSum_2", IType(1)) : GI(1);
      HandleBroadcasting(GO(0), b, reshape_output, b_axes, output);

      if (is_sub) {
        output.push_back(
            NodeDef("Neg",
                    {reshape_output},
                    {GI(1)}));
      }
    } else {
      if (is_sub) {
        output.push_back(
            NodeDef("Neg",
                    {GO(0)},
                    {GI(1)}));
      } else /*is_add*/ {
        output.push_back(
            NodeDef("Identity",
                    {GO(0)},
                    {GI(1)}));
      }
    }
  }
  return output;
}

IMPLEMENT_GRADIENT_BUILDER(GetMulGradient) {
  const ArgDef a = I(0), b = I(1);

  std::vector<Dimension> a_shape = GetShape(a);
  std::vector<Dimension> b_shape = GetShape(b);
  std::vector<int64_t> a_axes, b_axes;
  ComputeBroadcastBackwardAxes(a_shape, b_shape, &a_axes, &b_axes);

  std::vector<NodeDef> output;

  if (IsGradientRequiredForSrcNodeInput(0)) {
    output.push_back(
        NodeDef("Mul",
                {GO(0), I(1)},
                {IA("PreReduceGrad0", OType(0))}));

    if (a_axes.size() > 0) {
      HandleBroadcasting(IA("PreReduceGrad0", OType(0)), a, GI(0), a_axes, output);
    } else {
      output.push_back(
          NodeDef("Identity",
                  {IA("PreReduceGrad0", OType(0))},
                  {GI(0)}));
    }
  }

  if (IsGradientRequiredForSrcNodeInput(1)) {
    output.push_back(
        NodeDef("Mul",
                {GO(0), I(0)},
                {IA("PreReduceGrad1", OType(0))}));

    if (b_axes.size() > 0) {
      HandleBroadcasting(IA("PreReduceGrad1", OType(0)), b, GI(1), b_axes, output);
    } else {
      output.push_back(
          NodeDef("Identity",
                  {IA("PreReduceGrad1", OType(0))},
                  {GI(1)}));
    }
  }
  return output;
}

IMPLEMENT_GRADIENT_BUILDER(GetDivGradient) {
  if (IsGradientRequiredForSrcNodeInput(0) && IsGradientRequiredForSrcNodeInput(1)) {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"DivGrad", kMSDomain, 1},
                {GO(0), I(0), I(1)},
                {GI(0), GI(1)})};
  } else if (IsGradientRequiredForSrcNodeInput(0)) {
    // Y = A / B, dA = dY / B
    const ArgDef a = I(0), b = I(1);
    std::vector<int64_t> a_axes, b_axes;
    ComputeBroadcastBackwardAxes(GetShape(a), GetShape(b), &a_axes, &b_axes);

    std::vector<NodeDef> output;
    ArgDef tmp_grad = IA("PreReduceGrad0", OType(0));
    output.push_back(NodeDef("Div", {GO(0), I(1)}, {tmp_grad}));
    if (a_axes.size() > 0) {
      HandleBroadcasting(tmp_grad, a, GI(0), a_axes, output);
    } else {
      output.push_back(NodeDef("Identity", {tmp_grad}, {GI(0)}));
    }
    return output;
  } else if (IsGradientRequiredForSrcNodeInput(1)) {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"DivGrad", kMSDomain, 1},
                {GO(0), I(0), I(1)},
                // TODO: this IA("") does not cause kernel to know it is unneeded.
                // Gradient for the first input is still calculated.
                {IA(""), GI(1)})};
  } else {
    return std::vector<NodeDef>{};
  }
}

IMPLEMENT_GRADIENT_BUILDER(GetReduceMeanGradient) {
  std::vector<NodeDef> result;
  auto attributes = SrcNodeAttributes();
  bool keepdims = true;
  if (attributes.find("keepdims") != attributes.end() &&
      attributes.at("keepdims").has_i()) {
    keepdims = static_cast<bool>(attributes.at("keepdims").i());
  }

  ArgDef grad = GO(0);
  if (!keepdims && attributes.find("axes") != attributes.end()) {
    std::vector<int64_t> axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
    grad = IA("Unqueezed_Grad");
    result.push_back(NodeDef("Unsqueeze", {GO(0)}, {grad}, {MakeAttribute("axes", axes_values)}));
  }

  const int64_t type_float = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  result.push_back(NodeDef("Size", {I(0)}, {IA("Scale_Denominator")}));
  result.push_back(
      NodeDef("Cast",
              {IA("Scale_Denominator")},
              {IA("Casted_Scale_Denominator")},
              {MakeAttribute("to", type_float)}));
  result.push_back(NodeDef("Size", {GO(0)}, {IA("Scale_Numerator")}));
  result.push_back(
      NodeDef("Cast",
              {IA("Scale_Numerator")},
              {IA("Casted_Scale_Numerator")},
              {MakeAttribute("to", type_float)}));
  result.push_back(
      NodeDef("Div",
              {IA("Casted_Scale_Numerator"), IA("Casted_Scale_Denominator")},
              {IA("Scale")}));
  result.push_back(NodeDef("Mul", {grad, IA("Scale")}, {IA("Scaled_Grad")}));
  result.push_back(NodeDef("Shape", {I(0)}, {IA("Shaped_X")}));
  result.push_back(NodeDef("Expand", {IA("Scaled_Grad"), IA("Shaped_X")}, {GI(0)}));
  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetReduceSumGradient) {
  std::vector<NodeDef> result;
  auto attributes = SrcNodeAttributes();
  bool keepdims = true;
  if (attributes.find("keepdims") != attributes.end() &&
      attributes.at("keepdims").has_i()) {
    keepdims = static_cast<bool>(attributes.at("keepdims").i());
  }

  ArgDef grad = GO(0);
  if (!keepdims && attributes.find("axes") != attributes.end()) {
    std::vector<int64_t> axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
    grad = IA("Unqueezed_Grad");
    result.push_back(NodeDef("Unsqueeze", {GO(0)}, {grad}, {MakeAttribute("axes", axes_values)}));
  }

  result.push_back(NodeDef("Shape", {I(0)}, {IA("Shaped_X")}));
  result.push_back(NodeDef("Expand", {grad, IA("Shaped_X")}, {GI(0)}));
  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetPowGradient) {
  if (IsGradientRequiredForSrcNodeInput(1)) {
    ORT_THROW("GradientBuilder is not implemented for CUDA Pow's input exponent.");
  }
  return std::vector<NodeDef>{
      NodeDef("PowGrad",
              {GO(0), I(0), I(1)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetSoftmaxCrossEntropyGradient) {
  return std::vector<NodeDef>{
      NodeDef(OpDef{"SoftmaxCrossEntropyGrad", kMSDomain, 1},
              {GO(0), O(1), I(1)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetSparseSoftmaxCrossEntropyGradient) {
  if (GetSrcNodeInputSize() == 2) {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"SparseSoftmaxCrossEntropyGrad"},
                {GO(0), O(1), I(1)},
                {GI(0)},
                SrcNodeAttributes())};
  } else if (GetSrcNodeInputSize() == 3) {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"SparseSoftmaxCrossEntropyGrad"},
                {GO(0), O(1), I(1), I(2)},
                {GI(0)},
                SrcNodeAttributes())};
  } else {
    ORT_ENFORCE(false, "the number of input arguments must be 2 or 3");
  }
}

IMPLEMENT_GRADIENT_BUILDER(GetSoftmaxCrossEntropyLossGradient) {
  if (GetSrcNodeInputSize() == 2) {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"SoftmaxCrossEntropyLossGrad", kMSDomain, 1},
                {GO(0), O(1), I(1)},
                {GI(0)},
                SrcNodeAttributes())};
  } else if (GetSrcNodeInputSize() == 3) {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"SoftmaxCrossEntropyLossGrad", kMSDomain, 1},
                {GO(0), O(1), I(1), I(2)},
                {GI(0)},
                SrcNodeAttributes())};
  } else {
    ORT_THROW(false, "the number of input arguments must be 2 or 3");
  }
}

IMPLEMENT_GRADIENT_BUILDER(GetGlobalAveragePoolGradient) {
  const ArgDef X = I(0);

  // TODO: ONNX supports unknown shape for the input feed, e.g. [1, 3, -1, 28],
  // thus the shape of input might be missing at graph construction time.
  // However, in practice, we haven't seen a single model with unknown input shape.
  // We need to get the shape at runtime if this case need to be supported.
  // One way to do it is: scale = Size_Op(X, from=2); scaled_dY = Mul_Op(dY, scale)
  const auto& x_dims = X.type_proto->tensor_type().shape().dim();
  ORT_ENFORCE(x_dims.size() >= 3, "Input dimension cannot be less than 3.");
  int64_t scale = 1;
  for (auto dim = x_dims.begin() + 2; dim < x_dims.end(); dim++) {
    if (dim->has_dim_value()) {
      scale *= dim->dim_value();
    } else {
      ORT_ENFORCE(false, "Dimension missing");
    }
  }

  NodeDef scale_node = ConstantValueNode(1.0f / static_cast<float>(scale), Name("Scale"));
  ArgDef SCALE = scale_node.output_args[0];
  return std::vector<NodeDef>{
      scale_node,
      NodeDef("Mul",
              {GO(0), SCALE},
              {IA("scaled_dY")}),
      NodeDef("Shape",
              {X},
              {IA("x_shape")}),
      NodeDef("Expand",
              {IA("scaled_dY"), IA("x_shape")},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetGeluGradient) {
  return std::vector<NodeDef>{
      NodeDef(OpDef{"GeluGrad", kMSDomain, 1},
              {GO(0), I(0)},
              {GI(0)})};
}

namespace {
std::vector<NodeDef> GetBiasGeluGradNodes(
    bool use_approximation,
    const ArgDef& dY, const ArgDef& X, const ArgDef& B,  // inputs
    const ArgDef& dX, const ArgDef& dB) {                // outputs
  const auto B_shape = GetShape(B);
  ORT_ENFORCE(B_shape.size() == 1, "B must have exactly one dimension.");

  const std::vector<int64_t> B_axes = [&B_shape, &X]() {
    std::vector<int64_t> result{};
    ComputeBroadcastBackwardAxes(B_shape, GetShape(X), &result, nullptr);
    return result;
  }();

  return std::vector<NodeDef>{
      NodeDef(OpDef{use_approximation ? "BiasFastGeluGrad_dX" : "BiasGeluGrad_dX", kMSDomain, 1},
              {dY, X, B},
              {dX}),
      NodeDef("ReduceSum",
              {dX},
              {dB},
              {{"keepdims", MakeAttribute("keepdims", int64_t{0})},
               {"axes", MakeAttribute("axes", B_axes)}})};
}
}  // namespace

IMPLEMENT_GRADIENT_BUILDER(GetBiasGeluGradient) {
  const auto dY = GO(0), X = I(0), B = I(1),
             dX = GI(0), dB = GI(1);
  return GetBiasGeluGradNodes(false, dY, X, B, dX, dB);
}

IMPLEMENT_GRADIENT_BUILDER(GetFastGeluGradient) {
  const auto dY = GO(0), X = I(0),
             dX = GI(0);
  const auto num_src_node_inputs = GetSrcNodeInputSize();
  if (num_src_node_inputs == 2) {  // with bias
    // FastGeluGrad doesn't support bias - it needs to be composed with other ops
    const auto B = I(1),
               dB = GI(1);
    return GetBiasGeluGradNodes(true, dY, X, B, dX, dB);
  }
  if (num_src_node_inputs == 1) {  // without bias
    return std::vector<NodeDef>{
        NodeDef(OpDef{"FastGeluGrad", kMSDomain, 1},
                {dY, X},
                {dX})};
  }
  ORT_THROW("Unexpected number of FastGelu inputs: ", num_src_node_inputs);
}

IMPLEMENT_GRADIENT_BUILDER(GetLayerNormalizationGradient) {
  if (GetGradientGraphConfiguration().use_invertible_layernorm_grad) {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"InvertibleLayerNormalizationGrad", kMSDomain, 1},
                {GO(0), O(0), I(1), I(2), O(2)},
                {GI(0), GI(1), GI(2)},
                {SrcNodeAttributes()})};
  } else {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"LayerNormalizationGrad", kMSDomain, 1},
                {GO(0), I(0), I(1), O(1), O(2)},
                {GI(0), GI(1), GI(2)},
                {SrcNodeAttributes()})};
  }
}

IMPLEMENT_GRADIENT_BUILDER(GetBatchNormalizationGradient) {
  auto attributes = SrcNodeAttributes();
  if (attributes.find("epsilon") != attributes.end()) {
    float epsilon = attributes.at("epsilon").f();
    return std::vector<NodeDef>{
        NodeDef(OpDef{"BatchNormalizationGrad", kMSDomain, 1},
                {GO(0), I(0), I(1), O(3), O(4)},
                {GI(0), GI(1), GI(2)},
                {MakeAttribute("epsilon", epsilon)})};
  } else {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"BatchNormalizationGrad", kMSDomain, 1},
                {GO(0), I(0), I(1), O(3), O(4)},
                {GI(0), GI(1), GI(2)})};
  }
}

IMPLEMENT_GRADIENT_BUILDER(GetMegatronFGradient) {
  return std::vector<NodeDef>{
      NodeDef(OpDef{"NcclAllReduce", kMSDomain, 1},
              {GO(0)},
              {GI(0)},
              {MakeAttribute("group_type", static_cast<int64_t>(training::WorkerGroupType::HorizontalParallel))})};
}

IMPLEMENT_GRADIENT_BUILDER(GetMegatronGGradient) {
  return std::vector<NodeDef>{
      NodeDef("Identity",
              {GO(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetSliceGradient) {
  std::vector<ArgDef> inputs{GO(0), IA("I0_shape")};
  for (int i = 1; i < GetSrcNodeInputSize(); i++) {
    inputs.push_back(I(i));
  }

  return std::vector<NodeDef>{
      NodeDef("Shape",
              {I(0)},
              {IA("I0_shape")}),
      NodeDef(OpDef{"SliceGrad", kMSDomain, 1}, inputs, {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetWhereGradient) {
  std::vector<NodeDef> result;
  const int64_t data_type = static_cast<int64_t>(I(1).type_proto->tensor_type().elem_type());
  if (IsGradientRequiredForSrcNodeInput(1)) {
    result.push_back(NodeDef("Cast", {I(0)}, {IA("Positive_Mask")}, {MakeAttribute("to", data_type)}));
    result.push_back(NodeDef("Mul", {GO(0), IA("Positive_Mask")}, {GI(1)}));
  }

  if (IsGradientRequiredForSrcNodeInput(2)) {
    result.push_back(NodeDef("Not", {I(0)}, {IA("Not_Condition", IType(0))}));
    result.push_back(NodeDef("Cast", {IA("Not_Condition")}, {IA("Negative_Mask")}, {MakeAttribute("to", data_type)}));
    result.push_back(NodeDef("Mul", {GO(0), IA("Negative_Mask")}, {GI(2)}));
  }
  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetSendGradient) {
  // Send inputs: signal A, remote, data; outputs: signal B
  // Recv inputs: signal B, remote; outputs: signal A', data'

  std::vector<ArgDef> out_args;
  out_args.push_back(GI(0));  // Signal
  for (int i = 2; i < GetSrcNodeInputSize(); ++i) {
    out_args.push_back(GI(i));  // Data
  }

  return std::vector<NodeDef>{
      NodeDef(OpDef{"Recv", kMSDomain, 1},
              {O(0), I(1)},  // {Signal, Remote}
              out_args,
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetRecvGradient) {
  // Recv inputs: signal A, remote; outputs: signal B, data
  // Send inputs: signal B, remote, data'; outputs: signal A'

  std::vector<ArgDef> in_args;
  in_args.push_back(O(0));  // Signal
  in_args.push_back(I(1));  // Remote

  for (int i = 1; i < GetSrcNodeOutputSize(); ++i) {
    in_args.push_back(GO(i));  // Data
  }

  return std::vector<NodeDef>{
      NodeDef(OpDef{"Send", kMSDomain, 1},
              in_args,
              {GI(0)},  // Signal
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetExpandGradient) {
  ArgDef a = I(0), y = O(0);
  std::vector<Dimension> a_shape = GetShape(a);
  std::vector<Dimension> y_shape = GetShape(y);
  std::vector<int64_t> a_axes;
  ComputeBroadcastBackwardAxes(a_shape, y_shape, &a_axes, nullptr);

  std::vector<NodeDef> output;
  if (a_axes.size() > 0) {
    HandleBroadcasting(GO(0), a, GI(0), a_axes, output);
  } else {
    output.push_back(
        NodeDef("Identity",
                {GO(0)},
                {GI(0)}));
  }

  return output;
}

}  // namespace training
}  // namespace onnxruntime
