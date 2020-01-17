// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include "core/graph/training/gradient_builder.h"
#include "core/graph/training/gradient_builder_registry.h"
#include "core/graph/training/graph_augmenter.h"
#include "onnx/defs/attr_proto_util.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

#define IMPLEMENT_GRADIENT_BUILDER(name) \
  std::vector<NodeDef> name::GetGradientDefsImpl() const

IMPLEMENT_GRADIENT_BUILDER(GetCastGradient) {
  // TODO: handle invalid conversion cases
  const auto& data_type = I(0).type_proto->tensor_type().elem_type();
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
        result.push_back(
            NodeDef("Scale",
                    {IA("dC_reduced")},
                    {dC},
                    {MakeAttribute("scale", beta)}));
      } else {
        result.push_back(
            NodeDef("Identity", {IA("dC_reduced")}, {dC}));
      }
    } else {
      if (has_beta && beta != 1.0f) {
        result.push_back(
            NodeDef("Scale",
                    {dY},
                    {dC},
                    {MakeAttribute("scale", beta)}));
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
  //TODO: split attribute should be used!!!
  //AttributeProto split = MakeAttribute("split", std::vector<int64_t>());

  std::vector<ArgDef> outputs;
  for (int i = 0; i < GetSrcNodeInputSize(); ++i) {
    outputs.push_back(GI(i));
  }
  return std::vector<NodeDef>{
      NodeDef("Split",
              {GO(0)},
              outputs,
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetGatherNDGradient) {
  auto attributes = SrcNodeAttributes();
  ORT_ENFORCE(attributes.at("axis").has_i());
  auto axis = attributes.at("axis").i();
  return std::vector<NodeDef>{
      NodeDef("Shape",
              {I(0)},
              {IA("x_shape")}),
      NodeDef("GatherNDGrad",
              {IA("x_shape"), I(1), GO(0)},
              {GI(0)},
              {MakeAttribute("axis", axis)})};
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

IMPLEMENT_GRADIENT_BUILDER(GetTrainableDropoutGradient) {
  return std::vector<NodeDef>{
      NodeDef("TrainableDropoutGrad",
              {GO(0), O(1), I(1)},
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
      NodeDef("SoftmaxGrad",
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
      NodeDef("GatherGrad",
              {IA("I0_shape"), I(1), GO(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

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

  const ArgDef &a = I(0), b = I(1);

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
  const ArgDef &a = I(0), b = I(1);

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
                  {IA("PreReduceGrad0")},
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
                  {IA("PreReduceGrad1")},
                  {GI(1)}));
    }
  }
  return output;
}

IMPLEMENT_GRADIENT_BUILDER(GetDivGradient) {
  if (IsGradientRequiredForSrcNodeInput(0) && IsGradientRequiredForSrcNodeInput(1)) {
    return std::vector<NodeDef>{
        NodeDef("DivGrad",
                {GO(0), I(0), I(1)},
                {GI(0), GI(1)})};
  } else if (IsGradientRequiredForSrcNodeInput(0)) {
    // Y = A / B, dA = dY / B
    const ArgDef &a = I(0), b = I(1);
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
        NodeDef("DivGrad",
                {GO(0), I(0), I(1)},
                // TODO: this IA("") does not cause kernel to know it is unneeded.
                // Gradient for the first input is still calculated.
                {IA(""), GI(1)})};
  } else {
    return std::vector<NodeDef>{};
  }
}

IMPLEMENT_GRADIENT_BUILDER(GetReduceMeanGradient) {
  std::vector<Dimension> data_shape = GetShape(I(0));
  std::vector<NodeDef> result;

  auto attributes = SrcNodeAttributes();
  std::vector<int64_t> axes_values(data_shape.size());
  if (attributes.find("axes") != attributes.end()) {
    axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
  } else {
    std::iota(std::begin(axes_values), std::end(axes_values), 0);
  }

  bool keepdims = true;
  if (attributes.find("keepdims") != attributes.end() &&
      attributes.at("keepdims").has_i()) {
    keepdims = static_cast<bool>(attributes.at("keepdims").i());
  }

  ArgDef unsqueezed_Grad = GO(0);
  if (!keepdims) {
    unsqueezed_Grad = IA("Unqueezed_Grad");
    result.push_back(
        NodeDef("Unsqueeze",
                {GO(0)},
                {unsqueezed_Grad},
                {MakeAttribute("axes", axes_values)}));
  }

  std::vector<int64_t> repeats(data_shape.size(), 1);
  int64_t scale = 1;
  for (int64_t axis : axes_values) {
    if (axis < 0) {
      axis = data_shape.size() + axis;
    }

    if (data_shape[axis].has_dim_value()) {
      auto dim_value = data_shape[axis].dim_value();
      repeats[axis] = dim_value;
      scale *= dim_value;
    } else {
      ORT_THROW("Error: can't infer scale for ReduceMeanGrad");
    }
  }

  NodeDef repeats_node = ConstantValueNode(repeats, Name("repeats"));
  ArgDef REPEATS = repeats_node.output_args[0];
  result.push_back(repeats_node);
  result.push_back(
      NodeDef("Tile",
              {unsqueezed_Grad, REPEATS},
              {IA("Tiled_Grad", IType(0))}));

  NodeDef scale_node = ConstantValueNode(1.0f / static_cast<float>(scale), Name("Scale"));
  ArgDef SCALE = scale_node.output_args[0];
  result.push_back(scale_node);
  result.push_back(
      NodeDef("Mul",
              {IA("Tiled_Grad"), SCALE},
              {GI(0)}));

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
      NodeDef(OpDef{"SoftmaxCrossEntropyGrad"},
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

IMPLEMENT_GRADIENT_BUILDER(GetGlobalAveragePoolGradient) {
  const ArgDef& X = I(0);

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
  return std::vector<NodeDef>{
      NodeDef("Scale",
              {GO(0)},
              {IA("scaled_dY")},
              {MakeAttribute("scale", 1.0f / static_cast<float>(scale))}),
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

IMPLEMENT_GRADIENT_BUILDER(GetLayerNormalizationGradient) {
  return std::vector<NodeDef>{
      NodeDef("LayerNormalizationGrad",
              {GO(0), I(0), I(1), O(1), O(2)},
              {GI(0), GI(1), GI(2)},
              {SrcNodeAttributes()})};
}

IMPLEMENT_GRADIENT_BUILDER(GetBatchNormalizationGradient) {
  auto attributes = SrcNodeAttributes();
  if (attributes.find("epsilon") != attributes.end()) {
    float epsilon = attributes.at("epsilon").f();
    return std::vector<NodeDef>{
        NodeDef("BatchNormalizationGrad",
                {GO(0), I(0), I(1), O(3), O(4)},
                {GI(0), GI(1), GI(2)},
                {MakeAttribute("epsilon", epsilon)})};
  } else {
    return std::vector<NodeDef>{
        NodeDef("BatchNormalizationGrad",
                {GO(0), I(0), I(1), O(3), O(4)},
                {GI(0), GI(1), GI(2)})};
  }
}

}  // namespace training
}  // namespace onnxruntime
