// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/gradient_builder.h"

#include <cmath>
#include <numeric>
#include <list>

#include "onnx/defs/attr_proto_util.h"
#include "onnx/defs/tensor_proto_util.h"

#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/core/graph/gradient_builder_registry.h"
#include "orttraining/core/graph/graph_augmenter.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

static bool SimplifyReshape(const std::vector<Dimension>& target_shape,  // the output shape of Reshape
                            const std::vector<Dimension>& source_shape,  // the input shape of Reshape
                            TensorProto& shape_tensor) {                 // the simplified shape tensor if succeeded
  std::vector<int64_t> shape_const;
  std::list<std::string> target_dim_params;
  std::list<std::string> source_dim_params;
  auto get_dim_params = [](const std::vector<Dimension>& shape, std::list<std::string>& dim_params) {
    for (const auto& dim : shape) {
      if (utils::HasDimParam(dim)) {
        dim_params.push_back(dim.dim_param());
      } else if (utils::HasDimValue(dim)) {
        dim_params.push_back("");
      } else {
        return false;
      }
    }
    //trim empty strings in the tail of list
    while (!dim_params.empty() && dim_params.back().empty()) {
      dim_params.pop_back();
    }
    return true;
  };

  if (get_dim_params(target_shape, target_dim_params) &&
      get_dim_params(source_shape, source_dim_params) &&
      target_dim_params == source_dim_params) {
    for (const auto& dim : target_shape) {
      if (utils::HasDimParam(dim)) {
        shape_const.push_back(0);
      } else {
        shape_const.push_back(dim.dim_value());
      }
    }
    auto t = ToTensor<int64_t>(shape_const);
    t.add_dims(shape_const.size());
    shape_tensor.CopyFrom(t);
    return true;
  }
  return false;
}

#define IMPLEMENT_GRADIENT_BUILDER(name) \
  std::vector<NodeDef> name::GetGradientDefsImpl() const

IMPLEMENT_GRADIENT_BUILDER(GetCastGradient) {
  // TODO: handle invalid conversion cases
  return std::vector<NodeDef>{
      NodeDef("Cast",
              {GO(0)},
              {GI(0)},
              {MakeAttribute("to", int64_t(IElemType(0)))})};
}

IMPLEMENT_GRADIENT_BUILDER(GetSinGradient) {
  return std::vector<NodeDef>{
      NodeDef("SinGrad",
              {GO(0), I(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetLogGradient) {
  return std::vector<NodeDef>{
      NodeDef("Div",
              {GO(0), I(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetTanhGradient) {
  ArgDef Y = O(0);
  std::vector<NodeDef> result;
  NodeDef one_constant_node = OneConstantNode(OElemType(0));
  ArgDef one_arg = one_constant_node.output_args[0];
  result.push_back(one_constant_node);
  result.push_back(NodeDef("Mul", {Y, Y}, {IA("Squared_Y")}));
  result.push_back(NodeDef("Sub", {one_arg, IA("Squared_Y")}, {IA("Sub_Squared_Y")}));
  result.push_back(NodeDef("Mul", {GO(0), IA("Sub_Squared_Y")}, {GI(0)}));
  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetSqrtGradient) {
  std::vector<NodeDef> result;
  NodeDef half_constant_node = HalfConstantNode(OElemType(0));
  ArgDef half_arg = half_constant_node.output_args[0];
  result.push_back(half_constant_node);
  result.push_back(NodeDef("Div", {half_arg, O(0)}, {IA("Div_O0")}));
  result.push_back(NodeDef("Mul", {GO(0), IA("Div_O0")}, {GI(0)}));
  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetErfGradient) {
  ArgDef X = I(0);
  std::vector<NodeDef> result;
  NodeDef two_sqrt_pi_node = ConstantScalarNode(static_cast<float>(M_2_SQRTPI), Name("Two_Sqrt_Pi"), IElemType(0));
  ArgDef two_sqrt_pi_arg = two_sqrt_pi_node.output_args[0];
  result.push_back(two_sqrt_pi_node);
  result.push_back(NodeDef("Mul", {X, X}, {IA("Squared_X")}));
  result.push_back(NodeDef("Neg", {IA("Squared_X")}, {IA("Neg_Squared_X")}));
  result.push_back(NodeDef("Exp", {IA("Neg_Squared_X")}, {IA("Exp_Neg_Squared_X")}));
  result.push_back(NodeDef("Mul", {two_sqrt_pi_arg, IA("Exp_Neg_Squared_X")}, {IA("Mul_Exp_Neg_Squared_X")}));
  result.push_back(NodeDef("Mul", {GO(0), IA("Mul_Exp_Neg_Squared_X")}, {GI(0)}));
  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetMatMulGradient) {
  std::vector<NodeDef> result;

  ArgDef A = I(0), B = I(1), Y = O(0);
  std::vector<Dimension> A_shape, B_shape, Y_shape;
  const bool A_has_shape = GetShape(A, A_shape).IsOK();
  const bool B_has_shape = GetShape(B, B_shape).IsOK();
  const bool Y_has_shape = GetShape(Y, Y_shape).IsOK();

  auto dB_2d_case = [&]() {
    if (B_shape[0].has_dim_value() && B_shape[1].has_dim_value()) {
      // B[K, N] is a weight with known size
      int64_t K = B_shape[0].dim_value();
      int64_t N = B_shape[1].dim_value();

      std::vector<int64_t> A_shape_2d{-1, K};
      NodeDef A_target_shape_node = ConstantVectorNode(A_shape_2d, Name("A_target_shape"));

      std::vector<int64_t> dY_shape_2d{-1, N};
      NodeDef dY_target_shape_node = ConstantVectorNode(dY_shape_2d, Name("dY_target_shape"));

      return std::vector<NodeDef>{
          A_target_shape_node,
          dY_target_shape_node,

          // reshape A to 2D [M, K]
          NodeDef("Reshape", {A, A_target_shape_node.output_args[0]}, {IA("A_reshape_2d")}),

          // reshape dY to 2D [M, N]
          NodeDef("Reshape", {GO(0), dY_target_shape_node.output_args[0]}, {IA("dY_reshape_2d")}),

          // dB = A' * dY
          NodeDef("Gemm", {IA("A_reshape_2d"), IA("dY_reshape_2d")}, {GI(1)}, {MakeAttribute("transA", int64_t(1))})};
    } else {
      NodeDef zero_int64_const_node = ConstantScalarNode(int64_t{0}, {1}, Name("zero_int64"));
      NodeDef one_const_node = ConstantScalarNode(int64_t{1}, {1}, Name("one"));
      NodeDef neg_one_const_node = ConstantScalarNode(int64_t{-1}, {1}, Name("neg_one"));

      ArgDef ZERO_I = zero_int64_const_node.output_args[0];
      ArgDef ONE = one_const_node.output_args[0];
      ArgDef NEG_ONE = neg_one_const_node.output_args[0];

      return std::vector<NodeDef>{
          zero_int64_const_node,
          one_const_node,
          neg_one_const_node,

          NodeDef("Shape", {B}, {IA("B_shape")}),

          // reshape A to 2D [M, K]
          NodeDef("Gather", {IA("B_shape"), ZERO_I}, {IA("K_dim")}, {MakeAttribute("axis", int64_t(0))}),
          NodeDef("Concat", {NEG_ONE, IA("K_dim")}, {IA("A_target_shape")}, {MakeAttribute("axis", int64_t(0))}),
          NodeDef("Reshape", {A, IA("A_target_shape")}, {IA("A_reshape_2d")}),

          // reshape dY to 2D [M, N]
          NodeDef("Gather", {IA("B_shape"), ONE}, {IA("N_dim")}, {MakeAttribute("axis", int64_t(0))}),
          NodeDef("Concat", {NEG_ONE, IA("N_dim")}, {IA("dY_target_shape")}, {MakeAttribute("axis", int64_t(0))}),
          NodeDef("Reshape", {GO(0), IA("dY_target_shape")}, {IA("dY_reshape_2d")}),

          // dB = A' * dY
          NodeDef("Gemm", {IA("A_reshape_2d"), IA("dY_reshape_2d")}, {GI(1)}, {MakeAttribute("transA", int64_t(1))})};
    }
  };

  if (A_has_shape && B_has_shape && Y_has_shape &&
      A_shape.size() >= 2 && B_shape.size() >= 2) {
    std::vector<AttributeProto> shared_attributes;
    shared_attributes.push_back(MakeAttribute("beta", float(0)));
    AttributeProto transpose_first_input = MakeAttribute("transA", int64_t(1));
    AttributeProto transpose_second_input = MakeAttribute("transB", int64_t(1));

    if (A_shape.size() == 2 && B_shape.size() == 2) {
      // is GI(0) required
      if (IsGradientRequiredForSrcNodeInput(0)) {
        // dA = dY * B'
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_second_input);
        result.push_back(
            NodeDef("Gemm",
                    {GO(0), B},
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
                    {A, GO(0)},
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
        ComputeBroadcastBackwardAxes(A_shape, output_shape, &A_axes, nullptr, NodeName());

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
          AddReduceSumNode(IA("PreReduceGrad0"), IA("ReduceGrad0"), A_axes, true, result);
          result.push_back(NodeDef("Shape", {A}, {IA("A_shape")}));
          result.push_back(NodeDef("Reshape", {IA("ReduceGrad0"), IA("A_shape")}, {GI(0)}));
        }
      }
      if (IsGradientRequiredForSrcNodeInput(1)) {
        if (B_shape.size() == 2) {
          // for case: A[M1, M2, ... , K], B[K, N], Y[M1, M2, ..., N]
          const std::vector<NodeDef> dB_subgraph = dB_2d_case();
          result.insert(result.end(), dB_subgraph.begin(), dB_subgraph.end());
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
          ComputeBroadcastBackwardAxes(B_shape, output_shape, &B_axes, nullptr, NodeName());

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
            AddReduceSumNode(IA("PreReduceGrad1"), IA("ReduceGrad1"), B_axes, false, result);
            result.push_back(NodeDef("Shape", {B}, {IA("B_shape")}));
            result.push_back(NodeDef("Reshape", {IA("ReduceGrad1"), IA("B_shape")}, {GI(1)}));
          }
        }
      }
    }
  } else {
    //GetShape failed, build shape-independent gradient graph
    ArgDef a_axes, b_axes, a_shape, b_shape, ia_shape;
    a_shape = IA("Shape_" + A.name);
    b_shape = IA("Shape_" + B.name);

    if (IsGradientRequiredForSrcNodeInput(0)) {
      ArgDef pre_reduce_grad_0 = IA("PreReduceGrad0");
      result.push_back(
          NodeDef(OpDef{"FusedMatMul", kMSDomain, 1},
                  {GO(0), B},
                  {pre_reduce_grad_0},
                  {{"transB", MakeAttribute("transB", int64_t(1))}}));

      a_axes = IA("ReduceAxes_" + A.name + "_for_" + A.name);
      ia_shape = IA("Shape_" + pre_reduce_grad_0.name);
      ComputeBroadcastBackwardAxesDynamic(A, pre_reduce_grad_0, a_shape, ia_shape, &a_axes, nullptr, result);
      HandleBroadcastingDynamic(pre_reduce_grad_0, A, a_shape, GI(0), a_axes, result);
    }
    if (IsGradientRequiredForSrcNodeInput(1)) {
      if (B_has_shape && B_shape.size() == 2) {
        // for case: A[M1, M2, ... , K], B[K, N], Y[M1, M2, ..., N]
        const std::vector<NodeDef> dB_subgraph = dB_2d_case();
        result.insert(result.end(), dB_subgraph.begin(), dB_subgraph.end());
      } else {
        ArgDef pre_reduce_grad_1 = IA("PreReduceGrad1");
        result.push_back(
            NodeDef(OpDef{"FusedMatMul", kMSDomain, 1},
                    {A, GO(0)},
                    {pre_reduce_grad_1},
                    {{"transA", MakeAttribute("transA", int64_t(1))}}));

        b_axes = IA("ReduceAxes_" + B.name + "_for_" + B.name);
        ia_shape = IA("Shape_" + pre_reduce_grad_1.name);
        ComputeBroadcastBackwardAxesDynamic(pre_reduce_grad_1, B, ia_shape, b_shape, nullptr, &b_axes, result);
        HandleBroadcastingDynamic(pre_reduce_grad_1, B, b_shape, GI(1), b_axes, result);
      }
    }
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
  int elem_type = OElemType(0);
  AttributeProto transpose_first_input = MakeAttribute("transA", int64_t(1));
  AttributeProto transpose_second_input = MakeAttribute("transB", int64_t(1));

  std::vector<NodeDef> result;

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
        result.push_back(NodeDef("Gemm", {B, dY}, {dA}, attrs));
      }

      if (IsGradientRequiredForSrcNodeInput(1)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_first_input);
        attrs.push_back(transpose_second_input);
        result.push_back(NodeDef("Gemm", {dY, A}, {dB}, attrs));
      }
    } else {
      // Y = alpha * A' * B
      // dA = alpha * B * dY', dB = alpha * A * dY
      if (IsGradientRequiredForSrcNodeInput(0)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_second_input);
        result.push_back(NodeDef("Gemm", {B, dY}, {dA}, attrs));
      }

      if (IsGradientRequiredForSrcNodeInput(1)) {
        result.push_back(NodeDef("Gemm", {A, dY}, {dB}, shared_attributes));
      }
    }
  } else {
    if (transB) {
      // Y = alpha * A * B'
      // dA = alpha * dY * B, dB = alpha * dY' * A
      if (IsGradientRequiredForSrcNodeInput(0)) {
        result.push_back(NodeDef("Gemm", {dY, B}, {dA}, shared_attributes));
      }

      if (IsGradientRequiredForSrcNodeInput(1)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_first_input);
        result.push_back(NodeDef("Gemm", {dY, A}, {dB}, attrs));
      }
    } else {
      // Y = alpha * A * B
      // dA = alpha * dY * B', dB = alpha * A' * dY
      if (IsGradientRequiredForSrcNodeInput(0)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_second_input);
        result.push_back(NodeDef("Gemm", {dY, B}, {dA}, attrs));
      }

      if (IsGradientRequiredForSrcNodeInput(1)) {
        std::vector<AttributeProto> attrs(shared_attributes);
        attrs.push_back(transpose_first_input);
        result.push_back(NodeDef("Gemm", {A, dY}, {dB}, attrs));
      }
    }
  }

  if (IsGradientRequiredForSrcNodeInput(2)) {
    // Y = beta * C
    // dC = beta * dY
    bool has_beta = attributes.at("beta").has_f();
    float beta = attributes.at("beta").f();
    ORT_ENFORCE(beta != 0.0f);
    std::vector<Dimension> C_shape, dY_shape;
    if (GetShape(C, C_shape).IsOK() && GetShape(dY, dY_shape).IsOK()) {
      std::vector<int64_t> C_axes, dY_axes;
      ComputeBroadcastBackwardAxes(C_shape, dY_shape, &C_axes, &dY_axes, NodeName());

      if (C_axes.size() > 0) {
        HandleBroadcasting(dY, C, IA("dC_reduced"), C_axes, result);

        if (has_beta && beta != 1.0f) {
          NodeDef scale_node = ConstantScalarNode(beta, Name("Scale"), elem_type);
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
          NodeDef scale_node = ConstantScalarNode(beta, Name("Scale"), elem_type);
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
    } else {
      //GetShape failed, build shape-independent gradient graph
      ArgDef c_axes = IA("ReduceAxes_" + C.name);
      ArgDef c_shape = IA("Shape_" + C.name);
      ArgDef dy_shape = IA("Shape_" + dY.name);

      ComputeBroadcastBackwardAxesDynamic(C, dY, c_shape, dy_shape, &c_axes, nullptr, result);

      HandleBroadcastingDynamic(dY, C, c_shape, IA("dC_reduced"), c_axes, result);

      if (has_beta && beta != 1.0f) {
        NodeDef scale_node = ConstantScalarNode(beta, Name("Scale"), elem_type);
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
    std::vector<Dimension> data_shape;
    ORT_ENFORCE(GetShape(I(i), data_shape).IsOK());
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

IMPLEMENT_GRADIENT_BUILDER(GetConcatTrainingGradient) {
  auto attributes = SrcNodeAttributes();
  ORT_ENFORCE(utils::HasInt(attributes.at("axis")));
  auto axis = attributes.at("axis").i();

  std::vector<int64_t> split_attribute(GetSrcNodeInputSize());
  std::vector<ArgDef> outputs;
  bool known_shapes = true;
  for (int i = 0; i < GetSrcNodeInputSize(); ++i) {
    std::vector<Dimension> data_shape;
    if (GetShape(I(i), data_shape).IsOK()) {
      int64_t rank = static_cast<int64_t>(data_shape.size());
      int64_t axis_index = HandleNegativeAxis(axis, rank);
      if (data_shape[axis_index].has_dim_value()) {
        split_attribute[i] = data_shape[axis_index].dim_value();
      } else {
        known_shapes = false;
      }
    } else {
      known_shapes = false;
    }

    outputs.push_back(GI(i));
  }

  std::vector<AttributeProto> new_attributes;
  new_attributes.push_back(MakeAttribute("axis", axis));
  if (known_shapes) {
    new_attributes.push_back(MakeAttribute("split", split_attribute));
    return std::vector<NodeDef>{
        NodeDef("Split",
                {GO(0)},
                outputs,
                new_attributes)};
  } else {
    return std::vector<NodeDef>{
        NodeDef(OpDef{"SplitTraining", kMSDomain, 1},
                {GO(0), O(1)},
                outputs,
                new_attributes)};
  }
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
  std::vector<Dimension> target_shape;
  std::vector<Dimension> source_shape;
  if (GetShape(I(0), target_shape).IsOK() &&
      GetShape(GO(0), source_shape).IsOK()) {
    TensorProto shape_tensor;
    if (SimplifyReshape(target_shape,
                        source_shape,
                        shape_tensor)) {
      return std::vector<NodeDef>{
          NodeDef("Constant",
                  {},
                  {IA("x_shape")},
                  {MakeAttribute("value", shape_tensor)}),
          NodeDef("Reshape",
                  {GO(0), IA("x_shape")},
                  {GI(0)})};
    }
  }
  return std::vector<NodeDef>{
      NodeDef("Shape", {I(0)}, {IA("x_shape")}),
      NodeDef("Reshape", {GO(0), IA("x_shape")}, {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetTransposeGradient) {
  std::vector<int64_t> bw_perm;
  auto attributes = SrcNodeAttributes();
  std::vector<AttributeProto> new_attributes;
  if (attributes.empty()) {
    const TensorShapeProto& input_shape = I(0).type_proto->tensor_type().shape();
    if (input_shape.dim_size() > 0) {  //input_shape is available
      int n = input_shape.dim_size() - 1;
      bw_perm.resize(n + 1);
      std::generate(bw_perm.begin(), bw_perm.end(), [&n] { return n--; });
      new_attributes.push_back(MakeAttribute("perm", bw_perm));
    }
  } else {
    auto fw_perm = RetrieveValues<int64_t>(attributes.at("perm"));
    auto size = fw_perm.size();
    bw_perm.resize(size);
    for (int i = 0; i < static_cast<int>(size); ++i) {
      bw_perm[fw_perm[i]] = i;
    }
    new_attributes.push_back(MakeAttribute("perm", bw_perm));
  }

  return std::vector<NodeDef>{
      NodeDef("Transpose",
              {GO(0)},
              {GI(0)},
              new_attributes)};
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

IMPLEMENT_GRADIENT_BUILDER(GetConvGradient) {
  std::vector<ArgDef> outputs;
  for (int i = 0; i < GetSrcNodeInputSize(); i++) {
    if (IsGradientRequiredForSrcNodeInput(i)) {
      outputs.push_back(GI(i));
    } else {
      outputs.push_back(ArgDef("", nullptr));
    }
  }

  return std::vector<NodeDef>{
      NodeDef(OpDef{"ConvGrad", kMSDomain, 1},
              {GO(0), I(0), I(1)},
              outputs,
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetSigmoidGradient) {
  auto const_one = OneConstantNode(OElemType(0));
  return std::vector<NodeDef>{
      const_one,
      NodeDef("Sub",
              {const_one.output_args[0], O(0)},
              {IA("one_minus_output")}),
      NodeDef("Mul",
              {IA("one_minus_output"), O(0)},
              {IA("sigmoid_derivate")}),
      NodeDef("Mul",
              {IA("sigmoid_derivate"), GO(0)},
              {GI(0)})};
}
IMPLEMENT_GRADIENT_BUILDER(GetSoftmaxGradient) {
  return std::vector<NodeDef>{
      NodeDef(OpDef{"SoftmaxGrad", kMSDomain, 1},
              {GO(0), O(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetLogSoftmaxGradient) {
  return std::vector<NodeDef>{
      NodeDef(OpDef{"LogSoftmaxGrad", kMSDomain, 1},
              {GO(0), O(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetUnsqueezeGradient) {
  if (SrcNodeOpsetVersion() < 13) {
    return std::vector<NodeDef>{
        NodeDef("Squeeze",
                {GO(0)},
                {GI(0)},
                SrcNodeAttributes())};
  } else { // mandatory input 'axes' since opset 13
    return std::vector<NodeDef>{
        NodeDef(OpDef{"Squeeze", kOnnxDomain, 13},
                {GO(0), I(1)},
                {GI(0)},
                SrcNodeAttributes())};
  }
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
      NodeDef(OpDef{"ReluGrad", kMSDomain, 1},
              {GO(0), O(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetSqueezeGradient) {
  std::vector<NodeDef> result;
  size_t numInputs = GetSrcNodeInputSize();
  if (SrcNodeOpsetVersion() < 13) { //axes attribute
    auto attributes = SrcNodeAttributes();
    std::vector<int64_t> axes_values;
    if (attributes.find("axes") != attributes.end()) {
      axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
      result.push_back(
          NodeDef("Unsqueeze",
                  {GO(0)},
                  {GI(0)},
                  {MakeAttribute("axes", axes_values)}));
    } 
  } else if(numInputs == 2){ //optional input 'axes' is provided
    result.push_back(
        NodeDef(OpDef{"Unsqueeze", kOnnxDomain, 13},
                {GO(0), I(1)},
                {GI(0)}));
  } else { // if axes attribute/input not provided for squeeze
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
  std::vector<NodeDef> output;
  std::vector<Dimension> a_shape, b_shape;
  if (GetShape(a, a_shape).IsOK() && GetShape(b, b_shape).IsOK()) {
    std::vector<int64_t> a_axes, b_axes;
    ComputeBroadcastBackwardAxes(a_shape, b_shape, &a_axes, &b_axes, NodeName());
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
  } else {
    //GetShape failed, build shape-independent gradient graph
    ArgDef a_axes = IA("ReduceAxes_" + a.name);
    ArgDef b_axes = IA("ReduceAxes_" + b.name);
    ArgDef A_shape = IA("Shape_" + a.name);
    ArgDef B_shape = IA("Shape_" + b.name);
    ComputeBroadcastBackwardAxesDynamic(a, b, A_shape, B_shape, &a_axes, &b_axes, output);

    if (IsGradientRequiredForSrcNodeInput(0)) {
      HandleBroadcastingDynamic(GO(0), a, A_shape, GI(0), a_axes, output);
    }

    if (IsGradientRequiredForSrcNodeInput(1)) {
      ArgDef reshape_output = is_sub ? IA("ReshapeReduceSum_2", IType(1)) : GI(1);
      HandleBroadcastingDynamic(GO(0), b, B_shape, reshape_output, b_axes, output);

      if (is_sub) {
        output.push_back(
            NodeDef("Neg",
                    {reshape_output},
                    {GI(1)}));
      }
    }
  }
  return output;
}

IMPLEMENT_GRADIENT_BUILDER(GetMulGradient) {
  const ArgDef a = I(0), b = I(1);

  std::vector<NodeDef> output;
  std::vector<Dimension> a_shape, b_shape;
  if (GetShape(a, a_shape).IsOK() && GetShape(b, b_shape).IsOK()) {
    std::vector<int64_t> a_axes, b_axes;
    ComputeBroadcastBackwardAxes(a_shape, b_shape, &a_axes, &b_axes, NodeName());

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
  } else {
    //GetShape failed, build shape-independent gradient graph
    ArgDef a_axes = IA("ReduceAxes_" + a.name);
    ArgDef b_axes = IA("ReduceAxes_" + b.name);
    ArgDef A_shape = IA("Shape_" + a.name);
    ArgDef B_shape = IA("Shape_" + b.name);
    ComputeBroadcastBackwardAxesDynamic(a, b, A_shape, B_shape, &a_axes, &b_axes, output);

    if (IsGradientRequiredForSrcNodeInput(0)) {
      output.push_back(
          NodeDef("Mul",
                  {GO(0), I(1)},
                  {IA("PreReduceGrad0", OType(0))}));

      HandleBroadcastingDynamic(IA("PreReduceGrad0", OType(0)), a, A_shape, GI(0), a_axes, output);
    }

    if (IsGradientRequiredForSrcNodeInput(1)) {
      output.push_back(
          NodeDef("Mul",
                  {GO(0), I(0)},
                  {IA("PreReduceGrad1", OType(0))}));

      HandleBroadcastingDynamic(IA("PreReduceGrad1", OType(0)), b, B_shape, GI(1), b_axes, output);
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
    std::vector<NodeDef> output;
    std::vector<Dimension> a_shape, b_shape;
    if (GetShape(a, a_shape).IsOK() && GetShape(b, b_shape).IsOK()) {
      std::vector<int64_t> a_axes, b_axes;
      ComputeBroadcastBackwardAxes(a_shape, b_shape, &a_axes, &b_axes, NodeName());

      ArgDef tmp_grad = IA("PreReduceGrad0", OType(0));
      output.push_back(NodeDef("Div", {GO(0), I(1)}, {tmp_grad}));
      if (a_axes.size() > 0) {
        HandleBroadcasting(tmp_grad, a, GI(0), a_axes, output);
      } else {
        output.push_back(NodeDef("Identity", {tmp_grad}, {GI(0)}));
      }
    } else {
      //GetShape failed, build shape-independent gradient graph
      ArgDef a_axes = IA("ReduceAxes_" + a.name);
      ArgDef A_shape = IA("Shape_" + a.name);
      ArgDef B_shape = IA("Shape_" + b.name);

      ComputeBroadcastBackwardAxesDynamic(a, b, A_shape, B_shape, &a_axes, nullptr, output);

      ArgDef tmp_grad = IA("PreReduceGrad0", OType(0));
      output.push_back(NodeDef("Div", {GO(0), I(1)}, {tmp_grad}));
      HandleBroadcastingDynamic(tmp_grad, a, A_shape, GI(0), a_axes, output);
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

IMPLEMENT_GRADIENT_BUILDER(GetNegGradient) {
  return std::vector<NodeDef>{
      NodeDef("Neg", {GO(0)}, {GI(0)})};
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

  result.push_back(NodeDef("Size", {I(0)}, {IA("Sized_X")}));
  result.push_back(NodeDef("Size", {GO(0)}, {IA("Sized_Grad")}));
  result.push_back(NodeDef("Div", {IA("Sized_X"), IA("Sized_Grad")}, {IA("Scale")}));
  result.push_back(NodeDef(OpDef{"Scale", kMSDomain, 1},
                           {grad, IA("Scale")},
                           {IA("Scaled_Grad")},
                           {MakeAttribute("scale_down", int64_t(1))}));
  result.push_back(NodeDef("Shape", {I(0)}, {IA("Shaped_X")}));
  result.push_back(NodeDef("Expand", {IA("Scaled_Grad"), IA("Shaped_X")}, {GI(0)}));
  return result;
}

// Reference computation is pytorch's logsumexp_backward
// dx_i = exp(xi) / reduceSum(exp(xi))
// O(0) = log(reduceSum(exp(xi)))
// Self_Sub_Result = I(0) - O(0) = xi - log(sum(exp(xi))) = log( xi / reduceSum(exp(xi)))
// Gradient computation is re-using output and input from forward op, can be a recomputation candidate.
IMPLEMENT_GRADIENT_BUILDER(GetReduceLogSumExpGradient) {
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
    grad = IA("Unsqueezed_Grad");
    result.push_back(NodeDef("Unsqueeze", {GO(0)}, {grad}, {MakeAttribute("axes", axes_values)}));

    result.push_back(NodeDef("Unsqueeze", {O(0)}, {IA("Unsqueezed_Output")}, {MakeAttribute("axes", axes_values)}));
    result.push_back(NodeDef("Sub", {I(0), IA("Unsqueezed_Output")}, {IA("Self_Sub_Result")}));
  } else {
    result.push_back(NodeDef("Sub", {I(0), O(0)}, {IA("Self_Sub_Result")}));
  }

  result.push_back(NodeDef("Exp", {IA("Self_Sub_Result")}, {IA("Self_Sub_Result_Exp")}));

  result.push_back(NodeDef("Mul", {IA("Self_Sub_Result_Exp"), grad}, {GI(0)}));

  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetReduceL2Gradient) {
  std::vector<NodeDef> result;
  auto attributes = SrcNodeAttributes();
  bool keepdims = true;
  if (attributes.find("keepdims") != attributes.end() && attributes.at("keepdims").has_i()) {
    keepdims = static_cast<bool>(attributes.at("keepdims").i());
  }

  result.emplace_back(NodeDef("Div", {GO(0), O(0)}, {IA("Scaled_dY")}));

  // Handle 0 elements in Y.
  NodeDef zero_constant_node = ZeroConstantNode(IElemType(0));
  ArgDef ZERO = zero_constant_node.output_args[0];
  result.push_back(zero_constant_node);
  result.emplace_back(NodeDef("Equal", {O(0), ZERO}, {IA("Masked_Y")}));
  ArgDef scaled_dy_arg_def = IA("Masked_Scaled_dY");
  result.emplace_back(NodeDef("Where", {IA("Masked_Y"), ZERO, IA("Scaled_dY")}, {scaled_dy_arg_def}));

  if (!keepdims && attributes.find("axes") != attributes.end()) {
    std::vector<int64_t> axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
    scaled_dy_arg_def = IA("Unsqueezed_Masked_Scaled_dY");
    result.emplace_back(
        NodeDef("Unsqueeze", {IA("Masked_Scaled_dY")}, {scaled_dy_arg_def}, {MakeAttribute("axes", axes_values)}));
  }

  result.emplace_back(NodeDef("Mul", {I(0), scaled_dy_arg_def}, {GI(0)}));
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
  if (!keepdims) {
    size_t numInputs = GetSrcNodeInputSize();
    if (SrcNodeOpsetVersion() < 13) {  //axes is attribute
      if (attributes.find("axes") != attributes.end()) {
        std::vector<int64_t> axes_values = RetrieveValues<int64_t>(attributes.at("axes"));

        grad = IA("Unqueezed_Grad");
        result.push_back(NodeDef("Unsqueeze", {GO(0)}, {grad}, {MakeAttribute("axes", axes_values)}));

      } 
    } else if (numInputs == 2) {  //optional input 'axes' is available as input I(1)
      grad = IA("Unqueezed_Grad");
      result.push_back(NodeDef(OpDef{"Unsqueeze", kOnnxDomain, 13}, {GO(0), I(1)}, {grad}));
    } //axes is not available, the GO(0) is a scalar which can be expanded to required shape
  }

  result.push_back(NodeDef("Shape", {I(0)}, {IA("Shaped_X")}));
  result.push_back(NodeDef("Expand", {grad, IA("Shaped_X")}, {GI(0)}));
  return result;
}

IMPLEMENT_GRADIENT_BUILDER(GetPowGradient) {
  if (IsGradientRequiredForSrcNodeInput(1)) {
    ORT_THROW("GradientBuilder is not implemented for CUDA Pow's input exponent.");
  }

  std::vector<NodeDef> result;
  NodeDef one_constant_node = OneConstantNode(IElemType(0));
  ArgDef one_arg = one_constant_node.output_args[0];
  result.push_back(one_constant_node);
  result.push_back(NodeDef("Sub", {I(1), one_arg}, {IA("Sub_I1")}));
  result.push_back(NodeDef("Pow", {I(0), IA("Sub_I1")}, {IA("Pow_I0")}));
  result.push_back(NodeDef("Mul", {IA("Pow_I0"), I(1)}, {IA("Mul_Pow_I0_I1")}));
  result.push_back(NodeDef("Mul", {IA("Mul_Pow_I0_I1"), GO(0)}, {GI(0)}));
  return result;
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

  NodeDef scale_node = ConstantScalarNode(1.0f / static_cast<float>(scale), Name("Scale"), IElemType(0));
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

IMPLEMENT_GRADIENT_BUILDER(GetBiasGeluGradient) {
  const auto dY = GO(0), X = I(0), B = I(1),
             dX = GI(0), dB = GI(1);
  ArgDef b_axes = IA("ReduceAxes_" + B.name);
  ArgDef b_shape = IA("Shape_" + B.name);
  ArgDef x_shape = IA("Shape_" + X.name);
  return GetBiasGeluGradNodes(false, dY, X, B, dX, dB, b_axes, b_shape, x_shape, NodeName());
}

IMPLEMENT_GRADIENT_BUILDER(GetFastGeluGradient) {
  const auto dY = GO(0), X = I(0),
             dX = GI(0);
  const auto num_src_node_inputs = GetSrcNodeInputSize();
  if (num_src_node_inputs == 2) {  // with bias
    // FastGeluGrad doesn't support bias - it needs to be composed with other ops
    const auto B = I(1),
               dB = GI(1);
    ArgDef b_axes = IA("ReduceAxes_" + B.name);
    ArgDef b_shape = IA("Shape_" + B.name);
    ArgDef x_shape = IA("Shape_" + X.name);
    return GetBiasGeluGradNodes(true, dY, X, B, dX, dB, b_axes, b_shape, x_shape, NodeName());
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

IMPLEMENT_GRADIENT_BUILDER(GetSimplifiedLayerNormalizationGradient) {
  return std::vector<NodeDef>{
      NodeDef(OpDef{"SimplifiedLayerNormalizationGrad", kMSDomain, 1},
              {GO(0), I(0), I(1), O(1)},
              {GI(0), GI(1)},
              {SrcNodeAttributes()})};
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
  const int64_t data_type = static_cast<int64_t>(IElemType(1));
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
  std::vector<NodeDef> output;

  std::vector<Dimension> a_shape, y_shape;
  if (GetShape(a, a_shape).IsOK() && GetShape(y, y_shape).IsOK()) {
    std::vector<int64_t> a_axes;
    ComputeBroadcastBackwardAxes(a_shape, y_shape, &a_axes, nullptr, NodeName());

    if (a_axes.size() > 0) {
      HandleBroadcasting(GO(0), a, GI(0), a_axes, output);
    } else {
      output.push_back(
          NodeDef("Identity",
                  {GO(0)},
                  {GI(0)}));
    }
  } else {
    //GetShape failed, build shape-independent gradient graph
    ArgDef a_axes = IA("ReduceAxes_" + a.name);
    ArgDef A_shape = IA("Shape_" + a.name);
    ArgDef Y_shape = IA("Shape_" + y.name);

    ComputeBroadcastBackwardAxesDynamic(a, y, A_shape, Y_shape, &a_axes, nullptr, output);

    HandleBroadcastingDynamic(GO(0), a, A_shape, GI(0), a_axes, output);
  }

  return output;
}

IMPLEMENT_GRADIENT_BUILDER(GetExpGradient) {
  return std::vector<NodeDef>{
      NodeDef("Mul",
              {GO(0), O(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetFlattenGradient) {
  return std::vector<NodeDef>{
      NodeDef("Shape", {I(0)}, {IA("input_shape")}),
      NodeDef("Reshape", {GO(0), IA("input_shape")}, {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetTopKGradient) {
  // TopK's default axis is -1, which is different from GatherElements.
  auto attributes = SrcNodeAttributes();
  auto axis = utils::HasInt(attributes.at("axis")) ? attributes.at("axis").i() : -1;
  return std::vector<NodeDef>{
      NodeDef("Shape",
              {I(0)},
              {IA("x_shape")}),
      NodeDef(OpDef{"GatherElementsGrad", kMSDomain, 1},
              {GO(0), IA("x_shape"), O(1)},
              {GI(0)},
              {MakeAttribute("axis", axis)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetClipGradient) {
  std::vector<NodeDef> output;
  size_t numInputs = GetSrcNodeInputSize();
  bool has_i1 = false, has_i2 = false;
  ArgDef intermediate_arg_def = ArgDef("");
  // Gradients not defined on min and max, so we return the subgradient 1 for these cases.
  if (numInputs >= 2 && I(1).Exists()) {
    has_i1 = true;
    intermediate_arg_def = IA("Masked_Min");
    output.emplace_back(NodeDef("GreaterOrEqual", {I(0), I(1)}, {intermediate_arg_def}));
  }

  if (numInputs >= 3 && I(2).Exists()) {
    has_i2 = true;
    intermediate_arg_def = IA("Masked_Max");
    output.emplace_back(NodeDef("LessOrEqual", {I(0), I(2)}, {intermediate_arg_def}));
    if (has_i1) {
      intermediate_arg_def = IA("Masked_Min_Max");
      output.emplace_back(NodeDef("And", {IA("Masked_Min"), IA("Masked_Max")}, {intermediate_arg_def}));
    }
  }

  if (!has_i1 && !has_i2) {
    output.emplace_back(NodeDef("Identity", {GO(0)}, {GI(0)}));
  } else {
    output.emplace_back(
        NodeDef("Cast", {intermediate_arg_def}, {IA("Casted_Mask")}, {MakeAttribute("to", int64_t(IElemType(0)))}));
    output.emplace_back(NodeDef("Mul", {GO(0), IA("Casted_Mask")}, {GI(0)}));
  }

  return output;
}

IMPLEMENT_GRADIENT_BUILDER(GetAbsGradient) {
  return std::vector<NodeDef>{
      NodeDef("Sign", {I(0)}, {IA("Sign_Input")}),
      NodeDef("Mul", {GO(0), IA("Sign_Input")}, {GI(0)})
  };
}

}  // namespace training
}  // namespace onnxruntime
