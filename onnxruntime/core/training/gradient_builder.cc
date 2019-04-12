// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/training/gradient_builder.h"
#include "core/training/gradient_builder_registry.h"
#include "core/training/graph_augmenter.h"
#include "core/training/attr_proto_util.h"

namespace onnxruntime {
namespace training {

#define IMPLEMENT_GRADIENT_BUILDER(name) \
  std::vector<NodeDef> name::GetGradientDefs() const

IMPLEMENT_GRADIENT_BUILDER(GetSinGradient) {
  return std::vector<NodeDef>{
      NodeDef("Cos",
              {I(0)},
              {IA("cosx")}),
      NodeDef("Mul",
              {IA("cosx"), GO(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetMatMulGradient) {
  std::vector<NodeDef> result;

  NodeDef zero_contant_node = ZeroConstantNode();
  ArgDef ZERO = zero_contant_node.output_args[0];

  result.push_back(zero_contant_node);

  // is GI(0) required
  if (IsGradientRequiredForSrcNodeInput(0)) {
    // dA = dY * B'
    result.push_back(
        NodeDef("Gemm",
                {GO(0), I(1), ZERO},
                {GI(0)},
                {MakeAttribute("transB", int64_t(1))}));
  }

  // is GI(1) required
  if (IsGradientRequiredForSrcNodeInput(1)) {
    // dB = A' * dY
    result.push_back(
        NodeDef("Gemm",
                {I(0), GO(0), ZERO},
                {GI(1)},
                {MakeAttribute("transA", int64_t(1))}));
  }

  return result;
};

std::vector<int64_t> GetShape(const ArgDef& arg_def) {
  const auto& dims = arg_def.type_proto->tensor_type().shape().dim();
  std::vector<int64_t> shape;
  for (auto dim = dims.begin(); dim < dims.end(); dim++) {
    if (dim->has_dim_value()) {
      shape.push_back(dim->dim_value());
    } else {
      ORT_ENFORCE(false, "Dimension missing");
    }
  }
  return shape;
}

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

    std::vector<int64_t> C_shape = GetShape(C);
    std::vector<int64_t> dY_shape = GetShape(dY);

    std::vector<int64_t> C_axes, dY_axes;
    ComputeBroadcastBackwardAxes(C_shape, dY_shape, &C_axes, &dY_axes);

    if (C_axes.size() > 0) {
      result.push_back(
          NodeDef("ReduceSum",
                  {dY},
                  {IA("dY_ReduceSum")},
                  {{"keepdims", MakeAttribute("keepdims", int64_t(1))},
                   {"axes", MakeAttribute("axes", C_axes)}}));
      result.push_back(
          NodeDef("Shape",
                  {C},
                  {IA("c_shape")}));

      if (has_beta && beta != 1.0f) {
        result.push_back(
            NodeDef("Reshape",
                    {IA("dY_ReduceSum"), IA("c_shape")},
                    {IA("dC_reshaped")}));
        result.push_back(
            NodeDef("Scale",
                    {IA("dC_reshaped")},
                    {dC},
                    {MakeAttribute("scale", beta)}));
      } else {
        result.push_back(
            NodeDef("Reshape",
                    {IA("dY_ReduceSum"), IA("c_shape")},
                    {dC}));
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
    result.push_back(
        NodeDef("Concat",
                input_args,
                {GI(0)}));
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
              outputs)};
}

IMPLEMENT_GRADIENT_BUILDER(GetReshapeGradient) {
  return std::vector<NodeDef>{
      NodeDef("Shape",
              {I(0)},
              {IA("x_shape")}),
      NodeDef("Reshape",
              {GO(0), IA("x_shape")},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetAveragePoolGradient) {
  return std::vector<NodeDef>{
      NodeDef("AveragePoolGrad",
              {GO(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetMaxPoolGradient) {
  return std::vector<NodeDef>{
      NodeDef("MaxPoolGrad",
              {GO(0), O(1)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetPoolGradient) {
  return std::vector<NodeDef>{
      NodeDef(SrcNodeOpType() + "Grad",
              {GO(0), I(0), O(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetLRNGradient) {
  return std::vector<NodeDef>{
      NodeDef("LRNGrad",
              {GO(0), I(0), O(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetDropoutGradient) {
  // TODO: Add is_test to Dropout Op Schema
  bool is_test = false;
  if (is_test) {
    return std::vector<NodeDef>{
        NodeDef("DropoutGrad",
                {GO(0)},
                {GI(0)})};
  } else {
    std::vector<NodeDef> result;
    auto mask = O(1);

    // TODO: In latter version, when the mask type is enforced to tensor(float),
    // this conversion might not be needed anymore
    if (mask.type_proto->tensor_type().elem_type() != TensorProto_DataType_FLOAT) {
      mask = IA("f_mask");
      result.push_back(
          NodeDef("Cast",
                  {O(1)},
                  {mask},
                  {MakeAttribute("to", int64_t(TensorProto_DataType_FLOAT))}));
    }
    result.push_back(
        NodeDef("DropoutGrad",
                {GO(0), mask},
                {GI(0)}));
    return result;
  };
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
              outputs)};
}

IMPLEMENT_GRADIENT_BUILDER(GetSoftmaxGradient) {
  return std::vector<NodeDef>{
      NodeDef("SoftmaxGrad",
              {GO(0), O(0)},
              {GI(0)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetReluGradient) {
  return std::vector<NodeDef>{
      NodeDef("ReluGrad",
              {GO(0), I(0)},
              {GI(0)})};
}

void ComputeBroadcastBackwardAxes(
    const std::vector<int64_t>& A_dims,
    const std::vector<int64_t>& B_dims,
    std::vector<int64_t>* A_axes,
    std::vector<int64_t>* B_axes) {
  A_axes->clear();
  B_axes->clear();

  int ndim = int(std::max(A_dims.size(), B_dims.size()));
  int i = int(A_dims.size() - 1);
  int j = int(B_dims.size() - 1);
  int k = ndim - 1;

  for (; i >= 0 && j >= 0; --k) {
    ORT_ENFORCE(A_dims[i] == B_dims[j] || A_dims[i] == 1 || B_dims[j] == 1);

    if (A_dims[i] != B_dims[j]) {
      if (A_dims[i] == 1) {
        A_axes->push_back(gsl::narrow_cast<int64_t>(k));
      }

      if (B_dims[j] == 1) {
        B_axes->push_back(gsl::narrow_cast<int64_t>(k));
      }
    }

    --i;
    --j;
  }

  if (i < 0) {
    for (; k >= 0; --k) {
      A_axes->push_back(gsl::narrow_cast<int64_t>(k));
    }

  } else {
    for (; k >= 0; --k) {
      B_axes->push_back(gsl::narrow_cast<int64_t>(k));
    }
  }
}

IMPLEMENT_GRADIENT_BUILDER(GetAddGradient) {
  const ArgDef &a = I(0), b = I(1);

  std::vector<int64_t> a_shape = GetShape(a);
  std::vector<int64_t> b_shape = GetShape(b);

  std::vector<int64_t> a_axes, b_axes;
  ComputeBroadcastBackwardAxes(a_shape, b_shape, &a_axes, &b_axes);

  std::vector<NodeDef> output;

  if (IsGradientRequiredForSrcNodeInput(0)) {
    if (a_axes.size() > 0) {
      output.push_back(
          NodeDef("ReduceSum",
                  {GO(0)},
                  {IA("ReduceSum")},
                  {{"keepdims", MakeAttribute("keepdims", int64_t(1))},
                   {"axes", MakeAttribute("axes", a_axes)}}));

      output.push_back(
          NodeDef("Shape",
                  {a},
                  {IA("a_shape")}));

      output.push_back(
          NodeDef("Reshape",
                  {IA("ReduceSum"), IA("a_shape")},
                  {GI(0)}));
    } else {
      output.push_back(
          NodeDef("Identity",
                  {GO(0)},
                  {GI(0)}));
    }
  }

  if (IsGradientRequiredForSrcNodeInput(1)) {
    if (b_axes.size() > 0) {
      output.push_back(
          NodeDef("ReduceSum",
                  {GO(0)},
                  {IA("ReduceSum_2")},
                  {{"keepdims", MakeAttribute("keepdims", int64_t(1))},
                   {"axes", MakeAttribute("axes", b_axes)}}));

      output.push_back(
          NodeDef("Shape",
                  {b},
                  {IA("b_shape")}));

      output.push_back(
          NodeDef("Reshape",
                  {IA("ReduceSum_2"), IA("b_shape")},
                  {GI(1)}));
    } else {
      output.push_back(
          NodeDef("Identity",
                  {GO(0)},
                  {GI(1)}));
    }
  }
  return output;
}

IMPLEMENT_GRADIENT_BUILDER(GetSubGradient) {
  return std::vector<NodeDef>{
      NodeDef("SubGrad",
              {GO(0)},
              {GI(0), GI(1)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetReduceMeanGradient) {
  return std::vector<NodeDef>{
      NodeDef("ReduceMeanGrad",
              {GO(0)},
              {GI(0)},
              SrcNodeAttributes())};
}

IMPLEMENT_GRADIENT_BUILDER(GetPowGradient) {
  return std::vector<NodeDef>{
      NodeDef("PowGrad",
              {GO(0), I(0), I(1)},
              {GI(0), GI(1)})};
}

IMPLEMENT_GRADIENT_BUILDER(GetSoftmaxCrossEntropyGradient) {
  return std::vector<NodeDef>{
      NodeDef(OpDef{"SoftmaxCrossEntropyGrad", kMSDomain},
              {GO(0), I(0), I(1)},
              {GI(0)})};
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

}  // namespace training
}  // namespace onnxruntime
