// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/dml_ops/dml_defs.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "core/providers/dml/OperatorAuthorHelper/Attributes.h"

namespace ONNX_NAMESPACE {
  void convPoolShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    bool use_dilation, bool require_kernel_shape,
    int input1Idx,
    int input2Idx);
  void convTransposeShapeInference(InferenceContext& ctx);
  }  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace dml {
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;

void RegisterDmlSchemas() {

  MS_DML_OPERATOR_SCHEMA(FusedConv)
    .SetDomain(kMSDmlDomain)
    .SinceVersion(1)
    .SetDoc(R"DOC(DirectML fused Conv+Activation)DOC")
    .Input(0, "X", "", "T")
    .Input(1, "W", "", "T")
    .Input(2, "B", "", "T", OpSchema::Optional)
    .Output(0, "Y", "", "T")
    .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
    .Attr("kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
    .Attr("dilations", "", AttributeProto::INTS, OPTIONAL_VALUE)
    .Attr("strides", "", AttributeProto::INTS, OPTIONAL_VALUE)
    .Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
    .Attr("pads", "", AttributeProto::INTS, OPTIONAL_VALUE)
    .Attr("group", "", AttributeProto::INT, static_cast<int64_t>(1))
    .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
    .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
    .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
      ONNX_NAMESPACE::convPoolShapeInference(ctx, true, false, 0, 1);
    });

  MS_DML_OPERATOR_SCHEMA(FusedConvTranspose)
    .SetDomain(kMSDmlDomain)
    .SinceVersion(1)
    .SetDoc(R"DOC(DirectML fused ConvTranspose+Activation)DOC")
    .Input(0, "X", "", "T")
    .Input(1, "W", "", "T")
    .Input(2, "B", "", "T", OpSchema::Optional)
    .Output(0, "Y", "", "T")
    .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
    .Attr("kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
    .Attr("output_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
    .Attr("output_padding", "", AttributeProto::INTS, OPTIONAL_VALUE)
    .Attr("dilations", "", AttributeProto::INTS, OPTIONAL_VALUE)
    .Attr("strides", "", AttributeProto::INTS, OPTIONAL_VALUE)
    .Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
    .Attr("pads", "", AttributeProto::INTS, OPTIONAL_VALUE)
    .Attr("group", "", AttributeProto::INT, static_cast<int64_t>(1))
    .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
    .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
    .TypeAndShapeInferenceFunction(
        [](ONNX_NAMESPACE::InferenceContext& ctx) { ONNX_NAMESPACE::convTransposeShapeInference(ctx); });

  MS_DML_OPERATOR_SCHEMA(FusedInstanceNormalization)
    .SetDomain(kMSDmlDomain)
    .SinceVersion(1)
    .SetDoc(R"DOC(DirectML fused InstanceNormalization+Activation)DOC")
    .Attr("epsilon", "", AttributeProto::FLOAT, 1e-5f)
    .Input(0, "input", "", "T")
    .Input(1, "scale", "", "T")
    .Input(2, "B", "", "T")
    .Output(0, "output", "", "T")
    .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
    .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
    .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
    .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput(ctx);
    });
        
  MS_DML_OPERATOR_SCHEMA(FusedBatchNormalization)
    .SetDomain(kMSDmlDomain)
    .SinceVersion(1)
    .SetDoc(R"DOC(DirectML fused BatchNormalization+Activation)DOC")
    .NumOutputs({1, 5})
    .Attr("spatial", "", AttributeProto::INT, static_cast<int64_t>(1))
    .Attr("epsilon", "", AttributeProto::FLOAT, 1e-5f)
    .Attr("momentum", "", AttributeProto::FLOAT, 0.9f)
    .Input(0, "X", "", "T")
    .Input(1, "scale", "", "T")
    .Input(2, "B", "", "T")
    .Input(3, "mean", "", "T")
    .Input(4, "var", "", "T")
    .Output(0, "Y", "", "T")
    .Output(1, "mean", "", "T", OpSchema::Optional)
    .Output(2, "var", "", "T", OpSchema::Optional)
    .Output(3, "saved_mean", "", "T", OpSchema::Optional)
    .Output(4, "saved_var", "", "T", OpSchema::Optional)
    .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
    .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
    .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
    .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput(ctx);
      // TODO in training mode, it may be possible to infer some of
      // the other outputs as well.
    });
    
  MS_DML_OPERATOR_SCHEMA(FusedMeanVarianceNormalization)
    .SetDomain(kMSDmlDomain)
    .SinceVersion(1)
    .SetDoc(R"DOC(DirectML fused MeanVarianceNormalization+Activation)DOC")
    .Attr("across_channels", "", AttributeProto::INT, static_cast<int64_t>(0))
    .Attr("normalize_variance", "", AttributeProto::INT, static_cast<int64_t>(1))
    .Input(0, "input", "", "T")
    .Output(0, "output", "", "T")
    .TypeConstraint( "T", { "tensor(float16)", "tensor(float)", "tensor(double)" }, "")
    .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
    .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
    .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);
  
  MS_DML_OPERATOR_SCHEMA(FusedGemm)
    .SetDomain(kMSDmlDomain)
    .SinceVersion(1)
    .SetDoc(R"DOC(DirectML fused Gemm+Activation)DOC")
    .Input(0, "A", "", "T")
    .Input(1, "B", "", "T")
    .Input(2, "C", "", "T")
    .Output(0, "Y", "", "T")
    .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
    .Attr("transA", "", AttributeProto::INT, static_cast<int64_t>(0))
    .Attr("transB", "", AttributeProto::INT, static_cast<int64_t>(0))
    .Attr("alpha", "", AttributeProto::FLOAT, 1.0f)
    .Attr("beta", "", AttributeProto::FLOAT, 1.0f)
    .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
    .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
    .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2)) {
        auto transAAttr = ctx.getAttribute("transA");
        bool transA =
            transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
        auto transBAttr = ctx.getAttribute("transB");
        bool transB =
            transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
        auto& first_input_shape = getInputShape(ctx, 0);
        auto& second_input_shape = getInputShape(ctx, 1);
        if (first_input_shape.dim_size() != 2)
          fail_shape_inference("First input does not have rank 2");
        if (second_input_shape.dim_size() != 2)
          fail_shape_inference("Second input does not have rank 2");
        updateOutputShape(
            ctx,
            0,
            {first_input_shape.dim(transA ? 1 : 0),
              second_input_shape.dim(transB ? 0 : 1)});
      }
    });

  MS_DML_OPERATOR_SCHEMA(FusedMatMul)
    .SetDomain(kMSDmlDomain)
    .SinceVersion(1)
    .SetDoc(R"DOC(DirectML fused MatMul+Activation)DOC")
    .Input(0, "A", "", "T")
    .Input(1, "B", "", "T")
    .Output(0, "Y", "", "T")
    .TypeConstraint( "T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
    .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
    .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
    .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasNInputShapes(ctx, 2)) {
        return;
      }

      const auto shape0 = ctx.getInputType(0)->tensor_type().shape();
      const auto shape1 = ctx.getInputType(1)->tensor_type().shape();

      if (shape0.dim_size() == 0 || shape1.dim_size() == 0) {
        fail_shape_inference("Input tensors of wrong rank (0).");
      }

      ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

      // First promote each shape to at least rank-2. This logic is
      // specific to matmul, not generic broadcasting.
      {
        if (shape0.dim_size() == 1) {
          shapeL.add_dim()->set_dim_value(1);
          *shapeL.add_dim() = shape0.dim(0);
        } else {
          *shapeL.mutable_dim() = shape0.dim();
        }
        if (shape1.dim_size() == 1) {
          *shapeR.add_dim() = shape1.dim(0);
          shapeR.add_dim()->set_dim_value(1);
        } else {
          *shapeR.mutable_dim() = shape1.dim();
        }
      }

      // Check for compatible matrix multiply dimensions
      {
        auto dimL = shapeL.dim(shapeL.dim_size() - 1);
        auto dimR = shapeR.dim(shapeR.dim_size() - 2);
        if (dimL.has_dim_value() && dimR.has_dim_value() &&
            dimL.dim_value() != dimR.dim_value()) {
          fail_shape_inference(
              "Incompatible dimensions for matrix multiplication");
          ;
        }
      }

      ONNX_NAMESPACE::TensorShapeProto resultShape;

      // Now call out to generic multidimensional broadcasting for
      // the broadcastable prefixes.
      {
        ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
        for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
          *prefixShapeL.add_dim() = shapeL.dim(i);
        }
        for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
          *prefixShapeR.add_dim() = shapeR.dim(i);
        }
        bidirectionalBroadcastShapeInference(
            prefixShapeL, prefixShapeR, resultShape);
      }

      // Back to matmul-specific. Add the trailing dimensions back in.
      {
        if (shape0.dim_size() != 1) {
          *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
        }
        if (shape1.dim_size() != 1) {
          *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
        }
      }

      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() =
          resultShape;
    });

  MS_DML_OPERATOR_SCHEMA(FusedAdd)
    .SetDomain(kMSDmlDomain)
    .SinceVersion(1)
    .SetDoc(R"DOC(DirectML fused Add+Activation)DOC")
    .Input(0, "A", "", "T")
    .Input(1, "B", "", "T")
    .Output(0, "C", "", "T")
    .TypeConstraint("T", OpSchema::numeric_types_for_math_reduction(), "")
    .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
    .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
    .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });

  MS_DML_OPERATOR_SCHEMA(FusedSum)
    .SetDomain(kMSDmlDomain)
    .SinceVersion(1)
    .SetDoc(R"DOC(DirectML fused Sum+Activation)DOC")
    .Input(0, "data_0", "", "T", OpSchema::Variadic)
    .Output(0, "sum", "", "T")
    .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
    .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
    .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
    .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
    .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
    .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
      int num_inputs = static_cast<int>(ctx.getNumInputs());
      std::vector<const ONNX_NAMESPACE::TensorShapeProto*> shapes;
      for (int i = 0; i < num_inputs; ++i) {
        auto input_type = ctx.getInputType(i);
        if (nullptr == input_type || !input_type->has_tensor_type() ||
            !input_type->tensor_type().has_shape()) {
          return;
        }
        shapes.push_back(&input_type->tensor_type().shape());
      }

      multidirectionalBroadcastShapeInference(
          shapes,
          *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
  });

}
}  // namespace dml
}  // namespace onnxruntime
