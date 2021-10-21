// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/mlas/inc/mlas.h"

namespace ONNX_NAMESPACE {
void convPoolShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    bool use_dilation, bool require_kernel_shape,
    int input1Idx,
    int input2Idx);
void globalPoolTypeShapeInference(ONNX_NAMESPACE::InferenceContext& ctx);
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace contrib {

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::InferenceContext;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;

void NchwcPoolOpSchemaGenerator(OpSchema& schema) {
  schema.SetDomain(kMSNchwcDomain);
  schema.SinceVersion(1);
  schema.SetDoc(R"DOC(For internal use.)DOC");
  schema.Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"));
  schema.Attr("kernel_shape", "", AttributeProto::INTS);
  schema.Attr("dilations", "", AttributeProto::INTS, OPTIONAL_VALUE);
  schema.Attr("strides", "", AttributeProto::INTS, OPTIONAL_VALUE);
  schema.Attr("pads", "", AttributeProto::INTS, OPTIONAL_VALUE);
  schema.Attr("ceil_mode", "", AttributeProto::INT, static_cast<int64_t>(0));
  schema.Input(0, "X", "", "T");
  schema.Output(0, "Y", "", "T");
  schema.TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors");
  schema.TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
    ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
    ONNX_NAMESPACE::convPoolShapeInference(ctx, true, true, 0, 1);
  });
}

void NchwcGlobalPoolOpSchemaGenerator(OpSchema& schema) {
  schema.SetDomain(kMSNchwcDomain);
  schema.SinceVersion(1);
  schema.SetDoc(R"DOC(For internal use.)DOC");
  schema.Input(0, "X", "", "T");
  schema.Output(0, "Y", "", "T");
  schema.TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors");
  schema.TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
    ONNX_NAMESPACE::globalPoolTypeShapeInference(ctx);
  });
}

void RegisterNchwcSchemas() {
  ONNX_CONTRIB_OPERATOR_SCHEMA(ReorderInput)
      .SetDomain(kMSNchwcDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(For internal use.)DOC")
      .Attr("channels_last", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Input(0, "X", "", "T")
      .Output(0, "Y", "", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }

        const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

        auto input_rank = input_shape.dim_size();
        if (input_rank < 2) {
          fail_shape_inference("tensor rank too small");
        }

        auto channels_last = getAttribute(ctx, "channels_last", 0);

        // Copy the batch dimension.
        *output_shape->add_dim() = input_shape.dim(0);

        // Block align the channel dimension.
        const auto& input_channel_dim = input_shape.dim((channels_last == 0) ? 1 : input_rank - 1);
        auto* output_channel_dim = output_shape->add_dim();
        if (input_channel_dim.has_dim_value()) {
          const int64_t channels = input_channel_dim.dim_value();
          const int64_t nchwc_block_size = static_cast<int64_t>(MlasNchwcGetBlockSize());
          int64_t nchwc_channels = (channels + nchwc_block_size - 1) & ~(nchwc_block_size - 1);
          output_channel_dim->set_dim_value(nchwc_channels);
        }

        // Copy the spatial dimensions.
        int first_spatial_dim = (channels_last == 0) ? 2 : 1;
        for (int i = 0; i < input_rank - 2; i++) {
          *output_shape->add_dim() = input_shape.dim(first_spatial_dim + i);
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReorderOutput)
      .SetDomain(kMSNchwcDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(For internal use.)DOC")
      .Attr("channels", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("channels_last", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Input(0, "X", "", "T")
      .Output(0, "Y", "", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }

        const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

        auto input_rank = input_shape.dim_size();
        if (input_rank < 2) {
          fail_shape_inference("tensor rank too small");
        }

        // Update the output shape with the actual number of channels.
        auto channels = getAttribute(ctx, "channels", 0);
        if (channels <= 0) {
          fail_shape_inference("invalid channel count");
        }

        // Copy the batch dimension.
        *output_shape->add_dim() = input_shape.dim(0);

        auto channels_last = getAttribute(ctx, "channels_last", 0);
        if (channels_last == 0) {
          output_shape->add_dim()->set_dim_value(channels);
        }

        // Copy the spatial dimensions.
        for (int i = 0; i < input_rank - 2; i++) {
          *output_shape->add_dim() = input_shape.dim(2 + i);
        }

        if (channels_last != 0) {
          output_shape->add_dim()->set_dim_value(channels);
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(Conv)
      .SetDomain(kMSNchwcDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(For internal use.)DOC")
      .Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
      .Attr("kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("dilations", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("strides", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("pads", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("group", "", AttributeProto::INT, static_cast<int64_t>(1))
      .Attr("activation", "", AttributeProto::STRING, OPTIONAL_VALUE)
      .Attr("activation_params", "", AttributeProto::FLOATS, OPTIONAL_VALUE)
      .Input(0, "X", "", "T")
      .Input(1, "W", "", "T")
      .Input(2, "B", "", "T", OpSchema::Optional)
      .Input(3, "Sum", "", "T", OpSchema::Optional)
      .Output(0, "Y", "", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        ONNX_NAMESPACE::convPoolShapeInference(ctx, true, false, 0, 1);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(MaxPool)
      .FillUsing(NchwcPoolOpSchemaGenerator)
      .Attr("storage_order", "", AttributeProto::INT, static_cast<int64_t>(0));

  ONNX_CONTRIB_OPERATOR_SCHEMA(AveragePool)
      .FillUsing(NchwcPoolOpSchemaGenerator)
      .Attr("count_include_pad", "", AttributeProto::INT, static_cast<int64_t>(0));

  ONNX_CONTRIB_OPERATOR_SCHEMA(GlobalMaxPool)
      .FillUsing(NchwcGlobalPoolOpSchemaGenerator);

  ONNX_CONTRIB_OPERATOR_SCHEMA(GlobalAveragePool)
      .FillUsing(NchwcGlobalPoolOpSchemaGenerator);

  ONNX_CONTRIB_OPERATOR_SCHEMA(Upsample)
      .SetDomain(kMSNchwcDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(For internal use.)DOC")
      .Attr("scales", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("mode", "", AttributeProto::STRING, std::string("nearest"))
      .Attr("coordinate_transformation_mode", "", AttributeProto::STRING, std::string("asymmetric"))
      .Input(0, "X", "", "T")
      .Output(0, "Y", "", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (!hasNInputShapes(ctx, 1)) {
          return;
        }

        const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

        auto input_rank = input_shape.dim_size();
        if (input_rank < 2) {
          fail_shape_inference("tensor rank too small");
        }

        std::vector<int64_t> scales;
        if (!getRepeatedAttribute(ctx, "scales", scales)) {
          return;
        }
        if (static_cast<size_t>(input_rank) != scales.size()) {
          fail_shape_inference("invalid scales dimension");
        }

        for (int i = 0; i < input_rank; i++) {
          if (scales[i] <= 0) {
            fail_shape_inference("invalid scales value");
          }
          const auto& input_dim = input_shape.dim(i);
          auto* output_dim = output_shape->add_dim();
          if (input_dim.has_dim_value()) {
            output_dim->set_dim_value(input_dim.dim_value() * scales[i]);
          }
        }
      });
}

}  // namespace contrib
}  // namespace onnxruntime
