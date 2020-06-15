// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/nchwc_schema_defs.h"

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

void NhwcPoolOpSchemaGenerator(OpSchema& schema) {
  schema.SetDomain(kMSNhwcDomain);
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

void NhwcGlobalPoolOpSchemaGenerator(OpSchema& schema) {
  schema.SetDomain(kMSNhwcDomain);
  schema.SinceVersion(1);
  schema.SetDoc(R"DOC(For internal use.)DOC");
  schema.Input(0, "X", "", "T");
  schema.Output(0, "Y", "", "T");
  schema.TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors");
  schema.TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
    ONNX_NAMESPACE::globalPoolTypeShapeInference(ctx);
  });
}


void RegisterNhwcSchemas() {
  ONNX_CONTRIB_OPERATOR_SCHEMA(ReorderInput)
      .SetDomain(kMSNhwcDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(For internal use.)DOC")
      .Input(0, "X", "", "T")
      .Output(0, "Y", "", "T")
       .TypeConstraint(
          "T",
          {"tensor(float)", "tensor(int8)", "tensor(uint8)"},
          "Constrain input and output types to float/quantized tensors")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReorderOutput)
      .SetDomain(kMSNhwcDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(For internal use.)DOC")
      .Attr("channels", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Input(0, "X", "", "T")
      .Output(0, "Y", "", "T")
      .TypeConstraint(
          "T",
          {"tensor(float)", "tensor(int8)", "tensor(uint8)"},
          "Constrain input and output types to float/quantized tensors")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(Conv)
      .SetDomain(kMSNhwcDomain)
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
      .Output(0, "Y", "", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        ONNX_NAMESPACE::convPoolShapeInference(ctx, true, false, 0, 1);
      });

 ONNX_CONTRIB_OPERATOR_SCHEMA(FusedConv)
       .SetDomain(kMSNhwcDomain)
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
      .Output(0, "Y", "", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        ONNX_NAMESPACE::convPoolShapeInference(ctx, true, false, 0, 1);
      });


  ONNX_CONTRIB_OPERATOR_SCHEMA(MaxPool)
      .FillUsing(NhwcPoolOpSchemaGenerator)
      .Attr("storage_order", "", AttributeProto::INT, static_cast<int64_t>(0));

  ONNX_CONTRIB_OPERATOR_SCHEMA(AveragePool)
      .FillUsing(NhwcPoolOpSchemaGenerator)
      .Attr("count_include_pad", "", AttributeProto::INT, static_cast<int64_t>(0));

  ONNX_CONTRIB_OPERATOR_SCHEMA(GlobalMaxPool)
      .FillUsing(NhwcGlobalPoolOpSchemaGenerator);

  ONNX_CONTRIB_OPERATOR_SCHEMA(GlobalAveragePool)
      .FillUsing(NhwcGlobalPoolOpSchemaGenerator);
}

}  // namespace contrib
}  // namespace onnxruntime
