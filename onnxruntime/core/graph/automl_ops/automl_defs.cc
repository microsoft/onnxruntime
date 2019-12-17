// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/automl_ops/automl_defs.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace onnxruntime {
namespace automl {
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL;

static const char* DateTimeTransformer_ver1_doc = R"DOC(
        Extracts various datetime-related values from a UTC time_point.

        C++-style pseudo signature:
          TimePoint execute(std::chron::system_clock::time_point const &value);

        Examples:
          Given a time_point 'value' representing "November 17, 1976 12:27:04PM":

          "November 17, 1976 12:27:04PM" => {
            "year": 1976,
            "month": 11,
            "day": 17,
            "hour": 12,
            "minute": 27,
            "second": 04,
            "amPm": 2,        // PM
            "hour12": 12,
            "dayOfWeek": 3,   // Wednesday
            "dayOfQuarter": 48,
            "dayOfYear": 321,
            "weekOfMonth": 2,
            "quarterOfYear": 4,
            "halfOfYear": 2,
            "weekIso": 47,
            "yearIso": 1976,
            "monthLabel": "November",
            "amPmLabel": "pm",
            "dayOfWeekLabel": "Wednesday",
            "holidayName": "",
            "isPaidTimeOff": 0
          }
    )DOC";

void RegisterAutoMLSchemas() {
  MS_AUTOML_OPERATOR_SCHEMA(DateTimeTransformer)
      .SinceVersion(1)
      .SetDomain(kMSAutoMLDomain)
      .SetDoc(DateTimeTransformer_ver1_doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "InputT0")
      .Input(
          1,
          "Input",
          "The input represents a number of seconds passed since the epoch, suitable to properly construct"
          "an instance of std::chrono::system_clock::time_point",
          "InputT1")
      .Output(0, "year", "No information available", "OutputT0")
      .Output(1, "month", "No information available", "OutputT1")
      .Output(2, "day", "No information available", "OutputT1")
      .Output(3, "hour", "No information available", "OutputT1")
      .Output(4, "minute", "No information available", "OutputT1")
      .Output(5, "second", "No information available", "OutputT1")
      .Output(6, "amPm", "No information available", "OutputT1")
      .Output(7, "hour12", "No information available", "OutputT1")
      .Output(8, "dayOfWeek", "No information available", "OutputT1")
      .Output(9, "dayOfQuarter", "No information available", "OutputT1")
      .Output(10, "dayOfYear", "No information available", "OutputT2")
      .Output(11, "weekOfMonth", "No information available", "OutputT2")
      .Output(12, "quarterOfYear", "No information available", "OutputT1")
      .Output(13, "halfOfYear", "No information available", "OutputT1")
      .Output(14, "weekIso", "No information available", "OutputT1")
      .Output(15, "yearIso", "No information available", "OutputT0")
      .Output(16, "monthLabel", "No information available", "OutputT3")
      .Output(17, "amPmLabel", "No information available", "OutputT3")
      .Output(18, "dayOfWeekLabel", "No information available", "OutputT3")
      .Output(19, "holidayName", "No information available", "OutputT3")
      .Output(20, "isPaidTimeOff", "No information available", "OutputT1")
      .TypeConstraint(
          "InputT0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "InputT1",
          {"tensor(int64)"},
          "No information is available")
      .TypeConstraint(
          "OutputT0",
          {"tensor(int32)"},
          "No information is available")
      .TypeConstraint(
          "OutputT1",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "OutputT2",
          {"tensor(uint16)"},
          "No information is available")
      .TypeConstraint(
          "OutputT3",
          {"tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
            ctx.getOutputType(1)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(2)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(3)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(4)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(5)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(6)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(7)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(8)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(9)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(10)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT16);
            ctx.getOutputType(11)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT16);
            ctx.getOutputType(12)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(13)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(14)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(15)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
            ctx.getOutputType(16)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
            ctx.getOutputType(17)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
            ctx.getOutputType(18)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
            ctx.getOutputType(19)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
            ctx.getOutputType(20)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);

            for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
              *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
            }
          });
}
}  // namespace automl
}  // namespace onnxruntime
