// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/featurizers_ops/featurizers_defs.h"
#include "core/graph/op.h"

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

#define MS_FEATURIZERS_OPERATOR_SCHEMA(name) MS_FEATURIZERS_OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define MS_FEATURIZERS_OPERATOR_SCHEMA_UNIQ_HELPER(Counter, name) MS_FEATURIZERS_OPERATOR_SCHEMA_UNIQ(Counter, name)

#define MS_FEATURIZERS_OPERATOR_SCHEMA_UNIQ(Counter, name)       \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce( \
      op_schema_register_once##name##Counter) ONNX_UNUSED =      \
      ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__)

#define MS_FEATURIZERS_OPERATOR_SCHEMA_ELSEWHERE(name, schema_func) MS_FEATURIZERS_OPERATOR_SCHEMA_UNIQ_HELPER_ELSEWHERE(__COUNTER__, name, schema_func)
#define MS_FEATURIZERS_OPERATOR_SCHEMA_UNIQ_HELPER_ELSEWHERE(Counter, name, schema_func) MS_FEATURIZERS_OPERATOR_SCHEMA_UNIQ_ELSEWHERE(Counter, name, schema_func)

#define MS_FEATURIZERS_OPERATOR_SCHEMA_UNIQ_ELSEWHERE(Counter, name, schema_func) \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(                  \
      op_schema_register_once##name##Counter) ONNX_UNUSED =                       \
      schema_func(ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__))

namespace onnxruntime {
namespace featurizers {

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL;

// Forward declarations
static void RegisterCatImputerFeaturizerVer1();
static void RegisterDateTimeFeaturizerVer1();
static void RegisterMaxAbsScalarFeaturizerVer1();
static void RegisterStringFeaturizerVer1();

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
void RegisterMSFeaturizersSchemas() {
  RegisterCatImputerFeaturizerVer1();
  RegisterDateTimeFeaturizerVer1();
  RegisterMaxAbsScalarFeaturizerVer1();
  RegisterStringFeaturizerVer1();
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
void RegisterCatImputerFeaturizerVer1() {
  static const char* doc = R"DOC(
        Imputes (populates) values with the mode (most common value) encountered during
        training. This featurizer supports float and double for most (if not all) frameworks
        due to the existance of NaN in those types. Other types require 'optional' support
        within the host frameworks and programming languages.

        C++-style pseudo signature:
          std::float_t execute(std::float_t const &value);
          std::double_t execute(std::double_t const &value);
          template <typename T> T execute(std::optional<T> const &value);

        Examples (where 55.5 is the mode value):
          execute(1.0) -> 1.0
          execute(NaN) -> 55.5
          execute(2.0) -> 2.0
    )DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(CatImputerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "tensor(uint8)")
      .Input(
          1,
          "Input",
          "No information is available",
          "T")
      .Output(
          0,
          "Output",
          "No information is available",
          "T")
      .TypeConstraint(
          "T",
          {"tensor(float)", "tensor(double)", "tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromInputToOutput(ctx, 1, 0);
            if (!hasNInputShapes(ctx, 1)) {
              return;
            }
            propagateShapeFromInputToOutput(ctx, 1, 0);
          });
}

void RegisterDateTimeFeaturizerVer1() {
  static const char* doc = R"DOC(
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

  MS_FEATURIZERS_OPERATOR_SCHEMA(DateTimeTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "tensor(uint8)")
      .Input(
          1,
          "Input",
          "No information is available",
          "tensor(int64)")
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

void RegisterMaxAbsScalarFeaturizerVer1() {
  static const char* doc = R"DOC(
        Scales input based on the maximum absolute value of all data encountered during training.

        C++-style pseudo signature:
          std::float_t execute(std::uint16_t value);
          std::double_t execute(std::uint32_t value);

        Examples:
          Given a training set of [1.0, -2.0, 3.0, -4.0], where 4.0 is the absolute value of the
          maximum value encountered...

          execute(1.0) -> 1.0 / 4.0
          execute(-4.0) -> -4.0 / 4.0
          execute(100.0) -> 100 / 4.0
    )DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(MaxAbsScalarTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "tensor(uint8)")
      .Input(
          1,
          "Input",
          "No information is available",
          "InputT")
      .Output(
          0,
          "Output",
          "No information is available",
          "OutputT")
      .TypeConstraint(
          "InputT",
          {"tensor(int8)", "tensor(int16)", "tensor(uint8)", "tensor(uint16)", "tensor(float)", "tensor(int32)", "tensor(int64)", "tensor(uint32)", "tensor(uint64)", "tensor(double)"},
          "No information is available")
      .TypeConstraint(
          "OutputT",
          {"tensor(float)", "tensor(double)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            auto input_elem_type = ctx.getInputType(1)->tensor_type().elem_type();
            if (input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT8 ||
                input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT16 ||
                input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
                input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_UINT16 ||
                input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
              ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
            } else if (input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT32 ||
                       input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT64 ||
                       input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_UINT32 ||
                       input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_UINT64 ||
                       input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
              ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
            }

            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
          });
}

void RegisterStringFeaturizerVer1() {
  static const char* doc = R"DOC(
        Converts the input into a string representation based on the input's type.

        C++-style pseudo signature:
          template <typename T> std::string execute(T const &value);

        Examples:
          execute(1) -> "1"
          execute(3.14) -> "3.14"
    )DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(StringTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "tensor(uint8)")
      .Input(
          1,
          "Input",
          "No information is available",
          "InputT")
      .Output(
          0,
          "Output",
          "No information is available",
          "tensor(string)")
      .TypeConstraint(
          "InputT",
          {"tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(float)", "tensor(double)", "tensor(bool)", "tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_STRING, 0);

            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
          });
}

}  // namespace featurizers
}  // namespace onnxruntime
