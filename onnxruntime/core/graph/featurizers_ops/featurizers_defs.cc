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
static void RegisterHashOneHotVectorizerFeaturizerVer1();
static void RegisterImputationMarkerFeaturizerVer1();
static void RegisterLabelEncoderFeaturizerVer1();
static void RegisterMaxAbsScalarFeaturizerVer1();
static void RegisterMinMaxScalarFeaturizerVer1();
static void RegisterMissingDummiesFeaturizerVer1();
static void RegisterOneHotEncoderFeaturizerVer1();
static void RegisterRobustScalarFeaturizerVer1();
static void RegisterStringFeaturizerVer1();
static void RegisterTimeSeriesImputerFeaturizerVer1();

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
void RegisterMSFeaturizersSchemas() {
  RegisterCatImputerFeaturizerVer1();
  RegisterDateTimeFeaturizerVer1();
  RegisterHashOneHotVectorizerFeaturizerVer1();
  RegisterImputationMarkerFeaturizerVer1();
  RegisterLabelEncoderFeaturizerVer1();
  RegisterMaxAbsScalarFeaturizerVer1();
  RegisterMinMaxScalarFeaturizerVer1();
  RegisterMissingDummiesFeaturizerVer1();
  RegisterOneHotEncoderFeaturizerVer1();
  RegisterRobustScalarFeaturizerVer1();
  RegisterStringFeaturizerVer1();
  RegisterTimeSeriesImputerFeaturizerVer1();
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
          float execute(float const &value);
          double execute(double const &value);
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
          "T0")
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
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "T",
          {"tensor(float)", "tensor(double)", "tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromInputToOutput(ctx, 1, 0);
            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }
          });
}

void RegisterDateTimeFeaturizerVer1() {
  static const char* doc = R"DOC(
        Extracts various datetime-related values from a UTC time_point.

        C++-style pseudo signature:
          TimePoint execute(std::chrono::system_clock::time_point const &value);

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
          "T0")
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
          "T0",
          {"tensor(uint8)"},
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
            const bool has_shape = hasInputShape(ctx, 1);

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_INT32, 0);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 1);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 1);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 2);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 2);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 3);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 3);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 4);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 4);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 5);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 5);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 6);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 6);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 7);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 7);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 8);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 8);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 9);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 9);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT16, 10);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 10);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT16, 11);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 11);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 12);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 12);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 13);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 13);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 14);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 14);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_INT32, 15);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 15);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_STRING, 16);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 16);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_STRING, 17);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 17);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_STRING, 18);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 18);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_STRING, 19);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 19);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 20);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 20);
            }

          }
      );
}

void RegisterHashOneHotVectorizerFeaturizerVer1() {
  static const char* doc = R"DOC(
        Hashes the input to a categorical value, then produces a one hot encoded vector
        based on that value.

        C++-style pseudo signature:
            template <typename T> HashOneHotVectorizerStruct execute(T const &value);

        Examples:
          Assuming the hashing algorithm...
            "A" -> 1
            "B" -> 2
            "C" -> 5

          and 'numCols' set to 8:

            execute("A") -> [1, 0, 0, 0, 0, 0, 0, 0]
            execute("B") -> [0, 1, 0, 0, 0, 0, 0, 0]
            execute("C") -> [0, 0, 0, 0, 1, 0, 0, 0]
  )DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(HashOneHotVectorizerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "Input",
          "No information is available",
          "InputT")
      .Output(0, "NumElements", "No information available", "OutputT0")
      .Output(1, "Value", "No information available", "OutputT1")
      .Output(2, "Index", "No information available", "OutputT0")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "InputT",
          {"tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(float)", "tensor(double)", "tensor(bool)", "tensor(string)"},
          "No information is available")
      .TypeConstraint(
          "OutputT0",
          {"tensor(uint64)"},
          "No information is available")
      .TypeConstraint(
          "OutputT1",
          {"tensor(uint8)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            const bool has_shape = hasInputShape(ctx, 1);

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT64, 0);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 1);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 1);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT64, 2);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 2);
            }

          }
      );
}

void RegisterImputationMarkerFeaturizerVer1() {
  static const char* doc = R"DOC(
        Returns true if the input is null, false if it is not.

        C++-style pseudo signature:
          bool execute(float const &value);
          bool execute(double const &value);
          template <typename T> bool execute(std::optional<T> const &value);

        Examples:
          3.0 -> false
          NaN -> true
          "foo" -> false
          std::optional<string>() -> true
          std::optional<string>("bar") -> false
  )DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(ImputationMarkerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "Input",
          "No information is available",
          "InputT")
      .Output(
          0,
          "Output",
          "No information is available",
          "tensor(bool)")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "InputT",
          {"tensor(float)", "tensor(double)", "tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_BOOL, 0);
            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }
          });
}

void RegisterLabelEncoderFeaturizerVer1() {
  static const char* doc = R"DOC(
        Returns a unique id for the input based on all values encountered during training.

        C++-style pseudo signature:
          template <typename T> uint32 execute(T const &value);

        Examples:
          Assuming the training data of ["A", "B", "C"]...

          execute("A") -> 1
          execute("B") -> 2
          execute("C") -> 3
          execute("This value was not seen during training") -> 0
  )DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(LabelEncoderTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "Input",
          "No information is available",
          "InputT")
      .Output(
          0,
          "Output",
          "No information is available",
          "tensor(uint32)")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "InputT",
          {"tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(float)", "tensor(double)", "tensor(bool)", "tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT32, 0);
            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }
          });
}

void RegisterMaxAbsScalarFeaturizerVer1() {
  static const char* doc = R"DOC(
        Scales input based on the maximum absolute value of all data encountered during training.

        C++-style pseudo signature:
          float execute(uint16 value);
          double execute(uint32 value);

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
          "T0")
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
          "T0",
          {"tensor(uint8)"},
          "No information is available")
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
              propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 0);
            } else if (input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT32 ||
                       input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT64 ||
                       input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_UINT32 ||
                       input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_UINT64 ||
                       input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
              propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, 0);
            } else {
              fail_type_inference("input 1 is expected to have an accepted type");
            }

            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }

          });
}

void RegisterMinMaxScalarFeaturizerVer1() {
  static const char* doc = R"DOC(
        Scales input based on the scale that results from the minimum and maximum values encountered
        during training.

        C++-style pseudo signature:
            template <typeanem T> double(T const &value);

        Examples:
          Given the training data [1, 2, 3, 4, 5];
            min: 1
            max: 5
            scale (<max> - <min>): 4

          execute(2) = 2 / 4
          execute(20) = 20 / 4
  )DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(MinMaxScalarTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "Input",
          "No information is available",
          "InputT")
      .Output(
          0,
          "Output",
          "No information is available",
          "tensor(double)")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "InputT",
          {"tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(float)", "tensor(double)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, 0);
            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }
          });
}

void RegisterMissingDummiesFeaturizerVer1() {
  static const char* doc = R"DOC(
        Returns 1 if the input is null, 0 if it is not.

        C++-style pseudo signature:
            int8 execute(float const &value);
            int8 execute(double const &value);
            template <typename T> int8 execute(T const &value);

        Examples:
          1.0 -> 0
          NaN -> 1
          "foo" -> 0
          std::optional<string>() -> 1
          std::optional<string>("bar") -> 0
  )DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(MissingDummiesTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "Input",
          "No information is available",
          "InputT")
      .Output(
          0,
          "Output",
          "No information is available",
          "tensor(int8)")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "InputT",
          {"tensor(float)", "tensor(double)", "tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_INT8, 0);
            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }
          });
}

void RegisterOneHotEncoderFeaturizerVer1() {
  static const char* doc = R"DOC(
        Produces a one hot vector based on categories calculated during training.

        C++-style pseudo signature:
          template <typename T> OneHotVector execute(T const &value);

        Examples:
          Assuming the training data [10, 20, 30, 40]...

          execute(10) -> [0, 1, 0, 0, 0]
          execute(20) -> [0, 0, 1, 0, 0]
          execute(30) -> [0, 0, 0, 1, 0]
          execute(40) -> [0, 0, 0, 0, 1]
          execute(200) -> [1, 0, 0, 0, 0]
          execute(-1) -> [1, 0, 0, 0, 0]
  )DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(OneHotEncoderTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "Input",
          "No information is available",
          "InputT")
      .Output(0, "NumElements", "No information available", "OutputT0")
      .Output(1, "Value", "No information available", "OutputT1")
      .Output(2, "Index", "No information available", "OutputT0")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "InputT",
          {"tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(float)", "tensor(double)", "tensor(bool)", "tensor(string)"},
          "No information is available")
      .TypeConstraint(
          "OutputT0",
          {"tensor(uint64)"},
          "No information is available")
      .TypeConstraint(
          "OutputT1",
          {"tensor(uint8)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            const bool has_shape = hasInputShape(ctx, 1);

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT64, 0);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 1);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 1);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT64, 2);
            if(has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 2);
            }

          }
      );
}

void RegisterRobustScalarFeaturizerVer1() {
  static const char* doc = R"DOC(
        MinMaxScalarEstimator + centering?

        C++-style pseudo signature:
            TODO

        Examples:
          TODO
  )DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(RobustScalarTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
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
          "T0",
          {"tensor(uint8)"},
          "No information is available")
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
              propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 0);
            } else if (input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT32 ||
                       input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT64 ||
                       input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_UINT32 ||
                       input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_UINT64 ||
                       input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
              propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, 0);
            } else {
              fail_type_inference("input 1 is expected to have an accepted type");
            }

            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }

          });
}

void RegisterStringFeaturizerVer1() {
  static const char* doc = R"DOC(
        Converts the input into a string representation based on the input's type.

        C++-style pseudo signature:
          template <typename T> string execute(T const &value);

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
          "T0")
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
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "InputT",
          {"tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(float)", "tensor(double)", "tensor(bool)", "tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_STRING, 0);
            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }
          });
}

void RegisterTimeSeriesImputerFeaturizerVer1() {
  static const char* doc = R"DOC(
    Imputes rows and column values such that the generated output does not contain any
    time gaps per grain (based on the time gaps encountered during training) and that
    all missing column values are populated according to a strategy (forward fill,
    backward fill, mode, etc.).

    This Featurizer is unique in that it will produce 0:N rows per invocation, depending upon the
    input data.

    C++-style pseudo signature:
      template <typename... GrainColValueTs, typename... DataColValueTs>
      std::vector<
        std::tuple<
          bool, // true if the row was added
          std::chrono::system_clock::time_point,
          std::tuple<GrainColValueTs...>,
          std::tuple<DataColValueTs...>
        >
      > execute(
        std::chrono::system_clock::time_point const &value,
        std::tuple<GrainColValueTs...> const &grain,
        std::tuple<DataColValueTs...> const &colData
      );

    Examples:
      During training, the time period was found to be 1 day...

      Input:
        +------+-------+------------------+-------------------+
        | time | grain | forward fill col | backward fill col |
        +======+=======+==================+===================+
        | 1    | A     | 10               | None              |
        +------+-------+------------------+-------------------+
        | 2    | A     | None             | 200               |
        +------+-------+------------------+-------------------+
        | 1    | B     | -10              | -100              |
        +------+-------+------------------+-------------------+
        | 4    | A     | 40               | 400               |
        +------+-------+------------------+-------------------+
        | 6    | A     | 60               | 600               |
        +------+-------+------------------+-------------------+
        | 3    | B     | -30              | -300              |
        +------+-------+------------------+-------------------+

      Output:
        +-------+------+-------+------------------+-------------------+
        | Added | time | grain | forward fill col | backward fill col |
        +=======+======+=======+==================+===================+
        | false | 1    | A     | 10               | 200 (from 2)      |
        +-------+------+-------+------------------+-------------------+
        | false | 2    | A     | 10 (from 1)      | 200               |
        +-------+------+-------+------------------+-------------------+
        | true  | 3    | A     | 10 (from 2)      | 400 (from 4)      |
        +-------+------+-------+------------------+-------------------+
        | false | 4    | A     | 40               | 400               |
        +-------+------+-------+------------------+-------------------+
        | true  | 5    | A     | 40 (from 4)      | 600 (from 6)      |
        +-------+------+-------+------------------+-------------------+
        | false | 6    | A     | 60               | 600               |
        +-------+------+-------+------------------+-------------------+
        | false | 1    | B     | -10              | -100              |
        +-------+------+-------+------------------+-------------------+
        | true  | 2    | B     | -10 (from 1)     | -300 (from 3)     |
        +-------+------+-------+------------------+-------------------+
        | false | 3    | B     | -30              | -300              |
        +-------+------+-------+------------------+-------------------+
    )DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(TimeSeriesImputerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "Times",
          "Tensor of timestamps in seconds since epoch [R] where R is a number of rows.",
          "T1")
      .Input(
          2,
          "Keys",
          "Composite keys tensor of shape [R][K]. R is the same as Input(1)",
          "T2")
      .Input(
          3,
          "Data",
          "It is a data tensor of shape [R][C] where R - rows and C - columns. R must be the same with Input(1)",
          "T2")
      .Output(
          0,
          "Added",
          "Tensor of boolean with a shape of [IR]. Contains a boolean for each row in the result where true represents added row.",
          "T3")
      .Output(
          1,
          "ImputedTimes",
          "This is a tensor of timestamps in seconds since epoch of shape [IR], where IR is the number of output rows.",
          "T1")
      .Output(
          2,
          "ImputedKeys",
          "Contains keys along with the imputed keys. Tensor of shape [IR][K].",
          "T2")
      .Output(
          3,
          "ImputedData",
          "Tensor of shape [IR][C] where IR is the number of rows in the output."
          "C is the number of columns.",
          "T2")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "T1",
          {"tensor(int64)"},
          "Represents number of seconds since epoch")
      .TypeConstraint(
          "T2",
          {"tensor(string)"},
          "Output data")
      .TypeConstraint(
          "T3",
          {"tensor(bool)"},
          "Boolean Tensor")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_BOOL, 0);
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_INT64, 1);
            // Number of output rows is not known
            ONNX_NAMESPACE::TensorShapeProto shape_0_1;
            shape_0_1.add_dim();
            ONNX_NAMESPACE::updateOutputShape(ctx, 0, shape_0_1);
            ONNX_NAMESPACE::updateOutputShape(ctx, 1, shape_0_1);

            // Keys
            propagateElemTypeFromInputToOutput(ctx, 2, 2);
            // Keys shape
            if (hasInputShape(ctx, 2)) {
              const auto& input2_shape = getInputShape(ctx, 2);
              if (input2_shape.dim_size() != 2) {
                fail_shape_inference("Expecting keys to have 2 dimensions");
              }
              ONNX_NAMESPACE::TensorShapeProto shape;
              shape.add_dim();
              *shape.add_dim() = input2_shape.dim(1);
              ONNX_NAMESPACE::updateOutputShape(ctx, 2, shape);
            }

            // Data shape
            propagateElemTypeFromInputToOutput(ctx, 3, 3);
            if (hasInputShape(ctx, 3)) {
              const auto& input3_shape = getInputShape(ctx, 3);
              if (input3_shape.dim_size() != 2) {
                fail_shape_inference("Expecting data to have 2 dimensions");
              }
              ONNX_NAMESPACE::TensorShapeProto shape;
              shape.add_dim();
              *shape.add_dim() = input3_shape.dim(1);
              ONNX_NAMESPACE::updateOutputShape(ctx, 3, shape);
            }
          });
}

}  // namespace featurizers
}  // namespace onnxruntime
