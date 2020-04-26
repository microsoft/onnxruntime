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
using ONNX_NAMESPACE::OPTIONAL_VALUE;

// Forward declarations
static void RegisterCatImputerFeaturizerVer1();
static void RegisterCountVectorizerFeaturizerVer1();
static void RegisterDateTimeFeaturizerVer1();
static void RegisterForecastingPivotFeaturizerVer1();
static void RegisterFromStringFeaturizerVer1();
static void RegisterHashOneHotVectorizerFeaturizerVer1();
static void RegisterImputationMarkerFeaturizerVer1();
static void RegisterLabelEncoderFeaturizerVer1();
static void RegisterLagLeadOperatorFeaturizerVer1();
static void RegisterMaxAbsScalerFeaturizerVer1();
static void RegisterMeanImputerFeaturizerVer1();
static void RegisterMedianImputerFeaturizerVer1();
static void RegisterMinMaxImputerFeaturizerVer1();
static void RegisterMinMaxScalerFeaturizerVer1();
static void RegisterMissingDummiesFeaturizerVer1();
static void RegisterModeImputerFeaturizerVer1();
static void RegisterNumericalizeFeaturizerVer1();
static void RegisterOneHotEncoderFeaturizerVer1();
static void RegisterNormalizeFeaturizerVer1();
static void RegisterPCAFeaturizerVer1();
static void RegisterRobustScalerFeaturizerVer1();
static void RegisterRollingWindowFeaturizerVer1();
static void RegisterShortGrainDropperFeaturizerVer1();
static void RegisterStandardScaleWrapperFeaturizerVer1();
static void RegisterStringFeaturizerVer1();
static void RegisterTfidfVectorizerFeaturizerVer1();
static void RegisterTimeSeriesImputerFeaturizerVer1();
static void RegisterTruncatedSVDFeaturizerVer1();

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
void RegisterMSFeaturizersSchemas() {
  RegisterCatImputerFeaturizerVer1();
  RegisterCountVectorizerFeaturizerVer1();
  RegisterDateTimeFeaturizerVer1();
  RegisterForecastingPivotFeaturizerVer1();
  RegisterFromStringFeaturizerVer1();
  RegisterHashOneHotVectorizerFeaturizerVer1();
  RegisterImputationMarkerFeaturizerVer1();
  RegisterLabelEncoderFeaturizerVer1();
  RegisterLagLeadOperatorFeaturizerVer1();
  RegisterMaxAbsScalerFeaturizerVer1();
  RegisterMeanImputerFeaturizerVer1();
  RegisterMedianImputerFeaturizerVer1();
  RegisterMinMaxImputerFeaturizerVer1();
  RegisterMinMaxScalerFeaturizerVer1();
  RegisterMissingDummiesFeaturizerVer1();
  RegisterModeImputerFeaturizerVer1();
  RegisterNumericalizeFeaturizerVer1();
  RegisterOneHotEncoderFeaturizerVer1();
  RegisterPCAFeaturizerVer1();
  RegisterRobustScalerFeaturizerVer1();
  RegisterRollingWindowFeaturizerVer1();
  RegisterNormalizeFeaturizerVer1();
  RegisterShortGrainDropperFeaturizerVer1();
  RegisterStandardScaleWrapperFeaturizerVer1();
  RegisterStringFeaturizerVer1();
  RegisterTfidfVectorizerFeaturizerVer1();
  RegisterTimeSeriesImputerFeaturizerVer1();
  RegisterTruncatedSVDFeaturizerVer1();
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
void RegisterCatImputerFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      Imputes (populates) values with the mode (most common value) encountered during
  //      training. This featurizer supports float and double for most (if not all) frameworks
  //      due to the existance of NaN in those types. Other types require 'optional' support
  //      within the host frameworks and programming languages.

  //      C++-style pseudo signature:
  //        float execute(float const &value);
  //        double execute(double const &value);
  //        template <typename T> T execute(std::optional<T> const &value);

  //      Examples (where 55.5 is the mode value):
  //        execute(1.0) -> 1.0
  //        execute(NaN) -> 55.5
  //        execute(2.0) -> 2.0
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(CatImputerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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

void RegisterCountVectorizerFeaturizerVer1() {
  // static const char* doc = R"DOC(
  //     Returns the count of the number of occurrances of each distinct item according to a
  //     vocabulary established during training.

  //     C++-style pseudo signature:
  //       CountVector execute(std::string const &value);

  //     Examples:
  //       Assuming the training data is...
  //       ["orange apple orange grape", "grape carrot carrot apple", "peach banana orange banana"]

  //       The input data is...
  //       "banana grape grape apple apple apple orange"

  //       The result will be computed by...
  //         categorize and compute each word's number of apperance in input data, we have "apple -> 3", "banana -> 1", "grape -> 2", "orange -> 1"
  //         construct a dictionary and assign id for each unique word using training data, we have "apple -> 0", "banana -> 1", "grape -> 3", "orange -> 4"
  //         generate TFStruct by combining <word's id, word's number of apperance>

  //       The result is...
  //       [3, 1, 0, 2, 1]
  // )DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(CountVectorizerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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
          "T1")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "T",
          {"tensor(string)"},
          "No information is available")
      .TypeConstraint(
          "T1",
          {"tensor(uint32)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT32, 0);
            ONNX_NAMESPACE::TensorShapeProto shape_0;
            shape_0.add_dim();  // unknown at this time
            ONNX_NAMESPACE::updateOutputShape(ctx, 0, shape_0);
          });
}

void RegisterDateTimeFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      Extracts various datetime-related values from a UTC time_point.

  //      C++-style pseudo signature:
  //        TimePoint execute(std::chrono::system_clock::time_point const &value);

  //      Examples:
  //        Given a time_point 'value' representing "November 17, 1976 12:27:04PM":

  //        "November 17, 1976 12:27:04PM" => {
  //          "year": 1976,
  //          "month": 11,
  //          "day": 17,
  //          "hour": 12,
  //          "minute": 27,
  //          "second": 04,
  //          "amPm": 2,        // PM
  //          "hour12": 12,
  //          "dayOfWeek": 3,   // Wednesday
  //          "dayOfQuarter": 48,
  //          "dayOfYear": 321,
  //          "weekOfMonth": 2,
  //          "quarterOfYear": 4,
  //          "halfOfYear": 2,
  //          "weekIso": 47,
  //          "yearIso": 1976,
  //          "monthLabel": "November",
  //          "amPmLabel": "pm",
  //          "dayOfWeekLabel": "Wednesday",
  //          "holidayName": "",
  //          "isPaidTimeOff": 0
  //        }
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(DateTimeTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 1);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 1);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 2);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 2);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 3);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 3);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 4);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 4);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 5);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 5);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 6);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 6);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 7);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 7);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 8);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 8);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 9);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 9);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT16, 10);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 10);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT16, 11);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 11);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 12);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 12);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 13);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 13);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 14);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 14);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_INT32, 15);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 15);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_STRING, 16);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 16);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_STRING, 17);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 17);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_STRING, 18);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 18);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_STRING, 19);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 19);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 20);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 20);
            }
          });
}

void RegisterForecastingPivotFeaturizerVer1(){
  //static const char* doc = R"DOC(
  // Similar to an Excel pivot table, this featurizer will expand values in the given output
  //         where all "linked" values are not null/empty. "linked" means that all values in the same
  //         column across all matrixes are not null/empty.

  //         All rows across all matrixes must have the same length / same number of columns. The maximum number
  //         of rows generated by each invocation is equal to the number of columns of all the input matrixes.

  //         Better explained through examples, see below for a more detailed explaination of the
  //         functionality.

  //         C++-style pseudo signature:
  //         std::vector<double> execute(std::vector<Eigen::Matrix<double>> const &);
  //         std::vector<std::optional<std::string>> execute(std::vector<Eigen::Matrix<std::optional<std::string>>> const &);

  //         Examples:
  //           Given results produced by the RollingWindow- and LagLead-Transformers...

  //           +-------+--------------------------------+--------------------------------+
  //           | Index |     Rolling Window Results     |        Lag Lead Results        |
  //           +=======+================================+================================+
  //           | 0     | [ [na, na, na] ]               | [ [na, na, na], [na, na, na] ] |
  //           +-------+--------------------------------+--------------------------------+
  //           | 1     | [ [1, 2, 3] ]                  | [ [na, na, na], [na, na, na] ] |
  //           +-------+--------------------------------+--------------------------------+
  //           | 2     | [ [1, 2, 3] ]                  | [ [na, na, na], [na, na, na] ] |
  //           +-------+--------------------------------+--------------------------------+
  //           | 3     | [ [1, 2, 3] ]                  | [ [A, B, C], [na, na, na] ]    |
  //           +-------+--------------------------------+--------------------------------+
  //           | 4     | [ [1, 2, 3] ]                  | [ [A, B, C], [D, na, na] ]     |
  //           +-------+--------------------------------+--------------------------------+
  //           | 5     | [ [1, 2, 3] ]                  | [ [A, B, C], [D, na, F] ]      |
  //           +-------+--------------------------------+--------------------------------+

  //           Results:

  //             4: 1, A, D
  //             5: 1, A, D
  //             5: 3, C, F

  //           A more thourough description below uses the following notation:

  //             RW: Rolling Window Results
  //             LL: Lag Lead Results

  //             RW[row_index][col_index]
  //             LL[row_index][col_index]

  //             Using this notation for input index 5, we see:

  //               RW[0][0] == 1       LL[0][0] == A
  //               RW[0][1] == 2       LL[0][1] == B
  //                                   LL[1][0] == D
  //                                   LL[1][1] == na
  //                                   LL[1][2] == F

  //             For input at index N:

  //               0:
  //                 RW[0][0] == na, LL[0][0] == na, LL[1][0] == na;   na's found, nothing to output
  //                 RW[0][1] == na, LL[0][1] == na, LL[1][1] == na;   na's found, nothing to output
  //                 RW[0][2] == na, LL[0][2] == na, LL[1][2] == na;   na's found, nothing to output

  //               ...

  //               4:
  //                 RW[0][0] == 1, LL[0][0] == A, LL[1][0] == D;      no na's found - OUTPUT GENERATED (1, A, D)
  //                 RW[0][1] == 2, LL[0][1] == B, LL[1][1] == na;     na's found, nothing to output
  //                 RW[0][2] == 3, LL[0][2] == C, LL[1][2] == na;     na's found, nothing to output

  //               5:
  //                 RW[0][0] == 1, LL[0][0] == A, LL[1][0] == D;      no na's found - OUTPUT GENERATED (1, A, D)
  //                 RW[0][1] == 2, LL[0][1] == B, LL[1][1] == na;     na's found, nothing to output
  //                 RW[0][2] == 3, LL[0][2] == C, LL[1][2] == F;      no na's found - OUTPUT GENERATED (3, C, F)
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(ForecastingPivotTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .Attr("num_pivot_columns", "The first num_pivot_columns input in Input1 are pivoted", AttributeProto::INT)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "Inputs",
          "Variadic number of Input containing tensors of different size",
          "T",
          ONNX_NAMESPACE::OpSchema::FormalParameterOption::Variadic,
          false)
      .Output(
          0,
          "Output",
          "No information is available",
          "T",
          ONNX_NAMESPACE::OpSchema::FormalParameterOption::Variadic,
          false)
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "T",
          {"tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)",
           "tensor(float)", "tensor(double)", "tensor(bool)", "tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            //The first num_pivot_columns inputs of Input(1) only support float & double
            if (hasInputShape(ctx, 1)) {
              const auto& input_shape = getInputShape(ctx, 1);
              if (input_shape.dim_size() < 2) {
                fail_shape_inference("Expecting Inputs to have more than 2 dimensions");
              }
            }
            ONNX_NAMESPACE::TensorShapeProto shape;
            shape.add_dim();
            shape.add_dim();
            ONNX_NAMESPACE::updateOutputShape(ctx, 0, shape);
          });
}

void RegisterFromStringFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      Converts from a string to a scalar type.
  //      If destination type is a string, it is a passthrough.

  //      C++-style pseudo signature:
  //        int32 execute(std::string const &value);
  //        bool execute(std::string const &value);

  //      Examples:
  //        execute("True") -> true [bool]
  //        execute("10") -> 10 [int32]
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(FromStringTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .Attr(
          "result_type",
          "This is an integer that must represent one of the types that are enumerated in the OutputT constraint",
          AttributeProto::INT)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "Input",
          "Input string to be converted",
          "InputT")
      .Output(
          0,
          "Output",
          "A type converted from string",
          "OutputT")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "InputT",
          {"tensor(string)"},
          "Input string to be converted")
      .TypeConstraint(
          "OutputT",
          {"tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)",
           "tensor(float)", "tensor(double)", "tensor(bool)", "tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            using namespace ONNX_NAMESPACE;
            const auto* attr_proto = ctx.getAttribute("result_type");
            if (nullptr == attr_proto) {
              fail_type_inference("result_type is mandatory")
            }

            auto attr_value = attr_proto->i();
            if (!TensorProto::DataType_IsValid(static_cast<int>(attr_value))) {
              fail_type_inference("result_type value is not valid")
            }

            auto type_int = static_cast<TensorProto::DataType>(attr_value);
            switch (type_int) {
                // fall through
              case TensorProto_DataType_INT8:
              case TensorProto_DataType_UINT8:
              case TensorProto_DataType_INT16:
              case TensorProto_DataType_UINT16:
              case TensorProto_DataType_INT32:
              case TensorProto_DataType_UINT32:
              case TensorProto_DataType_INT64:
              case TensorProto_DataType_UINT64:
              case TensorProto_DataType_FLOAT:
              case TensorProto_DataType_DOUBLE:
              case TensorProto_DataType_BOOL:
              case TensorProto_DataType_STRING:
                break;
              default:
                fail_type_inference("attr result_type is expected to have an accepted type");
                break;
            }
            propagateElemTypeFromDtypeToOutput(ctx, type_int, 0);
            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }
          });
}

void RegisterHashOneHotVectorizerFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      Hashes the input to a categorical value, then produces a one hot encoded vector
  //      based on that value.

  //      C++-style pseudo signature:
  //          template <typename T> HashOneHotVectorizerStruct execute(T const &value);

  //      Examples:
  //        Assuming the hashing algorithm...
  //          "A" -> 1
  //          "B" -> 2
  //          "C" -> 5

  //        and 'numCols' set to 8:

  //          execute("A") -> [1, 0, 0, 0, 0, 0, 0, 0]
  //          execute("B") -> [0, 1, 0, 0, 0, 0, 0, 0]
  //          execute("C") -> [0, 0, 0, 0, 1, 0, 0, 0]
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(HashOneHotVectorizerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 1);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 1);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT64, 2);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 2);
            }
          });
}

void RegisterImputationMarkerFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      Returns true if the input is null, false if it is not.

  //      C++-style pseudo signature:
  //        bool execute(float const &value);
  //        bool execute(double const &value);
  //        template <typename T> bool execute(std::optional<T> const &value);

  //      Examples:
  //        3.0 -> false
  //        NaN -> true
  //        "foo" -> false
  //        std::optional<string>() -> true
  //        std::optional<string>("bar") -> false
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(ImputationMarkerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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
  //static const char* doc = R"DOC(
  //      Returns a unique id for the input based on all values encountered during training.

  //      C++-style pseudo signature:
  //        template <typename T> uint32 execute(T const &value);

  //      Examples:
  //        Assuming the training data of ["A", "B", "C"]...

  //        execute("A") -> 1
  //        execute("B") -> 2
  //        execute("C") -> 3
  //        execute("This value was not seen during training") -> 0
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(LabelEncoderTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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

void RegisterLagLeadOperatorFeaturizerVer1() {
  //static const char* doc = R"DOC(
      // Copying values from prior or future per grain. Works for general time series data sets.

      // The Horizon represents the maximum value in a range [1, N], where each element in that range is a delta applied to each offset. The resulting matrix will be in the form:

      // [
      // [value[offset[0] - N], value[offset[0] - (N - 1)], ..., value[offset[0] - 1]],
      // [value[offset[1] - N], value[offset[1] - (N - 1)], ..., value[offset[1] - 1]],
      // ...
      // [value[offset[K - 1] - N], value[offset[K - 1] - (N - 1)], ..., value[offset[K - 1] - 1]]
      // ]

      // The resulting matrix size is K rows x N cols, where K is the number of offsets and N is the horizon.

      // Horizon and offsets should be passed in during construction. Offsets are passed in as a vector of ints so multiple lag orders can be applied within one featurizer call.
      // Output type is a tuple of vector of string, which representing grains, and a matrix. The matrix is of optional<T> where rows are grouped by different offsets and columns are grouped by horizon.

      // C++-style pseudo signature:
      //   template <typename T> tuple<vector<string>,matrix<T?>> execute(std::vector<std::string> const &, T const &> const &value);

      // Examples:
      //     Since this featurizer is copying values per grain, we just use one type of grain in the following examples.

      //     A simple example would be horizon = 1 and we have offsets as [-3, 1] (which means lag 3 and lead 1)
      //     +-------+-------+---------------------+
      //     | grain | target| target_lag_3_lead_1 |
      //     +=======+=======+=====================+
      //     |Walmart| 8     | [[NAN], [  9]]      |
      //     +-------+-------+---------------------+
      //     |Walmart| 9     | [[NAN], [ 10]]      |
      //     +-------+-------+---------------------+
      //     |Walmart| 10    | [[NAN], [ 11]]      |
      //     +-------+-------+---------------------+
      //     |Walmart| 11    | [[  8], [NAN]]      |
      //     +-------+-------+---------------------+
      //     Values from the row above current row are copied.

      //     A more complex example would be, assuming we have horizon = 2 and we have offsets as [-2, 2, 1, -1] (which means lag 2, lead 2, lead 1 and lag 1)
      //     +-------+-------+-------------------------------------------------+
      //     | grain | target|        target_lag_2_lead_2_lead_1_lag_1         |
      //     +=======+=======+=================================================+
      //     |Walmart| 8     | [[NAN, NAN], [  9,  10], [NAN, NAN], [ 8,   9]] |
      //     +-------+-------+-------------------------------------------------+
      //     |Walmart| 9     | [[NAN, NAN], [ 10,  11], [NAN,   8], [ 9,  10]] |
      //     +-------+-------+-------------------------------------------------+
      //     |Walmart| 10    | [[NAN,   8], [ 11, NAN], [  8,   9], [10,  11]] |
      //     +-------+-------+-------------------------------------------------+
      //     |Walmart| 11    | [[  8,   9], [NAN, NAN], [  9,  10], [11, NAN]] |
      //     +-------+-------+-------------------------------------------------+
      //     Basically, if we have an offset of k for the row with row index t,
      //     target_lag_k[t] = target[t - horizon + k + 1]
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(LagLeadOperatorTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "Grains",
          "Grains tensor of shape [R][K].",
          "GrainT")
      .Input(
          2,
          "Target",
          "Target tensor of shape [R]",
          "T")
      .Output(
          0,
          "OutputGrains",
          "Grains tensor of shape [R][K]",
          "GrainT")
      .Output(
          1,
          "Output",
          "Output tensor of shape [R][P][Q]",
          "T")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "GrainT",
          {"tensor(string)"},
          "No information is available")
      .TypeConstraint(
          "T",
          {"tensor(float)", "tensor(double)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromInputToOutput(ctx, 1, 0);
            auto input_elem_type = ctx.getInputType(2)->tensor_type().elem_type();
            if (input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
              propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 1);
            } else if (input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
              propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, 1);
            }
            if (hasInputShape(ctx, 1)) {
              const auto& grains_shape = getInputShape(ctx, 1);
              if (grains_shape.dim_size() != 2) {
                fail_shape_inference("Expecting Grains to have 2 dimensions");
              }
              ONNX_NAMESPACE::TensorShapeProto shape;
              *shape.add_dim() = grains_shape.dim(0);
              shape.add_dim();
              shape.add_dim();
              ONNX_NAMESPACE::updateOutputShape(ctx, 1, shape);
            }
            if (hasInputShape(ctx, 2)) {
              const auto& target_shape = getInputShape(ctx, 2);
              if (target_shape.dim_size() != 1) {
                fail_shape_inference("Expecting Target to have 1 dimensions");
              }
            }
          });
}

void RegisterMaxAbsScalerFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      Scales input based on the maximum absolute value of all data encountered during training.

  //      C++-style pseudo signature:
  //        float execute(uint16 value);
  //        double execute(uint32 value);

  //      Examples:
  //        Given a training set of [1.0, -2.0, 3.0, -4.0], where 4.0 is the absolute value of the
  //        maximum value encountered...

  //        execute(1.0) -> 1.0 / 4.0
  //        execute(-4.0) -> -4.0 / 4.0
  //        execute(100.0) -> 100 / 4.0
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(MaxAbsScalerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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

void RegisterMeanImputerFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      Imputes (populates) values with the mean (average) value encountered during
  //      training.

  //      C++-style pseudo signature:
  //        float execute(float const &value);
  //        double execute(double const &value);
  //        template <typename T> T execute(std::optional<T> const &value);

  //      Examples (where 123.4 is the mean value):
  //        execute(1.0) -> 1.0
  //        execute(NaN) -> 123.4
  //        execute(2.0) -> 2.0
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(MeanImputerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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
          {"tensor(float)", "tensor(double)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, 0);
            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }
          });
}

void RegisterMedianImputerFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      Imputes (populates) values with the median value encountered during
  //      training.

  //      C++-style pseudo signature:
  //        float execute(float const &value);
  //        double execute(double const &value);
  //        template <typename T> T execute(std::optional<T> const &value);

  //      Examples (where 123.4 is the median value):
  //        execute(1.0) -> 1.0
  //        execute(NaN) -> 123.4
  //        execute(2.0) -> 2.0
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(MedianImputerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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
          {"tensor(float)", "tensor(double)", "tensor(string)"},
          "No information is available")
      .TypeConstraint(
          "OutputT",
          {"tensor(double)", "tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            auto input_elem_type = ctx.getInputType(1)->tensor_type().elem_type();
            if (input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
                input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
              propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, 0);
            } else if (input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
              propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_STRING, 0);
            } else {
              fail_type_inference("input 1 is expected to have an accepted type");
            }

            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }
          });
}

void RegisterMinMaxImputerFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      Imputes (populates) values with the minimum or maximum value encountered during
  //      training.

  //      C++-style pseudo signature:
  //        float execute(float const &value);
  //        double execute(double const &value);
  //        template <typename T> T execute(std::optional<T> const &value);

  //      Examples (where 123 is the minimum or maximum value (depending on configuration
  //      parameters set when creating the estimator):
  //        execute(1.0) -> 1.0
  //        execute(NaN) -> 123.4
  //        execute(2.0) -> 2.0
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(MinMaxImputerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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

void RegisterMinMaxScalerFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      Scales input based on the scale that results from the minimum and maximum values encountered
  //      during training.

  //      C++-style pseudo signature:
  //          template <typeanem T> double(T const &value);

  //      Examples:
  //        Given the training data [1, 2, 3, 4, 5];
  //          min: 1
  //          max: 5
  //          scale (<max> - <min>): 4

  //        execute(2) = 2 / 4
  //        execute(20) = 20 / 4
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(MinMaxScalerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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
  //static const char* doc = R"DOC(
  //      Returns 1 if the input is null, 0 if it is not.

  //      C++-style pseudo signature:
  //          int8 execute(float const &value);
  //          int8 execute(double const &value);
  //          template <typename T> int8 execute(T const &value);

  //      Examples:
  //        1.0 -> 0
  //        NaN -> 1
  //        "foo" -> 0
  //        std::optional<string>() -> 1
  //        std::optional<string>("bar") -> 0
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(MissingDummiesTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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

void RegisterModeImputerFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      Imputes (populates) values with the mode (most frequent) value encountered during
  //      training. The first value encountered with be used in the event that mutliple values
  //      we found the most number of times.

  //      C++-style pseudo signature:
  //        float execute(float const &value);
  //        double execute(double const &value);
  //        template <typename T> T execute(std::optional<T> const &value);

  //      Examples (where 123.4 is the mode value):
  //        execute(1.0) -> 1.0
  //        execute(NaN) -> 123.4
  //        execute(2.0) -> 2.0
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(ModeImputerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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

void RegisterNumericalizeFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      This is the LabelEncoder, but returns a null value for categories not encountered during training.

  //      C++-style pseudo signature:
  //        template <typename T> std::optional<uint32_t> execute(T const &value);

  //      Examples:
  //        Assuming the training data of ["Oh", "Huh", "Heh"]...
  //        execute("Oh") -> std::optional<uint32_t>(1)
  //        execute("Huh") -> std::optional<uint32_t>(2)
  //        execute("Heh") -> std::optional<uint32_t>(3)
  //        # This value was not seeing at training time
  //        execute("Oops") -> std::optional<uint32_t>()
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(NumericalizeTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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
          {"tensor(int8)", "tensor(uint8)", "tensor(int16)", "tensor(uint16)",
           "tensor(int32)", "tensor(uint32)", "tensor(int64)", "tensor(uint64)",
           "tensor(float)", "tensor(double)", "tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 0);
            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }
          });
}

void RegisterOneHotEncoderFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      Produces a one hot vector based on categories calculated during training.

  //      C++-style pseudo signature:
  //        template <typename T> OneHotVector execute(T const &value);

  //      Examples:
  //        Assuming the training data [10, 20, 30, 40]...

  //        execute(10) -> [0, 1, 0, 0, 0]
  //        execute(20) -> [0, 0, 1, 0, 0]
  //        execute(30) -> [0, 0, 0, 1, 0]
  //        execute(40) -> [0, 0, 0, 0, 1]
  //        execute(200) -> [1, 0, 0, 0, 0]
  //        execute(-1) -> [1, 0, 0, 0, 0]
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(OneHotEncoderTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT8, 1);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 1);
            }

            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT64, 2);
            if (has_shape) {
              propagateShapeFromInputToOutput(ctx, 1, 2);
            }
          });
}

void RegisterPCAFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //    Principal component analysis and matrix projection

  //    C++-style pseudo signature:
  //      template <typename MatrixT> MatrixT execute(MatrixT const &value);

  //    Examples:
  //      Assuming the training matrix A
  //      By applying PCA we get the eigenvector P[p, q].
  //      P is obtained via State input to deserialize the transformer
  //      the projecting matrix of an input matrix X[m][m] is X*P^T [m][p]
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(PCATransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "X",
          "matrix X[M][N]",
          "T")
      .Output(
          0,
          "Output",
          "matrix X*P^T [M][P]",
          "T")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "T",
          {"tensor(float)", "tensor(double)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 1, 0);
            if (hasInputShape(ctx, 1)) {
              const auto& input1_shape = getInputShape(ctx, 1);
              ONNX_NAMESPACE::TensorShapeProto shape_0;
              *shape_0.add_dim() = input1_shape.dim(0);
              shape_0.add_dim();  // unknown at this time
              ONNX_NAMESPACE::updateOutputShape(ctx, 0, shape_0);
            }
          });
}

void RegisterRobustScalerFeaturizerVer1() {
  //static const char* doc = R"DOC(
    // Remove the median and scales the data according to the quantile range.

    // C++-style pseudo signature:
    //     float execute(TInputFloat const &value);
    //     double execute(TInputDouble const &value);

    // Examples:
    //   Assuming the Training data [[ 1, -2, 2], [-2, 1, 3], [ 4, 1,-2]]

    //   There are 3 columns...
    //       column 1          column 2          column 3
    //          1                 -2                2
    //         -2                  1                3
    //          4                  1               -2

    //   For each column, we calculate the median value...
    //       column 1          column 2          column 3
    //          1                  1                2

    //   Also calculate the range for each column
    //       column 1          column 2          column 3
    //     4 - (-2) = 6      1 - (-2) = 3      3 - (-2) = 5

    //   Apply quantile range(QR) to the ranges, assuming the QR is [q_min = 25, q_max = 75], we get the scaling value = range * (q_max% - q_min%) for each column
    //       column 1          column 2          column 3
    //     6 * 50% = 3       3 * 50% = 1.5     5 * 50% = 2.5

    //   Remove the median and scales the original data per column
    //       column 1          column 2          column 3
    //     ( 1 - 1) / 3     (-2 - 1) / 1.5    ( 2 - 2) / 2.5
    //     (-2 - 1) / 3     ( 1 - 1) / 1.5    ( 3 - 2) / 2.5
    //     ( 4 - 1) / 3     ( 1 - 1) / 1.5    (-2 - 2) / 2.5

    //   The final result is
    //       column 1          column 2          column 3
    //          0                 -2                0
    //         -1                  0              0.4
    //          1                  0             -1.6
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(RobustScalerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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

void RegisterRollingWindowFeaturizerVer1() {
  //static const char* doc = R"DOC(
  // Calculates data based on a rolling window. Currently supports mean. Works for any data set that is already sorted.

  // Input type for this featurizer is a tuple of the grain columns and target value column. It is assumed that the data is sorted in the correct order.

  // C++-style pseudo signature:
  // template <typename T> matrix<double> execute(std::tuple<std::vector<std::string> const &, T const &> value);

  // Examples:
  //     A simple example would be horizon = 1, maxWindowSize = 2, and we want to take the mean.

  //     +-----------+-------+-------------------+
  //     | grain     | target| target_mean       |
  //     +===========+=======+===================+
  //     | A         | 10    | [[NAN]]           |
  //     +-----------+-------+-------------------+
  //     | A         | 4     | [[10]]            |
  //     +-----------+-------+-------------------+
  //     | A         | 6     | [[7]]             |
  //     +-----------+-------+-------------------+
  //     | A         | 11    | [[5]]             |
  //     +-----------+-------+-------------------+

  //     A more complex example would be, assuming we have horizon = 2, maxWindowSize = 2, min window size = 2, and we want the mean
  //     +-----------+-------+-------------------+
  //     | grain     | target| target_max        |
  //     +===========+=======+===================+
  //     | A         | 10    | [[NAN, NAN]]      |
  //     +-----------+-------+-------------------+
  //     | A         | 4     | [[NAN, NAN]]      |
  //     +-----------+-------+-------------------+
  //     | A         | 6     | [[NAN, 7]]        |
  //     +-----------+-------+-------------------+
  //     | A         | 11    | [[7, 5]]          |
  //     +-----------+-------+-------------------+
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(AnalyticalRollingWindowTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "Grains",
          "Grains tensor of shape [R][K].",
          "GrainT")
      .Input(
          2,
          "Target",
          "Target tensor of shape [R]",
          "T")
      .Output(
          0,
          "Output",
          "Output tensor of shape [R][M]",
          "OutputT")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "GrainT",
          {"tensor(string)"},
          "No information is available")
      .TypeConstraint(
          "T",
          {"tensor(int8)", "tensor(uint8)", "tensor(int16)",  "tensor(uint16)", "tensor(int32)", "tensor(uint32)", "tensor(int64)", "tensor(uint64)", "tensor(float)", "tensor(double)"},
          "No information is available")
      .TypeConstraint(
          "OutputT",
          {"tensor(double)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, 0);
            if (hasInputShape(ctx, 1)) {
              const auto& grains_shape = getInputShape(ctx, 1);
              if (grains_shape.dim_size() != 2) {
                fail_shape_inference("Expecting Grains to have 2 dimensions");
              }

              ONNX_NAMESPACE::TensorShapeProto shape;
              *shape.add_dim() = grains_shape.dim(0);
              shape.add_dim();
              shape.add_dim();
              ONNX_NAMESPACE::updateOutputShape(ctx, 0, shape);
            }
            if (hasInputShape(ctx, 2)) {
              const auto& target_shape = getInputShape(ctx, 2);
              if (target_shape.dim_size() != 1) {
                fail_shape_inference("Expecting Target to have 1 dimensions");
              }
            }
          });

  //static const char* doc = R"DOC(
      // Calculates data based on a rolling window. Currently supports minimum and maximum. Works for any data set that is already sorted.
      //
      // Input type for this featurizer is a tuple of the grain columns and target column to find the value. It is assumed that the data is sorted in the correct order.
      //
      // C++-style pseudo signature:
      //   template <typename T> matrix<T> execute(std::tuple<std::vector<std::string> const &, T const &> value);
      //
      // Examples:
      //     A simple example would be horizon = 1, maxWindowSize = 2, and we want to take the minimum.
      //     +-----------+-------+-------------------+
      //     | grain     | target| target_minimum    |
      //     +===========+=======+===================+
      //     | A         | 10    | [[NAN]]           |
      //     +-----------+-------+-------------------+
      //     | A         | 4     | [[10]]            |
      //     +-----------+-------+-------------------+
      //     | A         | 6     | [[4]]             |
      //     +-----------+-------+-------------------+
      //     | A         | 11    | [[4]]             |
      //     +-----------+-------+-------------------+
      //     A more complex example would be, assuming we have horizon = 2, maxWindowSize = 2, minWindowSize = 2, and we want the maximum value
      //     +-----------+-------+-------------------+
      //     | grain     | target| target_max        |
      //     +===========+=======+===================+
      //     | A         | 10    | [[NAN, NAN]]      |
      //     +-----------+-------+-------------------+
      //     | A         | 4     | [[NAN, NAN]]      |
      //     +-----------+-------+-------------------+
      //     | A         | 6     | [[NAN, 10]]       |
      //     +-----------+-------+-------------------+
      //     | A         | 11    | [[10, 6]]         |
      //     +-----------+-------+-------------------+
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(SimpleRollingWindowTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "Grains",
          "Grains tensor of shape [R][K].",
          "GrainT")
      .Input(
          2,
          "Target",
          "Target tensor of shape [R]",
          "T")
      .Output(
          0,
          "Output",
          "Output tensor of shape [R][M]",
          "OutputT")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "GrainT",
          {"tensor(string)"},
          "No information is available")
      .TypeConstraint(
          "T",
          {"tensor(float)", "tensor(double)"},
          "No information is available")
      .TypeConstraint(
          "OutputT",
          {"tensor(double)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, 0);
            if (hasInputShape(ctx, 1)) {
              const auto& grains_shape = getInputShape(ctx, 1);
              if (grains_shape.dim_size() != 2) {
                fail_shape_inference("Expecting Grains to have 2 dimensions");
              }

              ONNX_NAMESPACE::TensorShapeProto shape;
              *shape.add_dim() = grains_shape.dim(0);
              shape.add_dim();
              shape.add_dim();
              ONNX_NAMESPACE::updateOutputShape(ctx, 0, shape);
            }
            if (hasInputShape(ctx, 2)) {
              const auto& target_shape = getInputShape(ctx, 2);
              if (target_shape.dim_size() != 1) {
                fail_shape_inference("Expecting Target to have 1 dimensions");
              }
            }
          });
}

void RegisterNormalizeFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //    Computes the L1 norm for a provided data set and normalize every row so that
  //    its L1 norm is 1

  //    C++-style pseudo signature:
  //      template <typename IteratorT> std::vector<std::double_t> execute(std::pair<IteratorT, IteratorT> const &value);
  //      template <typename IteratorT> std::vector<std::double_t> execute(std::tuple<IteratorT, IteratorT> const &value);

  //    Examples:
  //      Given the training data
  //      [[4, 1, 2, 2],
  //       [1, 3, 9, 3],
  //       [5, 7, 5, 1]]

  //      L1 norms for each row are: [9, 16, 18]

  //      execute([4,1,2,2]) = [4/9, 1/9, 2/9, 2/9]
  //      execute([1,3,9,3]) = [1/16, 3/16, 9/16, 3/16]
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(NormalizeTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "Input",
          "Input broken by rows with shape [R][C] or [C] for a single row",
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
          {"tensor(int8)", "tensor(uint8)", "tensor(int16)", "tensor(uint16)", "tensor(int32)", "tensor(uint32)",
           "tensor(int64)", "tensor(uint64)", "tensor(float)", "tensor(double)"},
          "No information is available")
      .TypeConstraint(
          "OutputT",
          {"tensor(double)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, 0);
            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }
          });
}

void RegisterShortGrainDropperFeaturizerVer1() {
  // static const char* doc = R"DOC(
  //  Returns true to indicate that a row should be dropped if it wasn't encountered during training.

  //  C++-style pseudo signature:
  //    bool execute(std::vector<std::string> const &value);

  //  Examples:
  //    Consider the training data:

  //    [ ["one"], ["two"], ["two"], ["three"], ["three"], ["three"] ]

  //    and a ShortGrainDropper configured with minPoints set to 2. Grains ["two"] and ["three"] appear
  //    enough times in the training data to remain, while any other grain should be dropped:

  //    [ "one" ] -> true                         # drop
  //    [ "two" ] -> false                        # dont' drop
  //    [ "three" ] -> false                      # don't drop
  //    [ "never seen during training" ] -> true  # drop
  // )DOC";
  MS_FEATURIZERS_OPERATOR_SCHEMA(ShortGrainDropperTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "GrainInput",
          "String tensor of shape [R][K]. Grain-related tensor",
          "T1")
      .Input(
          2,
          "Input",
          "Variadic number of Input containing tensors of different size. Non-Grain-related tensors.",
          "T",
          ONNX_NAMESPACE::OpSchema::FormalParameterOption::Variadic,
          false)
      .Output(
          0,
          "GrainOutput",
          "String tensor of shape [P][K], P <= R. Grain-related tensor after imputed(dropped)",
          "T1")
      .Output(
          1,
          "Output",
          "Variadic number of Input containing tensors of different size. Non-Grain-related tensors after imputed(dropped).",
          "T",
          ONNX_NAMESPACE::OpSchema::FormalParameterOption::Variadic,
          false)
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "T1",
          {"tensor(string)"},
          "No information is available")
      .TypeConstraint(
          "T",
          {"tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)",
           "tensor(float)", "tensor(double)", "tensor(bool)", "tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_STRING, 0);
            if (hasInputShape(ctx, 1)) {
              const auto& input_shape = getInputShape(ctx, 1);
              if (input_shape.dim_size() != 2) {
                fail_shape_inference("Expecting Input1 to have 2 dimensions");
              }
              ONNX_NAMESPACE::TensorShapeProto shape;
              shape.add_dim();
              *shape.add_dim() = input_shape.dim(1);
              ONNX_NAMESPACE::updateOutputShape(ctx, 0, shape);
            }
            if (hasInputShape(ctx, 2)) {
              const auto& input_shape = getInputShape(ctx, 2);
              if (input_shape.dim_size() != 2) {
                fail_shape_inference("Expecting Input2 to have 2 dimensions");
              }
            }
          });
}

void RegisterStandardScaleWrapperFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      Standardize features by removing the mean and scaling to unit variance based on input flag with_mean and with_std
  //      standard score of a sample x is calculated as
  //        z = (x - u) / s
  //      where u is the mean of the training samples or 0 if with_mean is false, and s is the standard deviation of the training samples or 1 if with_std is false

  //      C++-style pseudo signature:
  //        template <typeanem T> double(T const &value);

  //      Examples:
  //        Given the training data [0, 0, 1, 1];
  //          mean: 0.5
  //          std: 0.5

  //        execute(2) = (2 - 0.5) / 0.5 = 3 if with_mean is true and with_std is true
  //        execute(2) = (2 - 0.5) / 1 = 1.5 if with_mean is true and with_std is false
  //        execute(2) = (2 - 0) / 0.5 = 4 if with_mean is false and with_std is true
  //        execute(2) = (2 - 0) / 1 = 2 if with_mean is false and with_std is false
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(StandardScaleWrapperTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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

void RegisterStringFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //      Converts the input into a string representation based on the input's type.

  //      C++-style pseudo signature:
  //        template <typename T> string execute(T const &value);

  //      Examples:
  //        execute(1) -> "1"
  //        execute(3.14) -> "3.14"
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(StringTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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

void RegisterTfidfVectorizerFeaturizerVer1() {
  // static const char* doc = R"DOC(
  //     Convert a collection of raw documents to a matrix of TF-IDF features

  //     C++-style pseudo signature:
  //       TfidfVector execute(std::string const &value);

  //     Examples:
  //       Assuming the training data is...
  //       ["this is the first document", "this document is the second document", "and this is the third one", "is this the first document"]

  //       Assuming the input data is...
  //       "this is the first document"
  //       The default result will be...
  //       [0. , 0.469791f, 0.580286f, 0.384085f, 0. , 0. , 0.384085f, 0. , 0.384085f]

  //       Assuming the input data is...
  //       "this document is the second document"
  //       The default result will be...
  //       [0. , 0.687624f, 0. , 0.281089f, 0. , 0.538648f, 0.281089f, 0. , 0.281089f]

  //       Assuming the input data is...
  //       "and this is the third one"
  //       The default result will be...
  //       [0.511849f, 0. , 0. , 0.267104f, 0.511849f, 0. , 0.267104f, 0.511849f, 0.267104f]

  //       Assuming the input data is...
  //       "is this the first document"
  //       The default result will be...
  //       [0. , 0.469791f, ,0.580286f, 0.384085f, 0. , 0. , 0.384085f, 0. , 0.384085f]
  // )DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(TfidfVectorizerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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
          "T1")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "T",
          {"tensor(string)"},
          "No information is available")
      .TypeConstraint(
          "T1",
          {"tensor(float)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 0);
            ONNX_NAMESPACE::TensorShapeProto shape_0;
            shape_0.add_dim();  // unknown at this time
            ONNX_NAMESPACE::updateOutputShape(ctx, 0, shape_0);
          });
}

void RegisterTimeSeriesImputerFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //  Imputes rows and column values such that the generated output does not contain any
  //  time gaps per grain (based on the time gaps encountered during training) and that
  //  all missing column values are populated according to a strategy (forward fill,
  //  backward fill, mode, etc.).

  //  This Featurizer is unique in that it will produce 0:N rows per invocation, depending upon the
  //  input data.

  //  C++-style pseudo signature:
  //    template <typename... GrainColValueTs, typename... DataColValueTs>
  //    std::vector<
  //      std::tuple<
  //        bool, // true if the row was added
  //        std::chrono::system_clock::time_point,
  //        std::tuple<GrainColValueTs...>,
  //        std::tuple<DataColValueTs...>
  //      >
  //    > execute(
  //      std::chrono::system_clock::time_point const &value,
  //      std::tuple<GrainColValueTs...> const &grain,
  //      std::tuple<DataColValueTs...> const &colData
  //    );

  //  Examples:
  //    During training, the time period was found to be 1 day...

  //    Input:
  //      +------+-------+------------------+-------------------+
  //      | time | grain | forward fill col | backward fill col |
  //      +======+=======+==================+===================+
  //      | 1    | A     | 10               | None              |
  //      +------+-------+------------------+-------------------+
  //      | 2    | A     | None             | 200               |
  //      +------+-------+------------------+-------------------+
  //      | 1    | B     | -10              | -100              |
  //      +------+-------+------------------+-------------------+
  //      | 4    | A     | 40               | 400               |
  //      +------+-------+------------------+-------------------+
  //      | 6    | A     | 60               | 600               |
  //      +------+-------+------------------+-------------------+
  //      | 3    | B     | -30              | -300              |
  //      +------+-------+------------------+-------------------+

  //    Output:
  //      +-------+------+-------+------------------+-------------------+
  //      | Added | time | grain | forward fill col | backward fill col |
  //      +=======+======+=======+==================+===================+
  //      | false | 1    | A     | 10               | 200 (from 2)      |
  //      +-------+------+-------+------------------+-------------------+
  //      | false | 2    | A     | 10 (from 1)      | 200               |
  //      +-------+------+-------+------------------+-------------------+
  //      | true  | 3    | A     | 10 (from 2)      | 400 (from 4)      |
  //      +-------+------+-------+------------------+-------------------+
  //      | false | 4    | A     | 40               | 400               |
  //      +-------+------+-------+------------------+-------------------+
  //      | true  | 5    | A     | 40 (from 4)      | 600 (from 6)      |
  //      +-------+------+-------+------------------+-------------------+
  //      | false | 6    | A     | 60               | 600               |
  //      +-------+------+-------+------------------+-------------------+
  //      | false | 1    | B     | -10              | -100              |
  //      +-------+------+-------+------------------+-------------------+
  //      | true  | 2    | B     | -10 (from 1)     | -300 (from 3)     |
  //      +-------+------+-------+------------------+-------------------+
  //      | false | 3    | B     | -30              | -300              |
  //      +-------+------+-------+------------------+-------------------+
  //  )DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(TimeSeriesImputerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
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
      // The first input in Input(3)(a variadic tensor that has multiple output) is the below commented input
      // for the ONNX requirement that allow 0 size of variadic input
      // .Input(
      //     3,
      //     "Data",
      //     "It is a data tensor of shape [R][C] where R - rows and C - columns. R must be the same with Input(1)",
      //     "T2")
      .Input(
          3,
          "Input",
          "Variadic number of Inputs containing tensors of different type",
          "T",
          ONNX_NAMESPACE::OpSchema::FormalParameterOption::Variadic,
          false)
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
      // The first output in Output(3)(a variadic tensor that has multiple output) is the below commented output
      // for the ONNX requirement that allow 0 size of variadic output
      // .Output(
      //     3,
      //     "ImputedData",
      //     "Tensor of shape [IR][C] where IR is the number of rows in the output."
      //     "C is the number of columns.",
      //     "T2")
      .Output(
          3,
          "Output",
          "Variadic number of Outputs containing tensors of different type",
          "T",
          ONNX_NAMESPACE::OpSchema::FormalParameterOption::Variadic,
          false)
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
      .TypeConstraint(
          "T",
          {"tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)",
           "tensor(float)", "tensor(double)", "tensor(bool)", "tensor(string)"},
          "No information is available")
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

            //Data Shape & Variadic I/O shapes
            propagateElemTypeFromInputToOutput(ctx, 3, 3);
            if (hasInputShape(ctx, 3)) {
              const auto& input3_shape = getInputShape(ctx, 3);
              if (input3_shape.dim_size() != 2) {
                fail_shape_inference("Expecting data and variadic inputs to have 2 dimensions");
              }
              ONNX_NAMESPACE::TensorShapeProto shape;
              shape.add_dim();
              *shape.add_dim() = input3_shape.dim(1);
              ONNX_NAMESPACE::updateOutputShape(ctx, 3, shape);
            }
          });
}

void RegisterTruncatedSVDFeaturizerVer1() {
  //static const char* doc = R"DOC(
  //    Dimensionality reduction using truncated SVD algorithm

  //    C++-style pseudo signature:
  //      template <typename MatrixT> MatrixT execute(MatrixT const &value);

  //    Examples:
  //      Assuming the training matrix A
  //      By applying TruncatedSVD we get the right singular vector P [P][Q]
  //      the projecting matrix of an input matrix X is X*P[M][Q]
  //)DOC";

  MS_FEATURIZERS_OPERATOR_SCHEMA(TruncatedSVDTransformer)
      .SinceVersion(1)
      .SetDomain(kMSFeaturizersDomain)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "T0")
      .Input(
          1,
          "X",
          "matrix X[M][N]",
          "T")
      .Output(
          0,
          "Output",
          "matrix X*P^T [M][Q]",
          "T")
      .TypeConstraint(
          "T0",
          {"tensor(uint8)"},
          "No information is available")
      .TypeConstraint(
          "T",
          {"tensor(float)", "tensor(double)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 1, 0);
            if (hasInputShape(ctx, 1)) {
              const auto& input1_shape = getInputShape(ctx, 1);
              ONNX_NAMESPACE::TensorShapeProto shape_0;
              *shape_0.add_dim() = input1_shape.dim(0);
              shape_0.add_dim();  // unknown at this time
              ONNX_NAMESPACE::updateOutputShape(ctx, 0, shape_0);
            }
          });
}

}  // namespace featurizers
}  // namespace onnxruntime
