// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
// TODO(): move compare_mlvalue.{h,cc} to test dir

#include <core/framework/ort_value.h>
#include <string>

namespace ONNX_NAMESPACE {
class ValueInfoProto;
}

namespace Ort {
struct Value;
}

namespace onnxruntime {

enum class COMPARE_RESULT {
  SUCCESS,
  RESULT_DIFFERS,
  TYPE_MISMATCH,
  SHAPE_MISMATCH,
  NOT_SUPPORT,
};

std::pair<COMPARE_RESULT, std::string> CompareOrtValue(const OrtValue& real, const OrtValue& expected,
                                                       double per_sample_tolerance,
                                                       double relative_per_sample_tolerance, bool post_processing);

// Compare two OrtValue numerically equal or not. The difference with CompareOrtValue is that this function
// will only check the numerical values of the OrtValue, and ignore the type, shape, etc.
//
// For example, if some tests run CPU EP baseline with float, and run CUDA test with float16, we could check
// them without converting the float16 to float. Be noted: we will try convert the float value to float16
// then back to float, to simulate rounding to half.
//
// If not equal, only return one single the differed value pair (to avoid multiple thread
// append into the same error_message string stream).
std::pair<COMPARE_RESULT, std::string> CompareOrtValueNumerals(const OrtValue& real,
                                                               const OrtValue& expected,
                                                               double per_sample_tolerance,
                                                               double relative_per_sample_tolerance);

// verify if the 'value' matches the 'expected' ValueInfoProto. 'value' is a model output
std::pair<COMPARE_RESULT, std::string> VerifyValueInfo(const ONNX_NAMESPACE::ValueInfoProto& expected,
                                                       const OrtValue* value);
}  // namespace onnxruntime
