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
enum class COMPARE_RESULT { SUCCESS,
                            RESULT_DIFFERS,
                            TYPE_MISMATCH,
                            SHAPE_MISMATCH,
                            NOT_SUPPORT };
std::pair<COMPARE_RESULT, std::string> CompareOrtValue(const OrtValue& real, const OrtValue& expected,
                                                       double per_sample_tolerance,
                                                       double relative_per_sample_tolerance, bool post_processing);

// verify if the 'value' matches the 'expected' ValueInfoProto. 'value' is a model output
std::pair<COMPARE_RESULT, std::string> VerifyValueInfo(const ONNX_NAMESPACE::ValueInfoProto& expected,
                                                       const OrtValue* value);
}  // namespace onnxruntime
