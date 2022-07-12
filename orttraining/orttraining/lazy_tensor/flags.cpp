// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "flags.h"
#include <cstdlib>
#include <cstring>
#include "core/common/common.h"

namespace onnxruntime {
namespace lazytensor {
bool IsEnvironmentVariableOne(const char* name) {
  const auto flag = std::getenv(name);
  if (flag == nullptr) {
    return false;
  }
  const auto is_one = std::strcmp(flag, "1") == 0;
  const auto is_zero = std::strcmp(flag, "0") == 0;
  ORT_ENFORCE(is_one || is_zero,
              "Must set ", name, "=0, ", name, "=1, or unset ", name);
  return is_one;
}

double GetEnvironmentVariableDoubleOrDefault(const char* name, const double default_value) {
  const auto number = std::getenv(name);
  if (!number) {
    return default_value;
  }
  return std::atof(number);
}

std::string RunType() {
  const auto run_type = std::getenv("ORT_LT_RUN_TYPE");
  if (!run_type) {
    return "ort";
  }
  return run_type;
}

bool DumpInputsOutputs() {
  return IsEnvironmentVariableOne("ORT_LT_DUMP_INPUTS_OUTPUTS");
}

bool DumpGraph() {
  return IsEnvironmentVariableOne("ORT_LT_DUMP_GRAPH");
}

bool CheckBaseline() {
  return IsEnvironmentVariableOne("ORT_LT_CHECK_BASELINE");
}

bool DumpAtenOpHistory() {
  return IsEnvironmentVariableOne("ORT_LT_DUMP_ATEN_OP_HISTORY");
}

bool CheckTensorContent() {
  ORT_ENFORCE(CheckBaseline(), "Must set ORT_LT_CHECK_BASELINE=1 to check tensor content.");
  return IsEnvironmentVariableOne("ORT_LT_CHECK_TENSOR_CONTENT");
}

double AbsoluteTolerance() {
  ORT_ENFORCE(CheckBaseline() && CheckTensorContent(),
              "Do not set ORT_LT_ABSOLUTE_TOLERANCE unless \
              ORT_LT_CHECK_TENSOR_CONTENT and ORT_LT_CHECK_BASELINE are set.");
  return GetEnvironmentVariableDoubleOrDefault("ORT_LT_ABSOLUTE_TOLERANCE", 1e-8);
}

double RelativeTolerance() {
  ORT_ENFORCE(CheckBaseline() && CheckTensorContent(),
              "Do not set ORT_LT_RELATIVE_TOLERANCE unless \
              ORT_LT_CHECK_TENSOR_CONTENT and ORT_LT_CHECK_BASELINE are set.");
  return GetEnvironmentVariableDoubleOrDefault("ORT_LT_RELATIVE_TOLERANCE", 1e-5);
}

bool DumpOnnxFusion() {
  return IsEnvironmentVariableOne("ORT_LT_DUMP_ONNX_FUSION");
}

}  // namespace lazytensor
}  // namespace onnxruntime
