// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <string_view>

#include "core/framework/ort_value.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace test {

struct ValidateOutputParams {
  std::optional<float> relative_error;
  std::optional<float> absolute_error;
  bool sort_output = false;
};

/// <summary>
/// General purpose function to check the equality of two OrtValue instances. All ONNX types are supported.
/// </summary>
/// <param name="name">Value name</param>
/// <param name="expected">Expected value.</param>
/// <param name="actual">Actual value.</param>
/// <param name="params">Optional parameters to adjust how the check is performed.</param>
/// <param name="provider_type">Execution provider type if relevant.</param>
void CheckOrtValuesAreEqual(std::string_view name, const OrtValue& expected, const OrtValue& actual,
                            const ValidateOutputParams& params = {}, const std::string& provider_type = "");

}  // namespace test
}  // namespace onnxruntime
