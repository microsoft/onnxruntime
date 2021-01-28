// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "op_builder.h"

namespace onnxruntime {
namespace coreml {

// Get the lookup table with IOpBuilder delegates for different onnx operators
// Note, the lookup table should have same number of entries as the result of CreateOpSupportCheckers()
// in op_support_checker.h
const std::unordered_map<std::string, const IOpBuilder*>& GetOpBuilders();

std::unique_ptr<IOpBuilder> CreateBinaryOpBuilder();

}  // namespace coreml
}  // namespace onnxruntime