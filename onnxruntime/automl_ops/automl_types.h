// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/data_types.h"
#include <functional>

namespace onnxruntime {
namespace automl {
void RegisterAutoMLTypes(const std::function<void(MLDataType)>& reg_fn);
const std::string* GetAutoMLCustomType(const ONNX_NAMESPACE::TypeProto& proto);
const std::string* GetAutoMLCustomType(const std::string&);
} // namespace automl
} // namespace onnxruntime
