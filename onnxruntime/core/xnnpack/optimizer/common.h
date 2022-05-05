// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <unordered_map>

#include "core/providers/common.h"
#include "onnx/onnx_pb.h"

namespace onnxruntime {
template <typename T>
static void AddAttribute(std::unordered_map<std::string, ::ONNX_NAMESPACE::AttributeProto>& attrs,
                         const std::string& name, const T& value) {
  attrs[name] = utils::MakeAttribute(name, value);
}
}