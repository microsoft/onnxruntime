// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnx/defs/schema.h"

namespace onnxruntime {
namespace contrib {
template <typename T>
::ONNX_NAMESPACE::OpSchema GetOpSchema();
}
}  // namespace onnxruntime