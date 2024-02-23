// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace contrib {
// ONNX namespace has the same function. We copy it to our namespace so that we can provide explicit specializations
// for it in onnxruntime::contrib namespace. Otherwise we will need to put a lot of our code in ONNX namespace.
template <typename T>
::ONNX_NAMESPACE::OpSchema GetOpSchema();
}  // namespace contrib
}  // namespace onnxruntime
