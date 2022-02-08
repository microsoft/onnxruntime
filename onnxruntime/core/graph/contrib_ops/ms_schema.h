// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/schema.h"
#else
#include "onnx/defs/data_type_utils.h"
#endif

namespace onnxruntime {
namespace contrib {
template <typename T>
::ONNX_NAMESPACE::OpSchema GetOpSchema();
}
}  // namespace onnxruntime