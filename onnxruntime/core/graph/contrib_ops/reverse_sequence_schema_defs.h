// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "onnx/defs/schema.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {
namespace contrib {

ONNX_NAMESPACE::OpSchema& RegisterReverseSequenceOpSchema(ONNX_NAMESPACE::OpSchema&& op_schema);

}  // namespace contrib
}  // namespace onnxruntime
