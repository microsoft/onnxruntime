// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/graph/onnx_protobuf.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace contrib {

void ValidateTypeAndShapeForScaleAndZP(
    ONNX_NAMESPACE::InferenceContext& ctx,
    int index,
    ::google::protobuf::int32 expectedType,
    bool isScalar,
    int expectedTensorSize = 0);

}
}  // namespace onnxruntime