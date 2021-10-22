// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/schema.h"
#else
#include "onnx/defs/data_type_utils.h"
#endif
#include "onnx/onnx_pb.h"
#include "onnx/onnx-operators_pb.h"
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