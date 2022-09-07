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

namespace onnxruntime {
namespace contrib {

void ValidateTypeAndShapeForScaleAndZP(ONNX_NAMESPACE::InferenceContext& ctx, int index,
                                       ::google::protobuf::int32 expectedType, bool isScalar,
                                       int expectedTensorSize = 0);

}

// Names follow the convention of BFP_<# sign bits>_<# mantissa bits>_<# exponent bits>_<size of bounding box>
enum class BFPType : int64_t {
  // 1 sign bit, 8 mantissa bits, 8 exponent bits, 16 numbers per bounding box
  BFP_1_8_8_16,

  // 1 sign bit, 8 mantissa bits, 8 exponent bits, 16 numbers per bounding box
  BFP_1_4_8_16,

  // Reserved for custom BFP types
  Custom_BFP_0,
  Custom_BFP_1
};
}  // namespace onnxruntime
