// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <stdexcept>

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace contrib {

enum class QuantParamTensorType : int {
  Scalar = 0,
  Tensor,
  Both,
};

void ValidateTypeAndShapeForScaleAndZP(ONNX_NAMESPACE::InferenceContext& ctx, int index,
                                       ::google::protobuf::int32 expectedType, QuantParamTensorType expectedScalar,
                                       int expectedTensorSize = 0);

// Names follow the convention of BFP_<# sign bits>_<# mantissa bits>_<# exponent bits>_<size of bounding box>
enum class BFPType : int64_t {
  // 1 sign bit, 8 mantissa bits, 8 exponent bits, 16 numbers per bounding box.
  BFP_1_8_8_16,

  // 1 sign bit, 8 mantissa bits, 8 exponent bits, 16 numbers per bounding box.
  BFP_1_4_8_16,

  // No sign bit, 8 mantissa bits, 8 exponent bits, 64 numbers per bounding box. Mantissa and exponent are signed.
  BFP_0_8_8_64,

  // No sign bit, 8 mantissa bits, 8 exponent bits, 128 numbers per bounding box. Mantissa and exponent are signed.
  BFP_0_8_8_128,

  // No sign bit, 8 mantissa bits, 8 exponent bits, 256 numbers per bounding box. Mantissa and exponent are signed.
  BFP_0_8_8_256,

  // No sign bit, 4 mantissa bits, 8 exponent bits, 128 numbers per bounding box. Mantissa and exponent are signed.
  BFP_0_4_8_128,

  // No sign bit, 4 mantissa bits, 8 exponent bits, 256 numbers per bounding box. Mantissa and exponent are signed.
  BFP_0_4_8_256,

  // No sign bit, 16 mantissa bits, 8 exponent bits, 64 numbers per bounding box. Mantissa and exponent are signed.
  BFP_0_16_8_64,

  // No sign bit, 16 mantissa bits, 8 exponent bits, 128 numbers per bounding box. Mantissa and exponent are signed.
  BFP_0_16_8_128,

  // No sign bit, 16 mantissa bits, 8 exponent bits, 256 numbers per bounding box. Mantissa and exponent are signed.
  BFP_0_16_8_256,

  // No sign bit, 16 mantissa bits, 8 exponent bits, 1 number per bounding box. Mantissa and exponent are signed.
  BFP_0_16_8_1,

  // No sign bit, 24 mantissa bits, 8 exponent bits, 1 number per bounding box. Mantissa and exponent are signed.
  BFP_0_24_8_1,

  // Reserved for custom BFP types
  Custom_BFP_0,
  Custom_BFP_1
};

inline size_t get_bounding_box_size(const BFPType& bfp_type) {
  switch (bfp_type) {
    case BFPType::BFP_0_16_8_1:
    case BFPType::BFP_0_24_8_1:
      return 1u;
    case BFPType::BFP_1_4_8_16:
    case BFPType::BFP_1_8_8_16:
      return 16u;
    case BFPType::BFP_0_8_8_64:
    case BFPType::BFP_0_16_8_64:
      return 64u;
    case BFPType::BFP_0_4_8_128:
    case BFPType::BFP_0_8_8_128:
    case BFPType::BFP_0_16_8_128:
      return 128u;
    case BFPType::BFP_0_4_8_256:
    case BFPType::BFP_0_8_8_256:
    case BFPType::BFP_0_16_8_256:
      return 256u;
    default:
      ONNX_THROW_EX(std::invalid_argument("Unsuported bfp_type case."));
  }
}
}  // namespace contrib
}  // namespace onnxruntime
