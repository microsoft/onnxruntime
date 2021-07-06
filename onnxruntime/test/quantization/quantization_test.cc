// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/quantization/quantization.h"

namespace onnxruntime {
namespace test {

TEST(Quantization, QuantizeSingleFloatInt8) {
  const float x = 1231.34f;

  quantization::Params<int8_t> params;
  params.zero_point = 0;
  params.scale = 0.123f;

  int8_t x_i8 = quantization::Quantize(x, params);
  EXPECT_EQ(x_i8, 127);
}

TEST(Quantization, QuantizeSingleFloatUInt8) {
  const float x = 1.25f;

  quantization::Params<uint8_t> params;
  params.zero_point = 85;
  params.scale = 0.0117647f;

  uint8_t x_u8 = quantization::Quantize(x, params);
  EXPECT_EQ(x_u8, 191);
}

}  // namespace test
}  // namespace onnxruntime
