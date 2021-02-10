// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/endian.h"
#include "core/framework/endian_utils.h"

#include <vector>

#include "gtest/gtest.h"

namespace onnxruntime {
namespace utils {
namespace test {

TEST(EndianTest, EndiannessDetection) {
  const uint16_t test_value = 0x1234;
  const uint8_t* test_value_first_byte = reinterpret_cast<const uint8_t*>(&test_value);
  if (endian::native == endian::little) {
    EXPECT_EQ(*test_value_first_byte, 0x34);
  } else if (endian::native == endian::big) {
    EXPECT_EQ(*test_value_first_byte, 0x12);
  }
}

TEST(EndianTest, SwapByteOrderCopy) {
  const auto src = std::vector<uint8_t>{
      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'};

  auto result = std::vector<uint8_t>(src.size());
  {
    SwapByteOrderCopy(3, gsl::make_span(src), gsl::make_span(result));
    const auto expected = std::vector<uint8_t>{
        'c', 'b', 'a',
        'f', 'e', 'd',
        'i', 'h', 'g',
        'l', 'k', 'j'};
    EXPECT_EQ(result, expected);
  }

  {
    SwapByteOrderCopy(4, gsl::make_span(src), gsl::make_span(result));
    const auto expected = std::vector<uint8_t>{
        'd', 'c', 'b', 'a',
        'h', 'g', 'f', 'e',
        'l', 'k', 'j', 'i'};
    EXPECT_EQ(result, expected);
  }
}

}  // namespace test
}  // namespace utils
}  // namespace onnxruntime
