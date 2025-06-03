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
  constexpr uint16_t test_value = 0x1234;
  const unsigned char* test_value_first_byte = reinterpret_cast<const unsigned char*>(&test_value);
  if constexpr (endian::native == endian::little) {
    EXPECT_EQ(*test_value_first_byte, 0x34);
  } else if constexpr (endian::native == endian::big) {
    EXPECT_EQ(*test_value_first_byte, 0x12);
  }
}

TEST(EndianTest, SwapByteOrderCopy) {
  const auto src = std::vector<unsigned char>{
      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'};

  auto result = std::vector<unsigned char>(src.size());
  {
    SwapByteOrderCopy(3, gsl::make_span(src), gsl::make_span(result));
    const auto expected = std::vector<unsigned char>{
        'c', 'b', 'a',
        'f', 'e', 'd',
        'i', 'h', 'g',
        'l', 'k', 'j'};
    EXPECT_EQ(result, expected);
  }

  {
    SwapByteOrderCopy(4, gsl::make_span(src), gsl::make_span(result));
    const auto expected = std::vector<unsigned char>{
        'd', 'c', 'b', 'a',
        'h', 'g', 'f', 'e',
        'l', 'k', 'j', 'i'};
    EXPECT_EQ(result, expected);
  }
}

// Test fixture for SwapByteOrderInplace tests
class SwapByteOrderInplaceTest : public ::testing::Test {};

TEST_F(SwapByteOrderInplaceTest, ElementSize1) {
  std::vector<unsigned char> data = {0x01, 0x02, 0x03, 0x04};
  std::vector<unsigned char> expected_data = {0x01, 0x02, 0x03, 0x04};
  gsl::span<unsigned char> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(1, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, ElementSize2_SingleElement) {
  std::vector<unsigned char> data = {0x01, 0x02};
  std::vector<unsigned char> expected_data = {0x02, 0x01};
  gsl::span<unsigned char> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(2, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, ElementSize2_MultipleElements) {
  std::vector<unsigned char> data = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06};
  std::vector<unsigned char> expected_data = {0x02, 0x01, 0x04, 0x03, 0x06, 0x05};
  gsl::span<unsigned char> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(2, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, ElementSize4_SingleElement) {
  std::vector<unsigned char> data = {0x01, 0x02, 0x03, 0x04};
  std::vector<unsigned char> expected_data = {0x04, 0x03, 0x02, 0x01};
  gsl::span<unsigned char> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(4, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, ElementSize4_MultipleElements) {
  std::vector<unsigned char> data = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
  std::vector<unsigned char> expected_data = {0x04, 0x03, 0x02, 0x01, 0x08, 0x07, 0x06, 0x05};
  gsl::span<unsigned char> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(4, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, ElementSize8_SingleElement) {
  std::vector<unsigned char> data = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
  std::vector<unsigned char> expected_data = {0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01};
  gsl::span<unsigned char> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(8, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, ElementSize8_MultipleElements) {
  std::vector<unsigned char> data = {
      0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
      0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18};
  std::vector<unsigned char> expected_data = {
      0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
      0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11};
  gsl::span<unsigned char> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(8, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, EmptyBuffer) {
  std::vector<unsigned char> data = {};
  std::vector<unsigned char> expected_data = {};
  gsl::span<unsigned char> data_span = gsl::make_span(data);

  // Should not crash or throw for valid element sizes, e.g., 2 or 4
  // The ORT_ENFORCE checks will pass as 0 % element_size == 0
  // The loop for swapping will not execute.
  utils::SwapByteOrderInplace(2, data_span);
  EXPECT_EQ(data, expected_data);

  utils::SwapByteOrderInplace(4, data_span);
  EXPECT_EQ(data, expected_data);
}

TEST_F(SwapByteOrderInplaceTest, ElementSize3_OddElementSize) {
  std::vector<unsigned char> data = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06};
  std::vector<unsigned char> expected_data = {0x03, 0x02, 0x01, 0x06, 0x05, 0x04};
  gsl::span<unsigned char> data_span = gsl::make_span(data);

  utils::SwapByteOrderInplace(3, data_span);
  EXPECT_EQ(data, expected_data);
}

}  // namespace test
}  // namespace utils
}  // namespace onnxruntime
