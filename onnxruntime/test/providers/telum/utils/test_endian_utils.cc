// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/providers/telum/utils/endian_utils.h"
#include "../test_utils.h"

namespace onnxruntime {
namespace test {
namespace telum {

using namespace onnxruntime::telum;

/**
 * @brief Test endianness detection
 */
TEST(TelumEndianUtilsTest, EndiannessDetection) {
  // Verify endianness detection works
  ASSERT_TRUE(VerifyEndianness());

  // Log detected endianness
  std::cout << "Detected endianness: " << GetEndiannessString() << std::endl;

  // On s390x, should be big-endian
#if defined(__s390x__) || defined(__s390__)
  EXPECT_TRUE(IsBigEndian());
#endif
}

/**
 * @brief Test byte swap operations
 */
TEST(TelumEndianUtilsTest, ByteSwap16) {
  uint16_t value = 0x1234;
  uint16_t swapped = ByteSwap16(value);
  EXPECT_EQ(swapped, 0x3412);

  // Double swap should return original
  EXPECT_EQ(ByteSwap16(swapped), value);
}

TEST(TelumEndianUtilsTest, ByteSwap32) {
  uint32_t value = 0x12345678;
  uint32_t swapped = ByteSwap32(value);
  EXPECT_EQ(swapped, 0x78563412);

  // Double swap should return original
  EXPECT_EQ(ByteSwap32(swapped), value);
}

TEST(TelumEndianUtilsTest, ByteSwap64) {
  uint64_t value = 0x123456789ABCDEF0ULL;
  uint64_t swapped = ByteSwap64(value);
  EXPECT_EQ(swapped, 0xF0DEBC9A78563412ULL);

  // Double swap should return original
  EXPECT_EQ(ByteSwap64(swapped), value);
}

/**
 * @brief Test host to network conversion
 */
TEST(TelumEndianUtilsTest, HostToNetwork) {
  // Test with different sizes
  uint8_t val8 = 0x12;
  EXPECT_EQ(HostToNetwork(val8), val8);  // No change for single byte

  uint16_t val16 = 0x1234;
  uint16_t net16 = HostToNetwork(val16);

  uint32_t val32 = 0x12345678;
  uint32_t net32 = HostToNetwork(val32);

  uint64_t val64 = 0x123456789ABCDEF0ULL;
  uint64_t net64 = HostToNetwork(val64);

  // On big-endian (s390x), values should be unchanged
  // On little-endian, values should be swapped
#if defined(__s390x__) || defined(__s390__)
  EXPECT_EQ(net16, val16);
  EXPECT_EQ(net32, val32);
  EXPECT_EQ(net64, val64);
#else
  EXPECT_EQ(net16, ByteSwap16(val16));
  EXPECT_EQ(net32, ByteSwap32(val32));
  EXPECT_EQ(net64, ByteSwap64(val64));
#endif
}

/**
 * @brief Test network to host conversion
 */
TEST(TelumEndianUtilsTest, NetworkToHost) {
  uint32_t net_value = 0x12345678;
  uint32_t host_value = NetworkToHost(net_value);

  // Converting back should give original
  EXPECT_EQ(HostToNetwork(host_value), net_value);
}

/**
 * @brief Test float conversion (important for tensor data)
 */
TEST(TelumEndianUtilsTest, FloatConversion) {
  float original = 3.14159f;

  // Convert to network byte order and back
  float network = HostToNetwork(original);
  float restored = NetworkToHost(network);

  // Should get back the same value
  EXPECT_FLOAT_EQ(restored, original);
}

/**
 * @brief Test with actual tensor-like data
 */
TEST(TelumEndianUtilsTest, TensorDataConversion) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> network_data(data.size());
  std::vector<float> restored_data(data.size());

  // Convert to network byte order
  for (size_t i = 0; i < data.size(); ++i) {
    network_data[i] = HostToNetwork(data[i]);
  }

  // Convert back to host byte order
  for (size_t i = 0; i < network_data.size(); ++i) {
    restored_data[i] = NetworkToHost(network_data[i]);
  }

  // Verify data integrity
  EXPECT_TRUE(VectorsAlmostEqual(data, restored_data));
}

/**
 * @brief Test endianness with known patterns
 */
TEST(TelumEndianUtilsTest, KnownPatterns) {
  // Test with pattern that's easy to verify
  union {
    uint32_t i;
    uint8_t bytes[4];
  } test;

  test.i = 0x01020304;

  // On big-endian: bytes = {01, 02, 03, 04}
  // On little-endian: bytes = {04, 03, 02, 01}

  if (IsBigEndian()) {
    EXPECT_EQ(test.bytes[0], 0x01);
    EXPECT_EQ(test.bytes[1], 0x02);
    EXPECT_EQ(test.bytes[2], 0x03);
    EXPECT_EQ(test.bytes[3], 0x04);
  } else {
    EXPECT_EQ(test.bytes[0], 0x04);
    EXPECT_EQ(test.bytes[1], 0x03);
    EXPECT_EQ(test.bytes[2], 0x02);
    EXPECT_EQ(test.bytes[3], 0x01);
  }
}

}  // namespace telum
}  // namespace test
}  // namespace onnxruntime

// Made with Bob
