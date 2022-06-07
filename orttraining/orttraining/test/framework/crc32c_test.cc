// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "orttraining/core/framework/tensorboard/crc32c.h"

using namespace onnxruntime::training::tensorboard;

namespace onnxruntime {
namespace test {

// RFC 3720 section B.4. CRC Examples
TEST(Crc32cTest, ChecksumTests) {
  char data[32];

  memset(data, 0x00, sizeof(data));
  ASSERT_EQ(static_cast<uint32_t>(0x8a9136aa), Crc32c(data, sizeof(data)));

  memset(data, 0xff, sizeof(data));
  ASSERT_EQ(static_cast<uint32_t>(0x62a8ab43), Crc32c(data, sizeof(data)));

  for (size_t i = 0; i < sizeof(data); i++) {
    data[i] = static_cast<char>(i);
  }
  ASSERT_EQ(static_cast<uint32_t>(0x46dd794e), Crc32c(data, sizeof(data)));

  for (size_t i = 0; i < sizeof(data); i++) {
    data[i] = 31 - static_cast<char>(i);
  }
  ASSERT_EQ(static_cast<uint32_t>(0x113fdb5c), Crc32c(data, sizeof(data)));

  uint8_t iscsi[] = {
    0x01, 0xc0, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0x14, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x04, 0x00,
    0x00, 0x00, 0x00, 0x14,
    0x00, 0x00, 0x00, 0x18,
    0x28, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
  };
  ASSERT_EQ(static_cast<uint32_t>(0xd9963a56), Crc32c(reinterpret_cast<const char*>(iscsi), sizeof(iscsi)));
}

}  // namespace test
}  // namespace onnxruntime
