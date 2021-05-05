// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/utf8_util.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

struct Sample {
  const char* sequence;
  bool valid;
};

const std::vector<Sample> samples = {
    {"a", true},
    {"\xc3\xb1", true},
    {"\xc3\x28", false},
    {"\xa0\xa1", false},
    {"\xe2\x82\xa1", true},
    {"\xe2\x28\xa1", false},
    {"\xe2\x82\x28", false},
    {"\xf0\x90\x8c\xbc", true},
    {"\xf0\x28\x8c\xbc", false},
    {"\xf0\x90\x28\xbc", false},
    {"\xf0\x28\x8c\x28", false},
    {"\xf8\xa1\xa1\xa1\xa1", false},       // valid but not Unicode
    {"\xfc\xa1\xa1\xa1\xa1\xa1", false}};  // valid but not Unicode

TEST(Utf8UtilTest, Validate) {
  using namespace utf8_util;
  for (auto& s : samples) {
    size_t utf8_len = 0;
    if (s.valid != utf8_validate(reinterpret_cast<const unsigned char*>(s.sequence), strnlen(s.sequence, 65535), utf8_len)) {
      ASSERT_TRUE(false);
    } else {
      if (s.valid) {
        ASSERT_EQ(1U, utf8_len);
      }
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
