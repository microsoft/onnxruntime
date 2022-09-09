// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env.h"

#include <fstream>

#include "gtest/gtest.h"

extern "C" char* __cxa_demangle(const char* mangled_name, char* buf, size_t* n, int* status);

namespace onnxruntime {
namespace test {

const char input[] = "_ZNSt6__ndk115basic_streambufIcNS_11char_traitsIcEEE5imbueERKNS_6localeE";

TEST(DummyCxaDemangleTest, InvalidArgs) {
  int status = 1000;
  ASSERT_EQ(__cxa_demangle(nullptr, nullptr, nullptr, &status), nullptr);
  ASSERT_EQ(status, -3);
}

TEST(DummyCxaDemangleTest, Alloc) {
  int status = 1000;
  char* output_buffer = __cxa_demangle(input, nullptr, nullptr, &status);
  ASSERT_EQ(status, 0);
  ASSERT_STREQ(output_buffer, input);
  std::free(output_buffer);

  // verify status can be omited
  char* output_buffer2 = __cxa_demangle(input, nullptr, nullptr, nullptr);
  ASSERT_STREQ(output_buffer2, input);
  std::free(output_buffer2);
}

TEST(DummyCxaDemangleTest, StackAllocatedBufferIsTooSmallButNoRealloc) {
  int status = 1000;
  const char* input = "0123456789";
  char buf[8];
  size_t buf_size = 8;
  char* output_buffer = __cxa_demangle(input, buf, &buf_size, &status);
  ASSERT_EQ(status, 0);
  ASSERT_EQ(output_buffer, buf);
  ASSERT_STREQ(output_buffer, "0123456");
  ASSERT_DEATH(std::free(output_buffer), ".*");
}

TEST(DummyCxaDemangleTest, ExistingBufferLargeEnough) {
  int status = -1;
  const char* input = "0123456789";
  char buf[50];
  size_t buf_size = 50;
  char* output_buffer = __cxa_demangle(input, buf, &buf_size, &status);
  ASSERT_EQ(status, 0);
  ASSERT_EQ(buf_size, 50);
  ASSERT_EQ(output_buffer, buf);
  ASSERT_STREQ(output_buffer, "0123456789");
  ASSERT_DEATH(std::free(output_buffer), ".*");
}

}  // namespace test
}  // namespace onnxruntime
