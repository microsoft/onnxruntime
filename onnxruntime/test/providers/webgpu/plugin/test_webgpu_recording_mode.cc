// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>

#include <gtest/gtest.h>

#include "core/providers/webgpu/ep/sync_stream.h"

namespace onnxruntime {
namespace test {

TEST(WebGpuPluginRecordingModeTest, VersionBoundariesAndSingleThreadOverride) {
  struct TestCase {
    uint32_t api_version;
    uint32_t patch_version;
    bool single_thread;
  };
  constexpr TestCase cases[] = {
      {24, 4, true},
      {24, 10, true},
      {25, 0, true},
      {26, 0, true},
      {27, 10, true},
      {28, 0, true},
      {28, 2, true},
      {28, 3, false},
      {28, 4, false},
      {28, 10, false},
      {29, 0, true},
      {29, 10, true},
      {30, 0, true},
      {30, 1, false},
      {30, 2, false},
      {30, 10, false},
      {31, 0, false},
      {31, 1, false},
      {32, 0, false},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "1." << test_case.api_version << "." << test_case.patch_version);
    EXPECT_EQ(webgpu::ep::ShouldUseSingleThreadMode(test_case.api_version, test_case.patch_version, false),
              test_case.single_thread);
    EXPECT_TRUE(webgpu::ep::ShouldUseSingleThreadMode(test_case.api_version, test_case.patch_version, true));
  }
}

}  // namespace test
}  // namespace onnxruntime
