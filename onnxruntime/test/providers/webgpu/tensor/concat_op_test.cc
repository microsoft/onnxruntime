// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef USE_WEBGPU
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#endif

namespace onnxruntime {
namespace test {

#ifdef USE_WEBGPU
TEST(ConcatOpTest, Concat1D_int32_4inputs) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  test.AddInput<int32_t>("input1", {1}, {1});
  test.AddInput<int32_t>("input2", {2}, {2, 3});
  test.AddInput<int32_t>("input3", {4}, {4, 5, 6, 7});
  test.AddInput<int32_t>("input4", {2}, {8, 9});
  test.AddOutput<int32_t>("concat_result", {9}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.Run();
}

TEST(ConcatOpTest, Concat1D_exceed_maxStorageBuffersPerShaderStage) {
  // maxStorageBuffersPerShaderStage==8
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  test.AddInput<int32_t>("input1", {1}, {1});
  test.AddInput<int32_t>("input2", {1}, {2});
  test.AddInput<int32_t>("input3", {1}, {3});
  test.AddInput<int32_t>("input4", {1}, {4});
  test.AddInput<int32_t>("input5", {1}, {5});
  test.AddInput<int32_t>("input6", {1}, {6});
  test.AddInput<int32_t>("input7", {1}, {7});
  test.AddInput<int32_t>("input8", {1}, {8});
  test.AddInput<int32_t>("input9", {1}, {9});
  test.AddOutput<int32_t>("concat_result", {9}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.Run();
}

TEST(ConcatOpTest, Concat2D_exceed_maxStorageBuffersPerShaderStage_axis0) {
  // maxStorageBuffersPerShaderStage==8
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  test.AddInput<int32_t>("input1", {1, 2}, {1, 2});
  test.AddInput<int32_t>("input2", {1, 2}, {3, 4});
  test.AddInput<int32_t>("input3", {1, 2}, {5, 6});
  test.AddInput<int32_t>("input4", {1, 2}, {7, 8});
  test.AddInput<int32_t>("input5", {1, 2}, {9, 10});
  test.AddInput<int32_t>("input6", {1, 2}, {11, 12});
  test.AddInput<int32_t>("input7", {1, 2}, {13, 14});
  test.AddInput<int32_t>("input8", {1, 2}, {15, 16});
  test.AddInput<int32_t>("input9", {1, 2}, {17, 18});
  test.AddOutput<int32_t>("concat_result", {9, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  test.Run();
}

TEST(ConcatOpTest, Concat2D_exceed_maxStorageBuffersPerShaderStage_axis1) {
  // maxStorageBuffersPerShaderStage==8
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  test.AddInput<int32_t>("input1", {1, 2}, {1, 2});
  test.AddInput<int32_t>("input2", {1, 2}, {3, 4});
  test.AddInput<int32_t>("input3", {1, 2}, {5, 6});
  test.AddInput<int32_t>("input4", {1, 2}, {7, 8});
  test.AddInput<int32_t>("input5", {1, 2}, {9, 10});
  test.AddInput<int32_t>("input6", {1, 2}, {11, 12});
  test.AddInput<int32_t>("input7", {1, 2}, {13, 14});
  test.AddInput<int32_t>("input8", {1, 2}, {15, 16});
  test.AddInput<int32_t>("input9", {1, 2}, {17, 18});
  test.AddOutput<int32_t>("concat_result", {1, 18}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_exceed_maxStorageBuffersPerShaderStage) {
  // maxStorageBuffersPerShaderStage==8
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  test.AddInput<int32_t>("input1", {2, 1, 1}, {1, 2});
  test.AddInput<int32_t>("input2", {2, 1, 1}, {3, 4});
  test.AddInput<int32_t>("input3", {2, 1, 1}, {5, 6});
  test.AddInput<int32_t>("input4", {2, 1, 1}, {7, 8});
  test.AddInput<int32_t>("input5", {2, 1, 1}, {9, 10});
  test.AddInput<int32_t>("input6", {2, 1, 1}, {11, 12});
  test.AddInput<int32_t>("input7", {2, 1, 1}, {13, 14});
  test.AddInput<int32_t>("input8", {2, 1, 1}, {15, 16});
  test.AddInput<int32_t>("input9", {2, 1, 1}, {17, 18});
  test.AddOutput<int32_t>("concat_result", {2, 9, 1}, {// batch 0
                                                       1, 3, 5, 7, 9, 11, 13, 15, 17,
                                                       // batch 1
                                                       2, 4, 6, 8, 10, 12, 14, 16, 18});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_exceed_maxStorageBuffersPerShaderStage_mixed_sizes) {
  // maxStorageBuffersPerShaderStage==8
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  test.AddInput<int32_t>("input1", {2, 1, 1}, {1, 2});
  test.AddInput<int32_t>("input2", {2, 3, 1}, {3, 4, 5, 6, 7, 8});
  test.AddInput<int32_t>("input3", {2, 2, 1}, {9, 10, 11, 12});
  test.AddInput<int32_t>("input4", {2, 1, 1}, {13, 14});
  test.AddOutput<int32_t>("concat_result", {2, 7, 1}, {// batch 0
                                                       1, 3, 4, 5, 9, 10, 13,
                                                       // batch 1
                                                       2, 6, 7, 8, 11, 12, 14});
  test.Run();
}

TEST(ConcatOpTest, Concat_int64_webgpu) {
  // int64 support on the WebGPU EP is gated behind the enableInt64 provider option.
  ConfigOptions provider_options{};
  ASSERT_STATUS_OK(provider_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(provider_options);
  if (provider == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }

  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});
  // Include values outside the int32 range (2^32, 2^33+1, a large negative, INT64_MAX) to prove
  // the full 64-bit value is preserved via a raw vec2<u32> storage copy rather than truncated to i32.
  test.AddInput<int64_t>("input1", {1, 2}, {1, 4294967296});                                        // 1, 2^32
  test.AddInput<int64_t>("input2", {2, 2}, {8589934593, -4294967297, 100, 9223372036854775807LL});  // 2^33+1, -(2^32+1), 100, INT64_MAX
  test.AddOutput<int64_t>("concat_result", {3, 2}, {1, 4294967296, 8589934593, -4294967297, 100, 9223372036854775807LL});

  // Disable CPU-EP fallback so the Concat node must run on the WebGPU kernel.
  SessionOptions so;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(so)
      .ConfigEp(std::move(provider))
      .RunWithConfig();
}
#endif  // USE_WEBGPU

}  // namespace test
}  // namespace onnxruntime
