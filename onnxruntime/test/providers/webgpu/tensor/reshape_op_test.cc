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
#include "core/framework/int4.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#endif

namespace onnxruntime {
namespace test {

#ifdef USE_WEBGPU
TEST(TensorOpTest, Reshape_int64_webgpu) {
  OpTester test("Reshape", 25);

  test.AddInput<int64_t>("data", {2, 3}, {1, 2, 3, 4, 5, 6});
  test.AddInput<int64_t>("shape", {3}, {3, 1, 2});
  test.AddOutput<int64_t>("reshaped", {3, 1, 2}, {1, 2, 3, 4, 5, 6});

  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}

TEST(TensorOpTest, Reshape_uint8_webgpu) {
  OpTester test("Reshape", 25);

  // Values span the full byte range.
  test.AddInput<uint8_t>("data", {2, 3}, {1, 2, 10, 100, 200, 255});
  test.AddInput<int64_t>("shape", {3}, {3, 1, 2});
  test.AddOutput<uint8_t>("reshaped", {3, 1, 2}, {1, 2, 10, 100, 200, 255});

  // Run on a WebGPU-only session and disable CPU-EP fallback. ORT implicitly adds a CPU EP for
  // fallback, so without this the Reshape node would silently run on CPU (and the test would still
  // pass) if WebGPU did not support uint8.
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(ConfigOptions{});
  if (provider == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }
  test.Config(options)
      .ConfigEp(std::move(provider))
      .RunWithConfig();
}

// Chromium's WebNN backend reshapes int4/uint4 (De)QuantizeLinear zero points, and the CPU Reshape kernel has
// no 4-bit types to constant fold them with. Odd element counts leave the last byte half used.
template <typename T>
void TestReshape4BitWebGpu(const std::vector<typename T::UnpackedType>& values, bool data_is_initializer) {
  auto pack = [](const std::vector<typename T::UnpackedType>& unpacked) {
    std::vector<T> packed(T::CalcNumInt4Pairs(unpacked.size()));
    for (size_t i = 0; i < unpacked.size(); ++i) {
      packed[i / 2].SetElem(i % 2, unpacked[i]);
    }
    return packed;
  };
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  {
    auto provider = WebGpuExecutionProviderWithOptions(ConfigOptions{});
    if (provider == nullptr) {
      GTEST_SKIP() << "WebGPU EP is not available";
    }
    OpTester test("Reshape", 21);
    const int64_t size = static_cast<int64_t>(values.size());
    test.AddInput<T>("data", {1, size}, pack(values), data_is_initializer);
    test.AddInput<int64_t>("shape", {1}, {size}, true);
    test.AddOutput<T>("reshaped", {size}, pack(values));
    test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
  }
  {
    // To a scalar, as for a per-tensor zero point.
    OpTester test("Reshape", 21);
    test.AddInput<T>("data", {1}, pack({values[0]}), data_is_initializer);
    test.AddInput<int64_t>("shape", {0}, {}, true);
    test.AddOutput<T>("reshaped", {}, pack({values[0]}));
    test.Config(options).ConfigEp(WebGpuExecutionProviderWithOptions(ConfigOptions{})).RunWithConfig();
  }
}

TEST(TensorOpTest, Reshape_int4_webgpu) {
  for (bool data_is_initializer : {false, true}) {
    SCOPED_TRACE(data_is_initializer);
    TestReshape4BitWebGpu<Int4x2>({-3, 7, -8, 0, 5}, data_is_initializer);
    TestReshape4BitWebGpu<UInt4x2>({13, 0, 15, 8, 2}, data_is_initializer);
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime
