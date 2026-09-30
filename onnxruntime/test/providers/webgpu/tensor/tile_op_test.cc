// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef USE_WEBGPU
#include "core/providers/webgpu/webgpu_provider_options.h"
#endif

namespace onnxruntime {
namespace test {

#if defined(USE_WEBGPU)
namespace {
template <typename T>
std::vector<T> InputData(size_t size) {
  std::vector<T> result(size);
  for (size_t i = 0; i < size; i++) {
    result[i] = static_cast<T>(i);
  }
  return result;
}
}  // namespace

// Runs a Tile test that must be assigned to the WebGPU kernel. The CPU EP is
// excluded so the test fails (rather than silently passing on a CPU fallback)
// if the WebGPU Tile kernel does not support the given element type. This
// exercises the WebGpuSupportedNumberTypes() constraint end-to-end for the
// newly added int32/uint32 support.
template <typename T>
void RunWebGpuTileTypeTest(const std::vector<int64_t>& input_dims,
                           const std::vector<int64_t>& repeats) {
  if (DefaultWebGpuExecutionProvider().get() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  size_t input_size =
      static_cast<size_t>(std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>()));
  std::vector<T> input_data = InputData<T>(input_size);
  size_t rank = input_dims.size();
  std::vector<int64_t> repeats_dims(1, static_cast<int64_t>(rank));
  std::vector<int64_t> output_dims(rank);
  for (size_t i = 0; i < rank; ++i) {
    output_dims[i] = input_dims[i] * repeats[i];
  }
  size_t output_size =
      static_cast<size_t>(std::accumulate(output_dims.begin(), output_dims.end(), 1LL, std::multiplies<int64_t>()));
  std::vector<T> output_data(output_size);
  std::vector<int64_t> input_strides(rank);
  std::vector<int64_t> output_strides(rank);
  if (rank >= 1) {
    input_strides[rank - 1] = output_strides[rank - 1] = 1;
    if (rank > 1) {
      for (size_t i = rank - 2;; --i) {
        input_strides[i] = input_dims[i + 1] * input_strides[i + 1];
        output_strides[i] = output_dims[i + 1] * output_strides[i + 1];
        if (i == 0) break;
      }
    }
  }
  for (size_t i = 0; i < output_size; ++i) {
    int64_t index = 0;
    int64_t remain = static_cast<int64_t>(i);
    for (size_t j = 0; j < rank; ++j) {
      index += (((remain / output_strides[j]) % input_dims[j]) * input_strides[j]);
      remain = remain % output_strides[j];
    }
    output_data[i] = input_data[static_cast<size_t>(index)];
  }

  OpTester test("Tile");
  test.AddInput<T>("input", input_dims, input_data);
  test.AddInput<int64_t>("repeats", repeats_dims, repeats);
  test.AddOutput<T>("output", output_dims, output_data);
  // Exclude the CPU EP so the node must run on the WebGPU Tile kernel.
  test.ConfigExcludeEps({kCpuExecutionProvider});
  test.RunWithConfig();
}

// int32 Tile must run on the WebGPU kernel (not fall back to CPU) and produce
// correct results across a range of shapes.
TEST(TensorOpTest, TileInt32TypeWebGpu) {
  RunWebGpuTileTypeTest<int32_t>({3}, {3});
  RunWebGpuTileTypeTest<int32_t>({2, 2}, {2, 1});
  RunWebGpuTileTypeTest<int32_t>({2, 3}, {2, 2});
  RunWebGpuTileTypeTest<int32_t>({2, 1, 3}, {1, 2, 1});
  RunWebGpuTileTypeTest<int32_t>({1, 2, 3, 4}, {2, 1, 2, 1});
}

// uint32 Tile must run on the WebGPU kernel (not fall back to CPU) and produce
// correct results across a range of shapes.
TEST(TensorOpTest, TileUint32TypeWebGpu) {
  RunWebGpuTileTypeTest<uint32_t>({3}, {3});
  RunWebGpuTileTypeTest<uint32_t>({2, 2}, {2, 1});
  RunWebGpuTileTypeTest<uint32_t>({2, 3}, {2, 2});
  RunWebGpuTileTypeTest<uint32_t>({2, 1, 3}, {1, 2, 1});
  RunWebGpuTileTypeTest<uint32_t>({1, 2, 3, 4}, {2, 1, 2, 1});
}

// The WebGPU Tile kernel stores per-axis repeat values in a uint32_t shader
// uniform. Repeat values that would truncate when cast to uint32_t must be
// rejected before reaching the shader. A zero-element input is used so that
// the earlier output-byte-size check (which requires dim > 0) does not fire
// first, exercising the explicit uint32_t-range guard.
TEST(TensorOpTest, TileRepeatExceedsUint32MaxWebGpu) {
  if (DefaultWebGpuExecutionProvider().get() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  OpTester test("Tile", 13);
  test.AddInput<float>("input", {0}, {});
  test.AddInput<int64_t>("repeats", {1}, {int64_t{4294967296}});  // 2^32
  test.AddOutput<float>("output", {0}, {});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "exceeds the WebGPU supported maximum",
           {}, nullptr, &execution_providers);
}

// The WebGPU Tile kernel validates that the 'repeats' input is 1-D, mirroring
// the CPU kernel's pre-existing check.
TEST(TensorOpTest, TileRepeatsMustBe1DWebGpu) {
  if (DefaultWebGpuExecutionProvider().get() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  OpTester test("Tile", 13);
  test.AddInput<float>("input", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<int64_t>("repeats", {1, 2}, {1, 1});
  test.AddOutput<float>("output", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "must be 1 dimensional",
           {}, nullptr, &execution_providers);
}

// The WebGPU Tile kernel validates that the 'repeats' length matches the
// input rank, mirroring the CPU kernel's pre-existing check.
TEST(TensorOpTest, TileRepeatsMustMatchInputRankWebGpu) {
  if (DefaultWebGpuExecutionProvider().get() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  OpTester test("Tile", 13);
  test.AddInput<float>("input", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<int64_t>("repeats", {1}, {1});
  test.AddOutput<float>("output", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "same length as the 'input' tensor",
           {}, nullptr, &execution_providers);
}

// int64 Tile must run on the WebGPU kernel (not fall back to CPU) when int64 is
// enabled via the session config option. Values with high-32-bit bits set are
// used to verify that the full 64-bit element is copied without truncation.
TEST(TensorOpTest, TileInt64TypeWebGpu) {
  if (DefaultWebGpuExecutionProvider().get() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  // Values chosen so the high 32 bits are non-zero, ensuring a correct raw
  // 64-bit copy (not a sign-extended i32 copy) is verified.
  const int64_t v0 = int64_t{0x0001000200030004LL};
  const int64_t v1 = int64_t{0x0005000600070008LL};
  const int64_t v2 = int64_t{-1LL};  // 0xFFFFFFFFFFFFFFFF

  // 1-D: input [v0, v1, v2], repeat 2 -> [v0, v1, v2, v0, v1, v2]
  {
    OpTester test("Tile", 13);
    test.AddInput<int64_t>("input", {3}, {v0, v1, v2});
    test.AddInput<int64_t>("repeats", {1}, {2});
    test.AddOutput<int64_t>("output", {6}, {v0, v1, v2, v0, v1, v2});
    ConfigOptions config_options{};
    ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
    auto provider = WebGpuExecutionProviderWithOptions(config_options);
    test.ConfigEp(std::move(provider))
        .ConfigExcludeEps({kCpuExecutionProvider})
        .RunWithConfig();
  }

  // 2-D: input [[v0, v1], [v2, v0]], repeat [2, 1]
  {
    OpTester test("Tile", 13);
    test.AddInput<int64_t>("input", {2, 2}, {v0, v1, v2, v0});
    test.AddInput<int64_t>("repeats", {2}, {2, 1});
    test.AddOutput<int64_t>("output", {4, 2}, {v0, v1, v2, v0, v0, v1, v2, v0});
    ConfigOptions config_options{};
    ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
    auto provider = WebGpuExecutionProviderWithOptions(config_options);
    test.ConfigEp(std::move(provider))
        .ConfigExcludeEps({kCpuExecutionProvider})
        .RunWithConfig();
  }

  // 3-D: input [[[v0, v1, v2]]], repeat [1, 2, 1]
  {
    OpTester test("Tile", 13);
    test.AddInput<int64_t>("input", {1, 1, 3}, {v0, v1, v2});
    test.AddInput<int64_t>("repeats", {3}, {1, 2, 1});
    test.AddOutput<int64_t>("output", {1, 2, 3}, {v0, v1, v2, v0, v1, v2});
    ConfigOptions config_options{};
    ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
    auto provider = WebGpuExecutionProviderWithOptions(config_options);
    test.ConfigEp(std::move(provider))
        .ConfigExcludeEps({kCpuExecutionProvider})
        .RunWithConfig();
  }
}

#endif  // defined(USE_WEBGPU)

}  // namespace test
}  // namespace onnxruntime
