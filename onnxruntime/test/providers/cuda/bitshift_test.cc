// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

template <typename T>
void RunBitShift(int opset, const char* direction,
                 const std::vector<int64_t>& x_shape, const std::vector<T>& x,
                 const std::vector<int64_t>& y_shape, const std::vector<T>& y,
                 const std::vector<int64_t>& z_shape, const std::vector<T>& z,
                 bool count_is_initializer = false) {
  SCOPED_TRACE(std::string(direction) + " opset " + std::to_string(opset));
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA execution provider is unavailable.";
  }
  OpTester test("BitShift", opset);
  test.AddAttribute("direction", std::string(direction));
  test.AddInput<T>("X", x_shape, x);
  test.AddInput<T>("Y", y_shape, y, count_is_initializer);
  test.AddOutput<T>("Z", z_shape, z);
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(options).ConfigEp(std::move(cuda_ep)).RunWithConfig();
}

// An independent oracle uses repeated multiply/divide, never a native shift.
template <typename T>
T ReferenceShift(T value, T count, bool left) {
  constexpr unsigned width = sizeof(T) * 8;
  if (static_cast<std::make_unsigned_t<T>>(count) >= width) {
    if constexpr (std::is_signed_v<T>) {
      if (!left && value < 0) {
        return T{-1};
      }
    }
    return T{0};
  }
  if (left) {
    auto bits = static_cast<std::make_unsigned_t<T>>(value);
    for (unsigned i = 0; i < static_cast<unsigned>(count); ++i) {
      bits = static_cast<std::make_unsigned_t<T>>(bits * 2);
    }
    T result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
  }
  for (unsigned i = 0; i < static_cast<unsigned>(count); ++i) {
    if constexpr (std::is_signed_v<T>) {
      const bool round_down = value < 0 && value % 2 != 0;
      value = static_cast<T>(value / 2 - (round_down ? 1 : 0));
    } else {
      value = static_cast<T>(value / 2);
    }
  }
  return value;
}

template <typename T>
std::vector<int> Opsets() {
  if constexpr (std::is_signed_v<T>) {
    return {28};
  }
  return {11, 27, 28};
}

template <typename T>
class CudaBitShiftTest : public ::testing::Test {};

using IntegerTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t,
                                      int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_SUITE(CudaBitShiftTest, IntegerTypes);

TYPED_TEST(CudaBitShiftTest, BoundariesAndMultidirectionalBroadcast) {
  using T = TypeParam;
  constexpr T width = sizeof(T) * 8;
  std::vector<T> values{0, 1, 3, std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()};
  std::vector<T> counts{0, 1, static_cast<T>(width - 1), width,
                        static_cast<T>(width + 1), std::numeric_limits<T>::max()};
  if constexpr (std::is_signed_v<T>) {
    values.insert(values.end(), {T{-1}, T{-3}, T{-8}});
    counts.insert(counts.end(), {T{-1}, std::numeric_limits<T>::lowest()});
  } else {
    values.push_back(static_cast<T>(std::numeric_limits<T>::max() / 2 + 1));
  }
  for (int opset : Opsets<T>()) {
    for (bool left : {false, true}) {
      std::vector<T> expected;
      for (T value : values) {
        for (T count : counts) {
          expected.push_back(ReferenceShift(value, count, left));
        }
      }
      RunBitShift(opset, left ? "LEFT" : "RIGHT",
                  {static_cast<int64_t>(values.size()), 1}, values,
                  {1, static_cast<int64_t>(counts.size())}, counts,
                  {static_cast<int64_t>(values.size()), static_cast<int64_t>(counts.size())}, expected);
    }
  }
}

TYPED_TEST(CudaBitShiftTest, ScalarsPerChannelAndTails) {
  using T = TypeParam;
  const std::vector<T> values{0, 1, 3, std::numeric_limits<T>::max(),
                              std::numeric_limits<T>::lowest(), 7};
  constexpr T width = sizeof(T) * 8;
  const std::vector<T> counts{0, 1, static_cast<T>(width - 1), width};
  for (int opset : Opsets<T>()) {
    for (bool left : {false, true}) {
      const char* direction = left ? "LEFT" : "RIGHT";
      std::vector<T> expected;
      for (T value : values) {
        expected.push_back(ReferenceShift(value, T{1}, left));
      }
      RunBitShift(opset, direction, {6}, values, {}, {1}, {6}, expected, true);
      expected.clear();
      for (T count : counts) {
        expected.push_back(ReferenceShift(std::numeric_limits<T>::max(), count, left));
      }
      RunBitShift(opset, direction, {}, {std::numeric_limits<T>::max()}, {4}, counts, {4}, expected);
      RunBitShift<T>(opset, direction, {}, {3}, {}, {1}, {}, {left ? T{6} : T{1}});

      expected.clear();
      for (size_t batch = 0; batch < 2; ++batch) {
        for (T count : counts) {
          for (size_t i = 0; i < 3; ++i) {
            expected.push_back(ReferenceShift(values[batch * 3 + i], count, left));
          }
        }
      }
      RunBitShift(opset, direction, {2, 1, 3}, values, {1, 4, 1}, counts, {2, 4, 3}, expected);

      for (int64_t batches : {int64_t{1}, int64_t{2}}) {
        std::vector<T> per_channel_values;
        expected.clear();
        for (int64_t batch = 0; batch < batches; ++batch) {
          per_channel_values.insert(per_channel_values.end(), values.begin(), values.end());
          for (size_t i = 0; i < values.size(); ++i) {
            expected.push_back(ReferenceShift(values[i], static_cast<T>(i / 3), left));
          }
        }
        RunBitShift(opset, direction, {batches, 2, 3}, per_channel_values,
                    {2, 1}, {0, 1}, {batches, 2, 3}, expected);
      }

      constexpr int64_t size = 2051;
      std::vector<T> tail_values;
      std::vector<T> tail_counts;
      expected.clear();
      for (int64_t i = 0; i < size; ++i) {
        const T value = values[static_cast<size_t>(i) % values.size()];
        const T count = counts[static_cast<size_t>(i) % counts.size()];
        tail_values.push_back(value);
        tail_counts.push_back(count);
        expected.push_back(ReferenceShift(value, count, left));
      }
      RunBitShift(opset, direction, {size}, tail_values, {size}, tail_counts, {size}, expected);
      RunBitShift<T>(opset, direction, {0}, {}, {}, {1}, {0}, {});
      RunBitShift<T>(opset, direction, {2, 0, 3}, {}, {1, 0, 1}, {}, {2, 0, 3}, {});
    }
  }
}

TEST(CudaBitShiftTest, Uint64HighBits) {
  const std::vector<uint64_t> values{0xfedcba9876543210ULL, 0x8000000000000001ULL,
                                     0xffffffffffffffffULL, 1, 0xffffffffffffffffULL};
  for (int opset : {11, 27, 28}) {
    RunBitShift<uint64_t>(opset, "RIGHT", {5}, values, {5}, {4, 63, 0, 64, 65}, {5},
                          {0x0fedcba987654321ULL, 1, 0xffffffffffffffffULL, 0, 0});
    RunBitShift<uint64_t>(opset, "LEFT", {5}, values, {5}, {4, 63, 0, 64, 65}, {5},
                          {0xedcba98765432100ULL, 0x8000000000000000ULL, 0xffffffffffffffffULL, 0, 0});
  }
}

TEST(CudaBitShiftTest, Int64HighBitsAndNegativeCounts) {
  constexpr int64_t min = std::numeric_limits<int64_t>::lowest();
  constexpr int64_t max = std::numeric_limits<int64_t>::max();
  RunBitShift<int64_t>(28, "RIGHT", {8}, {min, -3, max, -1, min, 7, min, -8},
                       {8}, {1, 1, 63, 0, 64, -1, min, -1}, {8},
                       {min / 2, -2, 0, -1, -1, 0, -1, -1});
  RunBitShift<int64_t>(28, "LEFT", {8}, {min, -3, max, -1, min, 7, min, 1},
                       {8}, {1, 1, 1, 63, 64, -1, min, 63}, {8},
                       {0, -6, -2, min, 0, 0, 0, min});
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
