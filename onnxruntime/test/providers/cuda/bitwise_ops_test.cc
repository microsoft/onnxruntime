// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

template <typename T>
void RunCudaBitwise(const char* op, int opset,
                    const std::vector<int64_t>& lhs_shape, const std::vector<T>& lhs,
                    const std::vector<int64_t>& rhs_shape, const std::vector<T>& rhs,
                    const std::vector<int64_t>& output_shape, const std::vector<T>& expected) {
  SCOPED_TRACE(std::string(op) + " opset=" + std::to_string(opset));
  OpTester test(op, opset);
  const bool unary = std::string(op) == "BitwiseNot" || std::string(op) == "Not";
  test.AddInput<T>(unary ? "X" : "A", lhs_shape, lhs);
  if (!unary) {
    test.AddInput<T>("B", rhs_shape, rhs);
  }
  test.AddOutput<T>(unary ? "Y" : "C", output_shape, expected);
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(options).ConfigEp(DefaultCudaExecutionProvider()).RunWithConfig();
}

template <typename T>
class CudaBitwiseOpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!DefaultCudaExecutionProvider()) {
      GTEST_SKIP() << "CUDA execution provider is unavailable";
    }
  }
};

using IntegerTypes = ::testing::Types<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>;
TYPED_TEST_SUITE(CudaBitwiseOpTest, IntegerTypes);

TYPED_TEST(CudaBitwiseOpTest, IntegerExtremaBroadcastAndTails) {
  using T = TypeParam;
  const std::vector<T> values{
      T{0}, T{1}, std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest(),
      static_cast<T>(-1), static_cast<T>(UINT64_C(0x5555555555555555) & std::numeric_limits<T>::max()),
      static_cast<T>(UINT64_C(0x2aaaaaaaaaaaaaaa) & std::numeric_limits<T>::max())};

  for (int opset : {18, 24}) {
    for (const char* op : {"BitwiseAnd", "BitwiseOr", "BitwiseNot"}) {
      const auto apply = [op](T a, T b) -> T {
        if (std::string(op) == "BitwiseAnd") {
          return static_cast<T>(a & b);
        }
        if (std::string(op) == "BitwiseOr") {
          return static_cast<T>(a | b);
        }
        return static_cast<T>(~a);
      };

      for (int64_t size : {0, 1, 7, 255, 256, 257, 1025}) {
        SCOPED_TRACE("size=" + std::to_string(size));
        std::vector<T> lhs, rhs, expected;
        lhs.reserve(static_cast<size_t>(size));
        rhs.reserve(static_cast<size_t>(size));
        expected.reserve(static_cast<size_t>(size));
        for (int64_t i = 0; i < size; ++i) {
          lhs.push_back(values[static_cast<size_t>(i) % values.size()]);
          rhs.push_back(values[(static_cast<size_t>(i) / values.size() + 2) % values.size()]);
          expected.push_back(apply(lhs.back(), rhs.back()));
        }
        RunCudaBitwise(op, opset, {size}, lhs, {size}, rhs, {size}, expected);
      }

      const T scalar = std::numeric_limits<T>::max();
      std::vector<T> expected;
      expected.reserve(values.size());
      for (T value : values) {
        expected.push_back(apply(value, scalar));
      }
      RunCudaBitwise(op, opset, {7}, values, {}, {scalar}, {7}, expected);
      RunCudaBitwise(op, opset, {}, {scalar}, {}, {scalar}, {}, {apply(scalar, scalar)});

      if (std::string(op) != "BitwiseNot") {
        RunCudaBitwise(op, opset, {}, {scalar}, {7}, values, {7}, expected);
        RunCudaBitwise<T>(op, opset, {2, 0, 3}, {}, {1, 3}, {T{0}, T{1}, scalar}, {2, 0, 3}, {});
        const std::vector<T> lhs(values.begin(), values.begin() + 6);
        const std::vector<T> rhs{values[2], values[3], values[4], values[5]};
        expected.clear();
        for (size_t batch = 0; batch < 2; ++batch) {
          for (T b : rhs) {
            for (size_t col = 0; col < 3; ++col) {
              expected.push_back(apply(lhs[batch * 3 + col], b));
            }
          }
        }
        RunCudaBitwise(op, opset, {2, 1, 3}, lhs, {1, 4, 1}, rhs, {2, 4, 3}, expected);
        RunCudaBitwise(op, opset, {1, 4, 1}, rhs, {2, 1, 3}, lhs, {2, 4, 3}, expected);
      }
    }
  }
}

TEST(CudaBitwiseLogicalTest, BooleanOpsUnchanged) {
  if (!DefaultCudaExecutionProvider()) {
    GTEST_SKIP() << "CUDA execution provider is unavailable";
  }
  const std::vector<bool> lhs{false, false, true, true};
  const std::vector<bool> rhs{false, true, false, true};
  RunCudaBitwise<bool>("And", 7, {4}, lhs, {4}, rhs, {4}, {false, false, false, true});
  RunCudaBitwise<bool>("Or", 7, {4}, lhs, {4}, rhs, {4}, {false, true, true, true});
  RunCudaBitwise<bool>("Not", 1, {4}, lhs, {}, {}, {4}, {true, true, false, false});
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
