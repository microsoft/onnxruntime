// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_FLOAT8_TYPES)

#include <vector>

#include "core/framework/float8.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "test_utils.h"
#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace test {

TEST(Float8_Tests, CastE4M3FN) {
  std::vector<std::pair<float, float>> cases{
      std::pair<float, float>(0.00439453125, 0.00390625),
      std::pair<float, float>(0.005859375, 0.005859375),
      std::pair<float, float>(0.005759375, 0.005859375),
      std::pair<float, float>(0.0046875, 0.00390625),
      std::pair<float, float>(0.001953125, 0.001953125),
      std::pair<float, float>(0.0029296875, 0.00390625),
      std::pair<float, float>(0.002053125, 0.001953125),
      std::pair<float, float>(0.00234375, 0.001953125),
      std::pair<float, float>(0.0087890625, 0.0078125),
      std::pair<float, float>(0.001171875, 0.001953125),
      std::pair<float, float>(1.8131605, 1.875)};
  for (auto it : cases) {
    auto f8 = onnxruntime::Float8E4M3FN(it.first);
    auto f8_32 = f8.ToFloat();
    EXPECT_EQ(it.second, f8_32);
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // DISABLE_FLOAT8_TYPES
