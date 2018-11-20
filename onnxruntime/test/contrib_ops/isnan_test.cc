// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include <cmath> // NAN

namespace onnxruntime {
namespace test {

TEST(ContribOpTest, IsNaN) {
  OpTester test("IsNaN", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, NAN, 2.0f, NAN});
  test.AddOutput<bool>("Y", dims, {false, true, false, true});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
