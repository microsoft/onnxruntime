// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(OptionalOpTest, OptionalHasElement) {
  OpTester test("OptionalHasElement", 1, onnxruntime::kMSDomain);

  test.AddInput<float>("A", {4, 2},
                       {-1.0856307f, 0.99734545f,
                        0.2829785f, -1.5062947f,
                        -0.5786002f, 1.6514366f,
                        -2.4266791f, -0.42891264f},
                       false,
                       nullptr,
                       /*is_optional_tensor*/ true);

  test.AddOutput<bool>("y", {}, {true});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
