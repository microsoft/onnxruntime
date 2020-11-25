// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(CropOpTest, Crop_Border) {
  OpTester test("Crop", 1, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {1, 1, 4, 4}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
  std::vector<int64_t> border{1, 1, 1, 1};
  test.AddAttribute("border", border);
  test.AddOutput<float>("y", {1, 1, 2, 2}, {6.0, 7.0, 10.0, 11.0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CropOpTest, Crop_Scale) {
  OpTester test("Crop", 1, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {1, 1, 4, 4}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});

  std::vector<int64_t> border{1, 1, 1, 1};
  test.AddAttribute("border", border);

  std::vector<int64_t> scale{2, 2};
  test.AddAttribute("scale", scale);

  test.AddOutput<float>("y", {1, 1, 2, 2}, {6.0, 7.0, 10.0, 11.0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
