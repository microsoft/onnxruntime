// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "core/util/math.h"
#include "test/providers/provider_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// test that a multi value constant has the correct shape
TEST(ConstantOpTest, MultiValueConstant_floats) {
  OpTester test("Constant", 14, kOnnxDomain);
  test.AddAttribute("value_floats", std::vector<float>{0.f, 1.f, 2.f, 3.f});
  test.AddOutput<float>("Y", {4}, {0.f, 1.f, 2.f, 3.f});
  test.Run();
}

TEST(ConstantOpTest, MultiValueConstant_ints) {
  OpTester test("Constant", 14, kOnnxDomain);
  test.AddAttribute("value_ints", std::vector<int64_t>{0, 1, 2, 3});
  test.AddOutput<int64_t>("Y", {4}, {0, 1, 2, 3});
  test.Run();
}

TEST(ConstantOpTest, MultiValueConstant_strings) {
  OpTester test("Constant", 14, kOnnxDomain);
  test.AddAttribute("value_strings", std::vector<std::string>{"zero", "one", "two", "three"});
  test.AddOutput<std::string>("Y", {4}, {"zero", "one", "two", "three"});
  test.Run();
}

// regression test - https://github.com/microsoft/onnxruntime/issues/11091
TEST(ConstantOpTest, GH11091) {
  auto model_uri = ORT_TSTR("testdata/constant_floats.onnx");
  SessionOptions so;
  so.session_logid = "ConstantOpTest.GH11091";
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_uri));
  ASSERT_STATUS_OK(session_object.Initialize());
}
}  // namespace test
}  // namespace onnxruntime
