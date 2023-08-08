// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

TEST(Identity, FloatType) {
  OpTester test("Identity", 9, kOnnxDomain);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddOutput<float>("Y", dims, {1.0f, 2.0f, 3.0f, 4.0f});
  test.Run();
}

TEST(Identity, StringType) {
  OpTester test("Identity", 10, kOnnxDomain);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<std::string>("X", dims, {"a", "b", "x", "y"});
  test.AddOutput<std::string>("Y", dims, {"a", "b", "x", "y"});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: unsupported data type
}

TEST(Identity, SequenceType) {
  OpTester test("Identity", 14, kOnnxDomain);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqInput("X", input);
  test.AddSeqOutput("Y", input);
  test.Run();
}

#if !defined(DISABLE_OPTIONAL_TYPE)

TEST(Identity, OptionalTensorType_NonNone) {
  OpTester test("Identity", 16, kOnnxDomain);

  std::initializer_list<float> data = {-1.0856307f, 0.99734545f};
  test.AddOptionalTypeTensorInput<float>("A", {2}, &data);
  test.AddOptionalTypeTensorOutput<float>("Y", {2}, &data);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: opset 16 is not supported yet
}

TEST(Identity, OptionalTensorType_None) {
  OpTester test("Identity", 16, kOnnxDomain);

  test.AddOptionalTypeTensorInput<float>("A", {}, nullptr);                            // None
  test.AddOptionalTypeTensorOutput<float>("Y", {}, nullptr);                           // None
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: opset 16 is not supported yet
}

TEST(Identity, OptionalTensorSequenceType_NonNone) {
  OpTester test("Identity", 16, kOnnxDomain);

  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

  test.AddOptionalTypeSeqInput<int64_t>("A", &input);
  test.AddOptionalTypeSeqOutput<int64_t>("Y", &input);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: opset 16 is not supported yet
}

TEST(Identity, OptionalTensorSequenceType_None) {
  OpTester test("Identity", 16, kOnnxDomain);

  test.AddOptionalTypeSeqInput<float>("A", nullptr);                                   // None
  test.AddOptionalTypeSeqOutput<float>("Y", nullptr);                                  // None
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: opset 16 is not supported yet
}

#endif

}  // namespace test
}  // namespace onnxruntime
