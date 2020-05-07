// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <bitset>
#include <cmath>
#include <random>
#include <thread>

#include "gtest/gtest.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/test_random_seed.h"
#include "orttraining/test/gradient/gradient_op_test_utils.h"

#include "onnx/defs/attr_proto_util.h"

namespace onnxruntime {
namespace test {

template <typename T1, typename T2>
void pass_through_multiple_tensors() {
  std::vector<T2> input0 = {10.0f};
  std::vector<T1> input1 = {2.2f, 4.7f, 9.6f};
  std::vector<T1> input2 = {0.6f, 4.3f};
  OpTester test_passthrough("PassThrough", 1, onnxruntime::kMSDomain);
  test_passthrough.AddInput<T2>("Input0", {}, input0);
  test_passthrough.AddInput<T1>("Input1", {static_cast<int64_t>(input1.size())}, input1);
  test_passthrough.AddInput<T1>("Input2", {static_cast<int64_t>(input2.size())}, input2);
  // We expect input and output are the same.
  test_passthrough.AddOutput<T2>("Output0", {}, input0);
  test_passthrough.AddOutput<T1>("Output1", {static_cast<int64_t>(input1.size())}, input1);
  test_passthrough.AddOutput<T1>("Output2", {static_cast<int64_t>(input2.size())}, input2);
  run_provider_specific_optest(test_passthrough);
}

// All inputs to PassThrough are float
TEST(PassThrough, PassThroughMultipleFloatTensors) {
  pass_through_multiple_tensors<float, float>();
}

// Input tensors have float and doule types
TEST(PassThrough, PassThroughFloatDoubleMixedInput) {
  pass_through_multiple_tensors<float, double>();
}

// Input tensors have float and doule types
TEST(PassThrough, PassThroughDoubleFloatMixedInput) {
  pass_through_multiple_tensors<double, float>();
}

// All inputs to PassThrough are double
TEST(PassThrough, PassThroughMultipleDoubleTensors) {
  pass_through_multiple_tensors<double, double>();
}

// All inputs to PassThrough are half
TEST(PassThrough, PassThroughMultipleHalfTensors) {
  std::vector<float> input0 = {10.0f};
  std::vector<float> input1 = {2.2f, 4.7f, 9.6f};
  std::vector<float> input2 = {0.6f, 4.3f};
  size_t input0_size = input0.size();
  size_t input1_size = input1.size();
  size_t input2_size = input2.size();
 
  std::vector<MLFloat16> input0_half(input0_size);
  std::vector<MLFloat16> input1_half(input1_size);
  std::vector<MLFloat16> input2_half(input2_size);

  ConvertFloatToMLFloat16(input0.data(),input0_half.data(), static_cast<int>(input0_size));
  ConvertFloatToMLFloat16(input1.data(), input1_half.data(), static_cast<int>(input1_size));
  ConvertFloatToMLFloat16(input2.data(), input2_half.data(), static_cast<int>(input2_size));

  OpTester test_passthrough("PassThrough", 1, onnxruntime::kMSDomain);
  test_passthrough.AddInput<MLFloat16>("Input0", {}, input0_half);
  test_passthrough.AddInput<MLFloat16>("Input1", {static_cast<int64_t>(input1_size)}, input1_half);
  test_passthrough.AddInput<MLFloat16>("Input2", {static_cast<int64_t>(input2_size)}, input2_half);
  // We expect input and output are the same.
  test_passthrough.AddOutput<MLFloat16>("Output0", {}, input0_half);
  test_passthrough.AddOutput<MLFloat16>("Output1", {static_cast<int64_t>(input1_size)}, input1_half);
  test_passthrough.AddOutput<MLFloat16>("Output2", {static_cast<int64_t>(input2_size)}, input2_half);
  run_provider_specific_optest(test_passthrough);
}

// Input tensors have float and half types
TEST(PassThrough, PassThroughHalfTensorFloatControl) {
  std::vector<float> input0 = {10.0f};
  std::vector<float> input1 = {2.2f, 4.7f, 9.6f};
  std::vector<float> input2 = {0.6f, 4.3f};
  size_t input1_size = input1.size();
  size_t input2_size = input2.size();
 
  std::vector<MLFloat16> input1_half(input1_size);
  std::vector<MLFloat16> input2_half(input2_size);

  ConvertFloatToMLFloat16(input1.data(), input1_half.data(), static_cast<int>(input1_size));
  ConvertFloatToMLFloat16(input2.data(), input2_half.data(), static_cast<int>(input2_size));

  OpTester test_passthrough("PassThrough", 1, onnxruntime::kMSDomain);
  test_passthrough.AddInput<float>("Input0", {}, input0);
  test_passthrough.AddInput<MLFloat16>("Input1", {static_cast<int64_t>(input1_size)}, input1_half);
  test_passthrough.AddInput<MLFloat16>("Input2", {static_cast<int64_t>(input2_size)}, input2_half);
  // We expect input and output are the same.
  test_passthrough.AddOutput<float>("Output0", {}, input0);
  test_passthrough.AddOutput<MLFloat16>("Output1", {static_cast<int64_t>(input1_size)}, input1_half);
  test_passthrough.AddOutput<MLFloat16>("Output2", {static_cast<int64_t>(input2_size)}, input2_half);
  run_provider_specific_optest(test_passthrough);
}

// Input tensors have half and bool types
TEST(PassThrough, PassThroughHalfBoolTensor) {
  std::initializer_list<bool> input0 = {true};
  std::vector<float> input1 = {2.2f, 4.7f, 9.6f};
  std::vector<float> input2 = {0.6f, 4.3f};
  size_t input1_size = input1.size();
  size_t input2_size = input2.size();
 
  std::vector<MLFloat16> input1_half(input1_size);
  std::vector<MLFloat16> input2_half(input2_size);

  ConvertFloatToMLFloat16(input1.data(), input1_half.data(), static_cast<int>(input1_size));
  ConvertFloatToMLFloat16(input2.data(), input2_half.data(), static_cast<int>(input2_size));

  OpTester test_passthrough("PassThrough", 1, onnxruntime::kMSDomain);
  test_passthrough.AddInput<bool>("Input0", {}, input0);
  test_passthrough.AddInput<MLFloat16>("Input1", {static_cast<int64_t>(input1_size)}, input1_half);
  test_passthrough.AddInput<MLFloat16>("Input2", {static_cast<int64_t>(input2_size)}, input2_half);
  // We expect input and output are the same.
  test_passthrough.AddOutput<bool>("Output0", {}, input0);
  test_passthrough.AddOutput<MLFloat16>("Output1", {static_cast<int64_t>(input1_size)}, input1_half);
  test_passthrough.AddOutput<MLFloat16>("Output2", {static_cast<int64_t>(input2_size)}, input2_half);
  run_provider_specific_optest(test_passthrough);
}

// Input tensors have float and half types, but only first 2 inputs are needed.
TEST(PassThrough, PassThroughHalfBoolTensorNeglectOneInput) {
  std::initializer_list<bool> input0 = {true};
  std::vector<float> input1 = {2.2f, 4.7f, 9.6f};
  std::vector<float> input2 = {0.6f, 4.3f};
  size_t input1_size = input1.size();
  size_t input2_size = input2.size();
 
  std::vector<MLFloat16> input1_half(input1_size);
  std::vector<MLFloat16> input2_half(input2_size);

  ConvertFloatToMLFloat16(input1.data(), input1_half.data(), static_cast<int>(input1_size));
  ConvertFloatToMLFloat16(input2.data(), input2_half.data(), static_cast<int>(input2_size));

  OpTester test_passthrough("PassThrough", 1, onnxruntime::kMSDomain);
  test_passthrough.AddInput<bool>("Input0", {}, input0);
  test_passthrough.AddInput<MLFloat16>("Input1", {static_cast<int64_t>(input1_size)}, input1_half);
  test_passthrough.AddInput<MLFloat16>("Input2", {static_cast<int64_t>(input2_size)}, input2_half);
  // We expect first 2 inputs and outputs are the same.
  test_passthrough.AddOutput<bool>("Output0", {}, input0);
  test_passthrough.AddOutput<MLFloat16>("Output1", {static_cast<int64_t>(input1_size)}, input1_half);
  run_provider_specific_optest(test_passthrough);
}

}  // namespace test
}  // namespace onnxruntime
