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

template <typename TTensor, typename TControl>
void pass_through_multiple_tensors() {
  std::vector<TControl> control = {10.0f};
  std::vector<TTensor> input1 = {2.2f, 4.7f, 9.6f};
  std::vector<TTensor> input2 = {0.6f, 4.3f};
  OpTester test_passthrough("PassThrough", 1, onnxruntime::kMSDomain);
  test_passthrough.AddInput<TControl>("control_signal", {}, control);
  test_passthrough.AddInput<TTensor>("Input1", {static_cast<int64_t>(input1.size())}, input1);
  test_passthrough.AddInput<TTensor>("Input2", {static_cast<int64_t>(input2.size())}, input2);
  // We expect input and output are the same.
  test_passthrough.AddOutput<TTensor>("Output1", {static_cast<int64_t>(input1.size())}, input1);
  test_passthrough.AddOutput<TTensor>("Output2", {static_cast<int64_t>(input2.size())}, input2);
  run_provider_specific_optest(test_passthrough);
}

TEST(PassThrough, PassThroughMultipleFloatTensors) {
  pass_through_multiple_tensors<float, float>();
}

TEST(PassThrough, PassThroughMultipleDoubleTensors) {
  pass_through_multiple_tensors<double, double>();
}

TEST(PassThrough, PassThroughMultipleHalfTensors) {
  std::vector<float> control = {10.0f};
  std::vector<float> input1 = {2.2f, 4.7f, 9.6f};
  std::vector<float> input2 = {0.6f, 4.3f};
  size_t control_size = control.size();
  size_t input1_size = input1.size();
  size_t input2_size = input2.size();
 
  std::vector<MLFloat16> control_half(control_size);
  std::vector<MLFloat16> input1_half(input1_size);
  std::vector<MLFloat16> input2_half(input2_size);

  ConvertFloatToMLFloat16(control.data(), control_half.data(), control_size);
  ConvertFloatToMLFloat16(input1.data(), input1_half.data(), input1_size);
  ConvertFloatToMLFloat16(input2.data(), input2_half.data(), input2_size);

  OpTester test_passthrough("PassThrough", 1, onnxruntime::kMSDomain);
  test_passthrough.AddInput<MLFloat16>("control_signal", {}, control_half);
  test_passthrough.AddInput<MLFloat16>("Input1", {static_cast<int64_t>(input1_size)}, input1_half);
  test_passthrough.AddInput<MLFloat16>("Input2", {static_cast<int64_t>(input2_size)}, input2_half);
  // We expect input and output are the same.
  test_passthrough.AddOutput<MLFloat16>("Output1", {static_cast<int64_t>(input1_size)}, input1_half);
  test_passthrough.AddOutput<MLFloat16>("Output2", {static_cast<int64_t>(input2_size)}, input2_half);
  run_provider_specific_optest(test_passthrough);
}

}  // namespace test
}  // namespace onnxruntime
