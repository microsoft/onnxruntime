// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include <type_traits>

using namespace ONNX_NAMESPACE;
namespace onnxruntime {
// Disable TensorRT in the tests due to lack of support of the operator.
namespace test {
TEST(ConstantOfShape, Float_Ones) {
  OpTester test("ConstantOfShape", 9);

  TensorProto t_proto;
  t_proto.set_data_type(TensorProto::FLOAT);
  t_proto.mutable_dims()->Add(1);
  t_proto.mutable_float_data()->Add(1.f);
  test.AddAttribute("value", t_proto);

  // We will input 1-D Tensor that will store 3 dimensions
  // and will provide shape for the output
  std::vector<int64_t> input_dims{3};
  std::vector<int64_t> input{4, 3, 2};
  test.AddInput<int64_t>("input", input_dims, input);

  std::vector<int64_t> output_dims(input);
  std::vector<float> output;
  output.resize(4 * 3 * 2);
  std::fill_n(output.begin(), 4 * 3 * 2, 1.f);

  test.AddOutput<float>("output", output_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ConstantOfShape, Int32_Zeros) {
  OpTester test("ConstantOfShape", 9);

  TensorProto t_proto;
  t_proto.set_data_type(TensorProto::INT32);
  t_proto.mutable_dims()->Add(1);
  t_proto.mutable_int32_data()->Add(0);
  test.AddAttribute("value", t_proto);

  std::vector<int64_t> input_dims{2};
  std::vector<int64_t> input{10, 6};
  test.AddInput<int64_t>("input", input_dims, input);

  std::vector<int64_t> output_dims(input);
  std::vector<int32_t> output;
  output.resize(10 * 6);
  std::fill_n(output.begin(), output.size(), 0);
  test.AddOutput<int32_t>("output", output_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ConstantOfShape, DefaultValue) {
  OpTester test("ConstantOfShape", 9);

  // By default the output will be FLOAT zeros
  std::vector<int64_t> input_dims{2};
  std::vector<int64_t> input{2, 6};
  test.AddInput<int64_t>("input", input_dims, input);

  std::vector<int64_t> output_dims(input);
  std::vector<float> output;
  output.resize(2 * 6);
  std::fill_n(output.begin(), output.size(), 0.f);
  test.AddOutput<float>("output", output_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

inline void SetValue(TensorProto& t_proto, float value) {
  t_proto.mutable_float_data()->Add(value);
}

inline void SetValue(TensorProto& t_proto, double value) {
  t_proto.mutable_double_data()->Add(value);
}

inline void SetValue(TensorProto& t_proto, MLFloat16 value) {
  t_proto.mutable_int32_data()->Add(value.val);
}

// This works for int64_t
template <class T>
inline void SetValue(TensorProto& t_proto, T value,
                     typename std::enable_if<std::is_same<T, int64_t>::value>::type* = nullptr) {
  t_proto.mutable_int64_data()->Add(value);
}

// For uint32 and uint64
template <class T>
inline void SetValue(TensorProto& t_proto, T value,
                     typename std::enable_if<std::is_same<T, uint64_t>::value ||
                                             std::is_same<T, uint32_t>::value>::type* = nullptr) {
  t_proto.mutable_uint64_data()->Add(value);
}

// For everything else except float, double and MLFloat16
template <class T>
inline void SetValue(TensorProto& t_proto, T value,
                     typename std::enable_if<!std::is_same<T, int64_t>::value &&
                                             !std::is_same<T, uint32_t>::value &&
                                             !std::is_same<T, uint64_t>::value>::type* = nullptr) {
  t_proto.mutable_int32_data()->Add(value);
}

template <class T>
void RunTypedTest(TensorProto::DataType dt, T value) {
  OpTester test("ConstantOfShape", 9);

  TensorProto t_proto;
  t_proto.set_data_type(dt);
  t_proto.mutable_dims()->Add(1);
  SetValue(t_proto, value);
  test.AddAttribute("value", t_proto);

  // By default the output will be FLOAT zeros
  std::vector<int64_t> input_dims{2};
  std::vector<int64_t> input{2, 6};
  test.AddInput<int64_t>("input", input_dims, input);

  std::vector<int64_t> output_dims(input);
  std::vector<T> output;
  output.resize(2 * 6);
  std::fill_n(output.begin(), output.size(), value);
  test.AddOutput<T>("output", output_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ConstantOfShape, TypeTests) {
  // bool can not be tested due to a shortcoming of
  // our test infrastructure which makes use of
  // std::vector<T> which has a specialization for bool
  // and does not have a continuous buffer implementation
  // RunTypedTest(TensorProto::BOOL, true);

  RunTypedTest(TensorProto::INT8, int8_t(8));
  RunTypedTest(TensorProto::INT16, int16_t(16));
  RunTypedTest(TensorProto::FLOAT, 1.f);
  RunTypedTest(TensorProto::FLOAT16, MLFloat16::FromBits(static_cast<uint16_t>(5)));
  RunTypedTest(TensorProto::DOUBLE, 1.0);
  RunTypedTest(TensorProto::INT32, int32_t(32));
  RunTypedTest(TensorProto::INT64, int64_t(64));
  RunTypedTest(TensorProto::UINT8, uint8_t(8U));
  RunTypedTest(TensorProto::UINT16, uint16_t(6U));
  RunTypedTest(TensorProto::UINT32, uint32_t(32U));
  RunTypedTest(TensorProto::UINT64, uint64_t(64U));
}

}  // namespace test
}  // namespace onnxruntime
