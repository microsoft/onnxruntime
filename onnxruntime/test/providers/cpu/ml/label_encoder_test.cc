// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename TInput, typename TOutput>
static void RunTest(const std::vector<int64_t>& dims, const std::vector<TInput>& input,
                    const std::vector<TOutput>& output) {
  OpTester test("LabelEncoder", 1, onnxruntime::kMLDomain);

  static const std::vector<std::string> labels = {"Beer", "Wine", "Tequila"};

  test.AddAttribute("classes_strings", labels);

  test.AddAttribute("default_string", "Water");
  test.AddAttribute<int64_t>("default_int64", 99);

  test.AddInput<TInput>("X", dims, input);
  test.AddOutput<TOutput>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, StringToInt) {
  std::vector<int64_t> dims{2, 2, 2};

  std::vector<std::string> input{"Beer", "Burger", "Tequila", "Burrito", "Wine", "Cheese", "Tequila", "Floor"};
  std::vector<int64_t> output{0, 99, 2, 99, 1, 99, 2, 99};

  RunTest(dims, input, output);
}

TEST(LabelEncoder, IntToString) {
  std::vector<int64_t> dims{2, 3};

  std::vector<int64_t> input{0, 10, 2, 3, 1, -1};
  std::vector<std::string> output{"Beer", "Water", "Tequila", "Water", "Wine", "Water"};

  RunTest(dims, input, output);
}

TEST(LabelEncoder, StringToIntOpset2) {
  std::vector<std::int64_t> dims{1, 5};

  std::vector<std::string> input{"AA", "BB", "CC", "DD", "AA"};
  std::vector<std::int64_t> output{9, 1, 5566, 4, 9};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const std::vector<std::string> keys{"AA", "BB", "DD"};
  const std::vector<std::int64_t> values{9, 1, 4};

  test.AddAttribute("keys_strings", keys);
  test.AddAttribute("values_int64s", values);
  test.AddAttribute("default_int64", (std::int64_t)5566);

  test.AddInput<std::string>("X", dims, input);
  test.AddOutput<std::int64_t>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, IntToStringOpset2) {
  std::vector<std::int64_t> dims{1, 5};

  std::vector<std::int64_t> input{9, 1, 5566, 4, 9};
  std::vector<std::string> output{"AA", "BB", "CC", "DD", "AA"};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const std::vector<std::int64_t> keys{9, 1, 4};
  const std::vector<std::string> values{"AA", "BB", "DD"};

  test.AddAttribute("keys_int64s", keys);
  test.AddAttribute("values_strings", values);
  test.AddAttribute<std::string>("default_string", "CC");

  test.AddInput<std::int64_t>("X", dims, input);
  test.AddOutput<std::string>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, FloatToStringOpset2) {
  std::vector<std::int64_t> dims{5, 1};

  std::vector<float> input{9.4f, 1.7f, 3.6f, 1.2f, 2.8f};
  std::vector<std::string> output{"AA", "BB", "DD", "CC", "CC"};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const std::vector<float> keys{9.4f, 1.7f, 3.6f};
  const std::vector<std::string> values{"AA", "BB", "DD"};

  test.AddAttribute("keys_floats", keys);
  test.AddAttribute("values_strings", values);
  test.AddAttribute<std::string>("default_string", "CC");

  test.AddInput<float>("X", dims, input);
  test.AddOutput<std::string>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, StringToFloatOpset2) {
  std::vector<std::int64_t> dims{5, 1};

  std::vector<std::string> input{"AA", "BB", "DD", "CC", "CC"};
  std::vector<float> output{9.4f, 1.7f, 3.6f, 55.66f, 55.66f};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const std::vector<std::string> keys{"AA", "BB", "DD"};
  const std::vector<float> values{9.4f, 1.7f, 3.6f};

  test.AddAttribute("keys_strings", keys);
  test.AddAttribute("values_floats", values);
  test.AddAttribute("default_float", 55.66f);

  test.AddInput<std::string>("X", dims, input);
  test.AddOutput<float>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, FloatToInt64Opset2) {
  std::vector<std::int64_t> dims{5};

  std::vector<float> input{9.4f, 1.7f, 3.6f, 55.66f, 55.66f};
  std::vector<std::int64_t> output{1, 9, 3, -8, -8};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const std::vector<float> keys{9.4f, 1.7f, 3.6f};
  const std::vector<std::int64_t> values{1, 9, 3};

  test.AddAttribute("keys_floats", keys);
  test.AddAttribute("values_int64s", values);
  test.AddAttribute("default_int64", (std::int64_t)-8);

  test.AddInput<float>("X", dims, input);
  test.AddOutput<std::int64_t>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, Int64ToFloatOpset2) {
  std::vector<std::int64_t> dims{5};

  std::vector<std::int64_t> input{3, 1, 9, -8, -8};
  std::vector<float> output{3.6f, 9.4f, 1.7f, 55.66f, 55.66f};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const std::vector<std::int64_t> keys{1, 9, 3};
  const std::vector<float> values{9.4f, 1.7f, 3.6f};

  test.AddAttribute("keys_int64s", keys);
  test.AddAttribute("values_floats", values);
  test.AddAttribute("default_float", 55.66f);

  test.AddInput<std::int64_t>("X", dims, input);
  test.AddOutput<float>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, Int64ToInt64Opset2) {
  std::vector<std::int64_t> dims{5};

  std::vector<std::int64_t> input{3, 5, 9, -8, -8};
  std::vector<std::int64_t> output{0, 1, -1, 2, 2};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const std::vector<std::int64_t> keys{3, 5, -8};
  const std::vector<std::int64_t> values{0, 1, 2};

  test.AddAttribute("keys_int64s", keys);
  test.AddAttribute("values_int64s", values);
  test.AddAttribute("default_int64", (std::int64_t)-1);

  test.AddInput<std::int64_t>("X", dims, input);
  test.AddOutput<std::int64_t>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, StringToStringOpset2) {
  std::vector<std::int64_t> dims{1, 5};

  std::vector<std::string> input{"A", "A", "C", "D", "E"};
  std::vector<std::string> output{"X", "X", "Z", "!", "!"};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const std::vector<std::string> keys{"A", "B", "C"};
  const std::vector<std::string> values{"X", "Y", "Z"};

  test.AddAttribute("keys_strings", keys);
  test.AddAttribute("values_strings", values);
  test.AddAttribute("default_string", "!");

  test.AddInput<std::string>("X", dims, input);
  test.AddOutput<std::string>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, FloatToFloatOpset2) {
  std::vector<std::int64_t> dims{1, 4};

  std::vector<float> input{-1.0f, 0.0f, 3.1427f, 7.25f};
  std::vector<float> output{1.0f, 0.0f, 2.718f, NAN};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const std::vector<float> keys{-1.0f, 0.0f, 7.25f};
  const std::vector<float> values{1.0f, 0.0f, NAN};

  test.AddAttribute("keys_floats", keys);
  test.AddAttribute("values_floats", values);
  test.AddAttribute("default_float", 2.718f);

  test.AddInput<float>("X", dims, input);
  test.AddOutput<float>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, Int64toInt64Opset4) {
  std::vector<std::int64_t> dims{1, 5};

  std::vector<int64_t> input{1, 2, 3, 4, 5};
  std::vector<int64_t> output{12, 13, 14, 15, 42};
  std::vector<int64_t> key_data{1, 2, 3, 4};
  std::vector<int64_t> value_data{12, 13, 14, 15};

  OpTester test("LabelEncoder", 4, onnxruntime::kMLDomain);

  test.AddAttribute("keys_int64s", key_data);
  test.AddAttribute("values_int64s", value_data);

  ONNX_NAMESPACE::TensorProto default_proto;
  default_proto.set_name("default_tensor");
  default_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  default_proto.add_dims(1);
  default_proto.add_int64_data(42);
  test.AddAttribute("default_tensor", default_proto);

  test.AddInput<int64_t>("X", dims, input);
  test.AddOutput<int64_t>("Y", dims, output);
  test.Run();
}

TEST(LabelEncoder, StringtoInt16Opset4) {
  std::vector<std::int64_t> dims{1, 5};

  const std::vector<std::string> input{"a", "b", "d", "c", "g"};
  const std::vector<int16_t> output{0, 1, 42, 2, 42};
  const std::vector<std::string> key_data{"a", "b", "c"};
  const std::vector<int16_t> value_data{0, 1, 2};

  OpTester test("LabelEncoder", 4, onnxruntime::kMLDomain);

  test.AddAttribute("keys_strings", key_data);

  ONNX_NAMESPACE::TensorProto values_proto;
  values_proto.set_name("values_tensor");
  values_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT16);
  values_proto.add_dims(value_data.size());
  for (const auto value : value_data) {
    values_proto.add_int32_data(value);
  }

  test.AddAttribute("values_tensor", values_proto);

  ONNX_NAMESPACE::TensorProto default_proto;
  default_proto.set_name("default_tensor");
  default_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT16);
  default_proto.add_dims(1);
  default_proto.add_int32_data(42);
  test.AddAttribute("default_tensor", default_proto);

  test.AddInput<std::string>("X", dims, input);
  test.AddOutput<int16_t>("Y", dims, output);
  test.Run();
}

TEST(LabelEncoder, Int64toStringOpset4) {
  std::vector<std::int64_t> dims{1, 5};

  std::vector<int64_t> input{1, 2, 3, 4, 5};
  std::vector<std::string> output{"Hello", "world", "_Unused", "onnxruntime", "!"};
  std::vector<int64_t> key_data{1, 2, 4, 5};
  std::vector<std::string> value_data{"Hello", "world", "onnxruntime", "!"};

  OpTester test("LabelEncoder", 4, onnxruntime::kMLDomain);

  ONNX_NAMESPACE::TensorProto keys_proto;
  keys_proto.set_name("keys_tensor");
  keys_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  keys_proto.add_dims(key_data.size());
  for (const auto key : key_data) {
    keys_proto.add_int64_data(key);
  }
  test.AddAttribute("keys_tensor", keys_proto);

  ONNX_NAMESPACE::TensorProto values_proto;
  values_proto.set_name("values_tensor");
  values_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
  values_proto.add_dims(value_data.size());
  for (const auto& value : value_data) {
    values_proto.add_string_data(value);
  }
  test.AddAttribute("values_tensor", values_proto);

  ONNX_NAMESPACE::TensorProto default_proto;
  default_proto.set_name("default_tensor");
  default_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
  default_proto.add_dims(1);
  default_proto.add_string_data("_Unused");
  test.AddAttribute("default_tensor", default_proto);

  test.AddInput<int64_t>("X", dims, input);
  test.AddOutput<std::string>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, StringToFloatOpset4) {
  std::vector<std::int64_t> dims{1, 5};

  std::vector<std::string> input{"Hello", "world", "Random", "onnxruntime", "!"};
  std::vector<float> output{3.14f, 2.0f, -0.0f, 2.718f, 5.0f};
  std::vector<std::string> key_data{"Hello", "world", "onnxruntime", "!"};
  std::vector<float> value_data{3.14f, 2.0f, 2.718f, 5.0f};

  OpTester test("LabelEncoder", 4, onnxruntime::kMLDomain);

  ONNX_NAMESPACE::TensorProto keys_proto;
  keys_proto.set_name("keys_tensor");
  keys_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
  keys_proto.add_dims(key_data.size());
  for (const auto& key : key_data) {
    keys_proto.add_string_data(key);
  }
  test.AddAttribute("keys_tensor", keys_proto);

  ONNX_NAMESPACE::TensorProto values_proto;
  values_proto.set_name("values_tensor");
  values_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  values_proto.add_dims(value_data.size());
  for (const auto& value : value_data) {
    values_proto.add_float_data(value);
  }
  test.AddAttribute("values_tensor", values_proto);

  ONNX_NAMESPACE::TensorProto default_proto;
  default_proto.set_name("default_tensor");
  default_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  default_proto.add_dims(1);
  default_proto.add_float_data(-0.0f);
  test.AddAttribute("default_tensor", default_proto);
  test.AddInput<std::string>("X", dims, input);
  test.AddOutput<float>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, StringToDoubleOpset4) {
  std::vector<std::int64_t> dims{1, 5};

  std::vector<std::string> input{"Hello", "world", "Random", "onnxruntime", "!"};
  std::vector<double> output{0.1, 1.1231e30, -0.0, 2.718, 5.0};
  std::vector<std::string> key_data{"Hello", "world", "onnxruntime", "!"};
  std::vector<double> value_data{0.1, 1.1231e30, 2.718, 5.0};

  OpTester test("LabelEncoder", 4, onnxruntime::kMLDomain);

  ONNX_NAMESPACE::TensorProto keys_proto;
  keys_proto.set_name("keys_tensor");
  keys_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
  keys_proto.add_dims(key_data.size());
  for (const auto& key : key_data) {
    keys_proto.add_string_data(key);
  }
  test.AddAttribute("keys_tensor", keys_proto);

  ONNX_NAMESPACE::TensorProto values_proto;
  values_proto.set_name("values_tensor");
  values_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
  values_proto.add_dims(value_data.size());
  for (const auto& value : value_data) {
    values_proto.add_double_data(value);
  }
  test.AddAttribute("values_tensor", values_proto);

  ONNX_NAMESPACE::TensorProto default_proto;
  default_proto.set_name("default_tensor");
  default_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
  default_proto.add_dims(1);
  default_proto.add_double_data(-0.0);
  test.AddAttribute("default_tensor", default_proto);
  test.AddInput<std::string>("X", dims, input);
  test.AddOutput<double>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, TensorBasedAttributesOpset4) {
  std::vector<std::int64_t> dims{1, 5};

  std::vector<int64_t> input{1, 2, 3, 4, 5};
  std::vector<int64_t> output{12, 13, 14, 15, 42};
  std::vector<int64_t> key_data{1, 2, 3, 4};
  std::vector<int64_t> value_data{12, 13, 14, 15};

  OpTester test("LabelEncoder", 4, onnxruntime::kMLDomain);

  ONNX_NAMESPACE::TensorProto keys_proto;
  keys_proto.set_name("keys_tensor");
  keys_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  keys_proto.add_dims(key_data.size());
  for (const auto key : key_data) {
    keys_proto.add_int64_data(key);
  }
  test.AddAttribute("keys_tensor", keys_proto);

  ONNX_NAMESPACE::TensorProto values_proto;
  values_proto.set_name("values_tensor");
  values_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  values_proto.add_dims(value_data.size());
  for (const auto value : value_data) {
    values_proto.add_int64_data(value);
  }
  test.AddAttribute("values_tensor", values_proto);

  ONNX_NAMESPACE::TensorProto default_proto;
  default_proto.set_name("default_tensor");
  default_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  default_proto.add_dims(1);
  default_proto.add_int64_data(42);
  test.AddAttribute("default_tensor", default_proto);

  test.AddInput<int64_t>("X", dims, input);
  test.AddOutput<int64_t>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, NaNsMappedTogetherOpset4) {
  std::vector<std::int64_t> dims{1, 6};
  std::vector<float> input{3.14f, std::nanf("1"), 2.718f, std::nanf("2"), 5.f, -1.f};
  std::vector<std::string> output{"a", "ONNX", "b", "ONNX", "c", "onnxruntime"};
  std::vector<float> key_data{3.14f, 2.718f, 5.0f, std::nanf("3")};
  std::vector<std::string> value_data{"a", "b", "c", "ONNX"};

  OpTester test("LabelEncoder", 4, onnxruntime::kMLDomain);

  test.AddAttribute("keys_floats", key_data);
  test.AddAttribute("values_strings", value_data);

  ONNX_NAMESPACE::TensorProto default_proto;
  default_proto.set_name("default_tensor");
  default_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
  default_proto.add_dims(1);
  default_proto.add_string_data("onnxruntime");
  test.AddAttribute("default_tensor", default_proto);

  test.AddInput<float>("X", dims, input);
  test.AddOutput<std::string>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, DoubleNaNsMappedTogetherOpset4) {
  std::vector<std::int64_t> dims{1, 6};
  std::vector<double> input{3.14, std::nan("1"), 2.718, std::nan("2"), 5.0, -1};
  std::vector<std::string> output{"a", "ONNX", "b", "ONNX", "c", "onnxruntime"};
  std::vector<double> key_data{3.14, 2.718, 5.0, std::nan("3")};
  std::vector<std::string> value_data{"a", "b", "c", "ONNX"};

  OpTester test("LabelEncoder", 4, onnxruntime::kMLDomain);

  ONNX_NAMESPACE::TensorProto keys_proto;
  keys_proto.set_name("keys_tensor");
  keys_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
  keys_proto.add_dims(key_data.size());
  for (const auto key : key_data) {
    keys_proto.add_double_data(key);
  }
  test.AddAttribute("keys_tensor", keys_proto);

  test.AddAttribute("values_strings", value_data);

  ONNX_NAMESPACE::TensorProto default_proto;
  default_proto.set_name("default_tensor");
  default_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
  default_proto.add_dims(1);
  default_proto.add_string_data("onnxruntime");
  test.AddAttribute("default_tensor", default_proto);

  test.AddInput<double>("X", dims, input);
  test.AddOutput<std::string>("Y", dims, output);

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
