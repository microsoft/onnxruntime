// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/framework/to_tensor_proto_element_type.h"
#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/framework/float16.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
inline std::vector<T> random(std::vector<int64_t> shape) {
  RandomValueGenerator generator{};
  return generator.Uniform<T>(shape, 0.0f, 1.0f);
}

template <>
inline std::vector<int64_t> random<int64_t>(std::vector<int64_t> shape) {
  int64_t size = 1;
  for (auto dim : shape)
    size *= dim;

  std::vector<int64_t> data(size);
  for (int64_t i = 0; i < size; i++)
    data[i] = static_cast<int64_t>(rand());
  return data;
}

template <>
inline std::vector<bool> random<bool>(std::vector<int64_t> shape) {
  int64_t size = 1;
  for (auto dim : shape)
    size *= dim;

  std::vector<bool> data(size);
  for (int64_t i = 0; i < size; i++)
    data[i] = bool(rand() % 2);
  return data;
}

template <>
inline std::vector<BFloat16> random<BFloat16>(std::vector<int64_t> shape) {
  auto floatdata = random<float>(shape);
  std::vector<BFloat16> data(floatdata.size());
  for (uint64_t i = 0; i < floatdata.size(); i++)
    data[i] = BFloat16(floatdata[i]);
  return data;
}

template <>
inline std::vector<MLFloat16> random<MLFloat16>(std::vector<int64_t> shape) {
  auto floatdata = random<float>(shape);
  std::vector<MLFloat16> data(floatdata.size());
  for (uint64_t i = 0; i < floatdata.size(); i++)
    data[i] = MLFloat16(floatdata[i]);
  return data;
}

struct FunctionTestCase {
 public:
  const char* domain;
  const char* opname;

  std::vector<NodeArg> input_args;
  std::vector<std::pair<std::string, OrtValue>> input_values;
  NameMLValMap input_value_map;

  std::vector<std::string> output_names;
  std::vector<NodeArg> output_args;

  NodeAttributes attributes;
  std::unique_ptr<IExecutionProvider> provider;

  std::unordered_map<std::string, int> opsets;

  FunctionTestCase(const char* _opname, const char* _domain = onnxruntime::kMSDomain) : domain(_domain), opname(_opname), provider(new CPUExecutionProvider(CPUExecutionProviderInfo())) {}

  template <typename T>
  void AddInput(std::string input_name, std::vector<int64_t> shape, std::vector<T> data, std::vector<std::string> symshape = {}) {
    auto arg_type = (symshape.size() > 0) ? TensorType(utils::ToTensorProtoElementType<T>(), symshape) : TensorType(utils::ToTensorProtoElementType<T>(), shape);
    input_args.emplace_back(input_name, &arg_type);

    OrtValue ort_value;
    CreateMLValue<T>(provider->CreatePreferredAllocators()[0], shape, data, &ort_value);
    input_values.push_back(std::make_pair(input_name, ort_value));
    input_value_map.insert(std::make_pair(input_name, ort_value));
  }

  void AddOpset(const char* opset_domain, int version) {
    opsets[opset_domain] = version;
  }

  template <typename T, bool GenData = true>
  void AddInput(std::string input_name, std::vector<int64_t> shape) {
    auto arg_type = TensorType(utils::ToTensorProtoElementType<T>(), shape);
    input_args.emplace_back(input_name, &arg_type);

    if (GenData) {
      std::vector<T> data = random<T>(shape);
      OrtValue ort_value;
      CreateMLValue<T>(provider->CreatePreferredAllocators()[0], shape, data, &ort_value);
      input_values.push_back(std::make_pair(input_name, ort_value));
      input_value_map.insert(std::make_pair(input_name, ort_value));
    }
  }

  template <typename T>
  void AddBoundedInput(const char* input_name, const std::vector<int64_t>& shape, T bound) {
    auto arg_type = TensorType(utils::ToTensorProtoElementType<T>(), shape);
    input_args.emplace_back(input_name, &arg_type);
    std::vector<T> data = random<T>(shape);
    for (size_t i = 0; i < data.size(); i++)
      data[i] = data[i] % bound;
    OrtValue ort_value;
    CreateMLValue<T>(provider->CreatePreferredAllocators()[0], shape, data, &ort_value);
    input_values.push_back(std::make_pair(input_name, ort_value));
    input_value_map.insert(std::make_pair(input_name, ort_value));
  }

  void AddOutput(std::string output_name);

  void AddAttribute(const char* attr_name, int64_t attr_val);
  void AddAttribute(const char* attr_name, const char* attr_val);

  onnxruntime::Node& AddCallNodeTo(onnxruntime::Graph& graph);

  std::unique_ptr<Model> CreateModel(bool inline_call = false);

  static std::vector<OrtValue> Run(onnxruntime::Model& model, NameMLValMap& feeds, std::vector<std::string> output_names);

  void RunTest();

  static void AssertEqual(const std::vector<OrtValue>& results1, const std::vector<OrtValue>& results2);

 private:
  ONNX_NAMESPACE::TypeProto TensorType(int32_t elem_type, std::vector<int64_t> dims);

  ONNX_NAMESPACE::TypeProto TensorType(int32_t elem_type, std::vector<std::string> dims);
};

}  // namespace test
}  // namespace onnxruntime
