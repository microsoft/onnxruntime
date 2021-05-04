// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <sstream>
#include <memory>
#include <cstdlib>

#include "function_test_util.h"

#include "gtest/gtest.h"

#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "test/common/tensor_op_test_utils.h"

namespace onnxruntime {
namespace test {

ONNX_NAMESPACE::TypeProto FunctionTestCase::TensorType(int32_t elem_type, std::vector<int64_t> dims) {
  ONNX_NAMESPACE::TypeProto typeProto;
  typeProto.mutable_tensor_type()->set_elem_type(elem_type);
  auto* shape = typeProto.mutable_tensor_type()->mutable_shape();
  for (auto dim : dims)
    shape->add_dim()->set_dim_value(dim);
  return typeProto;
}

ONNX_NAMESPACE::TypeProto FunctionTestCase::TensorType(int32_t elem_type, std::vector<std::string> dims) {
  ONNX_NAMESPACE::TypeProto typeProto;
  typeProto.mutable_tensor_type()->set_elem_type(elem_type);
  auto* shape = typeProto.mutable_tensor_type()->mutable_shape();
  for (auto dim : dims) {
    uint64_t dimval;
    std::istringstream s(dim);
    if (s >> dimval) {
      shape->add_dim()->set_dim_value(dimval);
    } else {
      shape->add_dim()->set_dim_param(dim);
    }
  }
  return typeProto;
}

std::vector<OrtValue> FunctionTestCase::Run(onnxruntime::Model& model, NameMLValMap& feeds, std::vector<std::string> output_names) {
  SessionOptions session_options;
  InferenceSession session_object{session_options, GetEnvironment()};

  std::string serialized_model;
  const bool serialization_status = model.ToProto().SerializeToString(&serialized_model);
  EXPECT_TRUE(serialization_status) << "Failed to serialize proto to string";
  std::stringstream sstr(serialized_model);
  auto status = session_object.Load(sstr);
  EXPECT_TRUE(status.IsOK());
  status = session_object.Initialize();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  RunOptions run_options;
  run_options.run_tag = session_options.session_logid;

  std::vector<OrtValue> fetches;

  status = session_object.Run(run_options, feeds, output_names, &fetches);
  EXPECT_TRUE(status.IsOK()) << "Session Run failed.";

  return fetches;
}

// Restricted to float tensors
template <typename T>
static void AssertEqualT(const Tensor& tensor1, const Tensor& tensor2) {
  auto size = tensor1.Shape().Size();
  auto* data1 = tensor1.template Data<T>();
  auto* data2 = tensor2.template Data<T>();

  T threshold = T(0.001f);

  for (int64_t i = 0; i < size; ++i) {
    ASSERT_NEAR(data1[i], data2[i], threshold) << "at position i:" << i;
  }
}

void FunctionTestCase::AssertEqual(const std::vector<OrtValue>& results1, const std::vector<OrtValue>& results2) {
  ASSERT_EQ(results1.size(), results2.size());
  for (size_t i = 0; i < results1.size(); i++) {
    auto& value1 = results1[i].Get<Tensor>();
    auto& value2 = results2[i].Get<Tensor>();
    // Currently, only float or double:
    if (value1.DataType() == DataTypeImpl::GetType<float>())
      AssertEqualT<float>(value1, value2);
    else
      AssertEqualT<double>(value1, value2);
  }
}

void FunctionTestCase::AddInput(std::string input_name, std::vector<int64_t> shape, std::vector<float> data, std::vector<std::string> symshape) {
  auto arg_type = (symshape.size() > 0) ? TensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, symshape) : TensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape);
  input_args.emplace_back(input_name, &arg_type);

  OrtValue ort_value;
  CreateMLValue<float>(provider->GetAllocator(0, OrtMemTypeDefault), shape, data, &ort_value);
  input_values.push_back(std::make_pair(input_name, ort_value));
  input_value_map.insert(std::make_pair(input_name, ort_value));
}

void FunctionTestCase::AddOutput(std::string output_name) {
  if (!output_name.empty()) output_names.emplace_back(output_name);
  output_args.emplace_back(output_name, nullptr);
}

void FunctionTestCase::AddAttribute(const char* attr_name, int64_t attr_val) {
  ONNX_NAMESPACE::AttributeProto axis_attr;
  axis_attr.set_name(attr_name);
  axis_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  axis_attr.set_i(attr_val);
  attributes[attr_name] = axis_attr;
}

onnxruntime::Node& FunctionTestCase::AddCallNodeTo(onnxruntime::Graph& graph) {
  std::vector<NodeArg*> input_arg_ptrs;

  for (auto& arg : input_args)
    input_arg_ptrs.push_back(&arg);

  std::vector<NodeArg*> output_arg_ptrs;
  for (auto& arg : output_args)
    output_arg_ptrs.push_back(&arg);

  return graph.AddNode("fncallnode", opname, "function call node", input_arg_ptrs, output_arg_ptrs, &attributes, domain);
}

std::unique_ptr<Model> FunctionTestCase::CreateModel(bool inline_call) {
  if (opsets.size() == 0) {
    // Default opsets
    opsets[kOnnxDomain] = 13;
    opsets[kMSDomain] = 1;
  }

  std::unique_ptr<Model> model(new Model("test", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                                         opsets, {}, DefaultLoggingManager().DefaultLogger()));

  onnxruntime::Graph& graph = model->MainGraph();
  auto& call_node = AddCallNodeTo(graph);

  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  if (inline_call) {
    graph.InlineFunction(call_node);
#if 0
    std::cout << graph << std::endl;
#endif
    status = graph.Resolve();
    EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  }

  return model;
}

void FunctionTestCase::RunTest() {
  auto model1 = CreateModel(false);
  auto results1 = Run(*model1, input_value_map, output_names);

  auto model2 = CreateModel(true);
  auto results2 = Run(*model2, input_value_map, output_names);

  AssertEqual(results1, results2);
}

}  // namespace test
}  // namespace onnxruntime