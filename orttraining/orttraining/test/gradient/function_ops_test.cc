// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <sstream>
#include <memory>

#include "gtest/gtest.h"
#include "core/framework/data_types.h"
#include "core/graph/model.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "orttraining/core/graph/training_op_defs.h"
#include "test/test_environment.h"

#include "core/session/inference_session.h"
#include "core/providers/cpu/cpu_execution_provider.h"

#include "test/framework/test_utils.h"
#include "test/common/tensor_op_test_utils.h"

#define _LOCAL_DEBUG_FLAG_ 1

using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace test {

typedef std::vector<onnxruntime::NodeArg*> ArgMap;

static void RegisterSchemas() {
  static bool registered = false;
  if (!registered) {
    onnxruntime::training::RegisterTrainingOpSchemas();
    registered = true;
  }
}

static ONNX_NAMESPACE::TypeProto TensorType(int32_t elem_type, std::vector<int64_t> dims) {
  ONNX_NAMESPACE::TypeProto typeProto;
  typeProto.mutable_tensor_type()->set_elem_type(elem_type);
  auto* shape = typeProto.mutable_tensor_type()->mutable_shape();
  for (auto dim : dims)
    shape->add_dim()->set_dim_value(dim);
  return typeProto;
}

static ONNX_NAMESPACE::TypeProto TensorType(int32_t elem_type, std::vector<std::string> dims) {
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

static std::vector<OrtValue>
Run(onnxruntime::Model& model, NameMLValMap& feeds, std::vector<std::string> output_names) {
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
static void AssertEqual(const Tensor& tensor1, const Tensor& tensor2) {
  auto size = tensor1.Shape().Size();
  auto* data1 = tensor1.template Data<T>();
  auto* data2 = tensor2.template Data<T>();

  T threshold = T(0.001f);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(data1[i], data2[i], threshold) << "as position i:" << i;
  }
}

static void AssertEqual(const std::vector<OrtValue>& results1, const std::vector<OrtValue>& results2) {
  ASSERT_EQ(results1.size(), results2.size());
  for (int i = 0; i < results1.size(); i++) {
    auto& value1 = results1[i].Get<Tensor>();
    auto& value2 = results2[i].Get<Tensor>();
    // Currently, only float or double:
    if (value1.DataType() == DataTypeImpl::GetType<float>())
      AssertEqual<float>(value1, value2);
    else
      AssertEqual<double>(value1, value2);
  }
}

struct FunctionTestCase {
  const char* opname;

  std::vector<NodeArg> input_args;
  std::vector<std::pair<std::string, OrtValue>> input_values;
  NameMLValMap input_value_map;

  std::vector<std::string> output_names;
  std::vector<NodeArg> output_args;

  NodeAttributes attributes;
  std::unique_ptr<IExecutionProvider> provider;

  std::unordered_map<std::string, int> opsets;

  FunctionTestCase(const char* _opname) : opname(_opname), provider(new CPUExecutionProvider(CPUExecutionProviderInfo())) {}

  void AddInput(std::string input_name, std::vector<int64_t> shape, std::vector<float> data, std::vector<std::string> symshape = {}) {
    auto arg_type = (symshape.size() > 0) ? TensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, symshape) : TensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape);
    input_args.emplace_back(input_name, &arg_type);

    OrtValue ort_value;
    CreateMLValue<float>(provider->GetAllocator(0, OrtMemTypeDefault), shape, data, &ort_value);
    input_values.push_back(std::make_pair(input_name, ort_value));
    input_value_map.insert(std::make_pair(input_name, ort_value));
  }

  template <typename T>
  void AddInput(std::string input_name, std::vector<int64_t> shape) {
    auto arg_type = TensorType(data_types_internal::ToTensorDataType<T>(), shape);
    input_args.emplace_back(input_name, &arg_type);

    RandomValueGenerator random{};
    std::vector<T> data = random.Uniform<T>(shape, 0.0f, 1.0f);

    OrtValue ort_value;
    CreateMLValue<T>(provider->GetAllocator(0, OrtMemTypeDefault), shape, data, &ort_value);
    input_values.push_back(std::make_pair(input_name, ort_value));
    input_value_map.insert(std::make_pair(input_name, ort_value));
  }

  template <>
  void AddInput<bool>(std::string input_name, std::vector<int64_t> shape) {
    auto arg_type = TensorType(ONNX_NAMESPACE::TensorProto_DataType_BOOL, shape);
    input_args.emplace_back(input_name, &arg_type);

    int64_t size = 1;
    for (auto dim : shape)
      size *= dim;

    std::vector<bool> data(size);
    for (int64_t i = 0; i < size; i++)
      data[i] = bool(i % 2);

    OrtValue ort_value;
    CreateMLValue<bool>(provider->GetAllocator(0, OrtMemTypeDefault), shape, data, &ort_value);
    input_values.push_back(std::make_pair(input_name, ort_value));
    input_value_map.insert(std::make_pair(input_name, ort_value));
  }

  void AddOutput(std::string output_name) {
    output_names.emplace_back(output_name);
    output_args.emplace_back(output_name, nullptr);
  }

  void AddAttribute(const char* attr_name, int64_t attr_val) {
    ONNX_NAMESPACE::AttributeProto axis_attr;
    axis_attr.set_name(attr_name);
    axis_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
    axis_attr.set_i(attr_val);
    attributes[attr_name] = axis_attr;
  }

  onnxruntime::Node& AddCallNodeTo(onnxruntime::Graph& graph) {
    std::vector<NodeArg*> input_arg_ptrs;

    for (auto& arg : input_args)
      input_arg_ptrs.push_back(&arg);

    std::vector<NodeArg*> output_arg_ptrs;
    for (auto& arg : output_args)
      output_arg_ptrs.push_back(&arg);

    return graph.AddNode("fncallnode", opname, "function call node", input_arg_ptrs, output_arg_ptrs, &attributes, onnxruntime::kMSDomain);
  }

  std::unique_ptr<Model> CreateModel(bool inline_call = false) {
    RegisterSchemas();
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
#if _LOCAL_DEBUG_FLAG_
      std::cout << graph << std::endl;
#endif
      status = graph.Resolve();
      EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
    }

    return model;
  }

  void RunTest() {
    auto model1 = CreateModel(false);
    auto results1 = Run(*model1, input_value_map, output_names);

    auto model2 = CreateModel(true);
    auto results2 = Run(*model2, input_value_map, output_names);

    AssertEqual(results1, results2);
  }
};

static void InitSoftmaxGradTestCase(FunctionTestCase& testCase, std::vector<int64_t> shape) {
  testCase.AddInput<float>("dY", shape);
  testCase.AddInput<float>("Y", shape);
  testCase.AddOutput("dX");
}

TEST(SoftmaxGradExpansionTest, DefaultAxis) {
  FunctionTestCase testCase("SoftmaxGrad");
  InitSoftmaxGradTestCase(testCase, {3, 2});
  testCase.RunTest();
}

TEST(SoftmaxGradExpansionTest, NegativeAxis) {
  FunctionTestCase testCase("SoftmaxGrad");
  InitSoftmaxGradTestCase(testCase, {3, 2});
  testCase.AddAttribute("axis", -1);
  testCase.RunTest();
}

TEST(SoftmaxGradExpansionTest, PositiveAxis) {
  FunctionTestCase testCase("SoftmaxGrad");
  InitSoftmaxGradTestCase(testCase, {3, 2});
  testCase.AddAttribute("axis", 1);
  testCase.RunTest();
}

TEST(SoftmaxGradExpansionTest, 3D) {
  FunctionTestCase testCase("SoftmaxGrad");
  InitSoftmaxGradTestCase(testCase, {3, 2, 2});
  testCase.RunTest();
}

TEST(SoftmaxGradExpansionTest, SymbolicShape) {
  FunctionTestCase testCase("SoftmaxGrad");
  std::vector<int64_t> shape{3, 2, 2};
  std::vector<std::string> sym_shape{"BatchSize", "SeqSize", "2"};
  int size = 12;
  std::vector<float> value(size);
  for (int64_t i = 0; i < size; i++)
    value[i] = float(i);

  testCase.AddInput("dY", shape, value, sym_shape);
  testCase.AddInput("Y", shape, value, sym_shape);
  testCase.AddOutput("dX");
  testCase.RunTest();
}

// Test (unexpanded) versions for both opset 12 and opset 13 models to ensure
// function-schema does not impact handling of opset 12 models. The current
// expansion requires opset 13, and no expansion should happen in opset 12
// models. Test is required since ORT currently generates function-expansion
// even when op is dispatched to a kernel.

TEST(SoftmaxGradExpansionTest, OpsetTest) {
  FunctionTestCase testCase("SoftmaxGrad");
  testCase.opsets[kOnnxDomain] = 12;
  testCase.opsets[kMSDomain] = 1;
  InitSoftmaxGradTestCase(testCase, {3, 2, 2});

  auto model1 = testCase.CreateModel();
  auto results1 = onnxruntime::test::Run(*model1, testCase.input_value_map, testCase.output_names);

  testCase.opsets[kOnnxDomain] = 13;
  testCase.opsets[kMSDomain] = 1;

  auto model2 = testCase.CreateModel();
  auto results2 = onnxruntime::test::Run(*model1, testCase.input_value_map, testCase.output_names);

  AssertEqual(results1, results2);
}

TEST(DropoutGradExpansionTest, WithoutRatio) {
  FunctionTestCase testCase("DropoutGrad");
  std::vector<int64_t> shape{16, 4, 4};
  testCase.AddInput<float>("dY", shape);
  testCase.AddInput<bool>("mask", shape);
  testCase.AddOutput("dX");
  testCase.RunTest();
}

TEST(DropoutGradExpansionTest, WithoutRatioDouble) {
  FunctionTestCase testCase("DropoutGrad");
  std::vector<int64_t> shape{16, 4, 4};
  testCase.AddInput<double>("dY", shape);
  testCase.AddInput<bool>("mask", shape);
  testCase.AddOutput("dX");
  testCase.RunTest();
}

TEST(DropoutGradExpansionTest, WithRatio) {
  FunctionTestCase testCase("DropoutGrad");
  std::vector<int64_t> shape{16, 4, 4};
  testCase.AddInput<float>("dY", shape);
  testCase.AddInput<bool>("mask", shape);
  testCase.AddInput("ratio", {}, {0.5f});
  testCase.AddOutput("dX");
  testCase.RunTest();
}

TEST(DropoutGradExpansionTest, WithRatioDouble) {
  FunctionTestCase testCase("DropoutGrad");
  std::vector<int64_t> shape{16, 4, 4};
  testCase.AddInput<double>("dY", shape);
  testCase.AddInput<bool>("mask", shape);
  testCase.AddInput("ratio", {}, {0.5f});
  testCase.AddOutput("dX");
  testCase.RunTest();
}


TEST(GeluGradExpansionTest, 2D) {
  FunctionTestCase testCase("GeluGrad");
  std::vector<int64_t> shape{16, 4};
  testCase.AddInput<float>("dY", shape);
  testCase.AddInput<float>("X", shape);
  testCase.AddOutput("dX");
  testCase.RunTest();
}

}  // namespace test
}  // namespace onnxruntime