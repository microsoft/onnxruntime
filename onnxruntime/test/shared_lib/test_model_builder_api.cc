// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gsl/gsl>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/common/narrow.h"
#include "core/graph/constants.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/shared_lib/test_fixture.h"
#include "test/shared_lib/utils.h"
#include "test/util/include/test_allocator.h"

#include "onnxruntime_config.h"  // generated file in build output dir

extern std::unique_ptr<Ort::Env> ort_env;

using namespace Ort;

namespace {

Ort::Session CreateSession(Ort::Env& env,
                           Model& graph_api_model,
                           Ort::SessionOptions* session_options_for_test = nullptr) {
  Ort::SessionOptions default_session_options;
  Ort::SessionOptions& session_options = session_options_for_test ? *session_options_for_test
                                                                  : default_session_options;

  // Set this to save the model if you want to debug.
  // session_options.SetOptimizedModelFilePath(ORT_TSTR("model_builder_output.onnx"));

  Ort::Session session(env, graph_api_model, session_options);

  // Session should not require the model to stay alive so free it now to validate.
  graph_api_model = Model(nullptr);

  return session;
}

template <typename ModelOutputT, typename ModelInputT = float>
void TestInference(Ort::Session& session,
                   const std::vector<Input<ModelInputT>>& inputs,
                   const char* output_name,
                   const std::vector<int64_t>& expected_dims,
                   const std::vector<ModelOutputT>& expected_values) {
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  // without preallocated output tensor
  RunSession<ModelOutputT, ModelInputT>(default_allocator.get(),
                                        session,
                                        inputs,
                                        output_name,
                                        expected_dims,
                                        expected_values,
                                        nullptr);
}

// Create OrtNode using the C API
OrtNode* CreateNode(const OrtModelEditorApi& api,
                    const char* operator_name, const char* node_name,
                    const gsl::span<const char*> input_names,
                    const gsl::span<const char*> output_names,
                    const gsl::span<OrtOpAttr*> attributes = {},
                    const char* domain_name = onnxruntime::kOnnxDomain) {
  OrtNode* node = nullptr;
  Ort::ThrowOnError(api.CreateNode(operator_name, domain_name, node_name,
                                   input_names.data(), input_names.size(),
                                   output_names.data(), output_names.size(),
                                   attributes.data(), attributes.size(),
                                   &node));
  return node;
}

// convenience func to convert initalizer lists to gsl::span
OrtNode* CreateNode(const OrtModelEditorApi& api,
                    const char* operator_name, const char* node_name,
                    const std::initializer_list<const char*> input_names,
                    const std::initializer_list<const char*> output_names,
                    const std::initializer_list<OrtOpAttr*> attributes = {},
                    const char* domain_name = onnxruntime::kOnnxDomain) {
  std::vector<const char*> inputs(input_names);
  std::vector<const char*> outputs(output_names);
  std::vector<OrtOpAttr*> attrs(attributes);
  return CreateNode(api, operator_name, node_name, inputs, outputs, attrs, domain_name);
}
}  // namespace

struct TestAllocator : public OrtAllocator {
  TestAllocator() {
    version = ORT_API_VERSION;
    Info = [](const struct OrtAllocator* this_ptr) -> const struct OrtMemoryInfo* {
      auto* test_allocator = static_cast<const TestAllocator*>(this_ptr);
      return test_allocator->memory_info;
    };

    Free = [](struct OrtAllocator* allocator, void* p) -> void {
      auto* test_allocator = static_cast<TestAllocator*>(allocator);
      // find the matching pointer and remove it
      auto it = std::find_if(test_allocator->weights.begin(), test_allocator->weights.end(),
                             [p](const std::unique_ptr<std::vector<float>>& v) { return v->data() == p; });
      if (it == test_allocator->weights.end()) {
        throw std::runtime_error("Free called with unknown pointer");
      }

      test_allocator->weights.erase(it);
    };

    Alloc = [](struct OrtAllocator* /*this*/, size_t /*size*/) -> void* {
      throw std::runtime_error("This should not be used");
    };

    Reserve = [](struct OrtAllocator* /*this*/, size_t /*size*/) -> void* {
      throw std::runtime_error("This should not be used");
    };
  }

  // initializers that are used directly by the model. as there's no copy they must remain valid.
  // we store them in the test allocator so we can validate that Free is called
  std::vector<std::unique_ptr<std::vector<float>>> weights;
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                                           OrtMemType::OrtMemTypeDefault);
};

// Test the ModelEditorAPI C api
// Uses the ORT C++ api for the rest for simplicity
TEST(ModelEditorAPITest, Basic_CApi) {
  const auto& api = Ort::GetApi();
  const auto& model_editor_api = Ort::GetModelEditorApi();

  TestAllocator deleter;

  // return void so we can use ASSERT_* in the lambda
  const auto build_model = [&](bool use_constant_node, OrtModel*& model) -> void {
    OrtGraph* graph = nullptr;
    Ort::ThrowOnError(model_editor_api.CreateGraph(&graph));

    //
    // Create OrtModel with a Gemm. X input is 3x4, Y input is 4x8, Z output is 3x8.
    // X is model input. Y is initializer.
    // Set the alpha attribute of the Gemm node to 2.0 to test attribute handling.
    //

    // model input
    OrtTensorTypeAndShapeInfo* tensor_type_info = nullptr;
    std::vector<int64_t> input_dims = {3, 4};
    // can use api.SetSymbolicDimensions to set symbolic dimensions.
    // the input array should have the same rank as the call to SetDimensions.
    // e.g. call SetDimensions with {-1, 3, 2} and SetSymbolicDimensions with {"N", nullptr, nullptr} to create
    //      a shape of {"N", 3, 2}

    Ort::ThrowOnError(api.CreateTensorTypeAndShapeInfo(&tensor_type_info));
    Ort::ThrowOnError(api.SetTensorElementType(tensor_type_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    Ort::ThrowOnError(api.SetDimensions(tensor_type_info, input_dims.data(), input_dims.size()));

    OrtTypeInfo* input_type_info = nullptr;
    Ort::ThrowOnError(model_editor_api.CreateTensorTypeInfo(tensor_type_info, &input_type_info));
    api.ReleaseTensorTypeAndShapeInfo(tensor_type_info);  // input_type_info took a copy

    // create ValueInfo and release the type info as CreateValueInfo takes a copy.
    OrtValueInfo* input_value_info = nullptr;
    Ort::ThrowOnError(model_editor_api.CreateValueInfo("X", input_type_info, &input_value_info));
    api.ReleaseTypeInfo(input_type_info);  // input_value_info took a copy
    tensor_type_info = nullptr;

    // model outputs
    OrtTypeInfo* output_type_info = nullptr;
    std::vector<int64_t> output_dims = {3, 8};

    Ort::ThrowOnError(api.CreateTensorTypeAndShapeInfo(&tensor_type_info));
    Ort::ThrowOnError(api.SetTensorElementType(tensor_type_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    Ort::ThrowOnError(api.SetDimensions(tensor_type_info, output_dims.data(), output_dims.size()));

    Ort::ThrowOnError(model_editor_api.CreateTensorTypeInfo(tensor_type_info, &output_type_info));
    api.ReleaseTensorTypeAndShapeInfo(tensor_type_info);  // input_type_info took a copy

    OrtValueInfo* output_value_info = nullptr;
    Ort::ThrowOnError(model_editor_api.CreateValueInfo("Z", output_type_info, &output_value_info));
    api.ReleaseTypeInfo(output_type_info);

    std::vector<OrtValueInfo*> graph_inputs = {input_value_info};
    std::vector<OrtValueInfo*> graph_outputs = {output_value_info};
    Ort::ThrowOnError(model_editor_api.SetGraphInputs(graph, graph_inputs.data(), graph_inputs.size()));
    Ort::ThrowOnError(model_editor_api.SetGraphOutputs(graph, graph_outputs.data(), graph_outputs.size()));
    input_value_info = nullptr;  // graph now owns the input/output values
    output_value_info = nullptr;

    //
    // Gemm node
    //

    OrtOpAttr* alpha_attr = nullptr;
    float alpha_value = 2.0;
    Ort::ThrowOnError(api.CreateOpAttr("alpha", &alpha_value, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT, &alpha_attr));

    std::vector<const char*> node_input_names = {"X", "Y"};
    const std::string gemm_output_name = use_constant_node ? "Z_temp" : "Z";
    std::vector<const char*> node_output_names = {gemm_output_name.c_str()};
    std::vector<OrtOpAttr*> node_attributes{alpha_attr};
    OrtNode* node = CreateNode(model_editor_api, "Gemm", "Gemm1", node_input_names, node_output_names, node_attributes);
    alpha_attr = nullptr;  // Node now owns

    Ort::ThrowOnError(model_editor_api.AddNodeToGraph(graph, node));
    node = nullptr;  // graph now owns node

    // Y input
    // As it's 128 bytes it could either be allocated using CreateTensorAsOrtValue or use existing memory.
    // Under 128 bytes must use CreateTensorAsOrtValue.
    std::vector<int64_t> y_dims = {4, 8};

    deleter.weights.emplace_back(std::make_unique<std::vector<float>>(32));
    auto& y_values = *deleter.weights.back();
    std::iota(y_values.begin(), y_values.end(), 1.0f);

    // create an initializer for the Y input. add to `weights` so the memory remains valid.
    OrtValue* y_tensor = nullptr;
    Ort::ThrowOnError(
        api.CreateTensorWithDataAndDeleterAsOrtValue(&deleter,
                                                     y_values.data(), y_values.size() * sizeof(y_values[0]),
                                                     y_dims.data(), y_dims.size(),
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                     &y_tensor));

    Ort::ThrowOnError(model_editor_api.AddInitializerToGraph(graph, "Y", y_tensor, /*data is external*/ true));
    y_tensor = nullptr;  // graph now owns

    if (use_constant_node) {
      // Test that a Constant node is converted to an initializer

      // create Constant nodes for min/max to limit output range
      OrtOpAttr* min_attr = nullptr;
      float min = 400.0f;
      Ort::ThrowOnError(api.CreateOpAttr("value", &min, sizeof(min), ORT_OP_ATTR_FLOAT, &min_attr));
      node = CreateNode(model_editor_api, "Constant", "clip_min", {}, {"min"}, {min_attr});
      Ort::ThrowOnError(model_editor_api.AddNodeToGraph(graph, node));
      node = nullptr;  // graph now owns node

      OrtOpAttr* max_attr = nullptr;
      float max = 900.0f;
      Ort::ThrowOnError(api.CreateOpAttr("value", &max, sizeof(max), ORT_OP_ATTR_FLOAT, &max_attr));
      node = CreateNode(model_editor_api, "Constant", "clip_max", {}, {"max"}, {max_attr});
      Ort::ThrowOnError(model_editor_api.AddNodeToGraph(graph, node));
      node = nullptr;  // graph now owns node

      node = CreateNode(model_editor_api, "Clip", "Clip1", {gemm_output_name.c_str(), "min", "max"}, {"Z"});
      Ort::ThrowOnError(model_editor_api.AddNodeToGraph(graph, node));
      node = nullptr;  // graph now owns node
    }

    std::vector<const char*> domain_names = {onnxruntime::kOnnxDomain};
    std::vector<int> opset_versions = {18};
    Ort::ThrowOnError(model_editor_api.CreateModel(domain_names.data(), opset_versions.data(), domain_names.size(),
                                                   &model));
    Ort::ThrowOnError(model_editor_api.AddGraphToModel(model, graph));
    graph = nullptr;  // model now owns
  };

  auto run_test = [&](bool use_constant_node) -> void {
    OrtModel* model = nullptr;
    build_model(use_constant_node, model);

    ASSERT_NE(model, nullptr) << "build_model should have created a model";

    std::vector<Input<float>> inputs(1);
    auto& input = inputs[0];
    input.name = "X";
    input.dims = {3, 4};
    input.values = {1.0f, 2.0f, 3.0f, 4.0f,
                    8.0f, 7.0f, 6.0f, 5.0f,
                    9.0f, 3.0f, 5.0f, 7.0f};

    std::vector<int64_t> expected_dims = {3, 8};
    Model cxx_model(model);
    auto session = CreateSession(*ort_env, cxx_model);

    std::vector<float> expected_output;
    if (use_constant_node) {
      // clipped with min 400 and max 900
      expected_output = {400.0f, 400.0f, 400.0f, 400.0f, 420.0f, 440.0f, 460.0f, 480.0f,
                         596.0f, 648.0f, 700.0f, 752.0f, 804.0f, 856.0f, 900.0f, 900.0f,
                         592.0f, 640.0f, 688.0f, 736.0f, 784.0f, 832.0f, 880.0f, 900.0f};
    } else {
      expected_output = {340.0f, 360.0f, 380.0f, 400.0f, 420.0f, 440.0f, 460.0f, 480.0f,
                         596.0f, 648.0f, 700.0f, 752.0f, 804.0f, 856.0f, 908.0f, 960.0f,
                         592.0f, 640.0f, 688.0f, 736.0f, 784.0f, 832.0f, 880.0f, 928.0f};
    }

    TestInference<float>(session, inputs, "Z", expected_dims, expected_output);

    api.ReleaseSession(session.release());

    ASSERT_EQ(deleter.weights.size(), size_t(0)) << "All weights should have been freed";
  };

  run_test(false);
  run_test(true);  // use Constant node for initializer
}

TEST(ModelEditorAPITest, Basic_CxxApi) {
  // initializers that are used directly by the model. as there's no copy they must remain valid
  std::vector<std::unique_ptr<std::vector<float>>> weights;

  Ort::Graph graph;

  //
  // Create OrtModel with a Gemm. X input is 3x4, Y input is 4x8, Z output is 3x8.
  // X is model input. Y is initializer.
  // Set the alpha attribute of the Gemm node to 2.0 to test attribute handling.
  //

  std::vector<ValueInfo> graph_inputs;
  std::vector<ValueInfo> graph_outputs;

  // model input. it's {3, 4} but use a symbolic dim to test that works.
  std::vector<int64_t> input_dims({-1, 4});
  std::vector<std::string> input_symbolic_dims({"multiple_of_3", ""});
  TensorTypeAndShapeInfo input_tensor_info(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                           input_dims,
                                           &input_symbolic_dims);
  auto input_type_info = TypeInfo::CreateTensorInfo(input_tensor_info.GetConst());
  graph_inputs.emplace_back("X", input_type_info.GetConst());

  // model outputs
  std::vector<int64_t> output_dims = {-1, 8};
  std::vector<std::string> output_symbolic_dims({"multiple_of_3", ""});
  TensorTypeAndShapeInfo output_tensor_info(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                            output_dims,
                                            &output_symbolic_dims);
  auto output_type_info = TypeInfo::CreateTensorInfo(output_tensor_info.GetConst());
  graph_outputs.emplace_back("Z", output_type_info.GetConst());

  graph.SetInputs(graph_inputs);
  graph.SetOutputs(graph_outputs);

  //
  // Gemm node
  //

  std::vector<OpAttr> attributes;
  float alpha_value = 2.0;
  attributes.push_back(OpAttr("alpha", &alpha_value, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT));

  Node node("Gemm", onnxruntime::kOnnxDomain, "Gemm1", {"X", "Y"}, {"Z"}, attributes);

  graph.AddNode(node);

  // create an initializer for the Y input.
  // add to `weights` so it remains valid for the lifetime of the session and we can avoid copying the data.
  // As it's 128 bytes it could either be allocated using CreateTensorAsOrtValue or use existing memory.
  // Under 128 bytes must use CreateTensorAsOrtValue.
  std::vector<int64_t> y_dims = {4, 8};

  weights.emplace_back(std::make_unique<std::vector<float>>(32));
  auto& y_values = *weights.back();
  std::iota(y_values.begin(), y_values.end(), 1.0f);

  auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // if you use this API the initializer data MUST remain valid for the lifetime of the InferenceSession
  auto y_tensor = Value::CreateTensor(info, y_values.data(), y_values.size(), y_dims.data(), y_dims.size());
  graph.AddInitializer("Y", y_tensor, /*data is external*/ true);

  std::vector<Model::DomainOpsetPair> opsets{{onnxruntime::kOnnxDomain, 18}};
  Model model(opsets);
  model.AddGraph(graph);

  std::vector<Input<float>> inputs(1);
  auto& input = inputs[0];
  input.name = "X";
  input.dims = {3, 4};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f,
                  8.0f, 7.0f, 6.0f, 5.0f,
                  9.0f, 3.0f, 5.0f, 7.0f};

  std::vector<int64_t> expected_dims = {3, 8};

  auto session = CreateSession(*ort_env, model);
  TestInference<float>(session, inputs, "Z", expected_dims,
                       {340.0f, 360.0f, 380.0f, 400.0f, 420.0f, 440.0f, 460.0f, 480.0f,
                        596.0f, 648.0f, 700.0f, 752.0f, 804.0f, 856.0f, 908.0f, 960.0f,
                        592.0f, 640.0f, 688.0f, 736.0f, 784.0f, 832.0f, 880.0f, 928.0f});
}

TEST(ModelEditorAPITest, BasicModelEdit_CxxApi) {
  //
  // Load existing model
  // Add Cast to change the model input from float to int64
  // Update model inputs to match
  // Run
  //

  SessionOptions so;

  // Set this to save the model if you want to debug.
  // so.SetOptimizedModelFilePath(ORT_TSTR("model_builder_edited.onnx"));

  Session session = Session::CreateModelEditorSession(*ort_env, TSTR("testdata/mnist.onnx"), so);

  ASSERT_EQ(session.GetOpset(""), 8);  // ONNX domain is empty string

  // we augment the original model with nodes, initializers and the updated model inputs/outputs from this model.
  // the original graph is unchanged. nodes can be added before/after it. initializers can be added.
  // new nodes must conform to the original domain:opset of the model.
  // additional operator domain:opset pairs can be added.
  std::vector<Model::DomainOpsetPair> opsets;  // no additional opsets required
  Model model(opsets);

  std::vector<ValueInfo> graph_inputs = session.GetInputs();
  ASSERT_EQ(graph_inputs.size(), size_t(1));
  ASSERT_EQ(graph_inputs[0].TypeInfo().GetTensorTypeAndShapeInfo().GetElementType(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  // typically this isn't needed. we replace this input but need to read info from it later on in the test
  // validation so we save the info locally to keep it accessible.
  auto orig_input_name = graph_inputs[0].Name();
  auto input_shape = graph_inputs[0].TypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  const std::string new_input_name = "Int64Input";

  // Add Cast node to convert input from float to int64
  std::vector<OpAttr> attributes;
  int64_t to = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  attributes.push_back(OpAttr("to", &to, 1, OrtOpAttrType::ORT_OP_ATTR_INT));

  Ort::Node node("Cast", onnxruntime::kOnnxDomain, new_input_name, {"Int64Input"},
                 // the existing node will now consume the output from the Cast instead of a graph input
                 {orig_input_name},
                 attributes);

  // we're replacing the only input. the shape is the same but the name and data type change.
  TensorTypeAndShapeInfo input_tensor_info(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                           input_shape);
  auto input_type_info = TypeInfo::CreateTensorInfo(input_tensor_info.GetConst());
  graph_inputs[0] = ValueInfo(new_input_name, input_type_info.GetConst());

  Graph graph;  // new info to augment the model with

  graph.AddNode(node);
  graph.SetInputs(graph_inputs);

  // the node we added does not require any new opsets.
  model.AddGraph(graph);
  session.FinalizeModelEditorSession(model, so);

  std::vector<Input<int64_t>> inputs(1);
  auto& input = inputs[0];
  input.name = new_input_name.c_str();
  input.dims = input_shape;

  auto num_values = std::accumulate(input.dims.begin(), input.dims.end(), int64_t(1), std::multiplies<int64_t>());
  input.values.resize(size_t(num_values));
  std::iota(input.values.begin(), input.values.end(), 1);

  std::vector<int64_t> expected_dims = {1, 10};
  std::vector<float> expected_output = {-48.5088f, -1040.2948f, -347.0959f, 101.7392f, 421.3352f,
                                        750.92145f, 231.5060f, -1694.4152f, 681.5623f, 378.1689f};

  TestInference<float>(session, inputs, session.GetOutputNames()[0].c_str(), expected_dims, expected_output);

  // double check with original model
  {
    SessionOptions expected_so;
    Session expected_session = Session(*ort_env, TSTR("testdata/mnist.onnx"), expected_so);
    std::vector<Input<float>> expected_inputs(1);
    auto& expected_input = expected_inputs[0];
    expected_input.name = orig_input_name.c_str();
    expected_input.dims = input_shape;
    expected_input.values.reserve(size_t(num_values));
    std::transform(input.values.begin(), input.values.end(), std::back_inserter(expected_input.values),
                   [&](int64_t value) { return float(value); });

    TestInference<float>(expected_session, expected_inputs, session.GetOutputNames()[0].c_str(),
                         expected_dims, expected_output);
  }
}

TEST(ModelEditorAPITest, InvalidDimension) {
  try {
    std::vector<int64_t> input_dims = {-2, 2};
    TensorTypeAndShapeInfo tensor_type_info(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                            input_dims);
    // invalid dim of -2 should cause exception
    TypeInfo::CreateTensorInfo(tensor_type_info.GetConst());
    FAIL() << "Expected exception for invalid dimension";
  } catch (const Ort::Exception& e) {
    ASSERT_STREQ(e.what(), "dim_values must be -1 (symbolic dimension) or larger.");
  }
}

TEST(ModelEditorAPITest, CreateInvalidModel_NoOpsets) {
  Ort::Graph graph;
  std::vector<ValueInfo> graph_inputs;
  std::vector<ValueInfo> graph_outputs;

  std::vector<int64_t> dims({4});
  TensorTypeAndShapeInfo tensor_info(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dims);
  auto type_info = TypeInfo::CreateTensorInfo(tensor_info.GetConst());
  graph_inputs.emplace_back("X", type_info.GetConst());
  graph_outputs.emplace_back("Z", type_info.GetConst());

  graph.SetInputs(graph_inputs);
  graph.SetOutputs(graph_outputs);

  Ort::Node node("Add", onnxruntime::kOnnxDomain, "Add1", {"X", "X"}, {"Z"});

  graph.AddNode(node);

  std::vector<Model::DomainOpsetPair> opsets;
  Model model(opsets);
  model.AddGraph(graph);

  try {
    auto session = CreateSession(*ort_env, model);
    FAIL();
  } catch (const Ort::Exception& e) {
    ASSERT_THAT(e.what(), ::testing::HasSubstr("Error No opset import for domain"));
  }
}

TEST(ModelEditorAPITest, CreateInvalidModel_MissingValue) {
  Ort::Graph graph;

  std::vector<ValueInfo> graph_inputs;
  std::vector<ValueInfo> graph_outputs;

  std::vector<int64_t> dims({4});
  TensorTypeAndShapeInfo tensor_info(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dims);
  auto type_info = TypeInfo::CreateTensorInfo(tensor_info.GetConst());
  graph_inputs.emplace_back("X", type_info.GetConst());
  graph_outputs.emplace_back("Z", type_info.GetConst());

  graph.SetInputs(graph_inputs);
  graph.SetOutputs(graph_outputs);

  Ort::Node node("Add", onnxruntime::kOnnxDomain, "Add1", {"X", "missing"}, {"Z"});
  graph.AddNode(node);

  std::vector<Model::DomainOpsetPair> opsets{{onnxruntime::kOnnxDomain, 18}};
  Model model(opsets);
  model.AddGraph(graph);

  try {
    auto session = CreateSession(*ort_env, model);
    FAIL();
  } catch (const Ort::Exception& e) {
    ASSERT_THAT(e.what(), ::testing::HasSubstr("Node input 'missing' is not a graph input, "
                                               "initializer, or output of a previous node."));
  }
}

TEST(ModelEditorAPITest, InvalidModelEdit) {
  // Add a node but make the edit invalid in various ways
  //   - add node but don't update graph inputs
  //   - add node with invalid domain
  const auto edit_model = [](bool invalid_domain) {
    SessionOptions so;

    // Set this to save the model if you want to debug.
    // so.SetOptimizedModelFilePath(ORT_TSTR("model_builder_edited.onnx"));

    Session session = Session::CreateModelEditorSession(*ort_env, TSTR("testdata/mnist.onnx"), so);

    ASSERT_EQ(session.GetOpset(""), 8);  // ONNX domain is empty string

    std::vector<Model::DomainOpsetPair> opsets;  // no additional opsets required
    Model model(opsets);
    Graph graph;  // new info to augment the model with

    const char* domain = invalid_domain ? "invalid_domain" : onnxruntime::kOnnxDomain;

    std::vector<ValueInfo> graph_inputs = session.GetInputs();
    ASSERT_EQ(graph_inputs.size(), size_t(1));
    ASSERT_EQ(graph_inputs[0].TypeInfo().GetTensorTypeAndShapeInfo().GetElementType(),
              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    const std::string new_input_name = "Int64Input";

    // Add Cast node to convert input from float to int64
    std::vector<OpAttr> attributes;
    int64_t to = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    attributes.push_back(OpAttr("to", &to, 1, OrtOpAttrType::ORT_OP_ATTR_INT));

    Node node("Cast", domain, "NewInputNode", {new_input_name},
              // the existing node will now consume the output from the Cast instead of a graph input
              {graph_inputs[0].Name()},
              attributes);
    graph.AddNode(node);

    if (invalid_domain) {
      // we're replacing the only input. the shape is the same but the name and data type change.
      TensorTypeAndShapeInfo input_tensor_info(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                               graph_inputs[0].TypeInfo().GetTensorTypeAndShapeInfo().GetShape());
      auto input_type_info = TypeInfo::CreateTensorInfo(input_tensor_info.GetConst());
      graph_inputs[0] = ValueInfo(new_input_name, input_type_info.GetConst());
      graph.SetInputs(graph_inputs);
    } else {
      // model should be invalid as we didn't connect the new node up to the graph inputs
    }

    // the node we added does not require any new opsets.
    model.AddGraph(graph);

    try {
      session.FinalizeModelEditorSession(model, so);
      FAIL() << "Should have failed to resolve graph due to invalid edits.";
    } catch (const Ort::Exception& e) {
      if (invalid_domain) {
        ASSERT_THAT(e.what(), ::testing::HasSubstr("Error No opset import for domain 'invalid_domain'"));
      } else {
        ASSERT_THAT(e.what(), ::testing::HasSubstr("This is an invalid model"));
      }
    }
  };

  edit_model(false);
  edit_model(true);  // add node with invalid domain
}

TEST(ModelEditorAPITest, CreateTypeInfo) {
  const auto& api = Ort::GetApi();
  const auto& model_editor_api = Ort::GetModelEditorApi();

  TensorTypeAndShapeInfo base_tensor_info(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                          {2, 4});

  OrtTypeInfo* base_tensor_type_info = nullptr;
  Ort::ThrowOnError(model_editor_api.CreateTensorTypeInfo(base_tensor_info, &base_tensor_type_info));

  ONNXType onnx_type = ONNX_TYPE_UNKNOWN;
  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  ONNXTensorElementDataType onnx_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

  // sparse tensor
  OrtTypeInfo* sparse_tensor_type_info = nullptr;
  Ort::ThrowOnError(model_editor_api.CreateSparseTensorTypeInfo(base_tensor_info, &sparse_tensor_type_info));
  Ort::ThrowOnError(api.GetOnnxTypeFromTypeInfo(sparse_tensor_type_info, &onnx_type));
  ASSERT_EQ(onnx_type, ONNXType::ONNX_TYPE_SPARSETENSOR);
  Ort::ThrowOnError(api.CastTypeInfoToTensorInfo(sparse_tensor_type_info, &tensor_info));
  Ort::ThrowOnError(api.GetTensorElementType(tensor_info, &onnx_element_type));
  ASSERT_EQ(onnx_element_type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  api.ReleaseTypeInfo(sparse_tensor_type_info);

  // sequence
  OrtTypeInfo* sequence_type_info = nullptr;
  const OrtSequenceTypeInfo* sequence_info = nullptr;
  OrtTypeInfo* sequence_element_type_info = nullptr;

  Ort::ThrowOnError(model_editor_api.CreateSequenceTypeInfo(base_tensor_type_info, &sequence_type_info));
  Ort::ThrowOnError(api.GetOnnxTypeFromTypeInfo(sequence_type_info, &onnx_type));
  ASSERT_EQ(onnx_type, ONNXType::ONNX_TYPE_SEQUENCE);
  Ort::ThrowOnError(api.CastTypeInfoToSequenceTypeInfo(sequence_type_info, &sequence_info));
  Ort::ThrowOnError(api.GetSequenceElementType(sequence_info, &sequence_element_type_info));
  Ort::ThrowOnError(api.GetOnnxTypeFromTypeInfo(sequence_element_type_info, &onnx_type));
  ASSERT_EQ(onnx_type, ONNXType::ONNX_TYPE_TENSOR);
  Ort::ThrowOnError(api.CastTypeInfoToTensorInfo(sequence_element_type_info, &tensor_info));
  Ort::ThrowOnError(api.GetTensorElementType(tensor_info, &onnx_element_type));
  ASSERT_EQ(onnx_element_type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  api.ReleaseTypeInfo(sequence_element_type_info);
  api.ReleaseTypeInfo(sequence_type_info);

  // map
  OrtTypeInfo* map_type_info = nullptr;
  const OrtMapTypeInfo* map_info = nullptr;
  ONNXTensorElementDataType map_key_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  OrtTypeInfo* map_value_type_info = nullptr;
  Ort::ThrowOnError(model_editor_api.CreateMapTypeInfo(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, base_tensor_type_info,
                                                       &map_type_info));  // clones map_type_info
  Ort::ThrowOnError(api.GetOnnxTypeFromTypeInfo(map_type_info, &onnx_type));
  ASSERT_EQ(onnx_type, ONNXType::ONNX_TYPE_MAP);
  Ort::ThrowOnError(api.CastTypeInfoToMapTypeInfo(map_type_info, &map_info));
  Ort::ThrowOnError(api.GetMapKeyType(map_info, &map_key_type));
  ASSERT_EQ(map_key_type, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  Ort::ThrowOnError(api.GetMapValueType(map_info, &map_value_type_info));
  Ort::ThrowOnError(api.GetOnnxTypeFromTypeInfo(map_value_type_info, &onnx_type));
  ASSERT_EQ(onnx_type, ONNXType::ONNX_TYPE_TENSOR);
  Ort::ThrowOnError(api.CastTypeInfoToTensorInfo(map_value_type_info, &tensor_info));
  Ort::ThrowOnError(api.GetTensorElementType(tensor_info, &onnx_element_type));
  ASSERT_EQ(onnx_element_type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  api.ReleaseTypeInfo(map_value_type_info);
  api.ReleaseTypeInfo(map_type_info);

  // optional
  OrtTypeInfo* optional_type_info = nullptr;
  const OrtOptionalTypeInfo* optional_info = nullptr;
  OrtTypeInfo* optional_contained_type_info = nullptr;
  Ort::ThrowOnError(model_editor_api.CreateOptionalTypeInfo(base_tensor_type_info, &optional_type_info));
  Ort::ThrowOnError(api.GetOnnxTypeFromTypeInfo(optional_type_info, &onnx_type));
  ASSERT_EQ(onnx_type, ONNXType::ONNX_TYPE_OPTIONAL);
  Ort::ThrowOnError(api.CastTypeInfoToOptionalTypeInfo(optional_type_info, &optional_info));
  Ort::ThrowOnError(api.GetOptionalContainedTypeInfo(optional_info, &optional_contained_type_info));
  Ort::ThrowOnError(api.GetOnnxTypeFromTypeInfo(optional_contained_type_info, &onnx_type));
  ASSERT_EQ(onnx_type, ONNXType::ONNX_TYPE_TENSOR);
  api.ReleaseTypeInfo(optional_contained_type_info);
  api.ReleaseTypeInfo(optional_type_info);

  api.ReleaseTypeInfo(base_tensor_type_info);
}
