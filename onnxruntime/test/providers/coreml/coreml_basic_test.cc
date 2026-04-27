// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "core/common/logging/logging.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/coreml/coreml_provider_factory_creator.h"
#include "core/providers/coreml/coreml_provider_factory.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/current_test_name.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/temp_dir.h"
#include "test/util/include/test_environment.h"
#include "test/util/include/test_utils.h"
#include "core/graph/onnx_protobuf.h"

#if !defined(ORT_MINIMAL_BUILD)
// if this is a full build we need the provider test utils
#include "test/providers/provider_test_utils.h"
#endif  // !(ORT_MINIMAL_BUILD)

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

// defined in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

static std::unordered_map<std::string, std::string> MakeCoreMLProviderOptions(std::string ModelFormat = "NeuralNetwork",
                                                                              std::string ComputeUnits = "CPUOnly",
                                                                              std::string ModelCacheDirectory = "") {
  std::unordered_map<std::string, std::string> provider_options = {{kCoremlProviderOption_MLComputeUnits, ComputeUnits},
                                                                   {kCoremlProviderOption_ModelFormat, ModelFormat},
                                                                   {kCoremlProviderOption_ModelCacheDirectory,
                                                                    ModelCacheDirectory}};
  return provider_options;
}

static std::unique_ptr<IExecutionProvider> MakeCoreMLExecutionProvider(
    std::string ModelFormat = "NeuralNetwork", std::string ComputeUnits = "CPUOnly", std::string ModelCacheDirectory = "") {
  std::unordered_map<std::string, std::string> provider_options = MakeCoreMLProviderOptions(ModelFormat,
                                                                                            ComputeUnits,
                                                                                            ModelCacheDirectory);
  return CoreMLProviderFactoryCreator::Create(provider_options)->CreateProvider();
}

#if !defined(ORT_MINIMAL_BUILD)

TEST(CoreMLExecutionProviderTest, TestAddEpUsingPublicApi) {
  auto session_has_ep = [](Ort::Session& session) -> bool {
    // Access the underlying InferenceSession.
    const OrtSession* ort_session = session;
    const InferenceSession* s = reinterpret_cast<const InferenceSession*>(ort_session);
    bool has_ep = false;

    for (const auto& provider : s->GetRegisteredProviderTypes()) {
      if (provider == kCoreMLExecutionProvider) {
        has_ep = true;
        break;
      }
    }
    return has_ep;
  };

  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/constant_floats.onnx");
  auto provider_options = MakeCoreMLProviderOptions("NeuralNetwork", "CPUOnly", "./tmp");

  {
    // Test C++ API to add CoreML EP with the short name 'CoreML'.
    Ort::SessionOptions so;
    so.AppendExecutionProvider("CoreML", provider_options);
    Ort::Session session(*ort_env, model_file_name, so);
    ASSERT_TRUE(session_has_ep(session)) << "CoreML EP was not found in registered providers for session.";
  }

  {
    // Test C++ API to add CoreML EP with the long canonical name 'CoreMLExecutionProvider'.
    Ort::SessionOptions so;
    so.AppendExecutionProvider(kCoreMLExecutionProvider, provider_options);
    Ort::Session session(*ort_env, model_file_name, so);
    ASSERT_TRUE(session_has_ep(session)) << "CoreML EP was not found in registered providers for session.";
  }
}

TEST(CoreMLExecutionProviderTest, FunctionTest) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("coreml_execution_provider_test_graph.onnx");

  {  // Create the model with 2 add nodes
    onnxruntime::Model model("graph_1", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    std::vector<onnxruntime::NodeArg*> inputs;
    std::vector<onnxruntime::NodeArg*> outputs;

    // FLOAT tensor.
    ONNX_NAMESPACE::TypeProto float_tensor;
    float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

    auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
    auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
    inputs.push_back(&input_arg_1);
    inputs.push_back(&input_arg_2);
    auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
    outputs.push_back(&output_arg);
    graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

    auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
    inputs.clear();
    inputs.push_back(&output_arg);
    inputs.push_back(&input_arg_3);
    auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
    outputs.clear();
    outputs.push_back(&output_arg_2);
    graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

    ASSERT_STATUS_OK(graph.Resolve());
    ASSERT_STATUS_OK(onnxruntime::Model::Save(model, model_file_name));
  }

#if defined(__APPLE__)
  std::vector<int64_t> dims_mul_x = {1, 1, 3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;

  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(allocator, dims_mul_x, values_mul_x, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(allocator, dims_mul_x, values_mul_x, &ml_value_z);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  RunAndVerifyOutputsWithEP(model_file_name, CurrentTestName(),
                            MakeCoreMLExecutionProvider(),
                            feeds);
#else
  TestModelLoad(model_file_name, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::Some);
#endif
}

// CoreML EP currently handles a special case for supporting ArgMax op:
// An ArgMax followed by a Cast to int32 type.
// Please see in <repo_root>/onnxruntime/core/providers/coreml/builders/impl/argmax_op_builder.cc
// and /cast_op_builder.cc. We have the following UT test here for this special case
// This test case can also be shared later if we want to support similar cases in NNAPI
TEST(CoreMLExecutionProviderTest, ArgMaxCastTest) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/coreml_argmax_cast_test.onnx");

#if defined(__APPLE__)
  std::vector<int64_t> dims_mul_x = {3, 2, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims_mul_x, values_mul_x, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  EPVerificationParams verification_params{};
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;

  RunAndVerifyOutputsWithEP(model_file_name, CurrentTestName(),
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            verification_params);
  RunAndVerifyOutputsWithEP(model_file_name, CurrentTestName(),
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds,
                            verification_params);
#else
  TestModelLoad(model_file_name, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::All);
#endif
}

TEST(CoreMLExecutionProviderTest, ArgMaxUnsupportedCastTest) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/coreml_argmax_unsupported_cast_test.onnx");

#if defined(__APPLE__)
  std::vector<int64_t> dims_mul_x = {3, 2, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims_mul_x, values_mul_x, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  const std::function<void(const Graph&)> graph_verifier = [](const Graph& graph) {
    GraphViewer graph_viewer{graph};
    const auto& node_indices_in_order = graph_viewer.GetNodesInTopologicalOrder();
    ASSERT_EQ(node_indices_in_order.size(), size_t{2});
    // second node should be an unsupported Cast
    const auto* cast_node = graph.GetNode(node_indices_in_order[1]);
    ASSERT_NE(cast_node, nullptr);
    ASSERT_EQ(cast_node->OpType(), "Cast");
    ASSERT_EQ(cast_node->GetExecutionProviderType(), kCpuExecutionProvider);
  };

  EPVerificationParams verification_params{};
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::Some;
  verification_params.graph_verifier = &graph_verifier;

  RunAndVerifyOutputsWithEP(model_file_name, CurrentTestName(),
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            verification_params);

  RunAndVerifyOutputsWithEP(model_file_name, CurrentTestName(),
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds,
                            verification_params);
#else
  TestModelLoad(model_file_name, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::Some);
#endif
}

TEST(CoreMLExecutionProviderTest, GatherWithScalarIndices) {
  // For scalar inputs, the input shape is modified from [] -> [1] before passing the input to CoreML.
  // This won't work for Gather because the output shape depends on the `indices` input shape which could be a scalar.
  // Currently, we expect the CoreML EP to only take the Shape node in this graph (Gather -> Shape).
  const auto model_file_name = ORT_TSTR("testdata/gather_with_scalar_indices_then_shape.onnx");

#if defined(__APPLE__)
  RandomValueGenerator gen{1234};
  std::vector<int64_t> X_shape = {5, 3, 4};
  std::vector<float> X_data = gen.Uniform<float>(X_shape, 0.0f, 1.0f);
  OrtValue X = CreateInputOrtValueOnCPU<float>(X_shape, X_data);
  OrtValue indices = CreateInputOrtValueOnCPU<int64_t>(AsSpan<int64_t>({}), AsSpan<int64_t>({1}));

  RunAndVerifyOutputsWithEP(model_file_name, CurrentTestName(),
                            MakeCoreMLExecutionProvider(),
                            {{"X", X}, {"indices", indices}});
#else
  TestModelLoad(model_file_name, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::Some);
#endif
}

TEST(CoreMLExecutionProviderTest, ShapeThenSliceAndGather) {
  // This is a simple test model that provides the output of Shape to Slice and Gather.
  // We expect the CoreML EP to support shape manipulations like this.
  const auto model_file_name = ORT_TSTR("testdata/shape_then_slice_and_gather.onnx");

#if defined(__APPLE__)
  RandomValueGenerator gen{1234};
  std::vector<int64_t> X_shape = {5, 3, 4, 1, 2};
  std::vector<float> X_data = gen.Uniform<float>(X_shape, 0.0f, 1.0f);
  OrtValue X = CreateInputOrtValueOnCPU<float>(X_shape, X_data);

  RunAndVerifyOutputsWithEP(model_file_name, CurrentTestName(),
                            MakeCoreMLExecutionProvider(),
                            {{"X", X}},
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
#else
  TestModelLoad(model_file_name, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::All);
#endif
}

#endif  // !(ORT_MINIMAL_BUILD)

TEST(CoreMLExecutionProviderTest, TestOrtFormatModel) {
  // mnist model that has only had basic optimizations applied. CoreML should be able to take at least some of the nodes
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/mnist.basic.ort");

#if defined(__APPLE__)
  RandomValueGenerator random{};
  const std::vector<int64_t> dims = {1, 1, 28, 28};
  std::vector<float> data = random.Gaussian<float>(dims, 0.0f, 1.f);

  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], dims, data, &ml_value);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("Input3", ml_value));

  RunAndVerifyOutputsWithEP(model_file_name, CurrentTestName(),
                            MakeCoreMLExecutionProvider(),
                            feeds);
#else
  TestModelLoad(model_file_name, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::Some);
#endif
}

#if defined(USE_COREML)
// Names in CoreML cannot start with [0-9] or contain anything but "[a-z][A-Z][0-9]_"
// Test that we fix invalid names in model inputs, initializers and outputs.
// This is only enforced for ML Program, so we only do name sanitization when creating an ML Program format model.
TEST(CoreMLExecutionProviderTest, TestNameSanitization) {
  OpTester test("Clip", 11);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("0", dims,
                       {-1.0f, 0.0f, 1.0f,
                        -6.0f, 0.0f, 6.0f,
                        -5.4f, 2.0f, 6.0f});
  test.AddInput<float>("1.min", {}, {-5}, true);  // add as initializers
  test.AddInput<float>("2/max", {}, {5}, true);
  test.AddOutput<float>("3", dims,
                        {-1.0f, 0.0f, 1.0f,
                         -5.0f, 0.0f, 5.0f,
                         -5.0f, 2.0f, 5.0f});

  // TensorRT does not support Clip opset 11 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
#endif

TEST(CoreMLExecutionProviderTest, TestModelCache) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/coreml_argmax_cast_test.onnx");

  onnx::ModelProto model;
  {
    std::ifstream in(model_file_name, std::ios_base::binary);
    model.ParseFromIstream(&in);
    in.close();
  }

  std::string out_string;
#if defined(__APPLE__)
  std::vector<int64_t> dims_mul_x = {3, 2, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims_mul_x, values_mul_x, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  std::string subgraph_name;
  const std::function<void(const Graph&)> graph_verifier = [&subgraph_name](const Graph& graph) {
    GraphViewer graph_viewer{graph};
    const auto& node_indices_in_order = graph_viewer.GetNodesInTopologicalOrder();
    const auto* node = graph.GetNode(node_indices_in_order[0]);
    auto _first = node->Name().find('_') + 1;
    auto _second = node->Name().find('_', _first);
    subgraph_name = node->Name().substr(_first, _second - _first);
  };
  EPVerificationParams verification_params{.graph_verifier = &graph_verifier};

  auto* metadata_props = model.add_metadata_props();
  metadata_props->set_key(kCOREML_CACHE_KEY);
  {  // test with valid model cache directory
    metadata_props->set_value("legalhash123");
    model.SerializeToString(&out_string);
    gsl::span<const std::byte> model_data{reinterpret_cast<const std::byte*>(out_string.data()), out_string.size()};
    RunAndVerifyOutputsWithEP(model_data, CurrentTestName(),
                              MakeCoreMLExecutionProvider("MLProgram", "CPUOnly", ORT_TSTR("./tmp/")),
                              feeds,
                              verification_params);
    ASSERT_EQ(std::filesystem::exists("./tmp/legalhash123"), true);
  }
  {
    // test with invalid model cache directory, only alphanumeric characters are allowed
    out_string.clear();
    metadata_props->set_key(kCOREML_CACHE_KEY);
    metadata_props->set_value("illegalhash__123");
    model.SerializeToString(&out_string);
    gsl::span<const std::byte> model_data{reinterpret_cast<const std::byte*>(out_string.data()), out_string.size()};
    RunAndVerifyOutputsWithEP(model_data, CurrentTestName(),
                              MakeCoreMLExecutionProvider("MLProgram", "CPUOnly", ORT_TSTR("./tmp")),
                              feeds,
                              verification_params);
    ASSERT_EQ(std::filesystem::exists("./tmp/illegalhash__123"), false);
    // the cache folder name should be the first part of the subgraph name
    ASSERT_EQ(std::filesystem::exists("./tmp/" + subgraph_name), true);
  }
  {
    // test with invalid model cache directory,  more than 64 characters
    out_string.clear();
    metadata_props->set_key(kCOREML_CACHE_KEY);
    metadata_props->set_value("modelhashwithmorethan64charactersmodelhashwithmorethan64charactersmodelhashwithmorethan64characters");
    model.SerializeToString(&out_string);
    gsl::span<const std::byte> model_data{reinterpret_cast<const std::byte*>(out_string.data()), out_string.size()};
    RunAndVerifyOutputsWithEP(model_data, CurrentTestName(),
                              MakeCoreMLExecutionProvider("MLProgram", "CPUOnly", ORT_TSTR("./tmp")),
                              feeds,
                              verification_params);
    ASSERT_EQ(std::filesystem::exists("./tmp/modelhashwithmorethan64charactersmodelhashwithmorethan64charactersmodelhashwithmorethan64characters"), false);
    // the cache folder name should be the first part of the subgraph name
    ASSERT_EQ(std::filesystem::exists("./tmp/" + subgraph_name), true);
  }
  {
    // test with invalid model cache directory,  empty
    out_string.clear();
    metadata_props->set_key(kCOREML_CACHE_KEY);
    metadata_props->set_value("");
    model.SerializeToString(&out_string);
    gsl::span<const std::byte> model_data{reinterpret_cast<const std::byte*>(out_string.data()), out_string.size()};
    RunAndVerifyOutputsWithEP(model_data, CurrentTestName(),
                              MakeCoreMLExecutionProvider("MLProgram", "CPUOnly", ORT_TSTR("./tmp")),
                              feeds,
                              verification_params);
    // the cache folder name should be the first part of the subgraph name
    ASSERT_EQ(std::filesystem::exists("./tmp/" + subgraph_name), true);
  }
  {
    // test with invalid model cache directory, caching shall be disabled
    out_string.clear();
    metadata_props->set_key(kCOREML_CACHE_KEY);
    metadata_props->set_value("");
    model.SerializeToString(&out_string);
    gsl::span<const std::byte> model_data{reinterpret_cast<const std::byte*>(out_string.data()), out_string.size()};
    RunAndVerifyOutputsWithEP(model_data, CurrentTestName(),
                              MakeCoreMLExecutionProvider("MLProgram", "CPUOnly", ORT_TSTR("/")),
                              feeds,
                              verification_params);
    // this folder can't be created
    ASSERT_EQ(std::filesystem::exists("/" + subgraph_name), false);
  }
#else
  model.SerializeToString(&out_string);
  gsl::span<const std::byte> model_data{reinterpret_cast<const std::byte*>(out_string.data()), out_string.size()};
  TestModelLoad(model_data, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::All);
#endif
}

// Test that CoreML EP can load a model with initializers stored in an external data file.
// Regression test for https://github.com/microsoft/onnxruntime/issues/28005
// The bug was that TensorProtoWithExternalDataToTensorProto passed a model file path
// (e.g. "/path/to/model.onnx") to ReadExternalDataForTensor which expects a directory,
// causing it to construct an invalid path like "/path/to/model.onnx/model.onnx_data".
#if !defined(ORT_MINIMAL_BUILD)
TEST(CoreMLExecutionProviderTest, ExternalDataInitializer) {
  // Create a temp directory for the model and external data file
  TemporaryDirectory tmp_dir(ORT_TSTR("coreml_external_data_test"));
  const auto model_path = std::filesystem::path(tmp_dir.Path()) / ORT_TSTR("model.onnx");
  const auto external_data_path = std::filesystem::path(tmp_dir.Path()) / ORT_TSTR("model.onnx_data");

  // Write external data file: 6 floats for a {1,1,3,2} initializer
  const std::vector<float> initializer_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  {
    std::ofstream ofs(external_data_path, std::ios::binary);
    ASSERT_TRUE(ofs.is_open());
    ofs.write(reinterpret_cast<const char*>(initializer_data.data()),
              initializer_data.size() * sizeof(float));
    ofs.close();
  }

  // Build a simple model: output = X + initializer (Add op)
  {
    ONNX_NAMESPACE::ModelProto model_proto;
    model_proto.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
    auto* opset = model_proto.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    auto* graph_proto = model_proto.mutable_graph();
    graph_proto->set_name("test_external_data");

    // Input X: {1,1,3,2} float tensor
    auto* input = graph_proto->add_input();
    input->set_name("X");
    auto* input_type = input->mutable_type()->mutable_tensor_type();
    input_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* input_shape = input_type->mutable_shape();
    input_shape->add_dim()->set_dim_value(1);
    input_shape->add_dim()->set_dim_value(1);
    input_shape->add_dim()->set_dim_value(3);
    input_shape->add_dim()->set_dim_value(2);

    // Output Y: {1,1,3,2} float tensor
    auto* output = graph_proto->add_output();
    output->set_name("Y");
    auto* output_type = output->mutable_type()->mutable_tensor_type();
    output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* output_shape = output_type->mutable_shape();
    output_shape->add_dim()->set_dim_value(1);
    output_shape->add_dim()->set_dim_value(1);
    output_shape->add_dim()->set_dim_value(3);
    output_shape->add_dim()->set_dim_value(2);

    // Initializer W with external data
    auto* initializer = graph_proto->add_initializer();
    initializer->set_name("W");
    initializer->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    initializer->add_dims(1);
    initializer->add_dims(1);
    initializer->add_dims(3);
    initializer->add_dims(2);
    initializer->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);

    auto* ext_location = initializer->add_external_data();
    ext_location->set_key("location");
    ext_location->set_value("model.onnx_data");
    auto* ext_offset = initializer->add_external_data();
    ext_offset->set_key("offset");
    ext_offset->set_value("0");
    auto* ext_length = initializer->add_external_data();
    ext_length->set_key("length");
    ext_length->set_value(std::to_string(initializer_data.size() * sizeof(float)));

    // Add node: Y = X + W
    auto* node = graph_proto->add_node();
    node->set_op_type("Add");
    node->add_input("X");
    node->add_input("W");
    node->add_output("Y");

    // Save model
    std::ofstream ofs(model_path, std::ios::binary);
    ASSERT_TRUE(ofs.is_open());
    ASSERT_TRUE(model_proto.SerializeToOstream(&ofs));
    ofs.close();
  }

  // Input data
  std::vector<int64_t> dims = {1, 1, 3, 2};
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  RunOptions run_options;
  run_options.run_tag = "ExternalDataInitializer";
  std::vector<std::string> output_names = {"Y"};

  // Load the model from a file path (not from memory) with the CoreML EP.
  // This is the scenario that triggers the bug: CoreML EP must resolve external data
  // relative to the model file's directory, not treat the model path as a directory.
  SessionOptions so;
  so.session_logid = "ExternalDataInitializer";
  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(MakeCoreMLExecutionProvider()));
  ASSERT_STATUS_OK(session.Load(model_path.native()));
  ASSERT_STATUS_OK(session.Initialize());

#if defined(__APPLE__)
  const auto& provider_types = session.GetRegisteredProviderTypes();
  EXPECT_NE(std::find(provider_types.begin(), provider_types.end(), kCoreMLExecutionProvider), provider_types.end());
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session.Run(run_options, feeds, output_names, &fetches));

  // Verify the output: Y = X + W = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6}
  ASSERT_EQ(fetches.size(), 1u);
  const auto& output_tensor = fetches[0].Get<Tensor>();
  auto output_data = output_tensor.DataAsSpan<float>();
  ASSERT_EQ(static_cast<size_t>(output_data.size()), input_data.size());
  for (size_t i = 0; i < input_data.size(); ++i) {
    EXPECT_NEAR(output_data[i], input_data[i] + initializer_data[i], 1e-5f)
        << "Mismatch at index " << i;
  }
#endif  // defined(__APPLE__)
}
#endif  // !(ORT_MINIMAL_BUILD)

// Verify that Pad(mode=reflect) is handled by the CoreML EP in ML Program mode
// instead of falling back to CPU.
// See https://github.com/microsoft/onnxruntime/issues/28022
#if !defined(ORT_MINIMAL_BUILD)
TEST(CoreMLExecutionProviderTest, PadReflectMLProgram) {
  // Build a model: output = Pad(X, pads, mode="reflect")
  // Input shape: {1,1,3,4}, pads on last 2 dims: H=[1,1], W=[2,1]
  // Expected output shape: {1,1,5,7}
  onnxruntime::Model model("pad_reflect_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  // Input X: {1,1,3,4} float tensor
  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type.mutable_tensor_type()->mutable_shape();
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(3);
  input_shape->add_dim()->set_dim_value(4);

  // Output Y: {1,1,5,7} float tensor
  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  output_shape->add_dim()->set_dim_value(1);
  output_shape->add_dim()->set_dim_value(1);
  output_shape->add_dim()->set_dim_value(5);
  output_shape->add_dim()->set_dim_value(7);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  // Pads initializer: [0,0,1,2, 0,0,1,1] -- pad H by (1,1), W by (2,1)
  ONNX_NAMESPACE::TensorProto pads_init;
  pads_init.set_name("pads");
  pads_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  pads_init.add_dims(8);
  // ONNX pads: [dim0_start, dim1_start, dim2_start, dim3_start, dim0_end, dim1_end, dim2_end, dim3_end]
  // Pads last two dims: H=(1,1), W=(2,1).
  const std::vector<int64_t> pads_data = {0, 0, 1, 2, 0, 0, 1, 1};
  for (auto v : pads_data) {
    pads_init.add_int64_data(v);
  }
  graph.AddInitializedTensor(pads_init);

  auto& pads_arg = graph.GetOrCreateNodeArg("pads", nullptr);

  auto& pad_node = graph.AddNode("pad_reflect", "Pad", "reflect pad",
                                 {&input_arg, &pads_arg}, {&output_arg});
  pad_node.AddAttribute("mode", "reflect");

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  // Input data for {1,1,3,4}:
  // [[[ 1,  2,  3,  4],
  //   [ 5,  6,  7,  8],
  //   [ 9, 10, 11, 12]]]
  std::vector<int64_t> dims = {1, 1, 3, 4};
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "PadReflectMLProgram",
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
#else
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

TEST(CoreMLExecutionProviderTest, PadConstantDefaultValueMLProgram) {
  // Build a model: output = Pad(X, pads, mode="constant"), no explicit constant_value input should default to 0.
  onnxruntime::Model model("pad_constant_default_value_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type.mutable_tensor_type()->mutable_shape();
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(3);
  input_shape->add_dim()->set_dim_value(4);

  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  output_shape->add_dim()->set_dim_value(1);
  output_shape->add_dim()->set_dim_value(1);
  output_shape->add_dim()->set_dim_value(5);
  output_shape->add_dim()->set_dim_value(7);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  ONNX_NAMESPACE::TensorProto pads_init;
  pads_init.set_name("pads");
  pads_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  pads_init.add_dims(8);
  // ONNX pads: [dim0_start, dim1_start, dim2_start, dim3_start, dim0_end, dim1_end, dim2_end, dim3_end]
  // Pads last two dims: H=(1,1), W=(2,1).
  const std::vector<int64_t> pads_data = {0, 0, 1, 2, 0, 0, 1, 1};
  for (auto v : pads_data) {
    pads_init.add_int64_data(v);
  }
  graph.AddInitializedTensor(pads_init);

  auto& pads_arg = graph.GetOrCreateNodeArg("pads", nullptr);
  auto& pad_node = graph.AddNode("pad_constant", "Pad", "constant pad default value",
                                 {&input_arg, &pads_arg}, {&output_arg});
  pad_node.AddAttribute("mode", "constant");

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 1, 3, 4};
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "PadConstantDefaultValueMLProgram",
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
#else
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

// Verify that ML Program supports padding on dimensions other than the last two.
// NeuralNetwork only supports padding on last two dims [H,W], but MIL pad op has no such restriction.
TEST(CoreMLExecutionProviderTest, PadAllDimsMLProgram) {
  // Build a model: output = Pad(X, pads, mode="constant")
  // Input shape: {2,3,4}, pad on dim 0: start=1, end=1 -> output shape: {4,3,4}
  onnxruntime::Model model("pad_all_dims_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  // Input X: {2,3,4} float tensor
  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type.mutable_tensor_type()->mutable_shape();
  input_shape->add_dim()->set_dim_value(2);
  input_shape->add_dim()->set_dim_value(3);
  input_shape->add_dim()->set_dim_value(4);

  // Output Y: {4,3,4} float tensor
  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  output_shape->add_dim()->set_dim_value(4);
  output_shape->add_dim()->set_dim_value(3);
  output_shape->add_dim()->set_dim_value(4);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  // Pads initializer: [1,0,0, 1,0,0] — pad dim 0 by (1,1), no padding on dims 1,2
  ONNX_NAMESPACE::TensorProto pads_init;
  pads_init.set_name("pads");
  pads_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  pads_init.add_dims(6);
  const std::vector<int64_t> pads_data = {1, 0, 0, 1, 0, 0};
  for (auto v : pads_data) {
    pads_init.add_int64_data(v);
  }
  graph.AddInitializedTensor(pads_init);

  // constant_value initializer: 0.0
  ONNX_NAMESPACE::TensorProto constant_value_init;
  constant_value_init.set_name("constant_value");
  constant_value_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  constant_value_init.add_float_data(0.0f);
  graph.AddInitializedTensor(constant_value_init);

  auto& pads_arg = graph.GetOrCreateNodeArg("pads", nullptr);
  auto& constant_value_arg = graph.GetOrCreateNodeArg("constant_value", nullptr);

  auto& pad_node = graph.AddNode("pad_all_dims", "Pad", "constant pad on dim 0",
                                 {&input_arg, &pads_arg, &constant_value_arg}, {&output_arg});
  pad_node.AddAttribute("mode", "constant");

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  // Input data for {2,3,4}
  std::vector<int64_t> dims = {2, 3, 4};
  std::vector<float> input_data(24);
  std::iota(input_data.begin(), input_data.end(), 1.0f);
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "PadAllDimsMLProgram",
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
#else
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

TEST(CoreMLExecutionProviderTest, Pad1DMLProgram) {
  // Build a model: output = Pad(X, pads, mode="constant")
  // 1D input shape: {5}, pad start=2, end=3 -> output shape: {10}
  onnxruntime::Model model("pad_1d_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  // Input X: {5} float tensor
  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type.mutable_tensor_type()->mutable_shape();
  input_shape->add_dim()->set_dim_value(5);

  // Output Y: {10} float tensor
  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  output_shape->add_dim()->set_dim_value(10);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  // Pads initializer: [2, 3] — pad dim 0 by (2,3)
  ONNX_NAMESPACE::TensorProto pads_init;
  pads_init.set_name("pads");
  pads_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  pads_init.add_dims(2);
  const std::vector<int64_t> pads_data = {2, 3};
  for (auto v : pads_data) {
    pads_init.add_int64_data(v);
  }
  graph.AddInitializedTensor(pads_init);

  // constant_value initializer: 0.0
  ONNX_NAMESPACE::TensorProto constant_value_init;
  constant_value_init.set_name("constant_value");
  constant_value_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  constant_value_init.add_float_data(0.0f);
  graph.AddInitializedTensor(constant_value_init);

  auto& pads_arg = graph.GetOrCreateNodeArg("pads", nullptr);
  auto& constant_value_arg = graph.GetOrCreateNodeArg("constant_value", nullptr);

  auto& pad_node = graph.AddNode("pad_1d", "Pad", "constant pad on 1D input",
                                 {&input_arg, &pads_arg, &constant_value_arg}, {&output_arg});
  pad_node.AddAttribute("mode", "constant");

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  // Input data for {5}
  std::vector<int64_t> dims = {5};
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "Pad1DMLProgram",
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
#else
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

TEST(CoreMLExecutionProviderTest, HardSigmoidTest) {
  // Single-node HardSigmoid model; verify it is claimed entirely by the
  // CoreML EP in both NeuralNetwork and MLProgram formats, and that the
  // output matches the CPU EP reference.
  onnxruntime::Model model("hard_sigmoid_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type.mutable_tensor_type()->mutable_shape();
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(3);
  input_shape->add_dim()->set_dim_value(2);
  input_shape->add_dim()->set_dim_value(4);

  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  output_shape->add_dim()->set_dim_value(1);
  output_shape->add_dim()->set_dim_value(3);
  output_shape->add_dim()->set_dim_value(2);
  output_shape->add_dim()->set_dim_value(4);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  auto& node = graph.AddNode("hard_sigmoid", "HardSigmoid", "HardSigmoid with non-default alpha/beta",
                             {&input_arg}, {&output_arg});
  // Use non-default values so the test catches any attribute-wiring bug.
  node.AddAttribute("alpha", 0.1f);
  node.AddAttribute("beta", 0.6f);

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  // Inputs span the three HardSigmoid regions (saturated-low, linear, saturated-high)
  // for alpha=0.1, beta=0.6: values < -6 clamp to 0, values > 4 clamp to 1, others are linear.
  std::vector<int64_t> dims = {1, 3, 2, 4};
  std::vector<float> input_data = {-10.0f, -7.0f, -6.0f, -5.0f, -3.0f, -1.0f, 0.0f, 1.0f,
                                   2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 10.0f, 0.5f, -0.5f,
                                   -4.0f, -2.0f, 1.5f, 2.5f, -1.5f, 3.5f, -3.5f, 4.5f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "HardSigmoidTest_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "HardSigmoidTest_MLProgram",
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
#else
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::All);
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}
#endif  // !(ORT_MINIMAL_BUILD)
}  // namespace test
}  // namespace onnxruntime
