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
#include "core/optimizer/graph_transformer_level.h"
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
  // The CoreML EP supports scalar 'indices' for Gather only when the 'data' input has a fully
  // static shape (it needs to claim a static intermediate shape for the post-gather squeeze).
  // This model's 'data' input is dynamic ([M, N, K]) so Gather still falls back to CPU and the
  // CoreML EP only takes the Shape node.
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

// GatherND on the ML Program path is only claimed when 'indices' is a constant initializer
// (see GatherNDOpBuilder::IsOpSupportedImpl -- CoreML's gather_nd miscomputes some shapes with a
// runtime indices input). This is the supported path: a multi-dimensional slice gather (index depth 1
// on rank-3 data) with constant indices must run on CoreML and match the CPU result.
TEST(CoreMLExecutionProviderTest, GatherNDConstantIndicesMLProgram) {
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}};
  onnxruntime::Model model("gnd_const", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto make_type = [](int32_t et, std::vector<int64_t> dims) {
    ONNX_NAMESPACE::TypeProto t;
    t.mutable_tensor_type()->set_elem_type(et);
    for (auto d : dims) t.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(d);
    return t;
  };
  const auto data_t = make_type(ONNX_NAMESPACE::TensorProto_DataType_INT64, {2, 2, 2});
  const auto out_t = make_type(ONNX_NAMESPACE::TensorProto_DataType_INT64, {2, 1, 2, 2});
  auto& data = graph.GetOrCreateNodeArg("data", &data_t);
  auto& out = graph.GetOrCreateNodeArg("Y", &out_t);
  ONNX_NAMESPACE::TensorProto idx;
  idx.set_name("indices");
  idx.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  idx.add_dims(2);
  idx.add_dims(1);
  idx.add_dims(1);
  idx.add_int64_data(1);
  idx.add_int64_data(0);
  graph.AddInitializedTensor(idx);
  auto& idx_arg = graph.GetOrCreateNodeArg("indices", nullptr);
  graph.AddNode("gnd", "GatherND", "", {&data, &idx_arg}, {&out});
  ORT_THROW_IF_ERROR(graph.Resolve());
  std::string md;
  model.ToProto().SerializeToString(&md);
  gsl::span<const std::byte> span{reinterpret_cast<const std::byte*>(md.data()), md.size()};
#if defined(__APPLE__)
  std::vector<int64_t> dims = {2, 2, 2};
  std::vector<int64_t> vals = {0, 1, 2, 3, 4, 5, 6, 7};
  OrtValue dv;
  CreateMLValue<int64_t>(CPUAllocator::DefaultInstance(), dims, vals, &dv);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("data", dv));
  RunAndVerifyOutputsWithEP(span, CurrentTestName(),
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
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

// Verify that Gemm with external data weight/bias works with CoreML EP.
// This exercises the GetTensorDataTransposed and bias unpacking paths in gemm_op_builder.cc
// which previously failed with "model_path must not be empty" for external data tensors.
TEST(CoreMLExecutionProviderTest, ExternalDataGemm) {
  TemporaryDirectory tmp_dir(ORT_TSTR("coreml_external_data_gemm_test"));
  const auto model_path = std::filesystem::path(tmp_dir.Path()) / ORT_TSTR("model.onnx");
  const auto external_data_path = std::filesystem::path(tmp_dir.Path()) / ORT_TSTR("model.onnx_data");

  // Gemm: Y = X * W + B, where X is {2,3}, W is {3,4}, B is {4}
  // Weight W: 3*4 = 12 floats, Bias B: 4 floats -> 16 floats total in external data
  const std::vector<float> weight_data = {0.1f, 0.2f, 0.3f, 0.4f,
                                          0.5f, 0.6f, 0.7f, 0.8f,
                                          0.9f, 1.0f, 1.1f, 1.2f};
  const std::vector<float> bias_data = {0.01f, 0.02f, 0.03f, 0.04f};

  // Write external data file: weight followed by bias
  {
    std::ofstream ofs(external_data_path, std::ios::binary);
    ASSERT_TRUE(ofs.is_open());
    ofs.write(reinterpret_cast<const char*>(weight_data.data()),
              weight_data.size() * sizeof(float));
    ofs.write(reinterpret_cast<const char*>(bias_data.data()),
              bias_data.size() * sizeof(float));
    ofs.close();
  }

  const size_t weight_byte_size = weight_data.size() * sizeof(float);
  const size_t bias_byte_size = bias_data.size() * sizeof(float);

  // Build model with Gemm op
  {
    ONNX_NAMESPACE::ModelProto model_proto;
    model_proto.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
    auto* opset = model_proto.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    auto* graph_proto = model_proto.mutable_graph();
    graph_proto->set_name("test_external_data_gemm");

    // Input X: {2,3} float tensor
    auto* input = graph_proto->add_input();
    input->set_name("X");
    auto* input_type = input->mutable_type()->mutable_tensor_type();
    input_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* input_shape = input_type->mutable_shape();
    input_shape->add_dim()->set_dim_value(2);
    input_shape->add_dim()->set_dim_value(3);

    // Output Y: {2,4} float tensor
    auto* output = graph_proto->add_output();
    output->set_name("Y");
    auto* output_type = output->mutable_type()->mutable_tensor_type();
    output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* output_shape = output_type->mutable_shape();
    output_shape->add_dim()->set_dim_value(2);
    output_shape->add_dim()->set_dim_value(4);

    // Initializer W {3,4} with external data
    auto* w_init = graph_proto->add_initializer();
    w_init->set_name("W");
    w_init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    w_init->add_dims(3);
    w_init->add_dims(4);
    w_init->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
    {
      auto* ext = w_init->add_external_data();
      ext->set_key("location");
      ext->set_value("model.onnx_data");
      ext = w_init->add_external_data();
      ext->set_key("offset");
      ext->set_value("0");
      ext = w_init->add_external_data();
      ext->set_key("length");
      ext->set_value(std::to_string(weight_byte_size));
    }

    // Initializer B {4} with external data (offset after W)
    auto* b_init = graph_proto->add_initializer();
    b_init->set_name("B");
    b_init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    b_init->add_dims(4);
    b_init->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
    {
      auto* ext = b_init->add_external_data();
      ext->set_key("location");
      ext->set_value("model.onnx_data");
      ext = b_init->add_external_data();
      ext->set_key("offset");
      ext->set_value(std::to_string(weight_byte_size));
      ext = b_init->add_external_data();
      ext->set_key("length");
      ext->set_value(std::to_string(bias_byte_size));
    }

    // Gemm node: Y = X * W + B (transB=0 by default, so W is {K,N} = {3,4})
    auto* node = graph_proto->add_node();
    node->set_op_type("Gemm");
    node->add_input("X");
    node->add_input("W");
    node->add_input("B");
    node->add_output("Y");

    // Save model
    std::ofstream ofs(model_path, std::ios::binary);
    ASSERT_TRUE(ofs.is_open());
    ASSERT_TRUE(model_proto.SerializeToOstream(&ofs));
    ofs.close();
  }

  // Input data: {2,3}
  std::vector<int64_t> dims = {2, 3};
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  RunOptions run_options;
  run_options.run_tag = "ExternalDataGemm";
  std::vector<std::string> output_names = {"Y"};

  SessionOptions so;
  so.session_logid = "ExternalDataGemm";
  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(MakeCoreMLExecutionProvider()));
  ASSERT_STATUS_OK(session.Load(model_path.native()));
  ASSERT_STATUS_OK(session.Initialize());

#if defined(__APPLE__)
  const auto& provider_types = session.GetRegisteredProviderTypes();
  EXPECT_NE(std::find(provider_types.begin(), provider_types.end(), kCoreMLExecutionProvider), provider_types.end());
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session.Run(run_options, feeds, output_names, &fetches));

  // Compute expected: Y = X * W + B
  // Row 0: [1*0.1+2*0.5+3*0.9+0.01, 1*0.2+2*0.6+3*1.0+0.02, 1*0.3+2*0.7+3*1.1+0.03, 1*0.4+2*0.8+3*1.2+0.04]
  //       = [3.81, 4.42, 5.03, 5.64]
  // Row 1: [4*0.1+5*0.5+6*0.9+0.01, 4*0.2+5*0.6+6*1.0+0.02, 4*0.3+5*0.7+6*1.1+0.03, 4*0.4+5*0.8+6*1.2+0.04]
  //       = [8.31, 9.82, 11.33, 12.84]
  const std::vector<float> expected = {3.81f, 4.42f, 5.03f, 5.64f, 8.31f, 9.82f, 11.33f, 12.84f};

  ASSERT_EQ(fetches.size(), 1u);
  const auto& output_tensor = fetches[0].Get<Tensor>();
  auto output_data = output_tensor.DataAsSpan<float>();
  ASSERT_EQ(static_cast<size_t>(output_data.size()), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5f) << "Mismatch at index " << i;
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

TEST(CoreMLExecutionProviderTest, QuickGeluTest) {
  // Single com.microsoft:QuickGelu node (produced by ORT's QuickGeluFusion pass
  // from the pattern x * sigmoid(alpha * x)). Verify the CoreML MLProgram path
  // claims the whole graph and produces the same output as CPU.
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* onnx_opset = model_proto.add_opset_import();
  onnx_opset->set_domain("");
  onnx_opset->set_version(13);
  auto* ms_opset = model_proto.add_opset_import();
  ms_opset->set_domain("com.microsoft");
  ms_opset->set_version(1);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("quick_gelu_test");

  auto* input = graph_proto->add_input();
  input->set_name("X");
  auto* input_shape = input->mutable_type()->mutable_tensor_type()->mutable_shape();
  input->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(3);
  input_shape->add_dim()->set_dim_value(2);
  input_shape->add_dim()->set_dim_value(4);

  auto* output = graph_proto->add_output();
  output->set_name("Y");
  auto* output_shape = output->mutable_type()->mutable_tensor_type()->mutable_shape();
  output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  output_shape->add_dim()->set_dim_value(1);
  output_shape->add_dim()->set_dim_value(3);
  output_shape->add_dim()->set_dim_value(2);
  output_shape->add_dim()->set_dim_value(4);

  auto* node = graph_proto->add_node();
  node->set_op_type("QuickGelu");
  node->set_domain("com.microsoft");
  node->add_input("X");
  node->add_output("Y");
  // Use a non-default alpha so the test catches any attribute-wiring bug.
  auto* alpha_attr = node->add_attribute();
  alpha_attr->set_name("alpha");
  alpha_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
  alpha_attr->set_f(1.5f);

  std::string model_data;
  ASSERT_TRUE(model_proto.SerializeToString(&model_data));
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 3, 2, 4};
  std::vector<float> input_data = {-10.0f, -3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f,
                                   10.0f, -5.0f, 5.0f, 2.0f, -2.0f, 4.0f, -4.0f, 0.25f,
                                   -0.25f, 7.0f, -7.0f, 1.5f, -1.5f, 0.1f, -0.1f, 20.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  RunAndVerifyOutputsWithEP(model_span, "QuickGeluTest_MLProgram",
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
#else
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

TEST(CoreMLExecutionProviderTest, QuickGeluTestAlphaOne) {
  // alpha == 1.0 triggers the "skip leading mul" optimization in the op
  // builder. Verify correctness on that branch — the emitted MIL graph is
  // sigmoid(x) -> mul(x, sigmoid(x)) instead of the 3-op decomposition.
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* onnx_opset = model_proto.add_opset_import();
  onnx_opset->set_domain("");
  onnx_opset->set_version(13);
  auto* ms_opset = model_proto.add_opset_import();
  ms_opset->set_domain("com.microsoft");
  ms_opset->set_version(1);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("quick_gelu_alpha_one_test");

  auto* input = graph_proto->add_input();
  input->set_name("X");
  auto* input_shape = input->mutable_type()->mutable_tensor_type()->mutable_shape();
  input->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(3);
  input_shape->add_dim()->set_dim_value(2);
  input_shape->add_dim()->set_dim_value(4);

  auto* output = graph_proto->add_output();
  output->set_name("Y");
  auto* output_shape = output->mutable_type()->mutable_tensor_type()->mutable_shape();
  output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  output_shape->add_dim()->set_dim_value(1);
  output_shape->add_dim()->set_dim_value(3);
  output_shape->add_dim()->set_dim_value(2);
  output_shape->add_dim()->set_dim_value(4);

  auto* node = graph_proto->add_node();
  node->set_op_type("QuickGelu");
  node->set_domain("com.microsoft");
  node->add_input("X");
  node->add_output("Y");
  auto* alpha_attr = node->add_attribute();
  alpha_attr->set_name("alpha");
  alpha_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
  alpha_attr->set_f(1.0f);

  std::string model_data;
  ASSERT_TRUE(model_proto.SerializeToString(&model_data));
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 3, 2, 4};
  std::vector<float> input_data = {-10.0f, -3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f,
                                   10.0f, -5.0f, 5.0f, 2.0f, -2.0f, 4.0f, -4.0f, 0.25f,
                                   -0.25f, 7.0f, -7.0f, 1.5f, -1.5f, 0.1f, -0.1f, 20.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  RunAndVerifyOutputsWithEP(model_span, "QuickGeluTestAlphaOne_MLProgram",
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
#else
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

TEST(CoreMLExecutionProviderTest, QuickGeluTestFp16) {
  // FLOAT16 variant of QuickGeluTest. Exercises the MLFloat16 branch of the
  // alpha-scalar wiring in QuickGeluOpBuilder::AddToModelBuilderImpl.
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* onnx_opset = model_proto.add_opset_import();
  onnx_opset->set_domain("");
  onnx_opset->set_version(13);
  auto* ms_opset = model_proto.add_opset_import();
  ms_opset->set_domain("com.microsoft");
  ms_opset->set_version(1);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("quick_gelu_fp16_test");

  auto* input = graph_proto->add_input();
  input->set_name("X");
  auto* input_shape = input->mutable_type()->mutable_tensor_type()->mutable_shape();
  input->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(3);
  input_shape->add_dim()->set_dim_value(2);
  input_shape->add_dim()->set_dim_value(4);

  auto* output = graph_proto->add_output();
  output->set_name("Y");
  auto* output_shape = output->mutable_type()->mutable_tensor_type()->mutable_shape();
  output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  output_shape->add_dim()->set_dim_value(1);
  output_shape->add_dim()->set_dim_value(3);
  output_shape->add_dim()->set_dim_value(2);
  output_shape->add_dim()->set_dim_value(4);

  auto* node = graph_proto->add_node();
  node->set_op_type("QuickGelu");
  node->set_domain("com.microsoft");
  node->add_input("X");
  node->add_output("Y");
  auto* alpha_attr = node->add_attribute();
  alpha_attr->set_name("alpha");
  alpha_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
  alpha_attr->set_f(1.5f);

  std::string model_data;
  ASSERT_TRUE(model_proto.SerializeToString(&model_data));
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 3, 2, 4};
  const std::vector<float> input_floats = {-10.0f, -3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f,
                                           10.0f, -5.0f, 5.0f, 2.0f, -2.0f, 4.0f, -4.0f, 0.25f,
                                           -0.25f, 7.0f, -7.0f, 1.5f, -1.5f, 0.1f, -0.1f, 20.0f};
  std::vector<MLFloat16> input_data;
  input_data.reserve(input_floats.size());
  for (float f : input_floats) input_data.emplace_back(f);

  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<MLFloat16>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  EPVerificationParams params{};
  params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  // fp16 accumulates larger absolute error than fp32 across the three-op
  // decomposition (mul, sigmoid, mul). Outputs are bounded by |x|, max ~20 in
  // this test; fp16 ulp at that magnitude is ~0.01, so 2e-2 leaves headroom.
  params.fp32_abs_err = 2e-2f;

  RunAndVerifyOutputsWithEP(model_span, "QuickGeluTestFp16_MLProgram",
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds, params);
#else
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

// Build a model: input -> Conv -> <op_chain...> -> output. The Conv anchors
// the partition so the trivial-partition heuristic keeps it; the chained ops
// land inside a single CoreML partition rather than fragmenting it.
namespace {
ONNX_NAMESPACE::ModelProto MakeConvWithTrivialChainModel(
    const std::string& trivial_op,
    bool tile_with_repeats /*for Tile only*/) {
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("conv_chain_test");

  auto add_value = [&](auto* proto, const char* name, const std::vector<int64_t>& shape) {
    proto->set_name(name);
    auto* tt = proto->mutable_type()->mutable_tensor_type();
    tt->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (int64_t d : shape) tt->mutable_shape()->add_dim()->set_dim_value(d);
  };
  const std::vector<int64_t> tile_reps =
      (trivial_op == "Tile" && tile_with_repeats) ? std::vector<int64_t>{1, 1, 2, 2}
                                                  : std::vector<int64_t>{1, 1, 1, 1};
  const std::vector<int64_t> output_shape = {1, 3, 3 * tile_reps[2], 3 * tile_reps[3]};
  add_value(graph_proto->add_input(), "X", {1, 2, 4, 4});
  add_value(graph_proto->add_output(), "Y", output_shape);

  // Conv weight initialiser
  auto* w = graph_proto->add_initializer();
  w->set_name("W");
  w->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (int64_t d : {3, 2, 2, 2}) w->add_dims(d);
  for (int i = 0; i < 24; ++i) w->add_float_data(0.05f * i - 0.4f);

  auto* conv = graph_proto->add_node();
  conv->set_op_type("Conv");
  conv->add_input("X");
  conv->add_input("W");
  conv->add_output("conv_out");
  auto* pads = conv->add_attribute();
  pads->set_name("pads");
  pads->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
  for (int64_t v : {0, 0, 0, 0}) pads->add_ints(v);

  if (trivial_op == "Tile") {
    auto* reps_init = graph_proto->add_initializer();
    reps_init->set_name("reps");
    reps_init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    reps_init->add_dims(static_cast<int64_t>(tile_reps.size()));
    for (int64_t v : tile_reps) reps_init->add_int64_data(v);
    auto* node = graph_proto->add_node();
    node->set_op_type("Tile");
    node->add_input("conv_out");
    node->add_input("reps");
    node->add_output("Y");
  } else {
    auto* node = graph_proto->add_node();
    node->set_op_type(trivial_op);
    node->add_input("conv_out");
    node->add_output("Y");
  }
  return model_proto;
}

void RunConvChainTest(const std::string& trivial_op, std::string_view log_id,
                      bool tile_with_repeats = false) {
  auto model_proto = MakeConvWithTrivialChainModel(trivial_op, tile_with_repeats);
  std::string model_data;
  ASSERT_TRUE(model_proto.SerializeToString(&model_data));
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 2, 4, 4};
  std::vector<float> x_data(32);
  for (size_t i = 0; i < x_data.size(); ++i) x_data[i] = static_cast<float>(i) * 0.1f - 1.5f;
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, x_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  RunAndVerifyOutputsWithEP(model_span, std::string(log_id),
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
#else
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}
}  // namespace

TEST(CoreMLExecutionProviderTest, IdentityWithConvAnchor) {
  // Conv → Identity → output. Conv anchors the partition; Identity must be
  // claimed (the trivial-partition heuristic keeps it because Conv is present).
  RunConvChainTest("Identity", "IdentityWithConvAnchor_MLProgram");
}

TEST(CoreMLExecutionProviderTest, CeilWithConvAnchor) {
  // Conv → Ceil → output. Same rationale; Ceil is also a unary MIL op.
  RunConvChainTest("Ceil", "CeilWithConvAnchor_MLProgram");
}

TEST(CoreMLExecutionProviderTest, TileWithConvAnchor) {
  // Conv → Tile(reps=[1,1,1,1]) → output. Validates the Tile builder claims
  // the node alongside the Conv anchor.
  RunConvChainTest("Tile", "TileWithConvAnchor_MLProgram");
}

TEST(CoreMLExecutionProviderTest, TileWithConvAnchorNonUnitRepeats) {
  // Conv → Tile(reps=[1,1,2,2]) → output. Exercises the non-trivial tile path
  // (output spatial dims doubled) end-to-end against the CPU reference.
  RunConvChainTest("Tile", "TileWithConvAnchorNonUnitRepeats_MLProgram",
                   /*tile_with_repeats=*/true);
}

// Helper for trivial-only chain tests. Builds a model with input X[dims] and
// output Y[dims], populates the graph body via `populate_chain`, and asserts
// the CoreML EP claims none of it. Graph optimisations are pinned to Default
// so passes like IdentityElimination / CastElimination do not pre-empt the
// trivial-partition heuristic in CoreMLExecutionProvider::GetCapability.
namespace {
void RunTrivialOnlyChainTest(
    std::string_view log_id,
    const std::vector<int64_t>& dims,
    const std::vector<float>& x_data,
    const std::function<void(ONNX_NAMESPACE::GraphProto*)>& populate_chain) {
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);
  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("trivial_only");

  auto add_value = [&](auto* proto, const char* name, const std::vector<int64_t>& shape) {
    proto->set_name(name);
    auto* tt = proto->mutable_type()->mutable_tensor_type();
    tt->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (int64_t d : shape) tt->mutable_shape()->add_dim()->set_dim_value(d);
  };
  add_value(graph_proto->add_input(), "X", dims);
  add_value(graph_proto->add_output(), "Y", dims);

  populate_chain(graph_proto);

  std::string model_data;
  ASSERT_TRUE(model_proto.SerializeToString(&model_data));
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

#if defined(__APPLE__)
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, x_data, &ml_value_x);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  auto disable_optimizations = [](SessionOptions& so) {
    so.graph_optimization_level = TransformerLevel::Default;
  };

  RunAndVerifyOutputsWithEP(model_span, std::string(log_id),
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::None},
                            disable_optimizations);
#else
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::None);
#endif
}
}  // namespace

TEST(CoreMLExecutionProviderTest, TrivialOnlyChainIsNotClaimedByCoreML) {
  // 3 chained Identity nodes with no compute-heavy anchor → heuristic drops the
  // partition so CPU runs it. Round-trip cost would exceed the saving otherwise.
  RunTrivialOnlyChainTest(
      "TrivialOnlyChainIsNotClaimedByCoreML_MLProgram",
      {1, 8},
      {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f},
      [](ONNX_NAMESPACE::GraphProto* graph) {
        auto* n1 = graph->add_node();
        n1->set_op_type("Identity");
        n1->add_input("X");
        n1->add_output("a");
        auto* n2 = graph->add_node();
        n2->set_op_type("Identity");
        n2->add_input("a");
        n2->add_output("b");
        auto* n3 = graph->add_node();
        n3->set_op_type("Identity");
        n3->add_input("b");
        n3->add_output("Y");
      });
}

TEST(CoreMLExecutionProviderTest, ReshapeOnlyChainIsNotClaimedByCoreML) {
  RunTrivialOnlyChainTest(
      "ReshapeOnlyChainIsNotClaimedByCoreML_MLProgram",
      {1, 8},
      {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f},
      [](ONNX_NAMESPACE::GraphProto* graph) {
        auto add_shape_init = [&](const char* name, const std::vector<int64_t>& shape) {
          auto* init = graph->add_initializer();
          init->set_name(name);
          init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
          init->add_dims(static_cast<int64_t>(shape.size()));
          for (int64_t v : shape) init->add_int64_data(v);
        };
        add_shape_init("shape_a", {2, 4});
        add_shape_init("shape_b", {1, 8});

        auto* n1 = graph->add_node();
        n1->set_op_type("Reshape");
        n1->add_input("X");
        n1->add_input("shape_a");
        n1->add_output("a");
        auto* n2 = graph->add_node();
        n2->set_op_type("Reshape");
        n2->add_input("a");
        n2->add_input("shape_b");
        n2->add_output("Y");
      });
}

TEST(CoreMLExecutionProviderTest, TransposeOnlyChainIsNotClaimedByCoreML) {
  RunTrivialOnlyChainTest(
      "TransposeOnlyChainIsNotClaimedByCoreML_MLProgram",
      {1, 8},
      {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f},
      [](ONNX_NAMESPACE::GraphProto* graph) {
        auto add_transpose = [&](const char* name, const char* in, const char* out,
                                 const std::vector<int64_t>& perm) {
          auto* node = graph->add_node();
          node->set_name(name);
          node->set_op_type("Transpose");
          node->add_input(in);
          node->add_output(out);
          auto* attr = node->add_attribute();
          attr->set_name("perm");
          attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
          for (int64_t v : perm) attr->add_ints(v);
        };
        // Two Transposes that compose back to the identity perm.
        add_transpose("t0", "X", "a", {1, 0});
        add_transpose("t1", "a", "Y", {1, 0});
      });
}

TEST(CoreMLExecutionProviderTest, TileOnlyIsNotClaimedByCoreML) {
  // Single Tile with reps=[1,1] — pure data movement, no compute anchor.
  RunTrivialOnlyChainTest(
      "TileOnlyIsNotClaimedByCoreML_MLProgram",
      {1, 8},
      {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f},
      [](ONNX_NAMESPACE::GraphProto* graph) {
        auto* reps = graph->add_initializer();
        reps->set_name("reps");
        reps->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
        reps->add_dims(2);
        reps->add_int64_data(1);
        reps->add_int64_data(1);
        auto* n = graph->add_node();
        n->set_op_type("Tile");
        n->add_input("X");
        n->add_input("reps");
        n->add_output("Y");
      });
}

TEST(CoreMLExecutionProviderTest, CeilOnlyIsNotClaimedByCoreML) {
  // Single Ceil — supported by the new Unary builder but trivial; heuristic drops it.
  RunTrivialOnlyChainTest(
      "CeilOnlyIsNotClaimedByCoreML_MLProgram",
      {1, 8},
      {0.1f, 0.6f, 1.4f, 1.9f, -0.6f, -1.4f, 2.5f, 3.1f},
      [](ONNX_NAMESPACE::GraphProto* graph) {
        auto* n = graph->add_node();
        n->set_op_type("Ceil");
        n->add_input("X");
        n->add_output("Y");
      });
}

TEST(CoreMLExecutionProviderTest, MixedTrivialChainIsNotClaimedByCoreML) {
  // Identity → Cast(float→float) → Reshape → Transpose. Different trivial ops in
  // sequence; with no compute-heavy anchor the heuristic drops the whole partition.
  RunTrivialOnlyChainTest(
      "MixedTrivialChainIsNotClaimedByCoreML_MLProgram",
      {1, 8},
      {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f},
      [](ONNX_NAMESPACE::GraphProto* graph) {
        auto* shape_init = graph->add_initializer();
        shape_init->set_name("reshape_shape");
        shape_init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
        shape_init->add_dims(2);
        shape_init->add_int64_data(8);
        shape_init->add_int64_data(1);

        auto* identity = graph->add_node();
        identity->set_op_type("Identity");
        identity->add_input("X");
        identity->add_output("a");

        auto* cast = graph->add_node();
        cast->set_op_type("Cast");
        cast->add_input("a");
        cast->add_output("b");
        auto* to_attr = cast->add_attribute();
        to_attr->set_name("to");
        to_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
        to_attr->set_i(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

        auto* reshape = graph->add_node();
        reshape->set_op_type("Reshape");
        reshape->add_input("b");
        reshape->add_input("reshape_shape");
        reshape->add_output("c");

        auto* transpose = graph->add_node();
        transpose->set_op_type("Transpose");
        transpose->add_input("c");
        transpose->add_output("Y");
        auto* perm_attr = transpose->add_attribute();
        perm_attr->set_name("perm");
        perm_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
        perm_attr->add_ints(1);
        perm_attr->add_ints(0);
      });
}

TEST(CoreMLExecutionProviderTest, ConvTrivialChainConvKeepsAllOnCoreML) {
  // Sandwich test: Conv → Identity → Cast → Reshape → Conv. The two Convs
  // make the partition non-trivial, so the heuristic keeps the trivial ops in
  // the same partition rather than splitting them off to CPU. Verifies the
  // "stay on GPU for GPU chains" half of the heuristic.
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);
  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("conv_trivial_conv_sandwich");

  auto add_value = [&](auto* proto, const char* name, const std::vector<int64_t>& shape) {
    proto->set_name(name);
    auto* tt = proto->mutable_type()->mutable_tensor_type();
    tt->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (int64_t d : shape) tt->mutable_shape()->add_dim()->set_dim_value(d);
  };
  add_value(graph_proto->add_input(), "X", {1, 2, 4, 4});
  add_value(graph_proto->add_output(), "Y", {1, 2, 3, 3});

  // Conv1: weight [3, 2, 2, 2], output [1, 3, 3, 3]
  auto* w1 = graph_proto->add_initializer();
  w1->set_name("W1");
  w1->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (int64_t d : {3, 2, 2, 2}) w1->add_dims(d);
  for (int i = 0; i < 24; ++i) w1->add_float_data(0.05f * i - 0.4f);

  // Conv2: weight [2, 3, 1, 1], output [1, 2, 3, 3]
  auto* w2 = graph_proto->add_initializer();
  w2->set_name("W2");
  w2->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (int64_t d : {2, 3, 1, 1}) w2->add_dims(d);
  for (int i = 0; i < 6; ++i) w2->add_float_data(0.1f * i - 0.25f);

  // Reshape shape initializer (no-op reshape: [1,3,3,3] → [1,3,3,3])
  auto* reshape_shape = graph_proto->add_initializer();
  reshape_shape->set_name("reshape_shape");
  reshape_shape->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  reshape_shape->add_dims(4);
  for (int64_t v : {1, 3, 3, 3}) reshape_shape->add_int64_data(v);

  auto add_pads_attr = [](ONNX_NAMESPACE::NodeProto* node) {
    auto* pads = node->add_attribute();
    pads->set_name("pads");
    pads->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
    for (int64_t v : {0, 0, 0, 0}) pads->add_ints(v);
  };

  auto* conv1 = graph_proto->add_node();
  conv1->set_op_type("Conv");
  conv1->add_input("X");
  conv1->add_input("W1");
  conv1->add_output("conv1_out");
  add_pads_attr(conv1);

  auto* identity = graph_proto->add_node();
  identity->set_op_type("Identity");
  identity->add_input("conv1_out");
  identity->add_output("ident_out");

  auto* cast = graph_proto->add_node();
  cast->set_op_type("Cast");
  cast->add_input("ident_out");
  cast->add_output("cast_out");
  auto* to_attr = cast->add_attribute();
  to_attr->set_name("to");
  to_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  to_attr->set_i(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto* reshape = graph_proto->add_node();
  reshape->set_op_type("Reshape");
  reshape->add_input("cast_out");
  reshape->add_input("reshape_shape");
  reshape->add_output("reshape_out");

  auto* conv2 = graph_proto->add_node();
  conv2->set_op_type("Conv");
  conv2->add_input("reshape_out");
  conv2->add_input("W2");
  conv2->add_output("Y");
  add_pads_attr(conv2);

  std::string model_data;
  ASSERT_TRUE(model_proto.SerializeToString(&model_data));
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 2, 4, 4};
  std::vector<float> x_data(32);
  for (size_t i = 0; i < x_data.size(); ++i) x_data[i] = static_cast<float>(i) * 0.1f - 1.5f;
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, x_data, &ml_value_x);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  // Disable optimisations so the trivial ops survive into partitioning and we
  // actually verify the heuristic (otherwise IdentityElimination / similar
  // passes could remove them before CoreML's GetCapability runs).
  auto disable_optimizations = [](SessionOptions& so) {
    so.graph_optimization_level = TransformerLevel::Default;
  };

  RunAndVerifyOutputsWithEP(model_span, "ConvTrivialChainConvKeepsAllOnCoreML_MLProgram",
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All},
                            disable_optimizations);
#else
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

namespace {
// Build a single-node com.microsoft:FusedConv model for the tests below.
// Input X is {1, 2, 4, 4}, weight W is {3, 2, 2, 2} (constant initializer, set
// to a simple pattern), no bias. stride=1, pad=0. Output is {1, 3, 3, 3}.
// When `add_z` is true, the optional 4th 'Z' (residual sum) input is added —
// used by the negative test that exercises CoreML's rejection path.
ONNX_NAMESPACE::ModelProto MakeFusedConvModel(const std::string& activation,
                                              const std::vector<float>& activation_params,
                                              bool add_z = false) {
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* onnx_opset = model_proto.add_opset_import();
  onnx_opset->set_domain("");
  onnx_opset->set_version(13);
  auto* ms_opset = model_proto.add_opset_import();
  ms_opset->set_domain("com.microsoft");
  ms_opset->set_version(1);

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name("fused_conv_test");

  auto add_tensor_value = [&](auto* proto, const char* name, const std::vector<int64_t>& shape) {
    proto->set_name(name);
    auto* tt = proto->mutable_type()->mutable_tensor_type();
    tt->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (int64_t d : shape) tt->mutable_shape()->add_dim()->set_dim_value(d);
  };
  add_tensor_value(graph_proto->add_input(), "X", {1, 2, 4, 4});
  if (add_z) {
    add_tensor_value(graph_proto->add_input(), "Z", {1, 3, 3, 3});
  }
  add_tensor_value(graph_proto->add_output(), "Y", {1, 3, 3, 3});

  // Weight initializer: {3, 2, 2, 2} = 24 floats, deterministic pattern.
  auto* w_init = graph_proto->add_initializer();
  w_init->set_name("W");
  w_init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (int64_t d : {3, 2, 2, 2}) w_init->add_dims(d);
  for (int i = 0; i < 3 * 2 * 2 * 2; ++i) {
    w_init->add_float_data(static_cast<float>(i) * 0.05f - 0.4f);
  }

  auto* node = graph_proto->add_node();
  node->set_op_type("FusedConv");
  node->set_domain("com.microsoft");
  node->add_input("X");
  node->add_input("W");
  if (add_z) {
    // FusedConv schema: X, W, B(optional), Z(optional). Skip B with "" so Z
    // lands in input slot 3.
    node->add_input("");
    node->add_input("Z");
  }
  node->add_output("Y");

  // Set pads explicitly since the CoreML conv builder's VALID-pad branch
  // omits the 'pad' input that the MIL op requires. Conv attrs otherwise
  // default: strides=[1,1].
  auto* pads_attr = node->add_attribute();
  pads_attr->set_name("pads");
  pads_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
  for (int64_t v : {0, 0, 0, 0}) pads_attr->add_ints(v);

  auto* act_attr = node->add_attribute();
  act_attr->set_name("activation");
  act_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  act_attr->set_s(activation);

  if (!activation_params.empty()) {
    auto* act_params_attr = node->add_attribute();
    act_params_attr->set_name("activation_params");
    act_params_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS);
    for (float v : activation_params) act_params_attr->add_floats(v);
  }

  return model_proto;
}

void RunFusedConvNegativeTest(const ONNX_NAMESPACE::ModelProto& model_proto, bool mlprogram) {
  std::string model_data;
  ASSERT_TRUE(model_proto.SerializeToString(&model_data));
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};
  auto provider = mlprogram ? MakeCoreMLExecutionProvider("MLProgram")
                            : MakeCoreMLExecutionProvider();
  TestModelLoad(model_span, std::move(provider), ExpectedEPNodeAssignment::None);
}

void RunFusedConvTest(const std::string& activation,
                      const std::vector<float>& activation_params,
                      std::string_view log_id) {
  auto model_proto = MakeFusedConvModel(activation, activation_params);
  std::string model_data;
  ASSERT_TRUE(model_proto.SerializeToString(&model_data));
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

#if defined(__APPLE__)
  std::vector<float> x_data(1 * 2 * 4 * 4);
  for (size_t i = 0; i < x_data.size(); ++i) x_data[i] = static_cast<float>(i) * 0.1f - 1.5f;
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, {1, 2, 4, 4}, x_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  RunAndVerifyOutputsWithEP(model_span, std::string(log_id),
                            MakeCoreMLExecutionProvider("MLProgram"),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
#else
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}
}  // namespace

TEST(CoreMLExecutionProviderTest, FusedConvTestRelu) {
  // Param-less activation. Exercises the Conv → activation wiring with no
  // `activation_params` attribute.
  RunFusedConvTest("Relu", {}, "FusedConvTestRelu_MLProgram");
}

TEST(CoreMLExecutionProviderTest, FusedConvTestHardSigmoid) {
  // Two-param activation (alpha, beta) with non-default values — catches any
  // activation_params-wiring bug. Depends on the HardSigmoid CoreML builder
  // landed in #28182.
  RunFusedConvTest("HardSigmoid", {0.15f, 0.55f}, "FusedConvTestHardSigmoid_MLProgram");
}

TEST(CoreMLExecutionProviderTest, FusedConvTestClip) {
  // Two-param activation where params map to alpha=min, beta=max in CoreML's
  // clip op. Covers the remaining parametric activation.
  RunFusedConvTest("Clip", {-0.5f, 0.5f}, "FusedConvTestClip_MLProgram");
}

TEST(CoreMLExecutionProviderTest, FusedConvTestLeakyRelu) {
  // Single-param activation (alpha). Heavily used by YOLOv3 — a CPU-optimized
  // YOLOv3 graph contains 72 Conv→LeakyRelu fusions, all of which would
  // otherwise fall back to CPU and fragment the CoreML partition.
  RunFusedConvTest("LeakyRelu", {0.1f}, "FusedConvTestLeakyRelu_MLProgram");
}

TEST(CoreMLExecutionProviderTest, FusedConvTestSigmoid) {
  // Param-less Sigmoid activation. Distinct from the Relu test only in the
  // emitted MIL op (`sigmoid` vs `relu`); guards against regressions in
  // op-name dispatch.
  RunFusedConvTest("Sigmoid", {}, "FusedConvTestSigmoid_MLProgram");
}

TEST(CoreMLExecutionProviderTest, FusedConvTestTanh) {
  // Param-less Tanh activation; same rationale as the Sigmoid test for the
  // remaining elementwise activation.
  RunFusedConvTest("Tanh", {}, "FusedConvTestTanh_MLProgram");
}

// Negative tests below cover the two gating cases that have a working CPU
// fallback (so TestModelLoad's Initialize() succeeds and the EP partition
// assignment can be verified). The arity-mismatch and unsupported-activation
// cases are also rejected by IsOpSupportedImpl, but the CPU FusedConv kernel
// rejects them too, so there's no end-to-end fallback to observe.

TEST(CoreMLExecutionProviderTest, FusedConvNeuralNetworkNotSupported) {
  // FusedConv is only implemented on the MLProgram path. The NeuralNetwork
  // builder must reject it so the node falls back to CPU rather than emit an
  // unfused Conv and silently lose the activation.
  RunFusedConvNegativeTest(MakeFusedConvModel("Relu", {}), /*mlprogram=*/false);
}

TEST(CoreMLExecutionProviderTest, FusedConvWithZInputNotSupported) {
  // The optional Z residual sum input (Y = activation(Conv(X,W,B) + Z)) is
  // not lowered by the MLProgram builder. Accepting such a node would
  // silently drop the residual add and produce wrong results, so it must be
  // rejected and fall back to CPU.
  RunFusedConvNegativeTest(MakeFusedConvModel("Relu", {}, /*add_z=*/true),
                           /*mlprogram=*/true);
}

TEST(CoreMLExecutionProviderTest, Split11UnevenAttribute) {
  // ai.onnx:Split-11 with `split` attribute carrying non-uniform sizes.
  // This is the form used by DWPose (`dw-ll_ucoco_384.onnx`); without
  // attribute support the node falls back to CPU and fragments the CoreML
  // partition.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 11}};
  onnxruntime::Model model("split11_uneven_attribute", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  // Input X: {1, 9} float
  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type.mutable_tensor_type()->mutable_shape();
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(9);

  // Outputs along axis=1 with split=[4, 3, 2]: {1,4}, {1,3}, {1,2}
  auto make_output_type = [](int64_t split_size) {
    ONNX_NAMESPACE::TypeProto t;
    t.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* s = t.mutable_tensor_type()->mutable_shape();
    s->add_dim()->set_dim_value(1);
    s->add_dim()->set_dim_value(split_size);
    return t;
  };
  ONNX_NAMESPACE::TypeProto out0_type = make_output_type(4);
  ONNX_NAMESPACE::TypeProto out1_type = make_output_type(3);
  ONNX_NAMESPACE::TypeProto out2_type = make_output_type(2);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_type);
  auto& out0_arg = graph.GetOrCreateNodeArg("Y0", &out0_type);
  auto& out1_arg = graph.GetOrCreateNodeArg("Y1", &out1_type);
  auto& out2_arg = graph.GetOrCreateNodeArg("Y2", &out2_type);

  auto& node = graph.AddNode("split11_uneven", "Split", "Split-11 with uneven 'split' attribute",
                             {&input_arg}, {&out0_arg, &out1_arg, &out2_arg});
  node.AddAttribute("axis", static_cast<int64_t>(1));
  node.AddAttribute("split", std::vector<int64_t>{4, 3, 2});

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 9};
  std::vector<float> input_data = {0.5f, -1.0f, 2.25f, -3.5f, 4.0f, -0.125f, 7.5f, -8.0f, 0.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "Split11UnevenAttribute_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "Split11UnevenAttribute_MLProgram",
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

TEST(CoreMLExecutionProviderTest, Split11EvenAttribute) {
  // Even sizes via attribute — exercises the split_sizes path with uniform
  // values rather than the fall-through num_splits path.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 11}};
  onnxruntime::Model model("split11_even_attribute", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type.mutable_tensor_type()->mutable_shape();
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(6);

  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  output_shape->add_dim()->set_dim_value(1);
  output_shape->add_dim()->set_dim_value(3);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_type);
  auto& out0_arg = graph.GetOrCreateNodeArg("Y0", &output_type);
  auto& out1_arg = graph.GetOrCreateNodeArg("Y1", &output_type);

  auto& node = graph.AddNode("split11_even", "Split", "Split-11 with even 'split' attribute",
                             {&input_arg}, {&out0_arg, &out1_arg});
  node.AddAttribute("axis", static_cast<int64_t>(1));
  node.AddAttribute("split", std::vector<int64_t>{3, 3});

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 6};
  std::vector<float> input_data = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "Split11EvenAttribute_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "Split11EvenAttribute_MLProgram",
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

TEST(CoreMLExecutionProviderTest, Split11NoAttributeEven) {
  // No `split` attribute, axis size divides evenly: must fall through to the
  // num_splits = num_outputs branch.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 11}};
  onnxruntime::Model model("split11_no_attribute_even", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type.mutable_tensor_type()->mutable_shape();
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(8);

  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  output_shape->add_dim()->set_dim_value(1);
  output_shape->add_dim()->set_dim_value(4);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_type);
  auto& out0_arg = graph.GetOrCreateNodeArg("Y0", &output_type);
  auto& out1_arg = graph.GetOrCreateNodeArg("Y1", &output_type);

  auto& node = graph.AddNode("split11_no_attr", "Split", "Split-11 with no 'split' attribute",
                             {&input_arg}, {&out0_arg, &out1_arg});
  node.AddAttribute("axis", static_cast<int64_t>(1));

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 8};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "Split11NoAttributeEven_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "Split11NoAttributeEven_MLProgram",
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

TEST(CoreMLExecutionProviderTest, Split13UnevenInput) {
  // Parity with Split11UnevenAttribute: same shapes and split sizes, but using
  // the opset-13 input form ('split' as a constant initializer) instead of the
  // pre-13 attribute form. Locks in that the existing input path still works.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}};
  onnxruntime::Model model("split13_uneven_input", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type.mutable_tensor_type()->mutable_shape();
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(9);

  auto make_output_type = [](int64_t split_size) {
    ONNX_NAMESPACE::TypeProto t;
    t.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* s = t.mutable_tensor_type()->mutable_shape();
    s->add_dim()->set_dim_value(1);
    s->add_dim()->set_dim_value(split_size);
    return t;
  };
  ONNX_NAMESPACE::TypeProto out0_type = make_output_type(4);
  ONNX_NAMESPACE::TypeProto out1_type = make_output_type(3);
  ONNX_NAMESPACE::TypeProto out2_type = make_output_type(2);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_type);
  auto& out0_arg = graph.GetOrCreateNodeArg("Y0", &out0_type);
  auto& out1_arg = graph.GetOrCreateNodeArg("Y1", &out1_type);
  auto& out2_arg = graph.GetOrCreateNodeArg("Y2", &out2_type);

  ONNX_NAMESPACE::TensorProto split_init;
  split_init.set_name("split_sizes");
  split_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  split_init.add_dims(3);
  for (auto v : std::vector<int64_t>{4, 3, 2}) {
    split_init.add_int64_data(v);
  }
  graph.AddInitializedTensor(split_init);
  auto& split_arg = graph.GetOrCreateNodeArg("split_sizes", nullptr);

  auto& node = graph.AddNode("split13_uneven", "Split", "Split-13 with uneven 'split' input",
                             {&input_arg, &split_arg}, {&out0_arg, &out1_arg, &out2_arg});
  node.AddAttribute("axis", static_cast<int64_t>(1));

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 9};
  std::vector<float> input_data = {0.5f, -1.0f, 2.25f, -3.5f, 4.0f, -0.125f, 7.5f, -8.0f, 0.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "Split13UnevenInput_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "Split13UnevenInput_MLProgram",
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

TEST(CoreMLExecutionProviderTest, Split13EvenInput) {
  // Parity with Split11EvenAttribute via the opset-13 input form.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}};
  onnxruntime::Model model("split13_even_input", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type.mutable_tensor_type()->mutable_shape();
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(6);

  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  output_shape->add_dim()->set_dim_value(1);
  output_shape->add_dim()->set_dim_value(3);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_type);
  auto& out0_arg = graph.GetOrCreateNodeArg("Y0", &output_type);
  auto& out1_arg = graph.GetOrCreateNodeArg("Y1", &output_type);

  ONNX_NAMESPACE::TensorProto split_init;
  split_init.set_name("split_sizes");
  split_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  split_init.add_dims(2);
  for (auto v : std::vector<int64_t>{3, 3}) {
    split_init.add_int64_data(v);
  }
  graph.AddInitializedTensor(split_init);
  auto& split_arg = graph.GetOrCreateNodeArg("split_sizes", nullptr);

  auto& node = graph.AddNode("split13_even", "Split", "Split-13 with even 'split' input",
                             {&input_arg, &split_arg}, {&out0_arg, &out1_arg});
  node.AddAttribute("axis", static_cast<int64_t>(1));

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 6};
  std::vector<float> input_data = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "Split13EvenInput_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "Split13EvenInput_MLProgram",
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

TEST(CoreMLExecutionProviderTest, Split13NoSplitInputEven) {
  // Parity with Split11NoAttributeEven: opset 13 with no 'split' input must
  // fall through to the SinceVersion() < 18 even-split branch (num_splits =
  // num_outputs) for both emitters.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}};
  onnxruntime::Model model("split13_no_split_input_even", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type.mutable_tensor_type()->mutable_shape();
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(8);

  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  output_shape->add_dim()->set_dim_value(1);
  output_shape->add_dim()->set_dim_value(4);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_type);
  auto& out0_arg = graph.GetOrCreateNodeArg("Y0", &output_type);
  auto& out1_arg = graph.GetOrCreateNodeArg("Y1", &output_type);

  auto& node = graph.AddNode("split13_no_split_input", "Split", "Split-13 with no 'split' input",
                             {&input_arg}, {&out0_arg, &out1_arg});
  node.AddAttribute("axis", static_cast<int64_t>(1));

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 8};
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "Split13NoSplitInputEven_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "Split13NoSplitInputEven_MLProgram",
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

TEST(CoreMLExecutionProviderTest, Split7UnevenAttribute) {
  // Opset 7 (≤ 10) parity check. The builder advertises support from opset 1
  // and reads the 'split' attribute; the Split11* tests cover opset 11, this
  // test covers the opset 7-10 range explicitly.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 7}};
  onnxruntime::Model model("split7_uneven_attribute", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type.mutable_tensor_type()->mutable_shape();
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(9);

  auto make_output_type = [](int64_t split_size) {
    ONNX_NAMESPACE::TypeProto t;
    t.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* s = t.mutable_tensor_type()->mutable_shape();
    s->add_dim()->set_dim_value(1);
    s->add_dim()->set_dim_value(split_size);
    return t;
  };
  ONNX_NAMESPACE::TypeProto out0_type = make_output_type(4);
  ONNX_NAMESPACE::TypeProto out1_type = make_output_type(3);
  ONNX_NAMESPACE::TypeProto out2_type = make_output_type(2);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_type);
  auto& out0_arg = graph.GetOrCreateNodeArg("Y0", &out0_type);
  auto& out1_arg = graph.GetOrCreateNodeArg("Y1", &out1_type);
  auto& out2_arg = graph.GetOrCreateNodeArg("Y2", &out2_type);

  auto& node = graph.AddNode("split7_uneven", "Split", "Split-7 with uneven 'split' attribute",
                             {&input_arg}, {&out0_arg, &out1_arg, &out2_arg});
  node.AddAttribute("axis", static_cast<int64_t>(1));
  node.AddAttribute("split", std::vector<int64_t>{4, 3, 2});

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 9};
  std::vector<float> input_data = {0.5f, -1.0f, 2.25f, -3.5f, 4.0f, -0.125f, 7.5f, -8.0f, 0.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "Split7UnevenAttribute_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "Split7UnevenAttribute_MLProgram",
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

TEST(CoreMLExecutionProviderTest, Split11ZeroSplitValueNotSupported) {
  // Negative: a zero entry in the 'split' attribute must be rejected so the
  // node falls back to CPU. Sum still equals the axis size, so this exercises
  // the non-positive value check specifically.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 11}};
  onnxruntime::Model model("split11_zero_split_value", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type.mutable_tensor_type()->mutable_shape();
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(9);

  auto make_output_type = [](int64_t split_size) {
    ONNX_NAMESPACE::TypeProto t;
    t.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* s = t.mutable_tensor_type()->mutable_shape();
    s->add_dim()->set_dim_value(1);
    s->add_dim()->set_dim_value(split_size);
    return t;
  };
  ONNX_NAMESPACE::TypeProto out0_type = make_output_type(3);
  ONNX_NAMESPACE::TypeProto out1_type = make_output_type(0);
  ONNX_NAMESPACE::TypeProto out2_type = make_output_type(6);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_type);
  auto& out0_arg = graph.GetOrCreateNodeArg("Y0", &out0_type);
  auto& out1_arg = graph.GetOrCreateNodeArg("Y1", &out1_type);
  auto& out2_arg = graph.GetOrCreateNodeArg("Y2", &out2_type);

  auto& node = graph.AddNode("split11_zero", "Split", "Split-11 with a zero 'split' entry",
                             {&input_arg}, {&out0_arg, &out1_arg, &out2_arg});
  node.AddAttribute("axis", static_cast<int64_t>(1));
  node.AddAttribute("split", std::vector<int64_t>{3, 0, 6});

  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::None);
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::None);
}

TEST(CoreMLExecutionProviderTest, Split11SingleOutputNotSupported) {
  // Negative: a Split node with only 1 output. CoreML SplitND requires ≥2,
  // so the attribute-form path's split_attr->size() < 2 check rejects it.
  // ONNX schema allows variadic ≥1 outputs and CPU's Split kernel accepts
  // a single output, so this case can be observed via partition assertion.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 11}};
  onnxruntime::Model model("split11_single_output", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type.mutable_tensor_type()->mutable_shape();
  input_shape->add_dim()->set_dim_value(1);
  input_shape->add_dim()->set_dim_value(5);

  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(5);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_type);
  auto& out0_arg = graph.GetOrCreateNodeArg("Y0", &output_type);

  auto& node = graph.AddNode("split11_single_output", "Split",
                             "Split-11 with a single output",
                             {&input_arg}, {&out0_arg});
  node.AddAttribute("axis", static_cast<int64_t>(1));
  node.AddAttribute("split", std::vector<int64_t>{5});

  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::None);
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::None);
}

namespace {
// int64 -> Cast(bool) -> Cast(float) [-> Sqrt]; the first Cast is fed directly
// by a graph input (no preceding node).
//
// append_nontrivial=false gives the all-Cast graph used by the NeuralNetwork
// negative test below. append_nontrivial=true appends a Sqrt: a CoreML partition
// made up only of trivial ops (Cast is marked trivial) is dropped, so the extra
// non-trivial op keeps the partition and lets the test below assert the bool
// Casts are claimed.
std::string MakeCastBoolModelData(bool append_nontrivial = false) {
  onnxruntime::Model model("cast_bool_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  auto make_type = [](int32_t elem_type) {
    ONNX_NAMESPACE::TypeProto t;
    t.mutable_tensor_type()->set_elem_type(elem_type);
    for (int64_t d : {1, 4}) t.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(d);
    return t;
  };
  const auto int64_type = make_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  const auto bool_type = make_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  const auto float_type = make_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto& x = graph.GetOrCreateNodeArg("X", &int64_type);
  auto& b = graph.GetOrCreateNodeArg("B", &bool_type);
  auto& y = graph.GetOrCreateNodeArg("Y", &float_type);

  auto& to_bool = graph.AddNode("cast_to_bool", "Cast", "int64 -> bool", {&x}, {&b});
  to_bool.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_BOOL));
  auto& to_float = graph.AddNode("cast_to_float", "Cast", "bool -> float", {&b}, {&y});
  to_float.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));

  if (append_nontrivial) {
    auto& z = graph.GetOrCreateNodeArg("Z", &float_type);
    graph.AddNode("sqrt", "Sqrt", "float -> float", {&y}, {&z});
  }

  ORT_THROW_IF_ERROR(graph.Resolve());
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  return model_data;
}

// Single-input model with both Sin and Cos consuming `X`, used by the
// Sin/Cos tests below.
std::string MakeSinCosModelData() {
  onnxruntime::Model model("sin_cos_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* shape = float_tensor.mutable_tensor_type()->mutable_shape();
  shape->add_dim()->set_dim_value(1);
  shape->add_dim()->set_dim_value(6);

  auto& x = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& sin_out = graph.GetOrCreateNodeArg("Sin_out", &float_tensor);
  auto& cos_out = graph.GetOrCreateNodeArg("Cos_out", &float_tensor);
  graph.AddNode("sin", "Sin", "sin node", {&x}, {&sin_out});
  graph.AddNode("cos", "Cos", "cos node", {&x}, {&cos_out});

  ORT_THROW_IF_ERROR(graph.Resolve());
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  return model_data;
}
}  // namespace

// On the NeuralNetwork format the Cast builder only supports a Cast that
// consumes an ArgMax, so these graph-input / Cast-fed Casts must fall back to
// CPU. Guards the IsOpSupportedImpl reordering that moved the preceding-node
// check into the NeuralNetwork branch.
TEST(CoreMLExecutionProviderTest, CastNonArgMaxNeuralNetworkNotSupported) {
  const std::string model_data = MakeCastBoolModelData();
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::None);
}

// Load-time partition check on the ML Program path: confirms the EP claims both
// bool Casts. A non-trivial Sqrt is appended so the partition isn't dropped as
// all-trivial (see MakeCastBoolModelData); all three nodes -- both Casts and the
// Sqrt -- must land on CoreML.
TEST(CoreMLExecutionProviderTest, CastBoolMLProgramPartition) {
  const std::string model_data = MakeCastBoolModelData(/*append_nontrivial=*/true);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
}

namespace {
ONNX_NAMESPACE::TypeProto MakeTensorType(int32_t elem_type, const std::vector<int64_t>& shape) {
  ONNX_NAMESPACE::TypeProto t;
  t.mutable_tensor_type()->set_elem_type(elem_type);
  for (int64_t d : shape) t.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(d);
  return t;
}

// Constant int64 indices initializer {{0},{2}} (shape [2,1]).
void AddGatherNDIndices(onnxruntime::Graph& graph) {
  ONNX_NAMESPACE::TensorProto indices;
  indices.set_name("indices");
  indices.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  indices.add_dims(2);
  indices.add_dims(1);
  for (int64_t v : {0, 2}) indices.add_int64_data(v);
  graph.AddInitializedTensor(indices);
}

// GatherND(data[4,3] float input, indices[2,1] const) -> out[2,3] float.
std::string MakeGatherNDModelData() {
  onnxruntime::Model model("gather_nd_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  const auto float_data = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {4, 3});
  const auto indices_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_INT64, {2, 1});
  const auto float_out = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});

  auto& data = graph.GetOrCreateNodeArg("data", &float_data);
  auto& indices = graph.GetOrCreateNodeArg("indices", &indices_type);
  auto& out = graph.GetOrCreateNodeArg("Out", &float_out);
  AddGatherNDIndices(graph);
  graph.AddNode("gather_nd", "GatherND", "gather rows", {&data, &indices}, {&out});

  ORT_THROW_IF_ERROR(graph.Resolve());
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  return model_data;
}

// data(int32 input) -> Cast(bool) -> GatherND -> Cast(float). Exercises the
// bool-data path, which the builder lowers as cast -> gather_nd -> cast (the
// bool tensors stay internal to the CoreML partition).
std::string MakeGatherNDBoolModelData() {
  onnxruntime::Model model("gather_nd_bool_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  const auto int32_data = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_INT32, {4, 3});
  const auto bool_data = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_BOOL, {4, 3});
  const auto indices_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_INT64, {2, 1});
  const auto bool_out = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_BOOL, {2, 3});
  const auto float_out = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});

  auto& src = graph.GetOrCreateNodeArg("Src", &int32_data);
  auto& data = graph.GetOrCreateNodeArg("data", &bool_data);
  auto& indices = graph.GetOrCreateNodeArg("indices", &indices_type);
  auto& gathered = graph.GetOrCreateNodeArg("gathered", &bool_out);
  auto& out = graph.GetOrCreateNodeArg("Out", &float_out);
  AddGatherNDIndices(graph);

  auto& to_bool = graph.AddNode("cast_to_bool", "Cast", "int32 -> bool", {&src}, {&data});
  to_bool.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_BOOL));
  graph.AddNode("gather_nd", "GatherND", "gather bool rows", {&data, &indices}, {&gathered});
  auto& to_float = graph.AddNode("cast_to_float", "Cast", "bool -> float", {&gathered}, {&out});
  to_float.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));

  ORT_THROW_IF_ERROR(graph.Resolve());
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  return model_data;
}

// GatherND with batch_dims=1: data[2,3] input, indices[2,1] const -> out[2].
std::string MakeGatherNDBatchDimsModelData() {
  onnxruntime::Model model("gather_nd_batchdims_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  const auto float_data = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 3});
  const auto indices_type = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_INT64, {2, 1});
  const auto float_out = MakeTensorType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2});

  auto& data = graph.GetOrCreateNodeArg("data", &float_data);
  auto& indices = graph.GetOrCreateNodeArg("indices", &indices_type);
  auto& out = graph.GetOrCreateNodeArg("Out", &float_out);

  ONNX_NAMESPACE::TensorProto indices_init;
  indices_init.set_name("indices");
  indices_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  indices_init.add_dims(2);
  indices_init.add_dims(1);
  for (int64_t v : {0, 1}) indices_init.add_int64_data(v);
  graph.AddInitializedTensor(indices_init);

  auto& node = graph.AddNode("gather_nd", "GatherND", "batched gather", {&data, &indices}, {&out});
  node.AddAttribute("batch_dims", static_cast<int64_t>(1));

  ORT_THROW_IF_ERROR(graph.Resolve());
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  return model_data;
}

// Where(cond, X, Y) with a constant bool `cond` initializer and X/Y graph
// inputs of the given element type. The constant `cond` keeps the bool
// internal -- a CoreML partition cannot have bool I/O.
std::string MakeWhereModelData(int32_t xy_elem_type) {
  onnxruntime::Model model("where_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  auto make_type = [](int32_t elem_type) {
    ONNX_NAMESPACE::TypeProto t;
    t.mutable_tensor_type()->set_elem_type(elem_type);
    for (int64_t d : {1, 4}) t.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(d);
    return t;
  };
  const auto bool_type = make_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  const auto xy_type = make_type(xy_elem_type);

  auto& cond = graph.GetOrCreateNodeArg("cond", &bool_type);
  auto& x = graph.GetOrCreateNodeArg("X", &xy_type);
  auto& y = graph.GetOrCreateNodeArg("Y", &xy_type);
  auto& out = graph.GetOrCreateNodeArg("Out", &xy_type);

  ONNX_NAMESPACE::TensorProto cond_init;
  cond_init.set_name("cond");
  cond_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  cond_init.add_dims(1);
  cond_init.add_dims(4);
  for (int32_t v : {1, 0, 1, 0}) cond_init.add_int32_data(v);  // ONNX stores bool in int32_data
  graph.AddInitializedTensor(cond_init);

  graph.AddNode("where", "Where", "select X where cond else Y", {&cond, &x, &y}, {&out});
  ORT_THROW_IF_ERROR(graph.Resolve());
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  return model_data;
}

// X1,X2(int64) -> Cast(bool) -> And -> Cast(float). And's bool inputs/output
// are internal -- bool cannot sit on a CoreML partition boundary.
std::string MakeAndChainModelData() {
  onnxruntime::Model model("and_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  auto make_type = [](int32_t elem_type) {
    ONNX_NAMESPACE::TypeProto t;
    t.mutable_tensor_type()->set_elem_type(elem_type);
    for (int64_t d : {1, 4}) t.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(d);
    return t;
  };
  const auto int64_type = make_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  const auto bool_type = make_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  const auto float_type = make_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto& x1 = graph.GetOrCreateNodeArg("X1", &int64_type);
  auto& x2 = graph.GetOrCreateNodeArg("X2", &int64_type);
  auto& a = graph.GetOrCreateNodeArg("A", &bool_type);
  auto& b = graph.GetOrCreateNodeArg("B", &bool_type);
  auto& c = graph.GetOrCreateNodeArg("C", &bool_type);
  auto& y = graph.GetOrCreateNodeArg("Y", &float_type);

  auto& cast_a = graph.AddNode("cast_a", "Cast", "int64 -> bool", {&x1}, {&a});
  cast_a.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_BOOL));
  auto& cast_b = graph.AddNode("cast_b", "Cast", "int64 -> bool", {&x2}, {&b});
  cast_b.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_BOOL));
  graph.AddNode("and", "And", "logical and", {&a, &b}, {&c});
  auto& cast_y = graph.AddNode("cast_y", "Cast", "bool -> float", {&c}, {&y});
  cast_y.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));

  ORT_THROW_IF_ERROR(graph.Resolve());
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  return model_data;
}

// Where(cond, X, Y) where the bool `cond` is a graph INPUT (not a constant), so
// the bool value sits on the CoreML partition boundary. RewriteBoolGraphIOBoundaries
// exposes it as an int32 feature and inserts an int32->bool cast, so the node is
// still claimed; model.mm does the bool<->int32 conversion at runtime.
std::string MakeWhereBoolInputModelData() {
  onnxruntime::Model model("where_bool_input_test", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  auto make_type = [](int32_t elem_type) {
    ONNX_NAMESPACE::TypeProto t;
    t.mutable_tensor_type()->set_elem_type(elem_type);
    for (int64_t d : {1, 4}) t.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(d);
    return t;
  };
  const auto bool_type = make_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  const auto float_type = make_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  auto& cond = graph.GetOrCreateNodeArg("cond", &bool_type);
  auto& x = graph.GetOrCreateNodeArg("X", &float_type);
  auto& y = graph.GetOrCreateNodeArg("Y", &float_type);
  auto& out = graph.GetOrCreateNodeArg("Out", &float_type);

  graph.AddNode("where", "Where", "select X where cond else Y", {&cond, &x, &y}, {&out});
  ORT_THROW_IF_ERROR(graph.Resolve());
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  return model_data;
}
}  // namespace

// GatherND is lowered to the ML Program 'gather_nd' op.
TEST(CoreMLExecutionProviderTest, GatherND_MLProgram) {
  const std::string model_data = MakeGatherNDModelData();
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};

#if defined(__APPLE__)
  std::vector<int64_t> dims = {4, 3};
  std::vector<float> values = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  OrtValue data_val;
  CreateMLValue<float>(CPUAllocator::DefaultInstance(), dims, values, &data_val);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("data", data_val));

  EPVerificationParams params{};
  params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  RunAndVerifyOutputsWithEP(model_span, CurrentTestName(),
                            MakeCoreMLExecutionProvider("MLProgram"), feeds, params);
#else
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

// Sin and Cos are lowered to the ML Program 'sin' / 'cos' ops.
TEST(CoreMLExecutionProviderTest, SinCos_MLProgram) {
  const std::string model_data = MakeSinCosModelData();
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 6};
  std::vector<float> values = {-2.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
  OrtValue x_val;
  CreateMLValue<float>(CPUAllocator::DefaultInstance(), dims, values, &x_val);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", x_val));

  EPVerificationParams params{};
  params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  RunAndVerifyOutputsWithEP(model_span, CurrentTestName(),
                            MakeCoreMLExecutionProvider("MLProgram"), feeds, params);
#else
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

TEST(CoreMLExecutionProviderTest, GatherScalarIndicesAxis1) {
  // ai.onnx:Gather with rank-0 (scalar) 'indices'. ONNX output rank =
  // data_rank + indices_rank - 1 = 2. The CoreML builder internally promotes
  // indices to [1], runs gather, then squeezes the inserted axis. Pattern
  // produced by StyleGAN-family generators (e.g. GFPGAN) that pick a
  // per-layer style code with a scalar index.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}};
  onnxruntime::Model model("gather_scalar_indices_axis1", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  // data X: {1, 4, 8} float
  ONNX_NAMESPACE::TypeProto data_type;
  data_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* data_shape = data_type.mutable_tensor_type()->mutable_shape();
  data_shape->add_dim()->set_dim_value(1);
  data_shape->add_dim()->set_dim_value(4);
  data_shape->add_dim()->set_dim_value(8);

  // output Y: {1, 8}
  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  output_shape->add_dim()->set_dim_value(1);
  output_shape->add_dim()->set_dim_value(8);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &data_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  // Scalar int64 index with value 2.
  ONNX_NAMESPACE::TensorProto idx_init;
  idx_init.set_name("idx");
  idx_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  // No dims => rank-0 tensor.
  idx_init.add_int64_data(2);
  graph.AddInitializedTensor(idx_init);
  auto& idx_arg = graph.GetOrCreateNodeArg("idx", nullptr);

  auto& node = graph.AddNode("gather_scalar", "Gather", "Gather with scalar indices",
                             {&input_arg, &idx_arg}, {&output_arg});
  node.AddAttribute("axis", static_cast<int64_t>(1));

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 4, 8};
  std::vector<float> input_data(1 * 4 * 8);
  for (size_t i = 0; i < input_data.size(); ++i) input_data[i] = static_cast<float>(i) * 0.25f - 1.0f;
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesAxis1_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesAxis1_MLProgram",
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

// CoreML's gather_nd rejects bool 'x', so the builder lowers a bool-data
// GatherND as cast(bool->int32) -> gather_nd -> cast(int32->bool). This
// Cast->GatherND->Cast chain must run fully on CoreML.
TEST(CoreMLExecutionProviderTest, GatherNDBoolData_MLProgram) {
  const std::string model_data = MakeGatherNDBoolModelData();
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};

#if defined(__APPLE__)
  std::vector<int64_t> dims = {4, 3};
  std::vector<int32_t> values = {0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1};
  OrtValue src_val;
  CreateMLValue<int32_t>(CPUAllocator::DefaultInstance(), dims, values, &src_val);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("Src", src_val));

  EPVerificationParams params{};
  params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  RunAndVerifyOutputsWithEP(model_span, CurrentTestName(),
                            MakeCoreMLExecutionProvider("MLProgram"), feeds, params);
#else
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

// Sin/Cos only have an ML Program lowering (the NeuralNetwork
// UnaryFunctionLayerParams has no sin/cos), so on the NeuralNetwork format
// they must fall back to CPU rather than be claimed.
TEST(CoreMLExecutionProviderTest, SinCosNeuralNetworkNotSupported) {
  const std::string model_data = MakeSinCosModelData();
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::None);
}

TEST(CoreMLExecutionProviderTest, GatherScalarIndicesAxis0) {
  // Scalar Gather along axis 0 — squeeze axis is 0; covers a different
  // squeeze position than the axis=1 test.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}};
  onnxruntime::Model model("gather_scalar_indices_axis0", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  // data X: {6, 5} float
  ONNX_NAMESPACE::TypeProto data_type;
  data_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* data_shape = data_type.mutable_tensor_type()->mutable_shape();
  data_shape->add_dim()->set_dim_value(6);
  data_shape->add_dim()->set_dim_value(5);

  // output Y: {5}
  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  output_shape->add_dim()->set_dim_value(5);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &data_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  ONNX_NAMESPACE::TensorProto idx_init;
  idx_init.set_name("idx");
  idx_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  idx_init.add_int64_data(4);
  graph.AddInitializedTensor(idx_init);
  auto& idx_arg = graph.GetOrCreateNodeArg("idx", nullptr);

  auto& node = graph.AddNode("gather_scalar_axis0", "Gather", "Gather scalar idx axis=0",
                             {&input_arg, &idx_arg}, {&output_arg});
  node.AddAttribute("axis", static_cast<int64_t>(0));

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {6, 5};
  std::vector<float> input_data(6 * 5);
  for (size_t i = 0; i < input_data.size(); ++i) input_data[i] = static_cast<float>(i) - 12.5f;
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesAxis0_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesAxis0_MLProgram",
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

// GatherND only has an ML Program lowering; on the NeuralNetwork format it
// must fall back to CPU.
TEST(CoreMLExecutionProviderTest, GatherNDNeuralNetworkNotSupported) {
  const std::string model_data = MakeGatherNDModelData();
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::None);
}

// The iOS15 gather_nd op has no batch_dims parameter, so GatherND with
// batch_dims != 0 must fall back to CPU.
TEST(CoreMLExecutionProviderTest, GatherNDBatchDimsNotSupported) {
  const std::string model_data = MakeGatherNDBatchDimsModelData();
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::None);
}

// Where is lowered to the ML Program 'select' op.
TEST(CoreMLExecutionProviderTest, Where_MLProgram) {
  const std::string model_data = MakeWhereModelData(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 4};
  OrtValue x_val, y_val;
  CreateMLValue<float>(CPUAllocator::DefaultInstance(), dims, {1.0f, 2.0f, 3.0f, 4.0f}, &x_val);
  CreateMLValue<float>(CPUAllocator::DefaultInstance(), dims, {-1.0f, -2.0f, -3.0f, -4.0f}, &y_val);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", x_val));
  feeds.insert(std::make_pair("Y", y_val));

  EPVerificationParams params{};
  params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  RunAndVerifyOutputsWithEP(model_span, CurrentTestName(),
                            MakeCoreMLExecutionProvider("MLProgram"), feeds, params);
#else
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

// Where only has an ML Program lowering ('select'); on the NeuralNetwork
// format it must fall back to CPU.
TEST(CoreMLExecutionProviderTest, WhereNeuralNetworkNotSupported) {
  const std::string model_data = MakeWhereModelData(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::None);
}

// Where's X/Y branches are restricted to float / float16; an int32 Where must
// fall back to CPU.
TEST(CoreMLExecutionProviderTest, WhereNonFloatBranchesNotSupported) {
  const std::string model_data = MakeWhereModelData(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::None);
}

// float16 X/Y variant of Where_MLProgram, exercising the float16 branch of
// HasSupportedInputsImpl and the 'select' lowering.
TEST(CoreMLExecutionProviderTest, WhereFloat16_MLProgram) {
  const std::string model_data = MakeWhereModelData(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 4};
  std::vector<MLFloat16> x_data{MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f), MLFloat16(4.0f)};
  std::vector<MLFloat16> y_data{MLFloat16(-1.0f), MLFloat16(-2.0f), MLFloat16(-3.0f), MLFloat16(-4.0f)};
  OrtValue x_val, y_val;
  CreateMLValue<MLFloat16>(CPUAllocator::DefaultInstance(), dims, x_data, &x_val);
  CreateMLValue<MLFloat16>(CPUAllocator::DefaultInstance(), dims, y_data, &y_val);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", x_val));
  feeds.insert(std::make_pair("Y", y_val));

  EPVerificationParams params{};
  params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  RunAndVerifyOutputsWithEP(model_span, CurrentTestName(),
                            MakeCoreMLExecutionProvider("MLProgram"), feeds, params);
#else
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

// A bool graph INPUT flowing into Where exercises the partition-boundary bool
// handling (RewriteBoolGraphIOBoundaries + model.mm int32<->bool conversion),
// rather than the constant/internal bool the other Where/And tests rely on.
TEST(CoreMLExecutionProviderTest, WhereBoolGraphInput_MLProgram) {
  const std::string model_data = MakeWhereBoolInputModelData();
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 4};
  OrtValue cond_val, x_val, y_val;
  CreateMLValue<bool>(CPUAllocator::DefaultInstance(), dims, std::vector<bool>{true, false, true, false}, &cond_val);
  CreateMLValue<float>(CPUAllocator::DefaultInstance(), dims, {1.0f, 2.0f, 3.0f, 4.0f}, &x_val);
  CreateMLValue<float>(CPUAllocator::DefaultInstance(), dims, {-1.0f, -2.0f, -3.0f, -4.0f}, &y_val);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("cond", cond_val));
  feeds.insert(std::make_pair("X", x_val));
  feeds.insert(std::make_pair("Y", y_val));

  EPVerificationParams params{};
  params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  RunAndVerifyOutputsWithEP(model_span, CurrentTestName(),
                            MakeCoreMLExecutionProvider("MLProgram"), feeds, params);
#else
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

// And is lowered to the ML Program 'logical_and' op.
TEST(CoreMLExecutionProviderTest, And_MLProgram) {
  const std::string model_data = MakeAndChainModelData();
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 4};
  OrtValue x1_val, x2_val;
  CreateMLValue<int64_t>(CPUAllocator::DefaultInstance(), dims, {1, 1, 0, 0}, &x1_val);
  CreateMLValue<int64_t>(CPUAllocator::DefaultInstance(), dims, {1, 0, 1, 0}, &x2_val);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X1", x1_val));
  feeds.insert(std::make_pair("X2", x2_val));

  EPVerificationParams params{};
  params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  RunAndVerifyOutputsWithEP(model_span, CurrentTestName(),
                            MakeCoreMLExecutionProvider("MLProgram"), feeds, params);
#else
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::All);
#endif
}

// And only has an ML Program lowering ('logical_and'); on the NeuralNetwork
// format the chain falls back to CPU.
TEST(CoreMLExecutionProviderTest, AndNeuralNetworkNotSupported) {
  const std::string model_data = MakeAndChainModelData();
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()),
                                        model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::None);
}

TEST(CoreMLExecutionProviderTest, GatherScalarIndicesNegativeAxis) {
  // Scalar Gather with negative axis (-1) — verifies HandleNegativeAxis is
  // applied when computing the squeeze axis.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}};
  onnxruntime::Model model("gather_scalar_indices_negative_axis", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  // data X: {2, 3, 4} float
  ONNX_NAMESPACE::TypeProto data_type;
  data_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* data_shape = data_type.mutable_tensor_type()->mutable_shape();
  data_shape->add_dim()->set_dim_value(2);
  data_shape->add_dim()->set_dim_value(3);
  data_shape->add_dim()->set_dim_value(4);

  // output Y: {2, 3} (axis=-1 == axis 2; output drops that axis)
  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  output_shape->add_dim()->set_dim_value(2);
  output_shape->add_dim()->set_dim_value(3);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &data_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  ONNX_NAMESPACE::TensorProto idx_init;
  idx_init.set_name("idx");
  idx_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  idx_init.add_int64_data(1);
  graph.AddInitializedTensor(idx_init);
  auto& idx_arg = graph.GetOrCreateNodeArg("idx", nullptr);

  auto& node = graph.AddNode("gather_scalar_neg_axis", "Gather", "Gather scalar idx axis=-1",
                             {&input_arg, &idx_arg}, {&output_arg});
  node.AddAttribute("axis", static_cast<int64_t>(-1));

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {2, 3, 4};
  std::vector<float> input_data(2 * 3 * 4);
  for (size_t i = 0; i < input_data.size(); ++i) input_data[i] = static_cast<float>(i) * 0.5f;
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesNegativeAxis_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesNegativeAxis_MLProgram",
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

TEST(CoreMLExecutionProviderTest, GatherScalarIndicesFloat16) {
  // FLOAT16 'data' input. HasSupportedInputsImpl restricts fp16 Gather to
  // MLProgram on CoreML 6+, so this test only runs the MLProgram path.
  // Exercises the MLFloat16 branch of the static intermediate shape claim.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}};
  onnxruntime::Model model("gather_scalar_indices_fp16", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto data_type;
  data_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  auto* data_shape = data_type.mutable_tensor_type()->mutable_shape();
  data_shape->add_dim()->set_dim_value(1);
  data_shape->add_dim()->set_dim_value(4);
  data_shape->add_dim()->set_dim_value(8);

  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  output_shape->add_dim()->set_dim_value(1);
  output_shape->add_dim()->set_dim_value(8);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &data_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  ONNX_NAMESPACE::TensorProto idx_init;
  idx_init.set_name("idx");
  idx_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  idx_init.add_int64_data(2);
  graph.AddInitializedTensor(idx_init);
  auto& idx_arg = graph.GetOrCreateNodeArg("idx", nullptr);

  auto& node = graph.AddNode("gather_scalar_fp16", "Gather", "Gather scalar idx fp16 data",
                             {&input_arg, &idx_arg}, {&output_arg});
  node.AddAttribute("axis", static_cast<int64_t>(1));

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {1, 4, 8};
  std::vector<MLFloat16> input_data;
  input_data.reserve(1 * 4 * 8);
  for (size_t i = 0; i < 1 * 4 * 8; ++i) {
    input_data.emplace_back(static_cast<float>(i) * 0.25f - 1.0f);
  }
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<MLFloat16>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesFloat16_MLProgram",
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

TEST(CoreMLExecutionProviderTest, GatherScalarIndicesInt64Data) {
  // INT64 'data' input. HasSupportedInputsImpl allows int64 in both NN and
  // MLProgram; verify both formats correctly route int64 through the
  // expand/gather/squeeze chain.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}};
  onnxruntime::Model model("gather_scalar_indices_int64_data", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto data_type;
  data_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  auto* data_shape = data_type.mutable_tensor_type()->mutable_shape();
  data_shape->add_dim()->set_dim_value(3);
  data_shape->add_dim()->set_dim_value(4);

  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  output_shape->add_dim()->set_dim_value(4);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &data_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  ONNX_NAMESPACE::TensorProto idx_init;
  idx_init.set_name("idx");
  idx_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  idx_init.add_int64_data(1);
  graph.AddInitializedTensor(idx_init);
  auto& idx_arg = graph.GetOrCreateNodeArg("idx", nullptr);

  auto& node = graph.AddNode("gather_scalar_int64", "Gather", "Gather scalar idx int64 data",
                             {&input_arg, &idx_arg}, {&output_arg});
  node.AddAttribute("axis", static_cast<int64_t>(0));

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {3, 4};
  std::vector<int64_t> input_data;
  input_data.reserve(3 * 4);
  for (int64_t i = 0; i < 3 * 4; ++i) input_data.push_back(i * 1000 - 5000);
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<int64_t>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesInt64Data_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesInt64Data_MLProgram",
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

TEST(CoreMLExecutionProviderTest, GatherScalarIndicesInt32Indices) {
  // INT32 'indices'. The other scalar-indices tests use INT64 indices (the
  // PyTorch default); this one exercises the INT32 branch through both the
  // dtype gating in IsOpSupportedImpl and the indices_dtype path-through to
  // the reshape's intermediate output dtype in AddToModelBuilderImpl.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}};
  onnxruntime::Model model("gather_scalar_indices_int32_indices", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto data_type;
  data_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* data_shape = data_type.mutable_tensor_type()->mutable_shape();
  data_shape->add_dim()->set_dim_value(3);
  data_shape->add_dim()->set_dim_value(4);

  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(4);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &data_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  ONNX_NAMESPACE::TensorProto idx_init;
  idx_init.set_name("idx");
  idx_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  idx_init.add_int32_data(2);
  graph.AddInitializedTensor(idx_init);
  auto& idx_arg = graph.GetOrCreateNodeArg("idx", nullptr);

  auto& node = graph.AddNode("gather_scalar_int32_idx", "Gather", "Gather scalar int32 idx",
                             {&input_arg, &idx_arg}, {&output_arg});
  node.AddAttribute("axis", static_cast<int64_t>(0));

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {3, 4};
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesInt32Indices_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesInt32Indices_MLProgram",
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

TEST(CoreMLExecutionProviderTest, GatherScalarIndicesRank4Data) {
  // Rank-4 'data' input — the supported maximum for scalar Gather (the
  // pre-squeeze intermediate is rank 4; CoreML's compiler rejects scalar
  // Gather at rank 5 with "Invalid rank: 6"). Output is rank 3.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}};
  onnxruntime::Model model("gather_scalar_indices_rank4", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto data_type;
  data_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* data_shape = data_type.mutable_tensor_type()->mutable_shape();
  for (int64_t d : {2, 5, 3, 4}) data_shape->add_dim()->set_dim_value(d);

  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  // Gather on axis=1 with scalar idx removes that axis: {2,3,4}
  for (int64_t d : {2, 3, 4}) output_shape->add_dim()->set_dim_value(d);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &data_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  ONNX_NAMESPACE::TensorProto idx_init;
  idx_init.set_name("idx");
  idx_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  idx_init.add_int64_data(3);
  graph.AddInitializedTensor(idx_init);
  auto& idx_arg = graph.GetOrCreateNodeArg("idx", nullptr);

  auto& node = graph.AddNode("gather_scalar_rank4", "Gather", "Gather scalar idx rank-4 data",
                             {&input_arg, &idx_arg}, {&output_arg});
  node.AddAttribute("axis", static_cast<int64_t>(1));

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {2, 5, 3, 4};
  std::vector<float> input_data(2 * 5 * 3 * 4);
  for (size_t i = 0; i < input_data.size(); ++i) input_data[i] = static_cast<float>(i) * 0.1f - 5.0f;
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesRank4Data_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesRank4Data_MLProgram",
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

TEST(CoreMLExecutionProviderTest, GatherScalarIndicesRank1Data) {
  // Rank-1 'data' input with scalar indices — output is rank-0 (the pre-squeeze
  // intermediate is rank 1, squeezed to a scalar). Confirms CoreML actually
  // produces a rank-0 result on both NN and MLProgram paths.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}};
  onnxruntime::Model model("gather_scalar_indices_rank1", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto data_type;
  data_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  data_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(6);

  // Output is rank-0: TypeProto with a shape that has no dims.
  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  output_type.mutable_tensor_type()->mutable_shape();

  auto& input_arg = graph.GetOrCreateNodeArg("X", &data_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  ONNX_NAMESPACE::TensorProto idx_init;
  idx_init.set_name("idx");
  idx_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  idx_init.add_int64_data(2);
  graph.AddInitializedTensor(idx_init);
  auto& idx_arg = graph.GetOrCreateNodeArg("idx", nullptr);

  auto& node = graph.AddNode("gather_scalar_rank1", "Gather", "Gather scalar idx rank-1 data",
                             {&input_arg, &idx_arg}, {&output_arg});
  node.AddAttribute("axis", static_cast<int64_t>(0));

  ASSERT_STATUS_OK(graph.Resolve());

#if defined(__APPLE__)
  std::vector<int64_t> dims = {6};
  std::vector<float> input_data(6);
  for (size_t i = 0; i < input_data.size(); ++i) input_data[i] = static_cast<float>(i) - 2.5f;
  OrtValue ml_value_x;
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  CreateMLValue<float>(allocator, dims, input_data, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};

  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesRank1Data_NN",
                            MakeCoreMLExecutionProvider(),
                            feeds,
                            EPVerificationParams{ExpectedEPNodeAssignment::All});
  RunAndVerifyOutputsWithEP(model_span, "GatherScalarIndicesRank1Data_MLProgram",
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

TEST(CoreMLExecutionProviderTest, GatherScalarIndicesDynamicDataNotSupported) {
  // The scalar-indices path emits a reshape-+squeeze chain whose intermediate
  // shape we have to claim statically. IsOpSupportedImpl rejects the node
  // when 'data' has any unknown dim so it falls back to CPU rather than
  // produce an ill-formed CoreML program.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}};
  onnxruntime::Model model("gather_scalar_indices_dynamic_data", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto data_type;
  data_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* data_shape = data_type.mutable_tensor_type()->mutable_shape();
  data_shape->add_dim()->set_dim_param("N");  // dynamic leading dim
  data_shape->add_dim()->set_dim_value(4);

  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("N");

  auto& input_arg = graph.GetOrCreateNodeArg("X", &data_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  ONNX_NAMESPACE::TensorProto idx_init;
  idx_init.set_name("idx");
  idx_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  idx_init.add_int64_data(0);
  graph.AddInitializedTensor(idx_init);
  auto& idx_arg = graph.GetOrCreateNodeArg("idx", nullptr);

  auto& node = graph.AddNode("gather_scalar_dyn", "Gather", "Gather scalar idx, dynamic data",
                             {&input_arg, &idx_arg}, {&output_arg});
  node.AddAttribute("axis", static_cast<int64_t>(1));

  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::None);
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::None);
}

TEST(CoreMLExecutionProviderTest, GatherScalarIndicesRank5DataNotSupported) {
  // Scalar-indices Gather caps data rank at 4 (CoreML compiler reports
  // "Invalid rank: 6" on the rank-5 reshape+gather intermediate). Rank-5
  // 'data' must fall back to CPU.
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}};
  onnxruntime::Model model("gather_scalar_indices_rank5", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto data_type;
  data_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* data_shape = data_type.mutable_tensor_type()->mutable_shape();
  for (int64_t d : {2, 3, 4, 5, 6}) data_shape->add_dim()->set_dim_value(d);

  ONNX_NAMESPACE::TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  // axis=2 with scalar idx removes that axis: {2,3,5,6}
  for (int64_t d : {2, 3, 5, 6}) output_shape->add_dim()->set_dim_value(d);

  auto& input_arg = graph.GetOrCreateNodeArg("X", &data_type);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_type);

  ONNX_NAMESPACE::TensorProto idx_init;
  idx_init.set_name("idx");
  idx_init.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  idx_init.add_int64_data(2);
  graph.AddInitializedTensor(idx_init);
  auto& idx_arg = graph.GetOrCreateNodeArg("idx", nullptr);

  auto& node = graph.AddNode("gather_scalar_rank5", "Gather", "Gather scalar idx rank-5 data",
                             {&input_arg, &idx_arg}, {&output_arg});
  node.AddAttribute("axis", static_cast<int64_t>(2));

  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  gsl::span<const std::byte> model_span{reinterpret_cast<const std::byte*>(model_data.data()), model_data.size()};
  TestModelLoad(model_span, MakeCoreMLExecutionProvider(), ExpectedEPNodeAssignment::None);
  TestModelLoad(model_span, MakeCoreMLExecutionProvider("MLProgram"), ExpectedEPNodeAssignment::None);
}

#endif  // !(ORT_MINIMAL_BUILD)
}  // namespace test
}  // namespace onnxruntime
