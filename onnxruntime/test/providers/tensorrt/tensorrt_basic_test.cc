// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"
#include "gtest/gtest.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"
#include "core/providers/tensorrt/tensorrt_provider_options.h"
#include "core/providers/tensorrt/tensorrt_execution_provider_utils.h"
#include <string>
#include <thread>

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

namespace test {
class TensorrtExecutionProviderCacheTest: public testing::TestWithParam<std::string> {};

template <typename T>
void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                   const std::vector<T>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<T> found(rtensor.Data<T>(), rtensor.Data<T>() + expected_values.size());
  ASSERT_EQ(expected_values, found);
}

/**
 * Create a simple model with dynamic or non-dynamic input shape.
 * \param model_name - model name
 * \param graph_name - graph name
 * \params dims - input dimensions
 *
 * input: "X", "Y" and "Z"
 *        you can specify input dimensions, for example (1, 3, 2), (1, 2) or (1, -1, -1)). Note: -1 means the dimension is dynamic.
 *        All three inputs have the same dimensions.
 * output: "M"
 *
 *      "X"  "Y"
 *        \  /
 *    "Z"  Add
 *      \  /
 *       Add
 *       /
 *     "M"
 *
 */
void CreateBaseModel(std::string model_name, std::string graph_name, std::vector<int> dims) {
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  for (auto dim: dims) {
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }

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

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  status = onnxruntime::Model::Save(model, model_name);
}

void RunSession(InferenceSession& session_object,
                RunOptions& run_options,
                NameMLValMap& feeds,
                std::vector<std::string> output_names,
                std::vector<int64_t> expected_dims,
                std::vector<float> expected_values) {
    std::vector<OrtValue> fetches;
    auto status = session_object.Run(run_options, feeds, output_names, &fetches);
    ASSERT_TRUE(status.IsOK());
    VerifyOutputs(fetches, expected_dims, expected_values);
}

void RunWithOneSessionSingleThreadInference(std::string model_name, std::string sess_log_id) {
  SessionOptions so;
  so.session_logid = sess_log_id;
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  InferenceSession session_object{so, GetEnvironment()};
  onnxruntime::AllocatorManager allocator_manager;
  auto cuda_provider = DefaultCudaExecutionProvider();
  cuda_provider->RegisterAllocator(allocator_manager);
  auto cpu_allocator = cuda_provider->GetAllocator(0, OrtMemTypeCPU);
  std::vector<int64_t> dims_mul_x = {1, 3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {1, 3, 2};
  std::vector<float> expected_values_mul_m = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};

  OrtTensorRTProviderOptionsV2 params{
      0,
      0,
      nullptr,
      1000,
      1,
      1 << 30,
      0,
      0,
      nullptr,
      0,
      0,
      0,
      0,
      0,
      nullptr,
      0,
      nullptr,
      0,
      0,
      0};

    params.trt_engine_cache_enable = 1;
    std::unique_ptr<IExecutionProvider> execution_provider = TensorrtExecutionProviderWithOptions(&params);
    EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());
    auto status = session_object.Load(model_name);
    ASSERT_TRUE(status.IsOK());
    status = session_object.Initialize();
    ASSERT_TRUE(status.IsOK());

    // run inference
    // TRT engine will be created and cached
    // TRT profile will be created and cached only for dynamic input shape
    // Data in profile,
    // X: 1, 3, 3, 2, 2, 2
    // Y: 1, 3, 3, 2, 2, 2
    // Z: 1, 3, 3, 2, 2, 2
    RunSession(session_object, run_options, feeds, output_names, expected_dims_mul_m, expected_values_mul_m);
}

void RunWithOneSessionMultiThreadsInference(std::string model_name, std::string sess_log_id) {
  SessionOptions so;
  so.session_logid = sess_log_id;
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  InferenceSession session_object{so, GetEnvironment()};
  onnxruntime::AllocatorManager allocator_manager;
  auto cuda_provider = DefaultCudaExecutionProvider();
  cuda_provider->RegisterAllocator(allocator_manager);
  auto cpu_allocator = cuda_provider->GetAllocator(0, OrtMemTypeCPU);
  std::vector<int64_t> dims_mul_x = {1, 3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {1, 3, 2};
  std::vector<float> expected_values_mul_m = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};

  OrtTensorRTProviderOptionsV2 params{
      0,
      0,
      nullptr,
      1000,
      1,
      1 << 30,
      0,
      0,
      nullptr,
      0,
      0,
      0,
      0,
      0,
      nullptr,
      0,
      nullptr,
      0,
      0,
      0};

    params.trt_engine_cache_enable = 1;
    std::unique_ptr<IExecutionProvider> execution_provider = TensorrtExecutionProviderWithOptions(&params);
    EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());
    auto status = session_object.Load(model_name);
    ASSERT_TRUE(status.IsOK());
    status = session_object.Initialize();
    ASSERT_TRUE(status.IsOK());

    // run inference with multi-threads
    // TRT engine will be created and cached
    // TRT profile will be created and cached only for dynamic input shape
    // Data in profile,
    // X: 1, 3, 3, 2, 2, 2
    // Y: 1, 3, 3, 2, 2, 2
    // Z: 1, 3, 3, 2, 2, 2

    std::vector<std::thread> threads;
    int num_thread = 5;
    for (int i = 0; i < num_thread; ++i)
      threads.push_back(std::thread(RunSession, std::ref(session_object), std::ref(run_options), std::ref(feeds), std::ref(output_names), std::ref(expected_dims_mul_m), std::ref(expected_values_mul_m)));

    for (auto& th : threads)
      th.join();
}

TEST(TensorrtExecutionProviderTest, SessionCreationWithMultiThreadsAndInferenceWithMultiThreads) {
  std::vector<std::thread> threads;
  std::string model_name = "trt_execution_provider_multithreading_test.onnx";
  std::string graph_name = "multithreading_test";
  std::string sess_log_id = "TRTEPMultiThreadingTestWithOneSessionSingleThread";
  std::vector<int> dims = {1, 3, 2};
  int num_thread = 5;

  CreateBaseModel(model_name, graph_name, dims);

  for (int i = 0; i < num_thread; ++i)
    threads.push_back(std::thread(RunWithOneSessionSingleThreadInference, model_name, sess_log_id));

  for (auto& th : threads)
    th.join();
}

TEST(TensorrtExecutionProviderTest, SessionCreationWithSingleThreadAndInferenceWithMultiThreads) {
  std::string model_name = "trt_execution_provider_multithreading_test.onnx";
  std::string graph_name = "multithreading_test";
  std::string sess_log_id = "TRTEPMultiThreadingTestWithOneSessionMultiThreads";
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model_name, graph_name, dims);
  RunWithOneSessionMultiThreadsInference(model_name, sess_log_id);
}

// Test loading same model in different way, when hash id is generated via model name/model content/env metadata
TEST(TensorrtExecutionProviderTest, TRTModelIdGeneratorUsingModelHashing) {
  auto model_path = ORT_TSTR("testdata/mnist.onnx");

  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_path, model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());

  Graph& graph = model->MainGraph();
  GraphViewer viewer(graph);

  // get the hash for the model when loaded from file
  HashValue model_hash;
  int id = TRTGenerateModelId(viewer, model_hash);
  ASSERT_EQ(id, 0);
  ASSERT_NE(model_hash, 0);

  // now load the model from bytes and check the hash differs
  std::ifstream model_file_stream(model_path, std::ios::in | std::ios::binary);

  std::shared_ptr<Model> model2;
  ONNX_NAMESPACE::ModelProto model_proto;
  ASSERT_STATUS_OK(Model::Load(model_file_stream, &model_proto));
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), PathString(), model2, nullptr,
                               DefaultLoggingManager().DefaultLogger()));

  Graph& graph2 = model2->MainGraph();
  GraphViewer viewer2(graph2);

  HashValue model_hash2;
  int id2 = TRTGenerateModelId(viewer2, model_hash2);

  // test comparing model 1 & 2
  ASSERT_EQ(id2, 0) << "id2 should be 0";

  // Test loading same model from different path, see if hash values are same as well
  model_path = ORT_TSTR("testdata/TRTEP_test_model/mnist.onnx");
  std::shared_ptr<Model> model3;
  ASSERT_TRUE(Model::Load(model_path, model3, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph3 = model3->MainGraph();
  GraphViewer viewer3(graph3);
  HashValue model_hash3;
  int id3 = TRTGenerateModelId(viewer3, model_hash3);
  ASSERT_EQ(model_hash, model_hash3) << "model 1&3 are same models and they have same hash, no matter where they are loaded";
  ASSERT_EQ(id3, 1) << "id3 should be 1 as model 1 & 3 have same hash";
}

// Compare on TRT subgraph id when repeatedly calling TRTGenerateModelId
TEST(TensorrtExecutionProviderTest, TRTSubgraphIdGeneratorUsingModelHashing) {
  // Load model
  auto model_path = ORT_TSTR("testdata/mnist.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_path, model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());

  Graph& main_graph = model->MainGraph();
  GraphViewer graph(main_graph);
  HashValue model_hash;

  // Graph id acquired
  int graph_id = TRTGenerateModelId(graph, model_hash);
  int asserted_subgraph_id = graph_id + 1;

  // mock fetching subgraphs and generate id by calling TRTGenerateModelId repeatedly
  const int number_of_ort_nodes = graph.NumberOfNodes();
  std::vector<size_t> nodes_vector(number_of_ort_nodes);
  std::iota(std::begin(nodes_vector), std::end(nodes_vector), 0);
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();

  for (const auto& index : nodes_vector) {
    const auto& node = graph.GetNode(node_index[index]);
    std::cout << "->" << node->Name(); 

    // Check if id increment each time TRTGenerateModelId is called
    int subgraph_id = TRTGenerateModelId(graph, model_hash);
    ASSERT_EQ(subgraph_id, asserted_subgraph_id) << "id will increment as TRTGenerateModelId is repeatedly called";
    asserted_subgraph_id++;
  }
}

TEST_P(TensorrtExecutionProviderCacheTest, Run) {
  // GetParam() returns the parameter of following format:
  // ##cache type##_##input shape type##
  std::string param = GetParam();
  size_t pos = param.find("_");
  std::string input_type = param.substr(pos + 1);
  ASSERT_NE(pos, std::string::npos);
  std::string cache_type = ToUTF8String(param.substr(0, pos));

  std::string model_name = "trt_execution_provider_" + cache_type + "caching_test_" + input_type + ".onnx";
  std::vector<int> dims;
  if (input_type.compare("dynamic") == 0) {
    dims = {1, -1, -1}; //dynamic shape input
  }
  else {
    dims = {1, 3, 2};
  }

  CreateBaseModel(model_name, cache_type + "cachingtest", dims);

  SessionOptions so;
  so.session_logid = "TensorrtExecutionProvider" + cache_type + "cacheTest";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  InferenceSession session_object{so, GetEnvironment()};
  onnxruntime::AllocatorManager allocator_manager;
  auto cuda_provider = DefaultCudaExecutionProvider();
  cuda_provider->RegisterAllocator(allocator_manager);
  auto cpu_allocator = cuda_provider->GetAllocator(0, OrtMemTypeCPU);
  std::vector<int64_t> dims_mul_x = {1, 3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {1, 3, 2};
  std::vector<float> expected_values_mul_m = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};

  OrtTensorRTProviderOptionsV2 params{
      0,
      0,
      nullptr,
      1000,
      1,
      1 << 30,
      0,
      0,
      nullptr,
      0,
      0,
      0,
      0,
      0,
      nullptr,
      0,
      nullptr,
      0,
      0,
      0};

  if (cache_type.compare("engine") == 0) {

    /* Following code block tests the functionality of engine and optimization profile of ORT TRT, including:
     * - engine cache serialization/de-serialization
     * - profile cache serialization/de-serialization
     * - engine/profile cache should be updated when the input shape changes
     * - min/max shape ranges of dynamic shape dimensions saved in profile cache
     * - read corrupted profile cache #TODO
     *
     */

    params.trt_engine_cache_enable = 1;
    std::unique_ptr<IExecutionProvider> execution_provider = TensorrtExecutionProviderWithOptions(&params);
    EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());
    auto status = session_object.Load(model_name);
    ASSERT_TRUE(status.IsOK());
    status = session_object.Initialize();
    ASSERT_TRUE(status.IsOK());

    // run inference
    // TRT engine will be created and cached
    // TRT profile will be created and cached only for dynamic input shape
    // Data in profile,
    // X: 1, 3, 3, 2, 2, 2
    // Y: 1, 3, 3, 2, 2, 2
    // Z: 1, 3, 3, 2, 2, 2
    status = session_object.Run(run_options, feeds, output_names, &fetches);
    ASSERT_TRUE(status.IsOK());
    VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);
    ASSERT_TRUE(IsCacheExistedByType("./", ".engine"));

    std::vector<fs::path> profile_files;

    // profile cache only being generated for dynamic input shape
    if (input_type.compare("static") == 0) {
      ASSERT_TRUE(!IsCacheExistedByType("./", ".profile"));
    } else {
      ASSERT_TRUE(IsCacheExistedByType("./", ".profile"));

      profile_files = GetCachesByType("./", ".profile");
      ASSERT_EQ(profile_files.size(), 1);
      std::ifstream profile_file(profile_files[0], std::ios::binary | std::ios::in);
      auto shape_ranges = DeserializeProfile(profile_file);

      // check min/max shape ranges of dynamic shape dimensions
      for(auto it = shape_ranges.cbegin(); it != shape_ranges.cend(); ++it) {
        auto ranges = it->second;
        for (auto it2 = ranges.cbegin(); it2 != ranges.cend(); ++it2) {
          if (it2->first == 1) {
            ASSERT_EQ(it2->second.first, 3);
            ASSERT_EQ(it2->second.second, 3);
          } else if (it2->first == 2) {
            ASSERT_EQ(it2->second.first, 2);
            ASSERT_EQ(it2->second.second, 2);
          }
        }
      }
    }

    // another inference run with input shape {1, 1, 6}
    // TRT engine and profile will be updated
    // Data in profile,
    // X: 1, 1, 3, 2, 2, 6
    // Y: 1, 1, 3, 2, 2, 6
    // Z: 1, 1, 3, 2, 2, 6
    dims_mul_x = {1, 1, 6};
    CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_x);
    CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_y);
    CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_z);
    feeds.clear();
    feeds.insert(std::make_pair("X", ml_value_x));
    feeds.insert(std::make_pair("Y", ml_value_y));
    feeds.insert(std::make_pair("Z", ml_value_z));

    // prepare outputs
    fetches.clear();

    // prepare expected inputs and outputs
    expected_dims_mul_m = {1, 1, 6};

    status = session_object.Run(run_options, feeds, output_names, &fetches);

    if (input_type.compare("static") == 0) {
      // Can't run inference since input shape changes but the engine is built with static input
      ASSERT_FALSE(status.IsOK());
    } else {
      ASSERT_TRUE(status.IsOK());
      VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);

      profile_files = GetCachesByType("./", ".profile");
      ASSERT_EQ(profile_files.size(), 1);
      std::ifstream profile_file2(profile_files[0], std::ios::binary | std::ios::in);
      auto shape_ranges2 = DeserializeProfile(profile_file2);

      // check min/max shape ranges of dynamic shape dimensions
      for(auto it = shape_ranges2.cbegin(); it != shape_ranges2.cend(); ++it) {
        auto ranges = it->second;
        for (auto it2 = ranges.cbegin(); it2 != ranges.cend(); ++it2) {
          if (it2->first == 1) {
            ASSERT_EQ(it2->second.first, 1);
            ASSERT_EQ(it2->second.second, 3);
          } else if (it2->first == 2) {
            ASSERT_EQ(it2->second.first, 2);
            ASSERT_EQ(it2->second.second, 6);
          }
        }
      }
    }
  } else if (cache_type.compare("timing") == 0) {
     // add test code here
  }

  // clean up caches
  RemoveCachesByType("./", ".engine");
  RemoveCachesByType("./", ".profile");
}

/*
 * The TensorrtExecutionProviderCacheTest aims to test the functionality of all the engine/profile/timing caches of ORT TRT.
 * It uses value-parameterized test and the parameter in the test is a composite parameter which has following format:
 * ##cache type##_##input shape type##
 * - cache type       (could be engine cache or timing cache. Note: profile cache will be tested along with engine cache)
 * - input shape type (could be dynamic input shape or static input shape)
 *
 * We have following test parameters:
 * - engine_static: engine cache enabled with non-dynamic input shape
 * - engine_dynamic: engine cache enabled with dynamic input shape
 * - timing_static: will be added
 * - timing_dynamic: will be added
 */
INSTANTIATE_TEST_SUITE_P(TensorrtExecutionProviderCacheTests, TensorrtExecutionProviderCacheTest, testing::Values("engine_static",
                                                                                                                  "engine_dynamic"),
                                                                                                  [](const ::testing::TestParamInfo<TensorrtExecutionProviderCacheTest::ParamType>& info) {return info.param;});

TEST(TensorrtExecutionProviderTest, FunctionTest) {
  onnxruntime::Model model("functiontest", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor.
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
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

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  std::string model_file_name = "trt_execution_provider_function_test.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "TensorrtExecutionProviderTest.FunctionTest";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  InferenceSession session_object{so, GetEnvironment()};

  onnxruntime::AllocatorManager allocator_manager;
  auto cuda_provider = DefaultCudaExecutionProvider();
  cuda_provider->RegisterAllocator(allocator_manager);
  auto cpu_allocator = cuda_provider->GetAllocator(0, OrtMemTypeCPU);

  std::vector<int64_t> dims_mul_x = {1, 3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {1, 3, 2};
  std::vector<float> expected_values_mul_m = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};

  std::unique_ptr<IExecutionProvider> execution_provider = DefaultTensorrtExecutionProvider();
  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());

  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);
}

TEST(TensorrtExecutionProviderTest, NodeIndexMappingTest) {
  onnxruntime::Model model("nodeindexmappingtest", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor.
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  // BOOL tensor.
  ONNX_NAMESPACE::TypeProto bool_tensor;
  bool_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  // UINT8 tensor.
  ONNX_NAMESPACE::TypeProto uint8_tensor;
  uint8_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &bool_tensor);
  inputs.push_back(&input_arg_1);
  auto& output_arg_1 = graph.GetOrCreateNodeArg("node_1_out", &uint8_tensor);
  outputs.push_back(&output_arg_1);
  auto& cast_node = graph.AddNode("cast1", "Cast", "node 1.", inputs, outputs);
  cast_node.AddAttribute("to", int64_t{2});

  inputs.clear();
  inputs.push_back(&output_arg_1);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &bool_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  auto& cast_node_2 = graph.AddNode("cast2", "Cast", "node 2.", inputs, outputs);
  cast_node_2.AddAttribute("to", int64_t{9});

  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
  inputs.clear();
  inputs.push_back(&input_arg_2);
  inputs.push_back(&input_arg_3);
  auto& output_arg_3 = graph.GetOrCreateNodeArg("N", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_3);
  graph.AddNode("sub", "Sub", "node 3.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  std::string model_file_name = "trt_execution_provider_nodeindexmapping_test.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "TensorrtExecutionProviderTest.NodeIndexMappingTest";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  InferenceSession session_object{so, GetEnvironment()};

  onnxruntime::AllocatorManager allocator_manager;
  auto cuda_provider = DefaultCudaExecutionProvider();
  cuda_provider->RegisterAllocator(allocator_manager);
  auto cpu_allocator = cuda_provider->GetAllocator(0, OrtMemTypeCPU);

  std::vector<int64_t> dims_mul_x = {1, 3, 2};
  std::vector<bool> values_mul_x = {true, false, true, false, true, false};
  std::vector<int64_t> dims_mul_y = {1, 3, 2};
  std::vector<float> values_mul_y = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  CreateMLValue<bool>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(cpu_allocator, dims_mul_y, values_mul_y, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(cpu_allocator, dims_mul_y, values_mul_y, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");
  output_names.push_back("N");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {1, 3, 2};
  std::vector<bool> expected_values_mul_m = {true, false, true, false, true, false};
  std::vector<int64_t> expected_dims_mul_n = {1, 3, 2};
  std::vector<float> expected_values_mul_n = {0, 0, 0, 0, 0, 0};

  std::unique_ptr<IExecutionProvider> execution_provider = DefaultTensorrtExecutionProvider();
  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());

  ASSERT_STATUS_OK(session_object.Load(model_file_name));
  ASSERT_STATUS_OK(session_object.Initialize());

  // Now run
  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &fetches));
  std::vector<OrtValue> fetche{fetches.back()};
  VerifyOutputs(fetche, expected_dims_mul_n, expected_values_mul_n);
}

TEST(TensorrtExecutionProviderTest, RemoveCycleTest) {
  onnxruntime::Model model("removecycletest", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor.
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  // BOOL tensor.
  ONNX_NAMESPACE::TypeProto bool_tensor;
  bool_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  // UINT8 tensor.
  ONNX_NAMESPACE::TypeProto uint8_tensor;
  uint8_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &bool_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &bool_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto& output_arg_1 = graph.GetOrCreateNodeArg("xor1_out", &bool_tensor);
  outputs.push_back(&output_arg_1);
  graph.AddNode("xor1", "Xor", "node 1.", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg_1);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("not_out", &bool_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  graph.AddNode("not", "Not", "node 2.", inputs, outputs);

  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &bool_tensor);
  inputs.clear();
  inputs.push_back(&output_arg_2);
  inputs.push_back(&input_arg_3);
  auto& output_arg_3 = graph.GetOrCreateNodeArg("xor2_out", &bool_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_3);
  graph.AddNode("xor2", "Xor", "node 3.", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg_2);
  inputs.push_back(&output_arg_3);
  auto& output_arg_4 = graph.GetOrCreateNodeArg("M", &bool_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_4);
  graph.AddNode("and", "And", "node 4.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  std::string model_file_name = "trt_execution_provider_removecycle_test.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);

  std::vector<int64_t> dims_mul_x = {1, 3, 2};
  std::vector<bool> values_mul_x = {true, false, true, false, true, false};
  std::vector<int64_t> dims_mul_y = {1, 3, 2};
  std::vector<bool> values_mul_y = {true, true, false, true, false, false};
  std::vector<int64_t> dims_mul_z = {1, 3, 2};
  std::vector<bool> values_mul_z = {true, false, true, false, true, false};

  SessionOptions so;
  so.session_logid = "TensorrtExecutionProviderTest.RemoveCycleTest";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  InferenceSession session_object{so, GetEnvironment()};

  onnxruntime::AllocatorManager allocator_manager;
  auto cuda_provider = DefaultCudaExecutionProvider();
  cuda_provider->RegisterAllocator(allocator_manager);
  auto cpu_allocator = cuda_provider->GetAllocator(0, OrtMemTypeCPU);

  OrtValue ml_value_x;
  CreateMLValue<bool>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<bool>(cpu_allocator, dims_mul_y, values_mul_y, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<bool>(cpu_allocator, dims_mul_y, values_mul_y, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {1, 3, 2};
  std::vector<bool> expected_values_mul_m = {false, false, false, false, false, true};

  std::unique_ptr<IExecutionProvider> execution_provider = DefaultTensorrtExecutionProvider();
  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());

  ASSERT_STATUS_OK(session_object.Load(model_file_name));
  ASSERT_STATUS_OK(session_object.Initialize());

  // Now run
  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &fetches));
  VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);
}
}  // namespace test
}  // namespace onnxruntime
