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
#include <filesystem>
#include <chrono>

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

namespace test {
class TensorrtExecutionProviderCacheTest : public testing::TestWithParam<std::string> {};

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
 * \param dims - input dimensions
 * \param add_non_zero_node - add NonZero node which makes the whole model partition into TRT EP and CUDA EP subgraphs.
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
 *     or
 *
 *      "X"  "Y"
 *        \  /
 *    "Z"  Add
 *      \  /
 *       Add
 *       /
 *    NonZero (This node will be placed on CUDA EP)
 *     /
 *   "M"
 */
void CreateBaseModel(std::string model_name,
                     std::string graph_name,
                     std::vector<int> dims,
                     bool add_non_zero_node = false) {
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  for (auto dim : dims) {
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

  if (add_non_zero_node) {
    auto& output_arg_2 = graph.GetOrCreateNodeArg("node_2_out_1", &float_tensor);
    outputs.clear();
    outputs.push_back(&output_arg_2);
    graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

    inputs.clear();
    inputs.push_back(&output_arg_2);
    ONNX_NAMESPACE::TypeProto int_tensor;
    int_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    auto& output_arg_3 = graph.GetOrCreateNodeArg("M", &int_tensor);
    outputs.clear();
    outputs.push_back(&output_arg_3);
    graph.AddNode("node_3", "NonZero", "node 3.", inputs, outputs);
  } else {
    auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
    outputs.clear();
    outputs.push_back(&output_arg_2);
    graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);
  }

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  status = onnxruntime::Model::Save(model, model_name);
}

bool HasCacheFileWithPrefix(const std::string& prefix, std::string file_dir = "") {
  std::filesystem::path target_dir;
  if (file_dir.empty()) {
    target_dir = std::filesystem::current_path();
  } else {
    target_dir = std::filesystem::path(file_dir);
  }

  for (const auto& entry : std::filesystem::directory_iterator(target_dir)) {
    if (entry.is_regular_file()) {
      std::string filename = entry.path().filename().string();
      if (filename.rfind(prefix, 0) == 0) {
        return true;
      }
    }
  }
  return false;
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

void RunSession2(InferenceSession& session_object,
                 RunOptions& run_options,
                 NameMLValMap& feeds,
                 std::vector<std::string> output_names,
                 std::vector<int64_t> expected_dims,
                 std::vector<int64_t> expected_values) {
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
  auto cuda_provider = DefaultCudaExecutionProvider();
  auto cpu_allocator = cuda_provider->CreatePreferredAllocators()[1];
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

  OrtTensorRTProviderOptionsV2 params;
  params.trt_engine_cache_enable = 1;
  params.trt_engine_cache_prefix = "TRTEP_Cache_Test";
  params.trt_dump_ep_context_model = 1;
  params.trt_ep_context_file_path = "EP_Context_model.onnx";
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

  // Verify on cache with customized prefix
  ASSERT_TRUE(HasCacheFileWithPrefix(params.trt_engine_cache_prefix));

  // Verify EP context model with user provided name
  ASSERT_TRUE(HasCacheFileWithPrefix(params.trt_ep_context_file_path));
}

void RunWithOneSessionMultiThreadsInference(std::string model_name, std::string sess_log_id, bool has_non_zero_node = false) {
  SessionOptions so;
  so.session_logid = sess_log_id;
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  InferenceSession session_object{so, GetEnvironment()};
  auto cuda_provider = DefaultCudaExecutionProvider();
  auto cpu_allocator = cuda_provider->CreatePreferredAllocators()[1];
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
  std::vector<int64_t> expected_dims_nonzero_m = {3, 6};
  std::vector<int64_t> expected_values_nonzero_m = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 1, 0, 1, 0, 1};

  OrtTensorRTProviderOptionsV2 params;
  params.trt_engine_cache_enable = 1;
  params.trt_engine_cache_prefix = "TRTEP_Cache_Test";
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
  for (int i = 0; i < num_thread; ++i) {
    if (has_non_zero_node)
      threads.push_back(std::thread(RunSession2, std::ref(session_object), std::ref(run_options), std::ref(feeds), std::ref(output_names), std::ref(expected_dims_nonzero_m), std::ref(expected_values_nonzero_m)));
    else
      threads.push_back(std::thread(RunSession, std::ref(session_object), std::ref(run_options), std::ref(feeds), std::ref(output_names), std::ref(expected_dims_mul_m), std::ref(expected_values_mul_m)));
  }

  for (auto& th : threads)
    th.join();

  // Verify on cache with customized prefix
  ASSERT_TRUE(HasCacheFileWithPrefix(params.trt_engine_cache_prefix));
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

  // In addition to the test case that whole model can be run by TRT, we also need to test the case where
  // the model is partitioned into TRT EP and CUDA EP subgraphs.
  // We did observe synchronization issue for TRT EP without PerContextThread implementation running those models.
  CreateBaseModel(model_name, graph_name, dims, true);
  RunWithOneSessionMultiThreadsInference(model_name, sess_log_id, true);
}

// Test loading same model in different way, when hash id is generated via model name/model content/env metadata
TEST(TensorrtExecutionProviderTest, TRTModelIdGeneratorUsingModelHashing) {
  auto model_path = ORT_TSTR("testdata/mnist.onnx");

  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_path, model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());

  Graph& graph = model->MainGraph();
  GraphViewer viewer(graph);

  // get the hash for the model when loaded from file
  HashValue model_hash = TRTGenerateId(viewer);
  ASSERT_NE(model_hash, 0);

  // now load the model from bytes and check the hash differs
  std::ifstream model_file_stream(model_path, std::ios::in | std::ios::binary);

  std::shared_ptr<Model> model2;
  ONNX_NAMESPACE::ModelProto model_proto;
  ASSERT_STATUS_OK(Model::Load(model_file_stream, &model_proto));
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), PathString(), model2, nullptr,
                               DefaultLoggingManager().DefaultLogger()));

  // Test loading same model from file and byte steam. Hash values should be different
  Graph& graph2 = model2->MainGraph();
  GraphViewer viewer2(graph2);
  HashValue model_hash2 = TRTGenerateId(viewer2);
  ASSERT_NE(model_hash, model_hash2);

  // Test loading same model from different path, see if hash values are same as well
  model_path = ORT_TSTR("testdata/TRTEP_test_model/mnist.onnx");
  std::shared_ptr<Model> model3;
  ASSERT_TRUE(Model::Load(model_path, model3, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph3 = model3->MainGraph();
  GraphViewer viewer3(graph3);
  HashValue model_hash3 = TRTGenerateId(viewer3);
  ASSERT_EQ(model_hash, model_hash3) << "model 1&3 are same models and they have same hash, no matter where they are loaded";
}

TEST(TensorrtExecutionProviderTest, EPContextNode) {
  std::string model_name = "EPContextNode_test.onnx";
  std::string graph_name = "EPContextNode_test";
  std::string sess_log_id = "EPContextNode_test";
  std::vector<int> dims = {1, 3, 2};
  CreateBaseModel(model_name, graph_name, dims);

  SessionOptions so;
  so.session_logid = sess_log_id;
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  InferenceSession session_object{so, GetEnvironment()};
  auto cuda_provider = DefaultCudaExecutionProvider();
  auto cpu_allocator = cuda_provider->CreatePreferredAllocators()[1];
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

  /*
   * Test case 1: Dump context model
   *
   * provider options=>
   *   trt_ep_context_file_path = "EP_Context_model.onnx"
   *
   * expected result =>
   *   context model "EP_Context_model.onnx" should be created in current directory
   *
   */
  OrtTensorRTProviderOptionsV2 params;
  params.trt_engine_cache_enable = 1;
  params.trt_dump_ep_context_model = 1;
  params.trt_ep_context_file_path = "EP_Context_model.onnx";
  std::unique_ptr<IExecutionProvider> execution_provider = TensorrtExecutionProviderWithOptions(&params);
  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());
  auto status = session_object.Load(model_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());
  ASSERT_TRUE(HasCacheFileWithPrefix(params.trt_ep_context_file_path));

  /*
   * Test case 2: Dump context model
   *
   * provider options=>
   *   trt_engine_cache_prefix = "TRT_engine_cache"
   *   trt_ep_context_file_path = "context_model_folder"
   *   trt_engine_cache_path = "engine_cache_folder"
   *
   * expected result =>
   *   engine cache "./context_model_folder/engine_cache_folder/TRT_engine_cache...engine" should be created
   *   context model "./context_model_folder/EPContextNode_test_ctx.onnx" should be created
   */
  InferenceSession session_object2{so, GetEnvironment()};
  OrtTensorRTProviderOptionsV2 params2;
  params2.trt_engine_cache_enable = 1;
  params2.trt_dump_ep_context_model = 1;
  params2.trt_engine_cache_prefix = "TRT_engine_cache";
  params2.trt_engine_cache_path = "engine_cache_folder";  // due to dump_ep_context_model = 1, the new cache path is ./context_model_folder/engine_cache_folder
  params2.trt_ep_context_file_path = "context_model_folder";
  execution_provider = TensorrtExecutionProviderWithOptions(&params2);
  EXPECT_TRUE(session_object2.RegisterExecutionProvider(std::move(execution_provider)).IsOK());
  status = session_object2.Load(model_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object2.Initialize();
  ASSERT_TRUE(status.IsOK());
  auto new_engine_cache_path = std::filesystem::path(params2.trt_ep_context_file_path).append(params2.trt_engine_cache_path).string();
  // Test engine cache path:
  // "./context_model_folder/engine_cache_folder/TRT_engine_cache...engine" should be created
  ASSERT_TRUE(HasCacheFileWithPrefix(params2.trt_engine_cache_prefix, new_engine_cache_path));
  // Test context model path:
  // "./context_model_folder/EPContextNode_test_ctx.onnx" should be created
  ASSERT_TRUE(HasCacheFileWithPrefix("EPContextNode_test_ctx.onnx", params2.trt_ep_context_file_path));

  /*
   * Test case 3: Run the dumped context model
   *
   * context model path = "./EP_Context_model.onnx" (created from case 1)
   *
   * expected result=>
   *   engine cache is also in the same current dirctory as "./xxxxx.engine"
   *   and the "ep_cache_context" attribute node of the context model should point to that.
   *
   */
  InferenceSession session_object3{so, GetEnvironment()};
  OrtTensorRTProviderOptionsV2 params3;
  model_name = params.trt_ep_context_file_path;
  params3.trt_engine_cache_enable = 1;
  execution_provider = TensorrtExecutionProviderWithOptions(&params3);
  EXPECT_TRUE(session_object3.RegisterExecutionProvider(std::move(execution_provider)).IsOK());
  status = session_object3.Load(model_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object3.Initialize();
  ASSERT_TRUE(status.IsOK());
  // run inference
  // TRT engine will be created and cached
  // TRT profile will be created and cached only for dynamic input shape
  // Data in profile,
  // X: 1, 3, 3, 2, 2, 2
  // Y: 1, 3, 3, 2, 2, 2
  // Z: 1, 3, 3, 2, 2, 2
  RunSession(session_object3, run_options, feeds, output_names, expected_dims_mul_m, expected_values_mul_m);

  /*
   * Test case 4: Run the dumped context model
   *
   * context model path = "./context_model_folder/EPContextNode_test_ctx.onnx" (created from case 2)
   *
   * expected result=>
   *   engine cache path is "./context_model_folder/engine_cache_folder/xxxxx.engine"
   *   and the "ep_cache_context" attribute node of the context model should point to "engine_cache_folder/xxxxx.engine".
   *
   */
  InferenceSession session_object4{so, GetEnvironment()};
  OrtTensorRTProviderOptionsV2 params4;
  model_name = "./context_model_folder/EPContextNode_test_ctx.onnx";
  execution_provider = TensorrtExecutionProviderWithOptions(&params4);
  EXPECT_TRUE(session_object4.RegisterExecutionProvider(std::move(execution_provider)).IsOK());
  status = session_object4.Load(model_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object4.Initialize();
  ASSERT_TRUE(status.IsOK());
  // run inference
  // TRT engine will be created and cached
  // TRT profile will be created and cached only for dynamic input shape
  // Data in profile,
  // X: 1, 3, 3, 2, 2, 2
  // Y: 1, 3, 3, 2, 2, 2
  // Z: 1, 3, 3, 2, 2, 2
  RunSession(session_object4, run_options, feeds, output_names, expected_dims_mul_m, expected_values_mul_m);

  /*
   * Test case 5: Dump context model with embed_model = 1
   */
  InferenceSession session_object5{so, GetEnvironment()};
  OrtTensorRTProviderOptionsV2 params5;
  params5.trt_dump_ep_context_model = 1;
  params5.trt_ep_context_embed_mode = 1;
  params5.trt_ep_context_file_path = "EP_Context_model_2.onnx";
  model_name = "EPContextNode_test.onnx";
  execution_provider = TensorrtExecutionProviderWithOptions(&params5);
  EXPECT_TRUE(session_object5.RegisterExecutionProvider(std::move(execution_provider)).IsOK());
  status = session_object5.Load(model_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object5.Initialize();
  ASSERT_TRUE(status.IsOK());

  /*
   * Test case 6: Run context model with embed_model = 1 (created from case 5)
   */
  InferenceSession session_object6{so, GetEnvironment()};
  OrtTensorRTProviderOptionsV2 params6;
  params6.trt_ep_context_embed_mode = 1;
  model_name = params5.trt_ep_context_file_path;
  execution_provider = TensorrtExecutionProviderWithOptions(&params6);
  EXPECT_TRUE(session_object6.RegisterExecutionProvider(std::move(execution_provider)).IsOK());
  status = session_object6.Load(model_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object6.Initialize();
  ASSERT_TRUE(status.IsOK());
  // run inference
  // TRT engine will be created and cached
  // TRT profile will be created and cached only for dynamic input shape
  // Data in profile,
  // X: 1, 3, 3, 2, 2, 2
  // Y: 1, 3, 3, 2, 2, 2
  // Z: 1, 3, 3, 2, 2, 2
  RunSession(session_object6, run_options, feeds, output_names, expected_dims_mul_m, expected_values_mul_m);
}

TEST(TensorrtExecutionProviderTest, TRTPluginsCustomOpTest) {
  std::string model_name = "testdata/trt_plugin_custom_op_test.onnx";
  SessionOptions so;
  so.session_logid = "TensorrtExecutionProviderTRTPluginsTest";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  InferenceSession session_object{so, GetEnvironment()};
  auto cuda_provider = DefaultCudaExecutionProvider();
  auto cpu_allocator = cuda_provider->CreatePreferredAllocators()[1];
  std::vector<int64_t> dims_op_x = {12, 256, 256};
  std::vector<float> values_op_x(1.0f, 786432);  // 786432=12*256*256
  OrtValue ml_value_x;
  CreateMLValue<float>(cpu_allocator, dims_op_x, values_op_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(cpu_allocator, dims_op_x, values_op_x, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(cpu_allocator, dims_op_x, values_op_x, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("input1", ml_value_x));
  feeds.insert(std::make_pair("input2", ml_value_y));
  feeds.insert(std::make_pair("input3", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("output");
  std::vector<OrtValue> fetches;

  OrtTensorRTProviderOptionsV2 params;
  std::unique_ptr<IExecutionProvider> execution_provider = TensorrtExecutionProviderWithOptions(&params);
  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());
  std::cout << model_name << std::endl;
  auto status = session_object.Load(model_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
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
    dims = {1, -1, -1};  // dynamic shape input
  } else {
    dims = {1, 3, 2};
  }

  CreateBaseModel(model_name, cache_type + "cachingtest", dims);

  SessionOptions so;
  so.session_logid = "TensorrtExecutionProvider" + cache_type + "cacheTest";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  InferenceSession session_object{so, GetEnvironment()};
  auto cuda_provider = DefaultCudaExecutionProvider();
  auto cpu_allocator = cuda_provider->CreatePreferredAllocators()[1];
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

  OrtTensorRTProviderOptionsV2 params;
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
    params.trt_engine_cache_prefix = "TRTEP_Cache_Test";
    params.trt_dump_ep_context_model = 1;
    params.trt_ep_context_file_path = "EP_Context_model.onnx";
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
    // X: 1, 3, 3, 3, 2, 2, 2, 2
    // Y: 1, 3, 3, 3, 2, 2, 2, 2
    // Z: 1, 3, 3, 3, 2, 2, 2, 2
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
      auto shape_ranges = DeserializeProfileV2(profile_file);

      // check min/max/opt shape ranges of dynamic shape dimensions
      for (auto it = shape_ranges.cbegin(); it != shape_ranges.cend(); ++it) {
        auto ranges = it->second;
        for (auto it2 = ranges.cbegin(); it2 != ranges.cend(); ++it2) {
          if (it2->first == 1) {
            ASSERT_EQ(it2->second[0][0], 3);
            ASSERT_EQ(it2->second[0][1], 3);
            ASSERT_EQ(it2->second[0][2], 3);
          } else if (it2->first == 2) {
            ASSERT_EQ(it2->second[0][0], 2);
            ASSERT_EQ(it2->second[0][1], 2);
            ASSERT_EQ(it2->second[0][2], 2);
          }
        }
      }
    }

    // another inference run with input shape {1, 1, 6}
    // TRT engine and profile will be updated
    // Data in profile,
    // X: 1, 1, 3, 3, 2, 2, 6, 6
    // Y: 1, 1, 3, 3, 2, 2, 6, 6
    // Z: 1, 1, 3, 3, 2, 2, 6, 6
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
      auto shape_ranges2 = DeserializeProfileV2(profile_file2);

      // check min/max/opt shape ranges of dynamic shape dimensions
      for (auto it = shape_ranges2.cbegin(); it != shape_ranges2.cend(); ++it) {
        auto ranges = it->second;
        for (auto it2 = ranges.cbegin(); it2 != ranges.cend(); ++it2) {
          if (it2->first == 1) {
            ASSERT_EQ(it2->second[0][0], 1);
            ASSERT_EQ(it2->second[0][1], 3);
            ASSERT_EQ(it2->second[0][2], 3);
          } else if (it2->first == 2) {
            ASSERT_EQ(it2->second[0][0], 2);
            ASSERT_EQ(it2->second[0][1], 6);
            ASSERT_EQ(it2->second[0][2], 6);
          }
        }
      }
    }

    // Test explicit min/max/opt profile shapes
    // create another session object with TRT EP provider options:
    // trt_profile_min_shapes=X:1x1x1,Y:1x1x1,Z:1x1x1
    // trt_profile_max_shapes=X:1x6x6,Y:1x6x6,Z:1x6x6
    // trt_profile_opt_shapes=X:1x2x3,Y:1x2x3,Z:1x2x3
    //
    // TRT engine and profile will be updated
    // Data in profile,
    // X: 1, 1, 6, 2, 2, 1, 6, 3
    // Y: 1, 1, 6, 2, 2, 1, 6, 3
    // Y: 1, 1, 6, 2, 2, 1, 6, 3
    InferenceSession session_object2{so, GetEnvironment()};
    params.trt_profile_min_shapes = "X:1x1x1,Y:1x1x1,Z:1x1x1";
    params.trt_profile_max_shapes = "X:1x6x6,Y:1x6x6,Z:1x6x6";
    params.trt_profile_opt_shapes = "X:1x2x3,Y:1x2x3,Z:1x2x3";
    std::unique_ptr<IExecutionProvider> execution_provider2 = TensorrtExecutionProviderWithOptions(&params);
    EXPECT_TRUE(session_object2.RegisterExecutionProvider(std::move(execution_provider2)).IsOK());
    status = session_object2.Load(model_name);
    ASSERT_TRUE(status.IsOK());
    status = session_object2.Initialize();
    ASSERT_TRUE(status.IsOK());

    status = session_object2.Run(run_options, feeds, output_names, &fetches);

    // Verify on cache with customized prefix
    ASSERT_TRUE(HasCacheFileWithPrefix(params.trt_engine_cache_prefix));

    // Verify EP context model with user provided name
    ASSERT_TRUE(HasCacheFileWithPrefix(params.trt_ep_context_file_path));

    if (input_type.compare("static") == 0) {
      // Can't run inference since input shape changes but the engine is built with static input
      ASSERT_FALSE(status.IsOK());
    } else {
      ASSERT_TRUE(status.IsOK());
      VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);

      profile_files = GetCachesByType("./", ".profile");
      ASSERT_EQ(profile_files.size(), 1);
      std::ifstream profile_file2(profile_files[0], std::ios::binary | std::ios::in);
      auto shape_ranges2 = DeserializeProfileV2(profile_file2);

      // check min/max/opt shape ranges of dynamic shape dimensions
      for (auto it = shape_ranges2.cbegin(); it != shape_ranges2.cend(); ++it) {
        auto ranges = it->second;
        for (auto it2 = ranges.cbegin(); it2 != ranges.cend(); ++it2) {
          if (it2->first == 1) {
            ASSERT_EQ(it2->second[0][0], 1);
            ASSERT_EQ(it2->second[0][1], 6);
            ASSERT_EQ(it2->second[0][2], 2);
          } else if (it2->first == 2) {
            ASSERT_EQ(it2->second[0][0], 1);
            ASSERT_EQ(it2->second[0][1], 6);
            ASSERT_EQ(it2->second[0][2], 3);
          }
        }
      }
    }

  } else if (cache_type.compare("timing") == 0) {
    /* Following code block tests the functionality of timing cache, including:
     * - timing cache cache serialization/de-serialization
     * - TODO: benefir of usign a timing cache no matter if dynamic / static input
     */

    // Temporarily disable comparing the engine build time until we find the model that can benefit from timing cache to get engine build time reduced.
    // uint64_t compilation_without_cache_ms, compilation_with_cache_ms;

    // First session is created with TRT EP with timing cache enabled
    // Not specifying a trt_timing_cache_path will result in using the working directory
    params.trt_timing_cache_enable = 1;
    {
      // auto start = chrono::steady_clock::now();
      std::unique_ptr<IExecutionProvider> execution_provider = TensorrtExecutionProviderWithOptions(&params);
      EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());
      auto status = session_object.Load(model_name);
      ASSERT_TRUE(status.IsOK());
      status = session_object.Initialize();
      ASSERT_TRUE(status.IsOK());

      status = session_object.Run(run_options, feeds, output_names, &fetches);
      // auto end = chrono::steady_clock::now();
      ASSERT_TRUE(status.IsOK());
      VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);
      ASSERT_TRUE(IsCacheExistedByType("./", ".timing"));
      // compilation_with_cache_ms = chrono::duration_cast<chrono::microseconds>(end - start).count();
    }

    // Second session is created with TRT EP without timing cache enabled
    params.trt_timing_cache_enable = 0;
    {
      InferenceSession session_object_new{so, GetEnvironment()};
      {
        // auto start = chrono::steady_clock::now();
        std::unique_ptr<IExecutionProvider> execution_provider = TensorrtExecutionProviderWithOptions(&params);
        EXPECT_TRUE(session_object_new.RegisterExecutionProvider(std::move(execution_provider)).IsOK());
        auto status = session_object_new.Load(model_name);
        ASSERT_TRUE(status.IsOK());
        status = session_object_new.Initialize();
        ASSERT_TRUE(status.IsOK());

        status = session_object_new.Run(run_options, feeds, output_names, &fetches);
        // TODO narrow down actual compilation section
        // auto end = chrono::steady_clock::now();

        ASSERT_TRUE(status.IsOK());
        VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);
        // compilation_without_cache_ms = chrono::duration_cast<chrono::microseconds>(end - start).count();
      }
    }

    // Temporarily disable comparing the engine build time until we find the model that can benefit from timing cache to get engine build time reduced.
    // ASSERT_TRUE(compilation_with_cache_ms <= compilation_without_cache_ms);
  }

  // clean up caches
  RemoveCachesByType("./", ".timing");
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
 * - timing_static: timing cache enabled, static input shape
 * - timing_dynamic: timing cache enabled, static input shape
 */
INSTANTIATE_TEST_SUITE_P(TensorrtExecutionProviderCacheTests, TensorrtExecutionProviderCacheTest, testing::Values("engine_static", "engine_dynamic", "timing_static", "timing_dynamic"),
                         [](const ::testing::TestParamInfo<TensorrtExecutionProviderCacheTest::ParamType>& info) { return info.param; });

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

  auto cuda_provider = DefaultCudaExecutionProvider();
  auto cpu_allocator = cuda_provider->CreatePreferredAllocators()[1];

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

TEST(TensorrtExecutionProviderTest, DISABLED_NodeIndexMappingTest) {  //  [W:onnxruntime:TensorrtExecutionProviderTest.NodeIndexMappingTest, model_load_utils.h:58 ValidateOpsetForDomain] ONNX Runtime only *guarantees* support for models stamped with official released onnx opset versions. Opset 19 is under development and support for this is limited. The operator schemas and or other functionality could possibly change before next ONNX release and in this case ONNX Runtime will not guarantee backward compatibility. Current official support for domain ai.onnx is till opset 18.
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

  auto cuda_provider = DefaultCudaExecutionProvider();
  auto cpu_allocator = cuda_provider->CreatePreferredAllocators()[1];

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

  auto cuda_provider = DefaultCudaExecutionProvider();
  auto cpu_allocator = cuda_provider->CreatePreferredAllocators()[1];

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
